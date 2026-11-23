"""Viability targets (DESIGN.md section 1.5). All PRISM values are from the 20Q2 release (CC BY 4.0).

Matching reuses the design-stage structure matches (data_probe/overlap_matches_prism20q2.csv, produced by
scripts/count_overlap.py with overlap.std_key); name-fallback matches are excluded. Only the ID columns of that
file are read (its name/pathway columns carry Repurposing Hub annotations and are never used or written).

  primary_lfc     PRISM primary replicate-collapsed LFC at 2.5 uM; HTS columns only where present, otherwise the
                  remaining screens (MTS004); median over the unit's columns
  residual_lfc    the same after subtracting, per column, the median LFC over all PRISM cell lines
  secondary_auc   official 'auc' from the secondary curve-parameter file, MTS010 rows preferred; median
  secondary_lfc   secondary replicate-collapsed LFC at a given dose (2.5 or 10 uM), MTS010 preferred; median
  gdsc_auc        GDSC2 AUC (A549); exploratory external check only, never written per compound
"""
import numpy as np
import pandas as pd

from . import paths

LINE = {"A549": "ACH-000681", "HEPG2": "ACH-000739", "U2OS": "ACH-000364"}
STRUCT = "std_key(prism smiles)"


def _matches(source):
    m = pd.read_csv(paths.probe_dir() / "overlap_matches_prism20q2.csv", usecols=["source", "drug_id", "units", "how"])
    ok = (m.how != "NAME_FALLBACK") if source == "GDSC2" else (m.how == STRUCT)
    m = m[(m.source == source) & m.units.notna() & ok]
    out = {}
    for r in m.itertuples():
        for un in str(r.units).split(";"):
            out.setdefault(un, set()).add(str(r.drug_id))
    return out


def _row(f):
    return pd.read_csv(f, index_col=0).iloc[0]


def primary_lfc(line, units, residual=False):
    """-> (Series unit -> value, info dict with screen accounting)."""
    P = paths.probe_dir() / "prism20q2"
    t = pd.read_csv(P / "primary-screen-replicate-collapsed-treatment-info.csv",
                    usecols=["broad_id", "dose", "column_name", "screen_id"])
    row = _row(P / f"primary_LFC_{line}.csv")
    if residual:
        med = pd.read_csv(paths.data_dir() / "prism20q2" / "primary_line_median.csv").set_index("column_name")
        row = row - med.median_lfc_all_lines.reindex(row.index)
    t = t.set_index("column_name").join(row.rename("lfc"), how="inner").dropna(subset=["lfc"])
    t = t[np.abs(np.log(t.dose.astype(float) / 2.5)) < 0.2]      # nominal 2.5 uM only (drops one 0.94 uM column)
    um = _matches("PRISM_primary")
    vals, acct = {}, dict(hts_only=0, hts_and_other=0, other_only=0, multi_column=0, doses=[])
    for un in units:
        g = t[t.broad_id.isin(um.get(un, ()))]
        if g.empty:
            continue
        hts = g[g.screen_id == "HTS"]
        if len(hts) and len(hts) < len(g):
            acct["hts_and_other"] += 1
        elif len(hts):
            acct["hts_only"] += 1
        else:
            acct["other_only"] += 1
        use = hts if len(hts) else g
        acct["multi_column"] += int(len(use) > 1)
        acct["doses"].extend(use.dose.tolist())
        vals[un] = float(use.lfc.median())
    d = np.array(acct.pop("doses"), float)
    acct.update(n=len(vals), dose_min=float(d.min()), dose_max=float(d.max()))
    return pd.Series(vals, dtype=float), acct


def secondary_auc(line, units):
    c = pd.read_csv(paths.data_dir() / "prism20q2" / "secondary_curve_params_targets.csv")
    c = c[(c.depmap_id == LINE[line]) & c.auc.notna()]
    um = _matches("PRISM_secondary")
    vals, n10 = {}, 0
    for un in units:
        g = c[c.broad_id.isin(um.get(un, ()))]
        if g.empty:
            continue
        g10 = g[g.screen_id == "MTS010"]
        n10 += int(len(g10) > 0)
        vals[un] = float((g10 if len(g10) else g).auc.median())
    return pd.Series(vals, dtype=float), dict(n=len(vals), n_mts010=n10)


def secondary_lfc(line, units, dose):
    P = paths.probe_dir() / "prism20q2"
    t = pd.read_csv(P / "secondary-screen-replicate-collapsed-treatment-info.csv",
                    usecols=["broad_id", "dose", "column_name", "screen_id"])
    pool = "PR500" if line == "A549" else "PR300"
    row = _row(P / f"secondary_LFC_{line}_{pool}.csv")
    t = t.set_index("column_name").join(row.rename("lfc"), how="inner").dropna(subset=["lfc"])
    t = t[np.abs(np.log(t.dose.astype(float) / dose)) < 0.2]
    um = _matches("PRISM_secondary")
    vals = {}
    for un in units:
        g = t[t.broad_id.isin(um.get(un, ()))]
        if g.empty:
            continue
        g10 = g[g.screen_id == "MTS010"]
        vals[un] = float((g10 if len(g10) else g).lfc.median())
    return pd.Series(vals, dtype=float), dict(n=len(vals), dose=dose)


def gdsc_auc(units):
    d = pd.read_csv(paths.probe_dir() / "gdsc" / "GDSC2_A549.csv", usecols=["DRUG_ID", "AUC"])
    um = _matches("GDSC2")
    vals = {}
    for un in units:
        v = d[d.DRUG_ID.astype(str).isin(um.get(un, ()))].AUC
        if len(v):
            vals[un] = float(v.mean())
    return pd.Series(vals, dtype=float)


TARGETS = {
    "prism_primary": lambda line, units: primary_lfc(line, units),
    "prism_primary_residual": lambda line, units: primary_lfc(line, units, residual=True),
    "prism_secondary_auc": lambda line, units: secondary_auc(line, units),
    "prism_secondary_lfc_2.5": lambda line, units: secondary_lfc(line, units, 2.5),
    "prism_secondary_lfc_10": lambda line, units: secondary_lfc(line, units, 10.0),
}


def availability(line, units, target):
    """Units with a value for the target, without returning the values (used before freezing)."""
    s, info = TARGETS[target](line, units)
    return list(s.index), info
