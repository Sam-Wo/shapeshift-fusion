"""Count compound overlap between viability sources (GDSC2, PRISM 19Q4) and ShapeShift step-2 units.

Inputs (all small; see DESIGN.md section 1 for provenance):
  data_probe/gdsc/{screened_compounds_rel_8.5_inchi.csv, GDSC2_fitted_dose_response_27Oct23.xlsx}
  data_probe/prism/{primary,secondary}-screen-replicate-collapsed-treatment-info.csv, *_LFC_<line>.csv
  ../shapeshift/step2/units.csv, tahoe_*.parquet, tahoe_consistency.csv
Matching reuses shapeshift/overlap.py (std_key, norm_name, _pc_get) - nothing is reimplemented.
  GDSC2 ships InChIKeys, not SMILES: InChIKey -> PubChem SMILES -> std_key (cached), plus a direct
  key comparison; name matching (norm_name) is a flagged fallback.
  PRISM ships SMILES: std_key directly; name fallback flagged.
Outputs: data_probe/overlap_matches.csv, data_probe/overlap_counts.csv, data_probe/overlap_summary.json
Usage: py -3.13 -I -X utf8 scripts/count_overlap.py [prism|prism20q2]   (PRISM release dir; default prism = 19Q4)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE = Path(__file__).resolve().parents[1]
SS = HERE.parent / "shapeshift"
sys.path.insert(0, str(SS))
import overlap as OV  # noqa: E402  (std_key, std_mol, norm_name, _pc_get)

DP = HERE / "data_probe"
S2 = SS / "step2"
PC_CACHE = DP / "pubchem_cache_fusion.json"
PRISM = sys.argv[1] if len(sys.argv) > 1 else "prism"
SFX = "" if PRISM == "prism" else "_" + PRISM
LINES = {"A549": ("SIDM00903", "ACH-000681", "CVCL_0023"),
         "HepG2C3A": ("SIDM01237", "ACH-000739", "CVCL_1098"),   # GDSC: C3A itself; PRISM: HepG2 parent
         "U2OS": ("SIDM01191", "ACH-000364", None)}


def split_names(s):
    return [x.strip() for x in str(s).replace(";", ",").split(",") if x.strip() and x.strip().lower() != "nan"]


# ------------------------------------------------------------------ units with morphology + Tahoe 5 uM
def load_units():
    u = pd.read_csv(S2 / "units.csv")
    u = u[u.tier.isin(["T1_LINCS_A549", "T2_EUOS_HepG2", "T3_JUMP_U2OS"])]
    sig = {"T1_LINCS_A549": pd.read_parquet(S2 / "tahoe_A549.parquet", columns=["unit", "dose_uM"]),
           "T2_EUOS_HepG2": pd.read_parquet(S2 / "tahoe_HepG2C3A.parquet", columns=["unit", "dose_uM"]),
           "T3_JUMP_U2OS": pd.read_parquet(SS / "tahoe_signatures.parquet", columns=["unit", "dose_uM"])}
    has5 = {t: set(d.loc[np.isclose(d.dose_uM.astype(float), 5.0), "unit"]) for t, d in sig.items()}
    u["tahoe5"] = [r.unit in has5[r.tier] for r in u.itertuples()]
    cons = pd.read_csv(S2 / "tahoe_consistency.csv")
    two = set(cons.loc[np.isclose(cons.dose_uM, 5.0) & (cons.n_plates >= 2), "unit"])
    u["two_plate_5uM"] = u.unit.isin(two)
    # names for the flagged fallback: Tahoe drug names + dataset names
    tdm = pd.read_parquet(SS / "data" / "tahoe" / "drug_metadata.parquet")
    tdm["unit"] = tdm.canonical_smiles.map(OV.std_key)
    names = {}
    for r in u.itertuples():
        for n in split_names(r.tahoe_drugs) + split_names(r.dataset_names):
            if OV.norm_name(n):
                names.setdefault(OV.norm_name(n), set()).add(r.unit)
    smiles = tdm.dropna(subset=["unit"]).groupby("unit").canonical_smiles.first()
    return u, names, smiles


# ------------------------------------------------------------------ GDSC2
def gdsc_keys(unitset, names):
    sc = pd.read_csv(DP / "gdsc" / "screened_compounds_rel_8.5_inchi.csv")
    cache = json.loads(PC_CACHE.read_text()) if PC_CACHE.exists() else {}
    rows = []
    for r in sc.itertuples():
        ik = r.INCHI_KEY if isinstance(r.INCHI_KEY, str) and len(r.INCHI_KEY) == 27 else None
        keys, how = set(), None
        if ik:
            j = OV._pc_get(cache, f"ik:{ik}", f"{OV.PUBCHEM}/compound/inchikey/{ik}/property/SMILES,IsomericSMILES/JSON")
            for p in (j or {}).get("PropertyTable", {}).get("Properties", []):
                k = OV.std_key(p.get("SMILES") or p.get("IsomericSMILES"))
                if k:
                    keys.add(k)
            if keys & unitset:
                how = "std_key(pubchem smiles from GDSC InChIKey)"
            elif ik in unitset:
                keys.add(ik); how = "inchikey_exact"
        hit = keys & unitset
        if not hit:  # flagged fallback
            for n in [r.DRUG_NAME] + split_names(r.SYNONYMS):
                h = names.get(OV.norm_name(n) or "", set())
                if h:
                    hit, how = h, "NAME_FALLBACK"; break
        rows.append(dict(source="GDSC2", drug_id=r.DRUG_ID, name=r.DRUG_NAME, units=";".join(sorted(hit)),
                         how=how if hit else None, target=r.TARGET, pathway=r.TARGET_PATHWAY))
    PC_CACHE.write_text(json.dumps(cache))
    return pd.DataFrame(rows)


def gdsc_response():
    d = pd.read_excel(DP / "gdsc" / "GDSC2_fitted_dose_response_27Oct23.xlsx")
    return d


# ------------------------------------------------------------------ PRISM
def prism_keys(screen, unitset, names):
    t = pd.read_csv(DP / PRISM / f"{screen}-screen-replicate-collapsed-treatment-info.csv")
    b = t.groupby("broad_id").agg(name=("name", "first"), smiles=("smiles", "first"), moa=("moa", "first")).reset_index()
    rows = []
    for r in b.itertuples():
        keys = {k for s in str(r.smiles).split(", ") if (k := OV.std_key(s))}
        hit, how = keys & unitset, "std_key(prism smiles)"
        if not hit:
            h = names.get(OV.norm_name(r.name) or "", set())
            if h:
                hit, how = h, "NAME_FALLBACK"
        rows.append(dict(source=f"PRISM_{screen}", drug_id=r.broad_id, name=r.name, units=";".join(sorted(hit)),
                         how=how if hit else None, target=None, pathway=r.moa))
    return pd.DataFrame(rows), t


def prism_measured(screen, line, t):
    """broad_id -> (n doses measured, response summary) in that line, from the range-read row."""
    fs = sorted((DP / PRISM).glob(f"{screen}_LFC_{line}.csv")) + sorted((DP / PRISM).glob(f"{screen}_LFC_{line}_*.csv"))
    if not fs:
        return None
    meta = t.set_index("column_name")[["broad_id", "dose", "screen_id"]]
    parts = []
    for f in fs:  # 20Q2 secondary has one row per pool (PR500 / PR300); pools cover different screens
        row = pd.read_csv(f, index_col=0).iloc[0]
        parts.append(meta.join(row.rename("lfc"), how="inner").dropna(subset=["lfc"]))
    df = pd.concat(parts)
    df = df[~df.index.duplicated()]
    if screen == "secondary":  # prefer the MTS010 redo where available (readme recommendation)
        pref = df.groupby("broad_id").screen_id.transform(lambda s: "MTS010" if (s == "MTS010").any() else s.iloc[0])
        df = df[df.screen_id == pref]
    g = df.groupby("broad_id").agg(n_doses=("lfc", "size"), min_lfc=("lfc", "min"), lfc_top=("lfc", "last"))
    return g


def scaffold(smi):
    m = OV.std_mol(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None


# ------------------------------------------------------------------ main
def main():
    u, names, smiles = load_units()
    unitset = set(u.unit)
    g = gdsc_keys(unitset, names)
    gd = gdsc_response()
    p1, t1 = prism_keys("primary", unitset, names)
    p2, t2 = prism_keys("secondary", unitset, names)
    allm = pd.concat([g, p1, p2], ignore_index=True)
    allm.to_csv(DP / f"overlap_matches{SFX}.csv", index=False)

    # unit -> response availability per source x line
    resp = {}
    for line, (sidm, ach, _) in LINES.items():
        sub = gd[gd.SANGER_MODEL_ID == sidm]
        ids = set(sub.DRUG_ID)
        resp[("GDSC2", line)] = {}
        for r in g[g.units.astype(bool) & g.drug_id.isin(ids)].itertuples():
            rr = sub[sub.DRUG_ID == r.drug_id].iloc[0]
            for un in r.units.split(";"):
                resp[("GDSC2", line)].setdefault(un, []).append(
                    dict(how=r.how, ln_ic50=rr.LN_IC50, auc=rr.AUC, max_conc=rr.MAX_CONC))
        for screen, pk, t in (("primary", p1, t1), ("secondary", p2, t2)):
            pl = {"HepG2C3A": "HEPG2"}.get(line, line)
            meas = prism_measured(screen, pl, t)
            key = (f"PRISM_{screen}", line)
            resp[key] = {}
            if meas is None:
                continue
            for r in pk[pk.units.astype(bool)].itertuples():
                if r.drug_id in meas.index and (screen == "primary" or meas.loc[r.drug_id, "n_doses"] >= 6):
                    for un in r.units.split(";"):
                        resp[key].setdefault(un, []).append(dict(how=r.how, min_lfc=meas.loc[r.drug_id, "min_lfc"]))

    routes = [("T1_LINCS_A549", "A549"), ("T2_EUOS_HepG2", "HepG2C3A"), ("T3_JUMP_U2OS", "U2OS"),
              ("T3_JUMP_U2OS", "A549"), ("T3_JUMP_U2OS", "HepG2C3A")]
    out = []
    for tier, line in routes:
        tu = u[(u.tier == tier) & u.tahoe5]
        for src in ("GDSC2", "PRISM_primary", "PRISM_secondary"):
            R = resp[(src, line)]
            m = tu[tu.unit.isin(R)]
            struct = m[[any(x["how"] != "NAME_FALLBACK" for x in R[un]) for un in m.unit]]
            ann = (m.moa.fillna("unclear") != "unclear")
            vc = m.moa[ann].value_counts()
            scaf = {scaffold(smiles.get(un)) for un in m.unit}
            rec = dict(tier=tier, target_line=line, source=src, n_units_tier_tahoe5=len(tu), n_overlap=len(m),
                       n_overlap_structure_only=len(struct), n_name_fallback=len(m) - len(struct),
                       n_two_plate_5uM=int(m.two_plate_5uM.sum()), n_annotated=int(ann.sum()),
                       n_moa_classes_ge3=int((vc >= 3).sum()), n_scaffolds=len(scaf - {None}))
            if src == "GDSC2" and len(m):
                v = [R[un][0] for un in m.unit]
                rec["frac_ic50_below_max_conc"] = float(np.mean([x["ln_ic50"] < np.log(x["max_conc"]) for x in v]))
                rec["auc_median"] = float(np.median([x["auc"] for x in v]))
                rec["auc_sd"] = float(np.std([x["auc"] for x in v]))
            if src.startswith("PRISM") and len(m):
                v = np.array([R[un][0]["min_lfc"] for un in m.unit])
                rec["frac_min_lfc_below_-1"] = float(np.mean(v < -1))
                rec["min_lfc_sd"] = float(v.std())
            out.append(rec)
    # union of viability sources per route (GDSC2 or PRISM secondary)
    for tier, line in routes:
        tu = u[(u.tier == tier) & u.tahoe5]
        anyv = tu.unit.isin(set(resp[("GDSC2", line)]) | set(resp[("PRISM_secondary", line)]))
        both = tu.unit.isin(set(resp[("GDSC2", line)]) & set(resp[("PRISM_secondary", line)]))
        out.append(dict(tier=tier, target_line=line, source="GDSC2_or_PRISMsec", n_units_tier_tahoe5=len(tu),
                        n_overlap=int(anyv.sum()), n_two_plate_5uM=int((anyv & tu.two_plate_5uM).sum()),
                        note=f"both sources: {int(both.sum())}"))
    c = pd.DataFrame(out)
    c.to_csv(DP / f"overlap_counts{SFX}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(c.to_string())
    summ = dict(
        gdsc2_compounds_matched_to_any_unit=int(g.units.astype(bool).sum()),
        gdsc2_match_how=g.how.value_counts().to_dict(),
        prism_primary_matched=int(p1.units.astype(bool).sum()), prism_primary_how=p1.how.value_counts().to_dict(),
        prism_secondary_matched=int(p2.units.astype(bool).sum()), prism_secondary_how=p2.how.value_counts().to_dict(),
        gdsc2_lines={k: int((gd.SANGER_MODEL_ID == v[0]).sum()) for k, v in LINES.items()},
        units_union=int(len(unitset)))
    (DP / f"overlap_summary{SFX}.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
