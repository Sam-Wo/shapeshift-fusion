"""Cheap target probes on the overlap sets (no model training):
 1. PRISM secondary AUC per compound in A549 / U2OS, from the range-read LFC rows (trapezoid on log10 dose,
    viability = 2^LFC clipped to [0, 1.5], normalised to the dose span).
 2. Cross-assay target reliability: GDSC2 AUC vs PRISM secondary AUC vs PRISM primary LFC (A549).
 3. Leakage probe: Tahoe 5 uM treated-cell-count ratio (n_cells_trt / plate x line median) and Tahoe
    signature magnitude vs PRISM / GDSC response.
 4. Target-mean baseline: fraction of target variance explained by MoA class (LOO class mean).
Usage: py -3.13 -I -X utf8 scripts/probe_targets.py  (run after count_overlap.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
SS = HERE.parent / "shapeshift"
S2 = SS / "step2"
DP = HERE / "data_probe"
LN = {"A549": ("SIDM00903", "A549", "CVCL_0023"), "HepG2C3A": ("SIDM01237", "HEPG2", "CVCL_1098"),
      "U2OS": ("SIDM01191", "U2OS", None)}


def unit_map(source):
    m = pd.read_csv(DP / "overlap_matches.csv")
    m = m[(m.source == source) & m.units.notna()]
    return {str(r.drug_id): r.units.split(";") for r in m.itertuples()}


def prism_targets(screen, line):
    f = DP / "prism" / f"{screen}_LFC_{line}.csv"
    if not f.exists():
        return pd.Series(dtype=float)
    t = pd.read_csv(DP / "prism" / f"{screen}-screen-replicate-collapsed-treatment-info.csv")
    row = pd.read_csv(f, index_col=0).iloc[0]
    df = t.set_index("column_name")[["broad_id", "dose", "screen_id"]].join(row.rename("lfc"), how="inner").dropna()
    um = unit_map(f"PRISM_{screen}")
    out = {}
    for b, g in df.groupby("broad_id"):
        if b not in um:
            continue
        if screen == "secondary":
            g = g[g.screen_id == ("MTS010" if (g.screen_id == "MTS010").any() else g.screen_id.iloc[0])]
            g = g.groupby("dose").lfc.median().sort_index()
            if len(g) < 6:
                continue
            x, v = np.log10(g.index.to_numpy(float)), np.clip(2.0 ** g.to_numpy(), 0, 1.5)
            val = np.trapezoid(v, x) / (x[-1] - x[0])
        else:
            val = g.lfc.median()
        for u in um[b]:
            out.setdefault(u, []).append(val)
    return pd.Series({u: float(np.mean(v)) for u, v in out.items()})


def gdsc_targets(line):
    d = pd.read_csv(DP / "gdsc" / f"GDSC2_{line}.csv")
    um = unit_map("GDSC2")
    out = {}
    for r in d.itertuples():
        for u in um.get(str(r.DRUG_ID), []):
            out.setdefault(u, []).append((r.AUC, r.LN_IC50))
    return pd.DataFrame({u: np.mean(v, 0) for u, v in out.items()}, index=["auc", "ln_ic50"]).T


def tahoe_covariates(cvcl):
    pl = pd.read_parquet(S2 / "tahoe_perline.parquet", columns=["unit", "dose_uM", "cell_line", "plates", "n_cells_trt"])
    pl = pl[pl.cell_line == cvcl].copy()
    pl["p1"] = pl.plates.astype(str).str.split(";").str[0]
    pl["ratio"] = pl.n_cells_trt / pl.groupby(["p1", "dose_uM"]).n_cells_trt.transform("median")
    pl = pl[np.isclose(pl.dose_uM.astype(float), 5.0)].set_index("unit")
    return np.log2(pl["ratio"].clip(lower=1e-3))


def corr(a, b):
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(j) < 8:
        return dict(n=len(j))
    r = spearmanr(j.iloc[:, 0], j.iloc[:, 1])
    return dict(n=len(j), rho=round(float(r.statistic), 3), p=float(f"{r.pvalue:.2g}"))


def moa_r2(y, moa):
    j = pd.concat([y.rename("y"), moa.rename("moa")], axis=1, join="inner").dropna()
    j = j[j.moa != "unclear"]
    if len(j) < 10:
        return None
    pred = []
    for i, r in j.iterrows():
        same = j[(j.moa == r.moa) & (j.index != i)].y
        pred.append(same.mean() if len(same) else j.drop(i).y.mean())
    pred = np.array(pred)
    return dict(n=len(j), loo_r2=round(float(1 - np.sum((j.y - pred) ** 2) / np.sum((j.y - j.y.mean()) ** 2)), 3))


def main():
    units = pd.read_csv(S2 / "units.csv")
    moa = units.drop_duplicates("unit").set_index("unit").moa
    res = {}
    T = {}
    for line, (_, pl, cvcl) in LN.items():
        T[(line, "prism_primary_lfc")] = prism_targets("primary", pl)
        T[(line, "prism_secondary_auc")] = prism_targets("secondary", pl)
        g = gdsc_targets(line if line != "HepG2C3A" else "C3A")
        T[(line, "gdsc2_auc")], T[(line, "gdsc2_ln_ic50")] = g.get("auc", pd.Series(dtype=float)), g.get("ln_ic50", pd.Series(dtype=float))
    a = "A549"
    res["A549_cross_assay"] = {
        "gdsc2_auc~prism_sec_auc": corr(T[(a, "gdsc2_auc")], T[(a, "prism_secondary_auc")]),
        "gdsc2_auc~prism_pri_lfc": corr(T[(a, "gdsc2_auc")], T[(a, "prism_primary_lfc")]),
        "prism_sec_auc~prism_pri_lfc": corr(T[(a, "prism_secondary_auc")], T[(a, "prism_primary_lfc")])}
    res["cross_line_prism_primary"] = {
        "A549~HepG2": corr(T[("A549", "prism_primary_lfc")], T[("HepG2C3A", "prism_primary_lfc")]),
        "A549~U2OS": corr(T[("A549", "prism_primary_lfc")], T[("U2OS", "prism_primary_lfc")])}
    res["cross_line_prism_secondary"] = {"A549~U2OS": corr(T[("A549", "prism_secondary_auc")], T[("U2OS", "prism_secondary_auc")])}
    cons = pd.read_csv(S2 / "tahoe_consistency.csv")
    cons = cons[np.isclose(cons.dose_uM, 5.0)].set_index("unit")
    for line in ("A549", "HepG2C3A"):
        cc = tahoe_covariates(LN[line][2])
        mag = pd.read_parquet(S2 / ("tahoe_A549.parquet" if line == "A549" else "tahoe_HepG2C3A.parquet"))
        mag = mag[np.isclose(mag.dose_uM.astype(float), 5.0)].set_index("unit")
        genes = [c for c in mag.columns if c not in ("dose_uM", "cell_line", "cell_name", "tahoe_drugs", "plates", "n_wells", "n_cells_trt")]
        norm = pd.Series(np.linalg.norm(mag[genes].to_numpy(float), axis=1), index=mag.index)
        for k in ("prism_primary_lfc", "prism_secondary_auc", "gdsc2_auc"):
            y = T[(line, k)]
            res[f"leak_{line}_{k}"] = {"tahoe_log2_cellcount_ratio": corr(y, cc), "tahoe_line_sig_norm": corr(y, norm),
                                       "tahoe_consensus_magnitude_centred": corr(y, cons.magnitude_centred)}
    for (line, k), y in T.items():
        if len(y):
            res[f"target_{line}_{k}"] = dict(n=len(y), median=round(float(y.median()), 3), sd=round(float(y.std()), 3),
                                              moa_mean_baseline=moa_r2(y, moa))
    (DP / "probe_targets.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
