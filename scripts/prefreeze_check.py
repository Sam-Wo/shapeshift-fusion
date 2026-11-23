"""Pre-freeze checks required by DESIGN.md section 4.4 ("Freezing"), run BEFORE any real target enters a model.

Re-runs the power_sim_rule.py logic on the REAL T1 feature matrices (183 structure-matched units; BASE = MoA
one-hot + Morgan SVD + nuisance, k = 5 PCs per block, in-fold PCA, nested inner CV) with scaffold-grouped folds and
SYNTHETIC targets drawn independently of the real ones:
  y = sqrt(0.30) * z(random linear combination of the global BASE design)
      + sqrt(0.08) * z(unique part of the other block's top-3 global PCs)
      + sqrt(delta) * z(unique part of the tested block's top-3 global PCs) + noise
Checks: the within-MoA Freedman-Lane block permutation has type-I <= 0.06 at delta = 0 (both blocks), and the
power of the full one-block rule (observed dR2 >= 0.03 and p < 0.05) at delta = 0.08.
Cost control (logged in DEVIATIONS.md D-11): one repeat of the 10-fold scaffold split per dataset (the analysis
uses 10 repeats), 99 permutations per dataset as in power_sim_rule.py.
Usage: py -3.13 -I -X utf8 scripts/prefreeze_check.py [n_datasets_null] [n_datasets_power] [n_jobs]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion import analysis as A  # noqa: E402
from fusion import engine as E  # noqa: E402
from fusion import paths  # noqa: E402
from fusion import splits as SP  # noqa: E402
from fusion.problem import tier_problem, target_series  # noqa: E402

N_NULL = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N_POW = int(sys.argv[2]) if len(sys.argv) > 2 else 100
N_JOBS = int(sys.argv[3]) if len(sys.argv) > 3 else 14
NPERM = 99


def z(v):
    return (v - v.mean()) / v.std()


def components(prob):
    n = prob.n
    idx = np.arange(n)
    oh = (prob.moa_code[:, None] == np.unique(prob.moa_code)[None, :]).astype(float)
    ch = E.gram_pca(prob.grams["_chem"], idx, idx, 16, centered=False)[0]
    N = np.where(np.isnan(prob.nuis), np.nanmedian(prob.nuis, 0), prob.nuis)
    base = np.hstack([oh, ch, N])
    base = (base - base.mean(0)) / np.maximum(base.std(0), 1e-12)
    pcs = {b: E.gram_pca(prob.grams[b], idx, idx, 5)[0] for b in ("morph", "tx")}

    def unique(b, other):
        Zr = np.column_stack([np.ones(n), base, pcs[other]])
        v = pcs[b][:, :3] / pcs[b][:, :3].std(0)
        v = v.sum(1)
        return z(v - Zr @ np.linalg.lstsq(Zr, v, rcond=None)[0])

    return base, {"tx": unique("tx", "morph"), "morph": unique("morph", "tx")}


def one(prob, groups, base, uniq, block, delta, seed):
    other = "morph" if block == "tx" else "tx"
    rng = np.random.default_rng(seed)
    y = (np.sqrt(0.30) * z(base @ rng.standard_normal(base.shape[1])) + np.sqrt(0.08) * uniq[other]
         + np.sqrt(delta) * uniq[block] + np.sqrt(1 - 0.30 - 0.08 - delta) * rng.standard_normal(prob.n))
    p = prob.with_y(y)
    plan = SP.make_plan(groups, "kfold", 10, 1, 5, seed=seed)
    defs = A.arms()
    red = "M" if block == "tx" else "T"
    r_red = float(E.r2(E.run_ridge(E.Featurizer(p), defs[red], plan)["oof"], y).mean())
    fl = A.fl_test(p, plan, defs["F"], r_red, block, [other], NPERM, seed + 1, n_jobs=1, chunk=NPERM)
    return dict(d=fl["obs"], p=fl["p"])


def main():
    from joblib import Parallel, delayed

    s, _ = target_series("T1", "prism_primary")
    units = list(s.index)              # availability only: the target VALUES are not used here
    prob, info = tier_problem("T1", units=units)
    groups = info["scaffold_group"]
    base, uniq = components(prob)
    t0 = time.time()
    out = {}
    cells = [("tx", 0.0, N_NULL), ("morph", 0.0, N_NULL), ("tx", 0.08, N_POW), ("morph", 0.08, N_POW)]
    for block, delta, nd in cells:
        rows = Parallel(n_jobs=N_JOBS)(delayed(one)(prob, groups, base, uniq, block, delta, 1000 * i + 7)
                                       for i in range(nd))
        d = np.array([r["d"] for r in rows])
        p = np.array([r["p"] for r in rows])
        rec = dict(n_datasets=nd, n_perm=NPERM, mean_obs_dR2=round(float(d.mean()), 4),
                   P_obs_ge_0p03=float(np.mean(d >= 0.03)), P_perm_p_lt_0p05=float(np.mean(p < 0.05)),
                   P_rule=float(np.mean((d >= 0.03) & (p < 0.05))),
                   mc_se_P_perm=round(float(np.sqrt(np.mean(p < 0.05) * (1 - np.mean(p < 0.05)) / nd)), 4))
        out[f"block={block},delta={delta}"] = rec
        print(block, delta, rec, round(time.time() - t0), "s", flush=True)
    out["setup"] = dict(n=prob.n, tier="T1", base_columns=int(base.shape[1]), k_block=5, folds="scaffold 10-fold x 1",
                        inner="5-fold grouped", seconds=round(time.time() - t0))
    out["type_I_ok"] = bool(all(out[f"block={b},delta=0.0"]["P_perm_p_lt_0p05"] <= 0.06 for b in ("tx", "morph")))
    (paths.results_dir() / "prefreeze_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
