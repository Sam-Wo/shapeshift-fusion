"""Run a full analysis (one tier x target x variant): ridge arms, GBM, NN, permutation tests, bootstrap CIs.

Per-compound out-of-fold predictions go to FUSION_RUNS (gitignored); only aggregates are returned.
"""
import json
import time

import numpy as np

from . import analysis as A
from . import engine as E
from . import gbm as G
from . import paths
from . import stats as ST

GBM_ARMS = {"GBM_BASE": (), "GBM_M": ("morph",), "GBM_T": ("tx",), "GBM_F": ("morph", "tx")}


def _gbm_chunk(prob, plan, idxs, arms):
    f = E.Featurizer(prob)
    out = []
    for si in idxs:
        rec = {}
        for nm, blocks in arms.items():
            rec[nm] = G.run_split(f, plan[si], si, blocks)
        out.append((si, rec))
    return out


def run_gbm(prob, plan, arms=GBM_ARMS, n_jobs=15, block_map=None):
    from joblib import Parallel, delayed

    if block_map:
        def _m(b):
            v = block_map.get(b, b)
            return tuple(v) if isinstance(v, (list, tuple)) else (v,)
        arms = {k: tuple(x for b in v for x in _m(b)) for k, v in arms.items()}
    R = max(s.repeat for s in plan) + 1
    order = list(range(len(plan)))
    chunks = [order[i::n_jobs * 2] for i in range(min(len(plan), n_jobs * 2))]
    vals = Parallel(n_jobs=n_jobs)(delayed(_gbm_chunk)(prob, plan, c, arms) for c in chunks)
    res = {nm: dict(oof=np.full((R, prob.n), np.nan), selected=[None] * len(plan)) for nm in arms}
    for part in vals:
        for si, rec in part:
            for nm, (pred, sel) in rec.items():
                res[nm]["oof"][plan[si].repeat, plan[si].test] = pred
                res[nm]["selected"][si] = sel
    return res


def _nn_chunk(prob, plan, idxs, groups, seeds):
    from . import nn as NN

    f = E.Featurizer(prob)
    return [(si,) + NN.run_split(f, plan[si], si, groups, seeds=seeds) for si in idxs]


def run_nn(prob, plan, groups, seeds=(0, 1, 2, 3, 4), n_jobs=15):
    from joblib import Parallel, delayed

    R = max(s.repeat for s in plan) + 1
    order = list(range(len(plan)))
    chunks = [order[i::n_jobs] for i in range(min(len(plan), n_jobs))]
    vals = Parallel(n_jobs=n_jobs)(delayed(_nn_chunk)(prob, plan, c, groups, seeds) for c in chunks)
    names = {"full": "NN_F", "zt_zeroed": "NN_F_zt0", "zm_zeroed": "NN_F_zm0"}
    res = {v: dict(oof=np.full((R, prob.n), np.nan), selected=[None] * len(plan)) for v in names.values()}
    for part in vals:
        for si, out, cfg in part:
            for k, v in names.items():
                res[v]["oof"][plan[si].repeat, plan[si].test] = out[k]
                res[v]["selected"][si] = cfg
    return res


def arm_table(res, y, boot, tperm=None, pos=None):
    rows = []
    for nm, r in res.items():
        r2r = E.r2(r["oof"], y)
        lo, hi = ST.ci(boot[nm])
        row = dict(arm=nm, R2=float(r2r.mean()), R2_sd_repeats=float(r2r.std()), R2_ci_lo=lo, R2_ci_hi=hi,
                   spearman=ST.spearman_oof(r["oof"], y))
        if tperm is not None and nm in tperm:
            row["target_perm_p"] = ST.perm_p(row["R2"], tperm[nm])
            row["target_perm_null_q95"] = float(np.quantile(tperm[nm], 0.95))
        if pos is not None and pos.sum() >= 3:
            sc = -r["oof"].mean(0)
            row["auroc_lfc_lt_-1"] = ST.auroc(sc, pos)
            row["auprc_lfc_lt_-1"] = ST.auprc(sc, pos)
            row["n_pos"] = int(pos.sum())
        rows.append(row)
    return rows


def delta(boot, res, y, full, reduced):
    d = float(E.r2(res[full]["oof"], y).mean() - E.r2(res[reduced]["oof"], y).mean())
    b = boot[full] - boot[reduced]
    lo, hi = ST.ci(b)
    return dict(contrast=f"{full} - {reduced}", dR2=d, ci_lo=lo, ci_hi=hi, lb95_one_sided=float(np.quantile(b, 0.05)),
                ub95_one_sided=float(np.quantile(b, 0.95)))


def save_oof(name, res, info, y):
    d = paths.runs_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "oof.npz", y=y, **{k: v["oof"] for k, v in res.items()})
    (d / "units.json").write_text(json.dumps(info["units"]))


def run_analysis(name, prob, info, arm_defs, plan, cfg, ridge_names=None, do_gbm=True, do_nn=False, n_perm_block=0,
                 n_perm_target=0, n_jobs=15, block_map=None, pos=None, log=print):
    """Returns dict(arms=[rows], deltas=[...], fl=..., timing=...)."""
    t0 = time.time()
    timing = {}
    res, feat = A.run_arms(prob, plan, arm_defs, names=ridge_names)
    timing["ridge"] = time.time() - t0
    log(f"[{name}] ridge done {timing['ridge']:.0f}s " +
        " ".join(f"{k}={v:.3f}" for k, v in A.r2_table(res, prob.y).items()))
    if do_gbm:
        t1 = time.time()
        res.update(run_gbm(prob, plan, n_jobs=n_jobs, block_map=block_map))
        timing["gbm"] = time.time() - t1
        log(f"[{name}] gbm done {timing['gbm']:.0f}s " +
            " ".join(f"{k}={E.r2(res[k]['oof'], prob.y).mean():.3f}" for k in GBM_ARMS))
    if do_nn:
        t1 = time.time()
        res.update(run_nn(prob, plan, info["scaffold_group"], seeds=tuple(cfg.get("nn_seeds", (0, 1, 2, 3, 4))),
                          n_jobs=n_jobs))
        timing["nn"] = time.time() - t1
        log(f"[{name}] nn done {timing['nn']:.0f}s " +
            " ".join(f"{k}={E.r2(res[k]['oof'], prob.y).mean():.3f}" for k in ("NN_F", "NN_F_zt0", "NN_F_zm0")))
    boot = ST.bootstrap({k: v["oof"] for k, v in res.items()}, prob.y, info["compound_group"],
                        cfg.get("n_boot", 2000), seed=cfg["seed"])
    tperm = None
    if n_perm_target:
        t1 = time.time()
        names = [n for n in res if n in arm_defs and arm_defs[n].kind in ("ridge", "moa_mean")]
        tperm = A.target_perm(prob, plan, arm_defs, names, n_perm_target, cfg["seed"] + 7, feat=feat)
        timing["target_perm"] = time.time() - t1
        log(f"[{name}] target permutations done {timing['target_perm']:.0f}s")
    out = dict(name=name, n=int(prob.n), arms=arm_table(res, prob.y, boot, tperm, pos))
    dl = [delta(boot, res, prob.y, "F", "M"), delta(boot, res, prob.y, "F", "T"),
          delta(boot, res, prob.y, "M", "BASE"), delta(boot, res, prob.y, "T", "BASE"),
          delta(boot, res, prob.y, "F", "BASE")]
    for a, b in (("F_late_avg", "M"), ("F_late_avg", "T"), ("F_late_stack", "M"), ("F_late_stack", "T")):
        if a in res:
            dl.append(delta(boot, res, prob.y, a, b))
    if do_gbm:
        dl += [delta(boot, res, prob.y, "GBM_F", "GBM_M"), delta(boot, res, prob.y, "GBM_F", "GBM_T"),
               delta(boot, res, prob.y, "GBM_M", "GBM_BASE"), delta(boot, res, prob.y, "GBM_T", "GBM_BASE")]
    if do_nn:
        dl += [delta(boot, res, prob.y, "NN_F", "NN_F_zt0"), delta(boot, res, prob.y, "NN_F", "NN_F_zm0")]
    mn = np.minimum(boot["F"] - boot["M"], boot["F"] - boot["T"])
    out["min_delta_ub95_one_sided"] = float(np.quantile(mn, 0.95))
    out["min_delta_ci"] = ST.ci(mn)
    out["deltas"] = dl
    if n_perm_block:
        t1 = time.time()
        bm = block_map or {}
        mb, tb = bm.get("morph", "morph"), bm.get("tx", "tx")
        r2 = A.r2_table(res, prob.y)
        flT = A.fl_test(prob, plan, arm_defs["F"], r2["M"], tb, [mb] if isinstance(mb, str) else list(mb),
                        n_perm_block, cfg["seed"] + 11, n_jobs=n_jobs)
        flM = A.fl_test(prob, plan, arm_defs["F"], r2["T"], mb, [tb], n_perm_block, cfg["seed"] + 13, n_jobs=n_jobs) \
            if isinstance(mb, str) else None
        out["fl"] = {"tx": {k: v for k, v in flT.items() if k != "null"}}
        out["fl_null"] = {"tx": flT["null"].tolist()}
        if flM is not None:
            out["fl"]["morph"] = {k: v for k, v in flM.items() if k != "null"}
            out["fl_null"]["morph"] = flM["null"].tolist()
        timing["fl"] = time.time() - t1
        log(f"[{name}] FL done {timing['fl']:.0f}s  p_tx={flT['p']:.4f}" + (f" p_morph={flM['p']:.4f}" if flM else ""))
    out["selected"] = {k: v["selected"][:3] for k, v in res.items() if v.get("selected")}
    timing["total"] = time.time() - t0
    out["timing_s"] = timing
    save_oof(name, res, info, prob.y)
    return out
