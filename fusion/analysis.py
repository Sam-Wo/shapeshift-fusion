"""Arms, CV runs and the pre-registered tests, on any Problem (real or synthetic)."""
import time

import numpy as np

from . import engine as E
from . import splits as SP
from . import stats as ST


def arms(k=5, k_grid=None, morph="morph", tx="tx", residualise=False, interaction=False, scales=None):
    """The pre-registered feature arms of DESIGN.md section 3.1 (ridge learner)."""
    base = dict(moa=True, chem=True, nuis=True)
    kw = dict(k=k, k_grid=k_grid, residualise=residualise, scales=scales)
    mb = tuple(morph) if isinstance(morph, (tuple, list)) else (morph,)
    tb = tuple(tx) if isinstance(tx, (tuple, list)) else (tx,)
    a = {
        "B0": E.Arm("B0", kind="mean"),
        "B1": E.Arm("B1", kind="moa_mean"),
        "B2_svd": E.Arm("B2_svd", chem=True),
        "B2_knn": E.Arm("B2_knn", kind="knn"),
        "B3": E.Arm("B3", nuis=True),
        "BASE": E.Arm("BASE", **base),
        "M": E.Arm("M", **base, blocks=mb, **kw),
        "T": E.Arm("T", **base, blocks=tb, **kw),
        "F": E.Arm("F", **base, blocks=mb + tb, interaction=interaction, **kw),
        "M_only": E.Arm("M_only", blocks=mb, **kw),
        "T_only": E.Arm("T_only", blocks=tb, **kw),
    }
    return a


def plans(groups, compound_groups, moa, cfg, seed):
    """scheme -> list of Split. groups: scaffold-cluster groups."""
    out = {}
    if "scaffold" in cfg["schemes"]:
        out["scaffold"] = SP.make_plan(groups, "kfold", cfg["n_folds"], cfg["n_repeats"], cfg["n_inner"], seed)
    if "loco" in cfg["schemes"]:
        out["loco"] = SP.make_plan(compound_groups, "logo", n_inner=cfg["n_inner"], seed=seed)
    if "moa" in cfg["schemes"]:
        out["moa"] = SP.make_plan(SP.moa_groups(moa, compound_groups), "logo", n_inner=cfg["n_inner"], seed=seed)
    return out


def run_arms(prob, plan, arm_defs, names=None, feat=None):
    """-> dict name -> result dict (oof (R, n), selected, ...). Late-fusion arms are added when M and T run."""
    feat = feat or E.Featurizer(prob)
    names = names or list(arm_defs)
    res = {}
    for nm in names:
        if nm in ("F_late_avg", "F_late_stack"):
            continue
        t0 = time.time()
        res[nm] = E.run_arm(feat, arm_defs[nm], plan, keep_inner=nm in ("M", "T"))
        res[nm]["seconds"] = time.time() - t0
    if "M" in res and "T" in res:
        res["F_late_avg"] = E.late_avg(res["M"], res["T"])
        res["F_late_stack"] = E.late_stack(prob, plan, res["M"], res["T"])
    return res, feat


def r2_table(res, y):
    return {k: float(E.r2(v["oof"], y).mean()) for k, v in res.items()}


# ------------------------------------------------------------------ Freedman-Lane block permutation
def _fl_chunk(prob, plan, arm, block, other, perms, k_other):
    H, _ = E.fl_design(prob, other, k_other=k_other)
    base = E.Featurizer(prob)
    out = []
    for perm in perms:
        Kp = E.fl_gram(prob.grams[block], H, perm)
        f = E.Featurizer(prob, grams={block: Kp}, parent=base)
        out.append(float(E.r2(E.run_ridge(f, arm, plan)["oof"], prob.y).mean()))
    return out


def fl_test(prob, plan, arm_full, r2_reduced, block, other, n_perm, seed, n_jobs=1, k_other=5, chunk=25):
    """Within-MoA Freedman-Lane permutation of one modality block. Returns observed dR2, null dR2 array, p,
    and diagnostics (rank of the regressor matrix, number of permutable compounds)."""
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    perms = [E.strata_perm(prob.moa_code, rng) for _ in range(n_perm)]
    chunks = [perms[i:i + chunk] for i in range(0, n_perm, chunk)]
    obs_full = float(E.r2(E.run_ridge(E.Featurizer(prob), arm_full, plan)["oof"], prob.y).mean())
    if n_jobs == 1:
        vals = [_fl_chunk(prob, plan, arm_full, block, other, c, k_other) for c in chunks]
    else:
        vals = Parallel(n_jobs=n_jobs)(delayed(_fl_chunk)(prob, plan, arm_full, block, other, c, k_other)
                                       for c in chunks)
    null = np.concatenate(vals) - r2_reduced
    obs = obs_full - r2_reduced
    H, rank = E.fl_design(prob, other, k_other=k_other)
    sizes = np.bincount(prob.moa_code)
    return dict(obs=obs, null=null, p=ST.perm_p(obs, null), regressor_rank=rank,
                n_permutable=int(sizes[sizes > 1].sum()), n=prob.n)


# ------------------------------------------------------------------ target permutation
def target_perm(prob, plan, arm_defs, names, n_perm, seed, feat=None, chunk=250):
    """y shuffled (free permutation), whole nested pipeline rerun for each arm; vectorised over permutations.
    Returns name -> null R^2 array."""
    rng = np.random.default_rng(seed)
    Y = np.column_stack([prob.y[rng.permutation(prob.n)] for _ in range(n_perm)])
    feat = feat or E.Featurizer(prob)
    out = {}
    for nm in names:
        vals = []
        for i in range(0, n_perm, chunk):
            Yc = Y[:, i:i + chunk]
            oof = E.run_arm(feat, arm_defs[nm], plan, Y=Yc)["oof"]
            vals.append(E.r2(oof, Yc).mean(0))
        out[nm] = np.concatenate(vals)
    return out
