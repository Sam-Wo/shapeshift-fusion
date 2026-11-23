"""Gradient-boosted trees, 'the baseline to beat' (DESIGN.md section 3.2), with LightGBM (DEVIATIONS.md D-02).

Features per split, all fitted on the training rows: MoA one-hot, Morgan SVD (k=16), nuisance covariates (raw;
LightGBM handles missing values), in-fold PCA scores of the arm's modality blocks (k=5), raw Morgan bits.
Grid (32 points, nested 5-fold inner CV, no early stopping):
  max_depth {2, 3} (num_leaves = 2**depth), learning_rate {0.03, 0.1}, n_estimators {100, 300},
  min_child_samples {10, 20}, reg_lambda {0, 1}.
n_estimators is evaluated from one 300-tree fit per remaining grid point (boosting is sequential, so the first
100 trees of a 300-tree fit are exactly the 100-tree model).
"""
import itertools

import numpy as np

GRID = [dict(max_depth=d, num_leaves=2 ** d, learning_rate=lr, min_child_samples=mcs, reg_lambda=l2)
        for d, lr, mcs, l2 in itertools.product((2, 3), (0.03, 0.1), (10, 20), (0.0, 1.0))]
N_ITERS = (100, 300)
FIXED = dict(objective="regression", n_estimators=300, subsample=1.0, colsample_bytree=1.0, min_split_gain=0.0,
             min_child_weight=1e-3, verbose=-1, n_jobs=1, deterministic=True, force_col_wise=True, seed=0)


def features(feat, S, E, skey, blocks, k=5, k_chem=16):
    P = feat.p
    xs, xe = [], []
    a, b = feat.moa(S, E)
    xs.append(a), xe.append(b)
    a, b = feat.chem(S, E, skey)
    xs.append(a[:, :k_chem]), xe.append(b[:, :k_chem])
    xs.append(P.nuis[S]), xe.append(P.nuis[E])
    for blk in blocks:
        a, b = feat.pcs(blk, S, E, skey, k)
        xs.append(a[:, :k]), xe.append(b[:, :k])
    xs.append(P.bits[S]), xe.append(P.bits[E])
    return np.hstack(xs), np.hstack(xe)


def _fit_predict(Xs, ys, Xe, only=None):
    """-> (len(GRID) * len(N_ITERS), |E|) predictions, in grid-major order (rows of other grid points are NaN
    when only= a single grid index)."""
    import lightgbm as lgb

    ds = lgb.Dataset(Xs, ys, params=dict(feature_pre_filter=False, min_data_in_bin=1, max_bin=63, verbose=-1),
                     free_raw_data=False)
    out = np.full((len(GRID) * len(N_ITERS), len(Xe)), np.nan)
    for gi, g in enumerate(GRID):
        if only is not None and gi != only:
            continue
        params = {**FIXED, **g}
        n_est = params.pop("n_estimators")
        params["lambda_l2"] = params.pop("reg_lambda")
        params["min_data_in_leaf"] = params.pop("min_child_samples")
        params["min_sum_hessian_in_leaf"] = params.pop("min_child_weight")
        params["bagging_fraction"] = params.pop("subsample")
        params["feature_fraction"] = params.pop("colsample_bytree")
        params["min_gain_to_split"] = params.pop("min_split_gain")
        bst = lgb.train(params, ds, num_boost_round=n_est)
        for j, it in enumerate(N_ITERS):
            out[gi * len(N_ITERS) + j] = bst.predict(Xe, num_iteration=it)
    return out


def run_split(feat, s, si, blocks):
    y = feat.p.y
    sse = np.zeros(len(GRID) * len(N_ITERS))
    for j, (tr, va) in enumerate(s.inner):
        Xs, Xe = features(feat, tr, va, (si, j, "gbm"), blocks)
        sse += ((_fit_predict(Xs, y[tr], Xe) - y[va][None]) ** 2).sum(1)
    best = int(sse.argmin())
    g, it = divmod(best, len(N_ITERS))
    Xs, Xe = features(feat, s.train, s.test, (si, -1, "gbm"), blocks)
    pred = _fit_predict(Xs, y[s.train], Xe, only=g)[best]
    return pred, dict(GRID[g], n_estimators=N_ITERS[it])
