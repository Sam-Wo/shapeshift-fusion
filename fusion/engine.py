"""Nested-CV engine for the ridge arms and the simple baselines (DESIGN.md sections 2.4, 3.1, 3.4).

Every transform is fitted on the training rows of a split only and then applied to the held-out rows:
  - per-modality PCA (centred) and the Morgan TruncatedSVD (uncentred) are computed from the Gram matrix
    restricted to the training rows (exact linear-kernel PCA: identical scores to sklearn PCA/TruncatedSVD up to
    component sign, which ridge is invariant to); held-out rows are projected onto the training components;
  - MoA one-hot uses the levels present in the training rows (unseen levels encode as all-zero);
  - nuisance covariates are median-imputed with training medians;
  - every column is standardised with training mean/SD;
  - ridge alpha (and the SVD k, block k, block scale when gridded) are chosen by inner grouped CV on the
    outer-training rows only, then refitted on all outer-training rows.
Working from Gram matrices makes a whole nested CV cheap enough to rerun 1,000x inside a permutation test.
tests/test_engine.py checks these transforms against sklearn PCA / TruncatedSVD / Ridge Pipelines.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

ALPHAS = np.logspace(0, 4, 8)


# ------------------------------------------------------------------ problem
class Problem:
    """All inputs for n compounds. Nothing in here depends on the target except ``y``."""

    def __init__(self, y, moa, nuis, bits, blocks, covars=None, extra=None):
        self.y = np.asarray(y, float)
        self.n = len(self.y)
        self.moa = np.asarray(moa, dtype=object)
        self.moa_levels, self.moa_code = np.unique(self.moa, return_inverse=True)
        self.nuis = np.asarray(nuis, float)
        self.bits = np.asarray(bits, np.float64)
        self.blocks = {k: np.asarray(v, np.float64) for k, v in blocks.items()}
        self.covars = covars or {}          # block -> nuisance column indices used for residualisation
        self.extra = {k: np.asarray(v, float) for k, v in (extra or {}).items()}   # raw standardised blocks
        self.grams = {k: v @ v.T for k, v in self.blocks.items()}
        self.grams["_chem"] = self.bits @ self.bits.T

    def subset(self, idx):
        idx = np.asarray(idx)
        return Problem(self.y[idx], self.moa[idx], self.nuis[idx], self.bits[idx],
                       {k: v[idx] for k, v in self.blocks.items()}, self.covars,
                       {k: v[idx] for k, v in self.extra.items()})

    def with_y(self, y):
        p = object.__new__(Problem)
        p.__dict__.update(self.__dict__)
        p.y = np.asarray(y, float)
        return p


# ------------------------------------------------------------------ fold-wise transforms
def gram_pca(K, S, E, k, centered=True):
    """Top-k PCA (centred) or TruncatedSVD (uncentred) scores of rows S (fit) and E (projected), from the Gram
    matrix K = X X^T. Uses only K[S, S] to fit and K[E, S] to project."""
    Kss = K[np.ix_(S, S)]
    Kes = K[np.ix_(E, S)]
    if centered:
        cm = Kss.mean(0)
        tm = cm.mean()
        Kc = Kss - cm[None, :] - cm[:, None] + tm
        Kec = Kes - Kes.mean(1, keepdims=True) - cm[None, :] + tm
    else:
        Kc, Kec = Kss, Kes
    m = len(S)
    k = min(k, m - 1)
    w, U = eigh(Kc, subset_by_index=[m - k, m - 1])
    w, U = w[::-1], U[:, ::-1]
    ok = w > 1e-10 * max(w[0], 1e-300)
    w = np.where(ok, w, 1.0)
    Zs = U * np.sqrt(w) * ok
    Ze = (Kec @ U) / np.sqrt(w) * ok
    return Zs, Ze


def standardize(As, Ae):
    mu = As.mean(0)
    sd = As.std(0)
    keep = sd > 1e-12
    return (As[:, keep] - mu[keep]) / sd[keep], (Ae[:, keep] - mu[keep]) / sd[keep]


def resid_on(Zs, Ze, Cs, Ce):
    """Residualise block scores on covariates with OLS fitted on the training rows."""
    Cs1 = np.column_stack([np.ones(len(Cs)), Cs])
    Ce1 = np.column_stack([np.ones(len(Ce)), Ce])
    B = np.linalg.lstsq(Cs1, Zs, rcond=None)[0]
    return Zs - Cs1 @ B, Ze - Ce1 @ B


class Featurizer:
    """Caches fold-wise transforms per (block, split-set key). ``grams`` overrides block Gram matrices (used by
    the block-permutation test); overridden blocks are cached separately from the parent's cache."""

    def __init__(self, prob, grams=None, parent=None):
        self.p = prob
        self.grams = dict(prob.grams)
        self.override = set()
        if grams:
            self.grams.update(grams)
            self.override = set(grams)
        self.cache = {} if parent is None else parent.cache
        self.local = {}

    def _get(self, key, fn):
        store = self.local if key[0] in self.override else self.cache
        if key not in store:
            store[key] = fn()
        return store[key]

    def pcs(self, block, S, E, skey, kmax=32):
        return self._get((block, skey, kmax), lambda: gram_pca(self.grams[block], S, E, kmax, centered=True))

    def chem(self, S, E, skey, kmax=32):
        return self._get(("_chem", skey, kmax), lambda: gram_pca(self.grams["_chem"], S, E, kmax, centered=False))

    def moa(self, S, E):
        lev = np.unique(self.p.moa_code[S])
        return (self.p.moa_code[S][:, None] == lev[None, :]).astype(float), \
            (self.p.moa_code[E][:, None] == lev[None, :]).astype(float)

    def nuis(self, S, E, cols=None):
        N = self.p.nuis if cols is None else self.p.nuis[:, cols]
        med = np.nanmedian(N[S], 0)
        med = np.where(np.isfinite(med), med, 0.0)
        fs = np.where(np.isnan(N[S]), med, N[S])
        fe = np.where(np.isnan(N[E]), med, N[E])
        return fs, fe

    def extra(self, name, S, E):
        X = self.p.extra[name]
        X = np.where(np.isfinite(X), X, np.nan)
        med = np.nanmedian(X[S], 0)
        med = np.where(np.isfinite(med), med, 0.0)
        return np.where(np.isnan(X[S]), med, X[S]), np.where(np.isnan(X[E]), med, X[E])


# ------------------------------------------------------------------ arms
@dataclass
class Arm:
    name: str
    kind: str = "ridge"                 # ridge | mean | moa_mean | knn | late_avg | late_stack
    moa: bool = False
    chem: bool = False
    nuis: bool = False
    blocks: tuple = ()                  # block names; each gets in-fold PCA with k components
    k: int = 5
    k_grid: tuple = None                # inner-CV choice of k (shared by all blocks of the arm)
    chem_k_grid: tuple = (16, 32)
    scales: tuple = None                # banded ridge: per-block penalty scale grid
    interaction: bool = False
    residualise: bool = False
    extra: tuple = ()
    knn_grid: tuple = (3, 5, 10)
    parts: tuple = ()                   # late fusion: names of the two arms combined
    meta: dict = field(default_factory=dict)

    @property
    def base(self):
        return self.moa or self.chem or self.nuis


def grid_of(arm):
    kc = arm.chem_k_grid if arm.chem else (0,)
    kb = arm.k_grid if (arm.k_grid and arm.blocks) else ((arm.k,) if arm.blocks else (0,))
    if arm.scales and arm.blocks:
        import itertools
        sc = list(itertools.product(arm.scales, repeat=len(arm.blocks)))
    else:
        sc = [None]
    return [(a, b, c) for a in kc for b in kb for c in sc]


def design(feat, arm, S, E, skey, kc, kb, scales):
    P = feat.p
    xs, xe = [], []
    if arm.moa:
        a, b = feat.moa(S, E)
        xs.append(a), xe.append(b)
    if arm.chem:
        a, b = feat.chem(S, E, skey)
        xs.append(a[:, :kc]), xe.append(b[:, :kc])
    if arm.nuis:
        a, b = feat.nuis(S, E)
        xs.append(a), xe.append(b)
    for name in arm.extra:
        a, b = feat.extra(name, S, E)
        xs.append(a), xe.append(b)
    Xs, Xe = standardize(np.hstack(xs), np.hstack(xe)) if xs else (np.zeros((len(S), 0)), np.zeros((len(E), 0)))
    bs, be = [], []
    for j, blk in enumerate(arm.blocks):
        kmax = max(arm.k_grid) if arm.k_grid else arm.k
        a, b = feat.pcs(blk, S, E, skey, max(kmax, 5))
        a, b = a[:, :kb], b[:, :kb]
        if arm.residualise and blk in P.covars:
            ca, cb = feat.nuis(S, E, P.covars[blk])
            a, b = resid_on(a, b, ca, cb)
        a, b = standardize(a, b)
        if scales is not None:
            a, b = a * np.sqrt(scales[j]), b * np.sqrt(scales[j])
        bs.append(a), be.append(b)
    if arm.interaction and len(bs) == 2:
        ia = (bs[0][:, :, None] * bs[1][:, None, :]).reshape(len(S), -1)
        ib = (be[0][:, :, None] * be[1][:, None, :]).reshape(len(E), -1)
        ia, ib = standardize(ia, ib)
        bs.append(ia), be.append(ib)
    return np.hstack([Xs] + bs), np.hstack([Xe] + be)


def ridge_multi(Xs, ys, Xe, alphas):
    """Ridge (intercept unpenalised) for every alpha at once. ys (m,) or (m, B) -> (A, |E|) or (A, |E|, B)."""
    ym = ys.mean(0)
    if Xs.shape[1] == 0:
        out = np.broadcast_to(ym, (len(alphas), len(Xe)) + np.shape(ym))
        return np.array(out)
    xm = Xs.mean(0)
    U, d, Vt = np.linalg.svd(Xs - xm, full_matrices=False)
    Uy = U.T @ (ys - ym)                               # (r,) or (r, B)
    XV = (Xe - xm) @ Vt.T                              # (E, r)
    f = d[None, :] / (d[None, :] ** 2 + np.asarray(alphas)[:, None])   # (A, r)
    if Uy.ndim == 1:
        return ym + (XV[None] * f[:, None, :]) @ Uy
    return ym + np.einsum("er,ar,rb->aeb", XV, f, Uy)


def _yv(Y, idx):
    return Y[idx]


def run_ridge(feat, arm, plan, alphas=ALPHAS, Y=None, keep_inner=False):
    """Nested CV for one ridge arm. Y: optional (n, B) matrix of targets (permutations) -> oof (R, n, B).
    Returns dict(oof, selected, inner_oof)."""
    P = feat.p
    Y = P.y if Y is None else Y
    multi = Y.ndim == 2
    grid = grid_of(arm)
    R = max(s.repeat for s in plan) + 1
    oof = np.full((R, P.n) + ((Y.shape[1],) if multi else ()), np.nan)
    selected, inner_oof = [], []
    A = len(alphas)
    for si, s in enumerate(plan):
        sse = np.zeros((len(grid), A) + ((Y.shape[1],) if multi else ()))
        store = {} if keep_inner else None
        for j, (tr, va) in enumerate(s.inner):
            key = (si, j, id(plan))
            for g, (kc, kb, sc) in enumerate(grid):
                Xs, Xe = design(feat, arm, tr, va, key, kc, kb, sc)
                pr = ridge_multi(Xs, Y[tr], Xe, alphas)
                err = pr - Y[va][None]
                sse[g] += (err ** 2).sum(1)
                if keep_inner:
                    store[(j, g)] = pr
        flat = sse.reshape(len(grid) * A, -1)
        best = flat.argmin(0)                          # per target column
        key = (si, -1, id(plan))
        preds = np.empty((len(grid), A, len(s.test)) + ((Y.shape[1],) if multi else ()))
        for g, (kc, kb, sc) in enumerate(grid):
            Xs, Xe = design(feat, arm, s.train, s.test, key, kc, kb, sc)
            preds[g] = ridge_multi(Xs, Y[s.train], Xe, alphas)
        pf = preds.reshape(len(grid) * A, len(s.test), -1)
        out = pf[best, :, np.arange(pf.shape[2])].T     # (E, B)
        oof[s.repeat, s.test] = out if multi else out[:, 0]
        if not multi:
            g, a = divmod(int(best[0]), A)
            selected.append(dict(kc=grid[g][0], kb=grid[g][1], scales=grid[g][2], alpha=float(alphas[a])))
            if keep_inner:
                io = np.full(P.n, np.nan)
                for j, (tr, va) in enumerate(s.inner):
                    io[va] = store[(j, g)][a]
                inner_oof.append(io)
    return dict(oof=oof, selected=selected, inner_oof=inner_oof)


# ------------------------------------------------------------------ simple baselines
def run_mean(feat, arm, plan, Y=None):
    P = feat.p
    Y = P.y if Y is None else Y
    R = max(s.repeat for s in plan) + 1
    oof = np.full((R,) + Y.shape, np.nan)
    for s in plan:
        if arm.kind == "mean":
            oof[s.repeat, s.test] = Y[s.train].mean(0)
        else:   # moa_mean: training-fold class mean, global training mean for unseen classes
            gm = Y[s.train].mean(0)
            for i in s.test:
                same = s.train[P.moa_code[s.train] == P.moa_code[i]]
                oof[s.repeat, i] = Y[same].mean(0) if len(same) else gm
    return dict(oof=oof, selected=[], inner_oof=[])


def tanimoto(bits):
    inter = bits @ bits.T
    cnt = bits.sum(1)
    return inter / np.maximum(cnt[:, None] + cnt[None, :] - inter, 1e-12)


def _knn_pred(T, S, E, y, k):
    sims = T[np.ix_(E, S)]
    o = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return y[S][o].mean(1)


def run_knn(feat, arm, plan, Y=None):
    P = feat.p
    Y = P.y if Y is None else Y
    T = feat.cache.setdefault(("_tanimoto",), tanimoto(P.bits))
    R = max(s.repeat for s in plan) + 1
    oof = np.full((R, P.n), np.nan)
    sel = []
    for s in plan:
        sse = np.zeros(len(arm.knn_grid))
        for tr, va in s.inner:
            for g, k in enumerate(arm.knn_grid):
                sse[g] += ((_knn_pred(T, tr, va, Y, min(k, len(tr))) - Y[va]) ** 2).sum()
        k = arm.knn_grid[int(sse.argmin())]
        oof[s.repeat, s.test] = _knn_pred(T, s.train, s.test, Y, min(k, len(s.train)))
        sel.append(dict(k=k))
    return dict(oof=oof, selected=sel, inner_oof=[])


def late_avg(res_a, res_b):
    return dict(oof=(res_a["oof"] + res_b["oof"]) / 2, selected=[], inner_oof=[])


def late_stack(prob, plan, res_a, res_b, meta_alpha=1.0):
    """Non-negative ridge meta-learner on the inner-CV out-of-fold predictions of the two arms, fitted inside
    each outer-training fold, applied to the two arms' outer-test predictions."""
    from sklearn.linear_model import Ridge

    oof = np.full_like(res_a["oof"], np.nan)
    wts = []
    for si, s in enumerate(plan):
        Za = np.column_stack([res_a["inner_oof"][si][s.train], res_b["inner_oof"][si][s.train]])
        m = Ridge(alpha=meta_alpha, positive=True).fit(Za, prob.y[s.train])
        Zt = np.column_stack([res_a["oof"][s.repeat, s.test], res_b["oof"][s.repeat, s.test]])
        oof[s.repeat, s.test] = m.predict(Zt)
        wts.append(m.coef_.tolist())
    return dict(oof=oof, selected=[dict(w=w) for w in wts], inner_oof=[])


def run_arm(feat, arm, plan, Y=None, keep_inner=False):
    if arm.kind == "ridge":
        return run_ridge(feat, arm, plan, Y=Y, keep_inner=keep_inner)
    if arm.kind in ("mean", "moa_mean"):
        return run_mean(feat, arm, plan, Y=Y)
    if arm.kind == "knn":
        return run_knn(feat, arm, plan, Y=Y)
    raise ValueError(arm.kind)


# ------------------------------------------------------------------ Freedman-Lane block permutation
def fl_design(prob, other_blocks, k_other=5, k_chem=16):
    """Low-rank regressors for the Freedman-Lane split of a block: intercept, MoA one-hot, global Morgan SVD
    (k_chem), median-imputed nuisance covariates, and the top-k_other global PCs of the other block(s).
    Unsupervised (no target). Returns the hat matrix H (n x n)."""
    n = prob.n
    idx = np.arange(n)
    cols = [np.ones((n, 1)), (prob.moa_code[:, None] == np.unique(prob.moa_code)[None, :]).astype(float)]
    cols.append(gram_pca(prob.grams["_chem"], idx, idx, k_chem, centered=False)[0])
    N = prob.nuis
    med = np.nanmedian(N, 0)
    cols.append(np.where(np.isnan(N), med, N))
    for b in other_blocks:
        cols.append(gram_pca(prob.grams[b], idx, idx, k_other, centered=True)[0])
    Z = np.hstack(cols)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=0), 1e-12)
    H = Z @ np.linalg.pinv(Z)
    return H, int(np.linalg.matrix_rank(Z))


def strata_perm(strata, rng):
    perm = np.arange(len(strata))
    for g in np.unique(strata):
        ix = np.flatnonzero(strata == g)
        if len(ix) > 1:
            perm[ix] = rng.permutation(ix)
    return perm


def fl_gram(K, H, perm):
    """Gram matrix of X* = H X + P (I - H) X, where (P R)_i = R_perm[i]; computed without touching X."""
    n = len(perm)
    I = np.eye(n)
    A = H + I[perm] @ (I - H)
    return A @ K @ A.T


def r2(oof, y):
    """Out-of-fold R^2 per repeat, SST over all evaluated compounds against the pooled mean. oof (R, n[, B])."""
    y = np.asarray(y)
    if oof.ndim == 3:
        sst = ((y - y.mean(0)) ** 2).sum(0)
        return 1 - ((oof - y[None]) ** 2).sum(1) / sst
    sst = ((y - y.mean()) ** 2).sum()
    return 1 - ((oof - y[None]) ** 2).sum(1) / sst
