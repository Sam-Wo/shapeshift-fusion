"""Synthetic compounds with the same structure as a real tier (for tests and CI; no real data).

y = BASE signal + morphology-unique signal + transcriptome-unique signal + noise, where each modality block is a
wide matrix whose leading latent directions carry the planted signal (so in-fold PCA can find it).
"""
import numpy as np

from .engine import Problem


def make(n=120, p_m=60, p_t=200, r2_base=0.3, r2_m=0.0, r2_t=0.0, n_moa=6, seed=0, latent=3):
    rng = np.random.default_rng(seed)
    moa = np.array(["unclear"] * (n // 2) + [f"moa{i % n_moa}" for i in range(n - n // 2)], dtype=object)
    rng.shuffle(moa)
    bits = (rng.random((n, 256)) < 0.08).astype(np.uint8)
    nuis = rng.standard_normal((n, 4))
    nuis[rng.random(nuis.shape) < 0.03] = np.nan
    Lm, Lt = rng.standard_normal((n, latent)), rng.standard_normal((n, latent))
    Lt[:, 0] = 0.5 * np.nan_to_num(nuis[:, 0]) + np.sqrt(0.75) * Lt[:, 0]     # block correlated with BASE
    Xm = Lm @ rng.standard_normal((latent, p_m)) * 3 + rng.standard_normal((n, p_m))
    Xt = Lt @ rng.standard_normal((latent, p_t)) * 3 + rng.standard_normal((n, p_t))
    moa_eff = {m: rng.standard_normal() for m in np.unique(moa)}
    base = np.array([moa_eff[m] for m in moa]) + np.nan_to_num(nuis[:, 1])

    def z(v):
        return (v - v.mean()) / v.std()

    def unique(L, other):
        Z = np.column_stack([np.ones(n), other])
        v = L.sum(1)
        return z(v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0])

    sig = np.sqrt(r2_base) * z(base)
    oth = np.column_stack([base, np.nan_to_num(nuis)])
    if r2_m:
        sig = sig + np.sqrt(r2_m) * unique(Lm, np.column_stack([oth, Lt]))
    if r2_t:
        sig = sig + np.sqrt(r2_t) * unique(Lt, np.column_stack([oth, Lm]))
    y = sig + np.sqrt(max(1 - r2_base - r2_m - r2_t, 0.05)) * rng.standard_normal(n)
    groups = np.arange(n) // 2                           # pairs of compounds share a scaffold group
    prob = Problem(y, moa, nuis, bits, {"morph": Xm, "tx": Xt}, covars={"tx": [0], "morph": [2]})
    return prob, groups
