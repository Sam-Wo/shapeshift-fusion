"""Inference on fixed out-of-fold predictions (DESIGN.md sections 3.5 and 4.4).

  - paired bootstrap over compound groups (2,000 resamples): every arm is scored on the same resample, so
    differences are paired; R^2 is recomputed per resample (SST against the resample's pooled mean) and averaged
    over CV repeats exactly like the point estimate;
  - permutation p-values (1 + #null >= obs) / (1 + B);
  - the intersection-union decision rule of section 4.4 and its verdict mapping (DEVIATIONS.md, D-07).
"""
import numpy as np
from scipy.stats import rankdata, spearmanr


def group_weights(groups, n_boot, rng):
    """(n_boot, n) resampling weights: groups drawn with replacement, every member of a drawn group counted."""
    groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    draws = rng.integers(0, len(ug), (n_boot, len(ug)))
    cnt = np.zeros((n_boot, len(ug)))
    np.add.at(cnt, (np.repeat(np.arange(n_boot), len(ug)), draws.ravel()), 1)
    return cnt[:, inv]


def r2_weighted(oof, y, W):
    """R^2 averaged over repeats for each weight vector. oof (R, n), y (n,), W (B, n) -> (B,)."""
    yw = (W @ y) / W.sum(1)
    sst = (W * (y[None, :] - yw[:, None]) ** 2).sum(1)
    sse = np.einsum("bn,rn->br", W, (oof - y[None]) ** 2)
    return (1 - sse / sst[:, None]).mean(1)


def bootstrap(oofs, y, groups, n_boot=2000, seed=0):
    """dict arm -> (n_boot,) bootstrap R^2 (repeat-averaged), all arms on the same resamples."""
    W = group_weights(groups, n_boot, np.random.default_rng(seed))
    return {k: r2_weighted(v, y, W) for k, v in oofs.items()}


def ci(x, level=0.95):
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def perm_p(obs, null):
    null = np.asarray(null)
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


def spearman_oof(oof, y):
    return float(np.mean([spearmanr(o, y).statistic for o in oof]))


def auroc(score, pos):
    """AUROC of 'score' for the positive class (Mann-Whitney)."""
    pos = np.asarray(pos, bool)
    r = rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(score, pos):
    pos = np.asarray(pos, bool)
    o = np.argsort(-np.asarray(score), kind="stable")
    rel = pos[o].astype(float)
    prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    return float((prec * rel).sum() / rel.sum())


def decision(dT, dM, pT, pM, gbm_dT, gbm_dM, ub_min, threshold=0.03, alpha=0.05):
    """Pre-registered rule (section 4.4) + verdict mapping fixed in DEVIATIONS.md D-07 before any result:
      POSITIVE      all three conditions hold ('jointly informative')
      INCONCLUSIVE  condition 1 holds (both point estimates >= threshold) but condition 2 or 3 fails
      NULL          condition 1 fails; qualified as 'bounded' if the one-sided 95% bootstrap upper bound of
                    min(dR2_T, dR2_M) is below the threshold, otherwise 'not bounded'."""
    c1 = min(dT, dM) >= threshold
    c2 = (pT < alpha) and (pM < alpha)
    c3 = (np.sign(gbm_dT) == np.sign(dT)) and (np.sign(gbm_dM) == np.sign(dM)) and gbm_dT != 0 and gbm_dM != 0
    if c1 and c2 and c3:
        verdict = "POSITIVE"
    elif c1:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NULL (bounded: joint gain < %.3f at 95%%)" % ub_min if ub_min < threshold else \
            "NULL (not bounded: 95%% upper bound of the joint gain %.3f >= %.2f)" % (ub_min, threshold)
    return dict(c1_min_dR2_ge_threshold=bool(c1), c2_both_perm_p_lt_alpha=bool(c2), c3_gbm_same_sign=bool(c3),
                verdict=verdict)
