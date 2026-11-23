"""Power / type-I of the ACTUAL pre-registered decision rule for one block (synthetic data only).

Rule for the transcriptome block: observed out-of-fold dR2_T >= 0.03 AND conditional permutation p < 0.05.
Generator as power_sim.py with K = 5 PCs per block, plus a BASE feature correlated with the transcriptome block
(BASE[:,0] = 0.6 * Xt[:,0] + 0.8 * noise), mimicking 'Tahoe magnitude' sitting in BASE.
Permutation schemes (99 permutations each):
  naive : shuffle Xt rows (breaks the Xt-BASE correlation)
  resid : Freedman-Lane style - regress Xt on [BASE, morph] (in-sample OLS), shuffle the residual rows, add back
          the fitted part (keeps the Xt-BASE correlation; only the unique part is permuted)
Usage: py -3.13 -I -X utf8 scripts/power_sim_rule.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1] / "data_probe" / "power_sim_rule.json"
ALPHAS = np.logspace(0, 4, 17)
K, NB, NPERM, REPS, N = 5, 10, 99, 100, 186


def simulate(n, delta, rng, r2_base=0.30, r2_morph=0.08):
    Xm, Xt = rng.standard_normal((n, K)), rng.standard_normal((n, K))
    Xb = rng.standard_normal((n, NB))
    Xb[:, 0] = 0.6 * Xt[:, 0] + 0.8 * rng.standard_normal(n)
    sig = (np.sqrt(r2_base / NB) * Xb.sum(1) + np.sqrt(r2_morph / 3) * Xm[:, :3].sum(1)
           + np.sqrt(delta / 3) * Xt[:, :3].sum(1))
    y = sig + np.sqrt(max(1 - r2_base - r2_morph - delta, 0.05)) * rng.standard_normal(n)
    return Xb, Xm, Xt, y


def r2_oof(X, y, seed):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    p = cross_val_predict(m, X, y, cv=KFold(10, shuffle=True, random_state=seed))
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def one(delta, rng, seed, schemes):
    Xb, Xm, Xt, y = simulate(N, delta, rng)
    red = np.hstack([Xb, Xm])
    r_red = r2_oof(red, y, seed)
    d = r2_oof(np.hstack([red, Xt]), y, seed) - r_red
    out = {"d": d}
    Z = np.column_stack([np.ones(N), red])
    fit = Z @ np.linalg.lstsq(Z, Xt, rcond=None)[0]
    res = Xt - fit
    for sc in schemes:
        null = []
        for _ in range(NPERM):
            pi = rng.permutation(N)
            Xp = Xt[pi] if sc == "naive" else fit + res[pi]
            null.append(r2_oof(np.hstack([red, Xp]), y, seed) - r_red)
        out[sc] = (1 + np.sum(np.array(null) >= d)) / (NPERM + 1)
    return out


def main():
    rng = np.random.default_rng(1)
    res = {}
    for delta, schemes in ((0.0, ("naive", "resid")), (0.05, ("resid",)), (0.08, ("resid",)), (0.12, ("resid",))):
        rows = [one(delta, rng, s, schemes) for s in range(REPS)]
        d = np.array([r["d"] for r in rows])
        rec = dict(mean_dR2=round(float(d.mean()), 4), P_obs_ge_0p03=float(np.mean(d >= 0.03)))
        for sc in schemes:
            p = np.array([r[sc] for r in rows])
            rec[f"P_perm_p_lt_0p05_{sc}"] = float(np.mean(p < 0.05))
            rec[f"P_rule_{sc}"] = float(np.mean((p < 0.05) & (d >= 0.03)))
        res[f"n={N},K={K},delta={delta}"] = rec
        print(delta, rec, flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
