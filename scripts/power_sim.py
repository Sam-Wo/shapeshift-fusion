"""Power simulation for the pre-registered primary comparison (synthetic data only, no real targets).

Model of the data (all standardised):
  y = base + morph_unique + tx_unique + noise
  base       : 10 'chemistry/MoA/nuisance' features, true R^2 = 0.30 (MoA-mean LOO R^2 in A549 was 0.32-0.38)
  morph      : 20 PCs, unique true R^2 = 0.08 spread over 3 PCs
  tx         : 20 PCs, unique true R^2 = delta spread over 3 PCs  (the quantity tested)
Primary statistic: out-of-fold dR^2 = R^2(base+morph+tx) - R^2(base+morph), ridge (RidgeCV inner alpha),
10-fold CV; one-sided test = paired bootstrap over compounds (2,000 resamples) of the SSE difference, lower 95% bound > 0.
Usage: py -3.13 -I -X utf8 scripts/power_sim.py [K]   K = PCs per modality block (default 20; signal always in 3)
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = Path(__file__).resolve().parents[1] / "data_probe" / ("power_sim.json" if K == 20 else f"power_sim_k{K}.json")
ALPHAS = np.logspace(0, 4, 17)  # alpha >= 1: GCV with alpha << 1 was unstable at p ~ n in a first run


def simulate(n, delta, rng, r2_base=0.30, r2_morph=0.08):
    Xb, Xm, Xt = rng.standard_normal((n, 10)), rng.standard_normal((n, K)), rng.standard_normal((n, K))
    sig = (np.sqrt(r2_base / 10) * Xb.sum(1) + np.sqrt(r2_morph / 3) * Xm[:, :3].sum(1)
           + np.sqrt(delta / 3) * Xt[:, :3].sum(1))
    y = sig + np.sqrt(1 - r2_base - r2_morph - delta) * rng.standard_normal(n)
    return Xb, Xm, Xt, y


def oof(X, y, seed):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    return cross_val_predict(m, X, y, cv=KFold(10, shuffle=True, random_state=seed))


def one(n, delta, rng, seed, B=2000):
    Xb, Xm, Xt, y = simulate(n, delta, rng)
    p_red = oof(np.hstack([Xb, Xm]), y, seed)
    p_full = oof(np.hstack([Xb, Xm, Xt]), y, seed)
    e_red, e_full = (y - p_red) ** 2, (y - p_full) ** 2
    sst = np.sum((y - y.mean()) ** 2)
    d = (e_red.sum() - e_full.sum()) / sst
    idx = rng.integers(0, n, (B, n))
    boot = (e_red[idx].sum(1) - e_full[idx].sum(1)) / (y[idx].var(1) * n)
    return d, np.quantile(boot, 0.05) > 0


def main():
    rng = np.random.default_rng(0)
    res = {}
    for n in ((61, 98, 140, 156, 186) if K == 20 else (98, 140, 186)):
        for delta in (0.0, 0.03, 0.05, 0.08, 0.12):
            out = [one(n, delta, rng, s) for s in range(200)]
            ds = np.array([o[0] for o in out])
            res[f"n={n},delta={delta}"] = dict(power=float(np.mean([o[1] for o in out])),
                                               mean_observed_dR2=round(float(ds.mean()), 4),
                                               sd_observed_dR2=round(float(ds.std()), 4))
            print(n, delta, res[f"n={n},delta={delta}"], flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
