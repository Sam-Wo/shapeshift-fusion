"""Small two-encoder PyTorch fusion model (DESIGN.md section 3.3; secondary, CPU).

  encoder per modality : LayerNorm(k) -> Linear(k, 64) -> GELU -> Dropout(0.3) -> Linear(64, 16)
  head                 : concat[z_m, z_t, BASE] -> Linear(., 32) -> GELU -> Dropout(0.3) -> Linear(32, 1)
                         + a linear skip term on BASE (the network learns a residual over a ridge-like base)
  regularisation       : AdamW (weight decay from the grid), dropout 0.3, modality dropout p = 0.25 (a whole
                         encoder output zeroed per sample), Gaussian input noise sigma = 0.1, Huber loss
  training             : batch 32, lr 1e-3 with cosine decay, <= 300 epochs, early stopping on a grouped
                         inner-validation split (patience 30, best weights restored), 5 seeds averaged
Inputs are fitted in-fold: modality PCA scores (k from the grid), BASE = MoA one-hot + Morgan SVD (16) +
median-imputed nuisance, all standardised on the training rows; y standardised on the training rows.
At test time the model also predicts with z_t zeroed (morphology-only path) and z_m zeroed (transcriptome-only
path); modality dropout trained the head for exactly that input.
Grid (4 configurations, chosen by 5-fold inner CV): weight decay {1e-3, 1e-2} x block k {5, 16} (D-10).
"""
import numpy as np

from .engine import standardize
from .splits import group_kfold

CONFIGS = [dict(wd=wd, k=k) for wd in (1e-3, 1e-2) for k in (5, 16)]
SEEDS = (0, 1, 2, 3, 4)


def _torch():
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def build_model(k_m, k_t, d_base, p_mod=0.25):
    torch = _torch()
    nn = torch.nn

    class Enc(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.f = nn.Sequential(nn.LayerNorm(k), nn.Linear(k, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 16))

        def forward(self, x):
            return self.f(x)

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.em, self.et = Enc(k_m), Enc(k_t)
            self.head = nn.Sequential(nn.Linear(32 + d_base, 32), nn.GELU(), nn.Dropout(0.3), nn.Linear(32, 1))
            self.skip = nn.Linear(d_base, 1)
            self.p_mod = p_mod

        def forward(self, xm, xt, xb, drop_m=None, drop_t=None):
            zm, zt = self.em(xm), self.et(xt)
            if self.training and self.p_mod > 0:
                keep_m = (torch.rand(len(xm), 1) >= self.p_mod).float()
                keep_t = (torch.rand(len(xt), 1) >= self.p_mod).float()
                zm, zt = zm * keep_m, zt * keep_t
            if drop_m:
                zm = zm * 0
            if drop_t:
                zt = zt * 0
            return (self.head(torch.cat([zm, zt, xb], 1)) + self.skip(xb)).squeeze(1)

    return Fusion()


def n_params(k_m=32, k_t=32, d_base=50):
    return sum(p.numel() for p in build_model(k_m, k_t, d_base).parameters())


def fit_predict(Xm, Xt, Xb, y, Em, Et, Eb, groups, wd, seed, max_epochs=300, patience=30, batch=32, noise=0.1):
    """Train one seed; returns predictions on E for (full, z_t zeroed, z_m zeroed), in original y units."""
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f = group_kfold(groups, 6, rng)                     # ~1/6 of training groups held out for early stopping
    va, tr = f == 0, f != 0
    mu, sd = y[tr].mean(), y[tr].std() + 1e-12
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    xm, xt, xb, yy = T(Xm), T(Xt), T(Xb), T((y - mu) / sd)
    model = build_model(Xm.shape[1], Xt.shape[1], Xb.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = torch.nn.HuberLoss()
    itr = np.flatnonzero(tr)
    ivl = torch.tensor(np.flatnonzero(va))
    best, best_state, wait = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        order = itr[rng.permutation(len(itr))]
        for b0 in range(0, len(order), batch):
            ib = torch.tensor(order[b0:b0 + batch])
            nz = lambda x: x[ib] + noise * torch.randn_like(x[ib])  # noqa: E731
            opt.zero_grad()
            loss = loss_fn(model(nz(xm), nz(xt), nz(xb)), yy[ib])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(((model(xm[ivl], xt[ivl], xb[ivl]) - yy[ivl]) ** 2).mean())
        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    em, et, eb = T(Em), T(Et), T(Eb)
    with torch.no_grad():
        out = [model(em, et, eb).numpy(), model(em, et, eb, drop_t=True).numpy(),
               model(em, et, eb, drop_m=True).numpy()]
    return [o * sd + mu for o in out]


def inputs(feat, S, E, skey, k, blocks=("morph", "tx")):
    P = feat.p
    a, b = feat.moa(S, E)
    c, d = feat.chem(S, E, skey)
    e, g = feat.nuis(S, E)
    Bs, Be = standardize(np.hstack([a, c[:, :16], e]), np.hstack([b, d[:, :16], g]))
    out = []
    for blk in blocks:
        zs, ze = feat.pcs(blk, S, E, skey, 16)
        out.append(standardize(zs[:, :k], ze[:, :k]))
    return out[0][0], out[1][0], Bs, out[0][1], out[1][1], Be


def run_split(feat, s, si, groups, seeds=SEEDS, configs=CONFIGS, max_epochs=300):
    y = feat.p.y
    sse = np.zeros(len(configs))
    for j, (tr, va) in enumerate(s.inner):
        for ci, cfg in enumerate(configs):
            Xm, Xt, Xb, Em, Et, Eb = inputs(feat, tr, va, (si, j, "nn"), cfg["k"])
            p = np.mean([fit_predict(Xm, Xt, Xb, y[tr], Em, Et, Eb, groups[tr], cfg["wd"], sd,
                                     max_epochs=max_epochs)[0] for sd in seeds], 0)
            sse[ci] += ((p - y[va]) ** 2).sum()
    cfg = configs[int(sse.argmin())]
    Xm, Xt, Xb, Em, Et, Eb = inputs(feat, s.train, s.test, (si, -1, "nn"), cfg["k"])
    outs = [fit_predict(Xm, Xt, Xb, y[s.train], Em, Et, Eb, groups[s.train], cfg["wd"], sd, max_epochs=max_epochs)
            for sd in seeds]
    full, m_only, t_only = (np.mean([o[i] for o in outs], 0) for i in range(3))
    return dict(full=full, zt_zeroed=m_only, zm_zeroed=t_only), cfg
