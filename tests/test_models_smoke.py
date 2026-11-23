"""CPU smoke tests: PyTorch fusion model (deterministic, ~7-9k parameters) and the LightGBM learner."""
import numpy as np
import pytest

from fusion import engine as E
from fusion import splits as SP
from fusion import synthetic


def test_torch_param_count_and_determinism():
    pytest.importorskip("torch")
    from fusion import nn

    assert 6000 < nn.n_params(32, 32, 50) < 10000
    rng = np.random.default_rng(0)
    Xm, Xt, Xb = rng.standard_normal((40, 5)), rng.standard_normal((40, 5)), rng.standard_normal((40, 8))
    y = Xb[:, 0] + rng.standard_normal(40)
    g = np.arange(40)
    a = nn.fit_predict(Xm, Xt, Xb, y, Xm[:5], Xt[:5], Xb[:5], g, 1e-2, seed=0, max_epochs=2)
    b = nn.fit_predict(Xm, Xt, Xb, y, Xm[:5], Xt[:5], Xb[:5], g, 1e-2, seed=0, max_epochs=2)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    assert not np.allclose(a[0], a[1])                 # zeroing z_t changes the prediction


def test_torch_run_split():
    pytest.importorskip("torch")
    from fusion import nn

    prob, groups = synthetic.make(n=60, seed=1)
    plan = SP.make_plan(groups, "kfold", 3, 1, 2, seed=0)
    out, cfg = nn.run_split(E.Featurizer(prob), plan[0], 0, groups, seeds=(0,), configs=nn.CONFIGS[:2],
                            max_epochs=3)
    assert out["full"].shape == plan[0].test.shape and np.isfinite(out["full"]).all()


def test_gbm_run_split():
    pytest.importorskip("lightgbm")
    from fusion import gbm

    prob, groups = synthetic.make(n=60, seed=2)
    plan = SP.make_plan(groups, "kfold", 3, 1, 2, seed=0)
    pred, sel = gbm.run_split(E.Featurizer(prob), plan[0], 0, ("morph", "tx"))
    assert pred.shape == plan[0].test.shape and np.isfinite(pred).all()
    assert sel["n_estimators"] in (100, 300)
