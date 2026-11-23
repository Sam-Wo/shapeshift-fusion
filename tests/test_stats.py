"""Planted-signal and null end-to-end checks of the primary pipeline, plus calibration of the statistics."""
import numpy as np

from fusion import analysis as A
from fusion import engine as E
from fusion import splits as SP
from fusion import stats as ST
from fusion import synthetic


def _primary(r2_m, r2_t, seed, n_perm=19):
    prob, groups = synthetic.make(n=150, r2_m=r2_m, r2_t=r2_t, seed=seed)
    plan = SP.make_plan(groups, "kfold", 5, 2, 3, seed=0)
    defs = A.arms()
    res, _ = A.run_arms(prob, plan, defs, names=["M", "T", "F"])
    tab = A.r2_table(res, prob.y)
    flT = A.fl_test(prob, plan, defs["F"], tab["M"], "tx", ["morph"], n_perm, seed)
    flM = A.fl_test(prob, plan, defs["F"], tab["T"], "morph", ["tx"], n_perm, seed + 1)
    return tab, flT, flM


def test_planted_signal_detected():
    tab, flT, flM = _primary(0.15, 0.15, seed=1)
    assert tab["F"] - tab["M"] >= 0.03 and tab["F"] - tab["T"] >= 0.03
    assert flT["p"] <= 0.05 and flM["p"] <= 0.05


def test_null_not_detected():
    tab, flT, flM = _primary(0.0, 0.0, seed=2)
    assert min(tab["F"] - tab["M"], tab["F"] - tab["T"]) < 0.03
    assert flT["p"] > 0.05 or flM["p"] > 0.05


def test_identity_permutation_reproduces_observed():
    prob, groups = synthetic.make(n=100, r2_t=0.1, seed=3)
    plan = SP.make_plan(groups, "kfold", 5, 1, 3, seed=0)
    arm = A.arms()["F"]
    obs = E.r2(E.run_ridge(E.Featurizer(prob), arm, plan)["oof"], prob.y).mean()
    H, rank = E.fl_design(prob, ["morph"])
    assert 5 < rank < prob.n - 20                   # residual is far from degenerate
    Kp = E.fl_gram(prob.grams["tx"], H, np.arange(prob.n))
    np.testing.assert_allclose(Kp, prob.grams["tx"], atol=1e-6 * np.abs(prob.grams["tx"]).max())
    f = E.Featurizer(prob, grams={"tx": Kp}, parent=E.Featurizer(prob))
    assert abs(E.r2(E.run_ridge(f, arm, plan)["oof"], prob.y).mean() - obs) < 1e-8


def test_fl_gram_equals_explicit_permuted_block():
    prob, _ = synthetic.make(n=60, seed=4)
    H, _ = E.fl_design(prob, ["morph"])
    perm = E.strata_perm(prob.moa_code, np.random.default_rng(0))
    X = prob.blocks["tx"]
    Xs = H @ X + ((np.eye(60) - H) @ X)[perm]
    np.testing.assert_allclose(E.fl_gram(prob.grams["tx"], H, perm), Xs @ Xs.T, rtol=1e-8,
                               atol=1e-8 * np.abs(Xs @ Xs.T).max())
    for g in np.unique(prob.moa_code):                 # permutation stays inside MoA strata
        ix = np.flatnonzero(prob.moa_code == g)
        assert set(perm[ix]) == set(ix)


def test_perm_p_calibrated_under_exchangeability():
    rng = np.random.default_rng(0)
    p = [ST.perm_p(rng.standard_normal(), rng.standard_normal(99)) for _ in range(4000)]
    assert abs(np.mean(np.array(p) < 0.05) - 0.04) < 0.015 and abs(np.mean(np.array(p) <= 0.05) - 0.05) < 0.015


def test_group_bootstrap_keeps_groups_together():
    groups = np.repeat(np.arange(20), 3)
    W = ST.group_weights(groups, 200, np.random.default_rng(0))
    assert np.allclose(W.sum(1), 60)
    for g in range(20):
        assert np.all(W[:, groups == g] == W[:, [3 * g]])


def test_bootstrap_r2_matches_point_estimate_with_unit_weights():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(50)
    oof = y[None] * 0.5 + rng.standard_normal((3, 50))
    np.testing.assert_allclose(ST.r2_weighted(oof, y, np.ones((1, 50)))[0], E.r2(oof, y).mean())


def test_decision_rule_mapping():
    assert ST.decision(0.05, 0.04, 0.01, 0.02, 0.01, 0.02, 0.1)["verdict"] == "POSITIVE"
    assert ST.decision(0.05, 0.04, 0.01, 0.20, 0.01, 0.02, 0.1)["verdict"] == "INCONCLUSIVE"
    assert ST.decision(0.05, 0.04, 0.01, 0.02, -0.01, 0.02, 0.1)["verdict"] == "INCONCLUSIVE"
    assert ST.decision(0.05, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02)["verdict"].startswith("NULL (bounded")
    assert ST.decision(0.05, 0.01, 0.01, 0.02, 0.01, 0.02, 0.05)["verdict"].startswith("NULL (not bounded")


def test_auroc_auprc():
    s = np.array([0.9, 0.8, 0.3, 0.2])
    pos = np.array([1, 0, 1, 0], bool)
    assert ST.auroc(s, pos) == 0.75
    assert abs(ST.auprc(s, pos) - (1 + 2 / 3) / 2) < 1e-12
