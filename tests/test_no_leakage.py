"""Transforms are fitted on training rows only; no compound or group is ever on both sides of a split."""
import numpy as np

from fusion import engine as E
from fusion import splits as SP
from fusion import synthetic


def test_train_transforms_ignore_test_rows():
    """Shuffling or perturbing the held-out rows' features must not change the training-fold transforms, and a
    held-out compound's prediction depends only on its own features."""
    prob, _ = synthetic.make(n=80, seed=4)
    S, Ev = np.arange(60), np.arange(60, 80)
    arm = E.Arm("F", moa=True, chem=True, nuis=True, blocks=("morph", "tx"), k=5)
    Xs, Xe = E.design(E.Featurizer(prob), arm, S, Ev, "a", 16, 5, None)
    p0 = E.ridge_multi(Xs, prob.y[S], Xe, [5.0])[0]

    rng = np.random.default_rng(0)
    perm = np.concatenate([S, rng.permutation(Ev)])           # shuffle the test rows' features among themselves
    blocks = {k: v[perm] for k, v in prob.blocks.items()}
    q = E.Problem(prob.y, prob.moa[perm], prob.nuis[perm], prob.bits[perm], blocks)
    Xs2, Xe2 = E.design(E.Featurizer(q), arm, S, Ev, "a", 16, 5, None)
    np.testing.assert_allclose(np.abs(Xs2), np.abs(Xs), atol=1e-8)
    p2 = E.ridge_multi(Xs2, prob.y[S], Xe2, [5.0])[0]
    inv = np.argsort(perm[60:] - 60)
    np.testing.assert_allclose(p2[inv], p0, atol=1e-8)

    blocks = {k: v.copy() for k, v in prob.blocks.items()}
    blocks["tx"][Ev] += 100 * rng.standard_normal(blocks["tx"][Ev].shape)   # wreck test rows
    blocks["tx"][Ev[0]] = prob.blocks["tx"][Ev[0]]                          # ... except the first
    q = E.Problem(prob.y, prob.moa, prob.nuis, prob.bits, blocks)
    Xs3, Xe3 = E.design(E.Featurizer(q), arm, S, Ev, "a", 16, 5, None)
    np.testing.assert_allclose(np.abs(Xs3), np.abs(Xs), atol=1e-8)
    p3 = E.ridge_multi(Xs3, prob.y[S], Xe3, [5.0])[0]
    assert abs(p3[0] - p0[0]) < 1e-8


def test_pca_fitted_on_training_fold_only():
    prob, _ = synthetic.make(n=80, seed=5)
    X = prob.blocks["tx"]
    S, Ev = np.arange(0, 80, 2), np.arange(1, 80, 2)
    Zs, _ = E.gram_pca(prob.grams["tx"], S, Ev, 5)
    Zs_alone, _ = E.gram_pca(X[S] @ X[S].T, np.arange(len(S)), np.arange(1), 5)
    np.testing.assert_allclose(np.abs(Zs), np.abs(Zs_alone), atol=1e-8)


def test_no_compound_or_group_in_train_and_test():
    groups = np.repeat(np.arange(40), 3)
    for plan in (SP.make_plan(groups, "kfold", 10, 3, 5, seed=1), SP.make_plan(groups, "logo", n_inner=5)):
        for s in plan:
            assert not np.intersect1d(s.train, s.test).size
            assert not np.intersect1d(groups[s.train], groups[s.test]).size
            for tr, va in s.inner:
                assert not np.intersect1d(groups[tr], groups[va]).size
                assert not np.intersect1d(va, s.test).size
                assert np.array_equal(np.sort(np.concatenate([tr, va])), np.sort(s.train))


def test_check_plan_catches_leak():
    groups = np.repeat(np.arange(10), 2)
    plan = SP.make_plan(groups, "kfold", 5, 1, 2, seed=0)
    plan[0].train = np.append(plan[0].train, plan[0].test[0])
    try:
        SP.check_plan(plan, groups, 20)
    except AssertionError:
        return
    raise AssertionError("leak not detected")
