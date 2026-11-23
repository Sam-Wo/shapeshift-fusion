import numpy as np
import pandas as pd

from fusion import splits as SP
from fusion.data import compound_groups


def test_prodrug_pairs_merged_and_never_straddle():
    u = pd.DataFrame({"unit": list("abcdef"),
                      "dataset_ids": ["X1", "X2;X3", "X3", "X4", "X5", "X1;X6"]})
    cg = compound_groups(u)
    assert cg[0] == cg[5] and cg[1] == cg[2] and len(set(cg)) == 4
    scaf = np.array(["s1", "s2", "s3", "s1", "s4", "s5"], dtype=object)
    g = SP.merge_labels(scaf, cg)
    assert g[0] == g[3] == g[5]
    for s in SP.make_plan(g, "kfold", 3, 4, 2, seed=0):
        assert not set(cg[s.train]) & set(cg[s.test])


def test_group_kfold_balanced_and_repeats_differ():
    groups = np.repeat(np.arange(60), np.r_[np.full(30, 1), np.full(30, 4)])
    f1 = SP.group_kfold(groups, 10, np.random.default_rng(0))
    f2 = SP.group_kfold(groups, 10, np.random.default_rng(1))
    sizes = np.bincount(f1)
    assert sizes.max() - sizes.min() <= 4
    assert (f1 != f2).any()


def test_moa_groups_unclear_singletons():
    moa = np.array(["a", "a", "unclear", "unclear", "b"], dtype=object)
    g = SP.moa_groups(moa, np.arange(5))
    assert g[0] == g[1] and g[2] != g[3] and len(set(g)) == 4
