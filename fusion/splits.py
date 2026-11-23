"""Grouped outer/inner splits (DESIGN.md section 3.4).

A split scheme is defined by one integer group label per compound; no group ever straddles a train/test
boundary, in the outer folds or in the inner (hyperparameter) folds, which use the same grouping rule.

  scaffold : compound groups merged with generic-Murcko-scaffold Butina clusters; 10-fold, 10 repeats
  loco     : leave-one-compound-group-out; inner 5-fold over compound groups
  moa      : leave-one-MoA-class-out (each 'unclear' compound its own group, merged with compound groups)
"""
from dataclasses import dataclass, field

import numpy as np


def union_find(n, pairs):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = [find(i) for i in range(n)]
    _, lab = np.unique(roots, return_inverse=True)
    return lab.astype(int)


def merge_labels(*labels):
    """Connected components of 'shares any of these labels' (each label array: one value per compound)."""
    labels = [np.asarray(lab) for lab in labels]
    n = len(labels[0])
    pairs = []
    for lab in labels:
        first = {}
        for i, v in enumerate(lab):
            if v in first:
                pairs.append((first[v], i))
            else:
                first[v] = i
    return union_find(n, pairs)


def moa_groups(moa, compound_groups):
    moa = np.asarray(moa, dtype=object)
    lab = np.array([f"moa:{m}" if m != "unclear" else f"single:{i}" for i, m in enumerate(moa)], dtype=object)
    return merge_labels(lab, compound_groups)


def group_kfold(groups, n_folds, rng):
    """Random group-to-fold assignment, balanced by compound count: groups are visited in random order and
    each goes to the fold that currently holds the fewest compounds. Returns fold id per compound."""
    groups = np.asarray(groups)
    ug, sizes = np.unique(groups, return_counts=True)
    n_folds = min(n_folds, len(ug))
    load = np.zeros(n_folds)
    gfold = {}
    for j in rng.permutation(len(ug)):
        f = int(np.flatnonzero(load == load.min())[rng.integers((load == load.min()).sum())])
        gfold[ug[j]] = f
        load[f] += sizes[j]
    return np.array([gfold[g] for g in groups])


@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray
    inner: list = field(default_factory=list)   # list of (train, val) index arrays (into the full data)
    repeat: int = 0
    fold: int = 0


def _inner(train, groups, n_inner, rng):
    f = group_kfold(groups[train], n_inner, rng)
    return [(train[f != k], train[f == k]) for k in range(f.max() + 1)]


def make_plan(groups, scheme, n_folds=10, n_repeats=10, n_inner=5, seed=0):
    """List of Split objects. scheme: 'kfold' (repeated grouped k-fold) or 'logo' (leave-one-group-out)."""
    groups = np.asarray(groups)
    n = len(groups)
    idx = np.arange(n)
    plan = []
    if scheme == "kfold":
        for r in range(n_repeats):
            rng = np.random.default_rng([seed, r])
            fold = group_kfold(groups, n_folds, rng)
            for k in range(fold.max() + 1):
                tr, te = idx[fold != k], idx[fold == k]
                plan.append(Split(tr, te, _inner(tr, groups, n_inner, np.random.default_rng([seed, r, k])), r, k))
    elif scheme == "logo":
        for k, g in enumerate(np.unique(groups)):
            tr, te = idx[groups != g], idx[groups == g]
            plan.append(Split(tr, te, _inner(tr, groups, n_inner, np.random.default_rng([seed, 0, k])), 0, k))
    else:
        raise ValueError(scheme)
    check_plan(plan, groups, n)
    return plan


def check_plan(plan, groups, n):
    """Raise if any compound or group is in both train and test (outer or inner), or if a repeat does not
    cover every compound exactly once."""
    groups = np.asarray(groups)
    reps = {}
    for s in plan:
        if np.intersect1d(s.train, s.test).size or np.intersect1d(groups[s.train], groups[s.test]).size:
            raise AssertionError("outer split leaks a compound/group")
        for tr, va in s.inner:
            if np.intersect1d(tr, va).size or np.intersect1d(groups[tr], groups[va]).size:
                raise AssertionError("inner split leaks a compound/group")
            if np.setdiff1d(np.union1d(tr, va), s.train).size or np.intersect1d(va, s.test).size:
                raise AssertionError("inner split uses outer-test compounds")
        reps.setdefault(s.repeat, []).append(s.test)
    for r, tests in reps.items():
        allt = np.concatenate(tests)
        if len(allt) != n or len(np.unique(allt)) != n:
            raise AssertionError(f"repeat {r} does not cover every compound exactly once")
