"""Build an engine.Problem for a tier x target from the cached tier features (fusion.data) and fusion.targets."""
import numpy as np

from . import data as D
from . import targets as TG
from .engine import Problem

PRISM_LINE = {"T1": "A549", "T2": "HEPG2", "T3": "U2OS"}


def _relabel(g):
    return np.unique(np.asarray(g), return_inverse=True)[1]


def tier_problem(tier, y_by_unit=None, units=None, subset=None):
    """y_by_unit: Series unit -> target (units without a value are dropped). units: explicit unit list (used by
    the pre-freeze check with a synthetic y). subset: optional boolean filter name ('two_plate')."""
    arr, meta = D.load_tier(tier)
    all_units = meta["units"]
    if y_by_unit is not None:
        keep = [i for i, u in enumerate(all_units) if u in y_by_unit.index and np.isfinite(y_by_unit[u])]
        y = np.array([y_by_unit[all_units[i]] for i in keep], float)
    else:
        keep = [all_units.index(u) for u in units]
        y = np.zeros(len(keep))
    if subset == "two_plate":
        sel = [j for j, i in enumerate(keep) if meta["two_plate"][i]]
        keep, y = [keep[j] for j in sel], y[sel]
    idx = np.array(keep)
    nc = meta["nuis_cols"]
    tx = arr["tx"][idx].astype(np.float64)
    blocks = {"morph": arr["morph"][idx], "tx": tx,
              "tx_dir": tx / np.maximum(np.linalg.norm(tx, axis=1, keepdims=True), 1e-12)}
    for i in range(6):
        if f"morph_d{i}" in arr:
            blocks[f"morph_d{i}"] = arr[f"morph_d{i}"][idx]
    covars = {"tx": [nc.index("tx_magnitude"), nc.index("tx_log2_cellratio")], "morph": [nc.index("morph_norm")]}
    covars["tx_dir"] = covars["tx"]
    desc = arr["desc"][idx]
    desc = np.where(np.isfinite(desc) & (np.abs(desc) < 1e8), desc, np.nan)
    prob = Problem(y, np.array(meta["moa"], dtype=object)[idx], arr["nuis"][idx], arr["bits"][idx], blocks,
                   covars=covars, extra={"desc": desc})
    info = dict(units=[all_units[i] for i in idx], two_plate=np.array(meta["two_plate"])[idx],
                scaffold_group=_relabel(arr["scaffold_group"][idx]),
                compound_group=_relabel(arr["compound_group"][idx]), nuis_cols=nc, meta=meta)
    return prob, info


def target_series(tier, target):
    arr, meta = D.load_tier(tier)
    s, acct = TG.TARGETS[target](PRISM_LINE[tier], meta["units"])
    return s, acct
