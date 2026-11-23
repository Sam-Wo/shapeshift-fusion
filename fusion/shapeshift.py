"""Lazy imports of the public ShapeShift code (structure matching and Tahoe plate centring).

Nothing is re-implemented: ``overlap.std_key`` / ``std_mol`` and ``compare_tiers.Tahoe`` / ``load_morph`` are
imported from SHAPESHIFT_CODE. ``compare_tiers`` derives its data root from its own file location, so after
import its module globals ``ROOT`` and ``S2`` are pointed at SHAPESHIFT_DATA (the Step-2 tables live there, not
in the code checkout). Imports are lazy so that the synthetic-data tests run without either directory.
"""
import importlib
import sys

from . import paths

_MODS = {}


def _add(p):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def overlap():
    if "overlap" not in _MODS:
        code = paths.shapeshift_code()
        if not (code / "overlap.py").exists():
            raise FileNotFoundError(f"overlap.py not found under SHAPESHIFT_CODE={code}")
        _add(code)
        _MODS["overlap"] = importlib.import_module("overlap")
    return _MODS["overlap"]


def compare_tiers():
    if "ct" not in _MODS:
        code = paths.shapeshift_code()
        if not (code / "step2" / "compare_tiers.py").exists():
            raise FileNotFoundError(f"step2/compare_tiers.py not found under SHAPESHIFT_CODE={code}")
        _add(code / "step2")
        ct = importlib.import_module("compare_tiers")
        data = paths.shapeshift_data()
        if not (data / "step2" / "units.csv").exists():
            raise FileNotFoundError(f"step2/units.csv not found under SHAPESHIFT_DATA={data}")
        ct.ROOT, ct.S2 = data, data / "step2"
        _MODS["ct"] = ct
    return _MODS["ct"]


def code_commit():
    """git HEAD of the ShapeShift code checkout (recorded in the run manifest), or None."""
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(paths.shapeshift_code()), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None
