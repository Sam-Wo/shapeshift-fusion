"""Locations of inputs and outputs. Nothing here is machine-specific.

Every location can be set by an environment variable (or by the matching ``run_all.py`` CLI flag, which sets the
variable). Defaults are relative to this repository, assuming the sibling layout

    <parent>/shapeshift-fusion/   this repository
    <parent>/shapeshift/          ShapeShift Step-2 data root (tables produced by the ShapeShift pipelines)
    <parent>/shapeshift-repo/     public ShapeShift code (overlap.py, step2/compare_tiers.py)

Expected layout of SHAPESHIFT_DATA (all read-only; see README "Data layout"):

    step2/units.csv
    step2/morph_lincs.parquet, step2/morph_lincs_activity.csv
    step2/morph_euos.parquet,  step2/morph_euos_activity.csv
    step2/morph_activity.csv, step2/morph_wells_{high,low}.parquet       (T3, read by compare_tiers.load_morph)
    morph_profiles_high.parquet, morph_profiles_low.parquet                (T3 JUMP profiles)
    tahoe_signatures.parquet
    step2/tahoe_perline.parquet, step2/tahoe_plate_line_mean.parquet
    step2/tahoe_consistency.csv
    data/tahoe/drug_metadata.parquet                                      (Tahoe-100M metadata, CC0)

FUSION_PROBE holds the design-stage probe outputs (data_probe/: PRISM rows, overlap matches, GDSC probe tables).
FUSION_DATA holds new downloads and caches (data/, gitignored). FUSION_RUNS holds per-compound run outputs
(runs/, gitignored). Only aggregate tables go to results/.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _p(var, default):
    v = os.environ.get(var)
    return Path(v).expanduser().resolve() if v else default.resolve()


def shapeshift_data():
    return _p("SHAPESHIFT_DATA", REPO.parent / "shapeshift")


def shapeshift_code():
    return _p("SHAPESHIFT_CODE", REPO.parent / "shapeshift-repo")


def probe_dir():
    return _p("FUSION_PROBE", REPO / "data_probe")


def data_dir():
    d = _p("FUSION_DATA", REPO / "data")
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir():
    d = _p("FUSION_RUNS", REPO / "runs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir():
    d = REPO / "results"
    d.mkdir(exist_ok=True)
    return d


def figures_dir():
    d = REPO / "figures"
    d.mkdir(exist_ok=True)
    return d
