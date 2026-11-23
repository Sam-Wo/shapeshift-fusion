"""End-to-end smoke run of pipeline.run_analysis + report on synthetic data (writes only into tmp dirs)."""
import json

import numpy as np
import pytest

from fusion import analysis as A
from fusion import pipeline as PL
from fusion import splits as SP
from fusion import stats as ST
from fusion import synthetic


def test_run_analysis_and_report(tmp_path, monkeypatch):
    pytest.importorskip("lightgbm")
    monkeypatch.setenv("FUSION_RUNS", str(tmp_path / "runs"))
    from fusion import paths, report

    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path)
    monkeypatch.setattr(paths, "figures_dir", lambda: tmp_path)
    prob, groups = synthetic.make(n=60, r2_t=0.2, r2_m=0.2, seed=0)
    info = dict(units=[f"u{i}" for i in range(60)], scaffold_group=groups, compound_group=np.arange(60))
    plan = SP.make_plan(groups, "kfold", 3, 1, 2, seed=0)
    cfg = dict(seed=1, n_boot=50)
    monkeypatch.setattr(PL.G, "GRID", PL.G.GRID[:2])
    out = PL.run_analysis("primary_scaffold", prob, info, A.arms(), plan, cfg, do_gbm=True, n_perm_block=5,
                          n_perm_target=5, n_jobs=1, pos=prob.y < -1, log=lambda m: None)
    dT = next(d for d in out["deltas"] if d["contrast"] == "F - M")
    out["decision"] = ST.decision(dT["dR2"], dT["dR2"], out["fl"]["tx"]["p"], out["fl"]["morph"]["p"], 0.1, 0.1,
                                  out["min_delta_ub95_one_sided"])
    d = tmp_path / "runs" / "primary_scaffold"
    (d / "summary.json").write_text(json.dumps(out, default=float))
    report.build(log=lambda m: None)
    assert (tmp_path / "primary_decision.csv").exists()
    assert (tmp_path / "fig1_primary_r2_by_arm.png").exists()
    txt = (tmp_path / "arms_all_analyses.csv").read_text()
    assert "u0" not in txt                              # no per-compound rows in results
