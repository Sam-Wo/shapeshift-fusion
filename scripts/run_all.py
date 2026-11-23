"""Rebuild everything from the inputs: features -> primary analysis -> secondaries -> exploratory -> report.

Usage:
  py -3.13 -I -X utf8 scripts/run_all.py [--data DIR] [--code DIR] [--probe DIR] [--jobs N] [--force]
                                         [--only primary|secondary|exploratory|report] [--quick]
  --data   ShapeShift Step-2 data root (sets SHAPESHIFT_DATA)     default ../shapeshift
  --code   public ShapeShift code checkout (sets SHAPESHIFT_CODE)  default ../shapeshift-repo
  --probe  design-stage probe outputs (sets FUSION_PROBE)          default ./data_probe
  --quick  smoke run: 2 repeats, 20 permutations, no NN (results are NOT the pre-registered run)
Each analysis is checkpointed to runs/<name>/summary.json and skipped when present (unless --force).
New downloads (PRISM curve parameters, full primary LFC matrix) are fetched by scripts/fetch_targets.py,
called automatically when they are missing.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"), ap.add_argument("--code"), ap.add_argument("--probe")
    ap.add_argument("--jobs", type=int, default=15)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["primary", "secondary", "exploratory", "report"])
    a = ap.parse_args()
    for flag, var in (("data", "SHAPESHIFT_DATA"), ("code", "SHAPESHIFT_CODE"), ("probe", "FUSION_PROBE")):
        if getattr(a, flag):
            os.environ[var] = str(Path(getattr(a, flag)).resolve())
    if a.quick:
        os.environ.setdefault("FUSION_RUNS", str(ROOT / "runs" / "quick"))
    return a


ARGS = parse()

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from fusion import analysis as A  # noqa: E402
from fusion import data as D  # noqa: E402
from fusion import engine as E  # noqa: E402
from fusion import paths, report  # noqa: E402
from fusion import pipeline as PL  # noqa: E402
from fusion import shapeshift as SS  # noqa: E402
from fusion import stats as ST  # noqa: E402
from fusion import targets as TG  # noqa: E402
from fusion.problem import target_series, tier_problem  # noqa: E402

LOG = paths.runs_dir() / "run_all.log"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def git_sha():
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "fusion", "scripts", "configs"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return None


def cfg_load(name):
    f = ROOT / "configs" / name
    txt = f.read_text()
    c = yaml.safe_load(txt)
    c["_sha256"] = hashlib.sha256(txt.encode()).hexdigest()
    if ARGS.quick:
        c.update(n_repeats=2, n_perm_block=20, n_perm_target=20, n_boot=200)
        if "decision" in c:
            c["decision"]["n_perm_block"] = 20
    return c


def done(name):
    f = paths.runs_dir() / name / "summary.json"
    return f.exists() and not ARGS.force


def save(name, out):
    d = paths.runs_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    out["git_sha"] = git_sha()
    (d / "summary.json").write_text(json.dumps(out, indent=1, default=float))


def load(name):
    return json.loads((paths.runs_dir() / name / "summary.json").read_text())


def plan_cfg(c):
    return dict(schemes=["scaffold", "loco", "moa"], n_folds=c["n_folds"], n_repeats=c["n_repeats"],
                n_inner=c["n_inner"])


def problem_for(tier, target, subset=None):
    s, acct = target_series(tier, target)
    prob, info = tier_problem(tier, y_by_unit=s, subset=subset)
    return prob, info, acct


# ------------------------------------------------------------------ stages
def stage_downloads():
    """Fetch the two PRISM 20Q2 files (scripts/fetch_targets.py) if they are not under FUSION_DATA yet."""
    d = paths.data_dir() / "prism20q2"
    if not ((d / "secondary_curve_params_targets.csv").exists() and (d / "primary_line_median.csv").exists()):
        log("fetching PRISM 20Q2 curve parameters + primary matrix (scripts/fetch_targets.py)")
        sys.path.insert(0, str(ROOT / "scripts"))
        import fetch_targets
        fetch_targets.main()


def stage_features():
    for t in D.TIERS:
        if not (paths.data_dir() / "cache" / f"features_{t}.npz").exists() or ARGS.force:
            log(f"building features {t}")
            D.build_tier(t, force=True)


def manifest(cp, cs):
    metas = {t: D.load_tier(t)[1] for t in D.TIERS}
    import lightgbm
    import sklearn
    import torch
    m = dict(git_sha=git_sha(), shapeshift_code_commit=SS.code_commit(), python=platform.python_version(),
             platform=platform.platform(), numpy=np.__version__, sklearn=sklearn.__version__,
             lightgbm=lightgbm.__version__, torch=torch.__version__,
             config_sha256={"primary.yaml": cp["_sha256"], "secondary.yaml": cs["_sha256"]},
             step2_inputs_sha256=metas["T1"]["inputs_sha256"],
             new_downloads=json.loads((paths.data_dir() / "MANIFEST.json").read_text()),
             quick=bool(ARGS.quick))
    h = hashlib.sha256(json.dumps(m["step2_inputs_sha256"], sort_keys=True).encode()).hexdigest()
    m["data_manifest_sha256"] = h
    (paths.results_dir() / "run_manifest.json").write_text(json.dumps(m, indent=1))
    return m


def stage_primary(cp):
    prob, info, acct = problem_for(cp["tier"], cp["target"])
    log(f"primary: n={prob.n} accounting={acct}")
    if prob.n != cp["expected_n"]:
        raise SystemExit(f"primary n={prob.n} != pre-registered {cp['expected_n']}: log a deviation first")
    defs = A.arms(k=cp["k"])
    pc = plan_cfg(cp)
    pl = A.plans(info["scaffold_group"], info["compound_group"], prob.moa, pc, cp["seed"])
    pos = prob.y < -1.0
    for scheme in cp["schemes"]:
        name = f"primary_{scheme}"
        if done(name):
            log(f"skip {name}")
            continue
        main = scheme == "scaffold"
        out = PL.run_analysis(name, prob, info, defs, pl[scheme], cp, do_gbm=scheme in cp["gbm"]["schemes"],
                              do_nn=main and not ARGS.quick and scheme in cp["nn"]["schemes"],
                              n_perm_block=cp["decision"]["n_perm_block"] if main else 0,
                              n_perm_target=cp["n_perm_target"] if main else 0, n_jobs=ARGS.jobs, pos=pos, log=log)
        out.update(scheme=scheme, tier=cp["tier"], target=cp["target"], accounting=acct, n_splits=len(pl[scheme]),
                   n_groups=int(len(np.unique(info["scaffold_group"]))), config_sha256=cp["_sha256"])
        if main:
            dT = next(d for d in out["deltas"] if d["contrast"] == "F - M")
            dM = next(d for d in out["deltas"] if d["contrast"] == "F - T")
            gT = next(d for d in out["deltas"] if d["contrast"] == "GBM_F - GBM_M")
            gM = next(d for d in out["deltas"] if d["contrast"] == "GBM_F - GBM_T")
            out["decision"] = ST.decision(dT["dR2"], dM["dR2"], out["fl"]["tx"]["p"], out["fl"]["morph"]["p"],
                                          gT["dR2"], gM["dR2"], out["min_delta_ub95_one_sided"],
                                          cp["decision"]["threshold"], cp["decision"]["alpha"])
            log(f"PRIMARY DECISION: {out['decision']}")
        save(name, out)


def stage_secondary(cs, cp):
    for tier, target in cs["targets"]:
        name = f"secondary_{tier}_{target}"
        if done(name):
            log(f"skip {name}")
            continue
        prob, info, acct = problem_for(tier, target)
        log(f"{name}: n={prob.n} accounting={acct}")
        pl = A.plans(info["scaffold_group"], info["compound_group"], prob.moa, dict(plan_cfg(cs), schemes=["scaffold"]),
                     cs["seed"])
        pos = prob.y < cs["binary_threshold"] if target.startswith("prism_primary") else None
        out = PL.run_analysis(name, prob, info, A.arms(k=cp["k"]), pl["scaffold"], cs, do_gbm=True,
                              n_perm_block=cs["n_perm_block"], n_jobs=ARGS.jobs, pos=pos, log=log)
        out.update(tier=tier, target=target, accounting=acct, kind="secondary target")
        save(name, out)
    for vname, v in cs["variants"].items():
        name = f"secondary_T1_variant_{vname}"
        if done(name):
            log(f"skip {name}")
            continue
        v = dict(v)
        subset = v.pop("subset", None)
        prob, info, acct = problem_for("T1", "prism_primary", subset=subset)
        log(f"{name}: n={prob.n}")
        pl = A.plans(info["scaffold_group"], info["compound_group"], prob.moa, dict(plan_cfg(cs), schemes=["scaffold"]),
                     cs["seed"])
        kw = dict(k=cp["k"], **v)
        defs = A.arms(**kw)
        bm = {k: v[k] for k in ("morph", "tx") if k in v}
        nperm = 0 if vname in ("k_inner_cv", "banded_ridge", "interaction") else cs["n_perm_block"]
        out = PL.run_analysis(name, prob, info, defs, pl["scaffold"], cs, do_gbm=subset == "two_plate",
                              n_perm_block=nperm, n_jobs=ARGS.jobs, block_map=bm or None, pos=prob.y < -1.0, log=log)
        out.update(tier="T1", target="prism_primary", variant=vname, kind="secondary variant")
        save(name, out)


def stage_exploratory(cs, cp):
    """Exploratory (not pre-registered): GDSC2 A549 external check, two-plate scoring of the primary OOF
    predictions, an RDKit-descriptor chemistry arm."""
    name = "exploratory"
    if done(name):
        log(f"skip {name}")
        return
    out = {}
    prob, info, _ = problem_for("T1", "prism_primary")
    z = np.load(paths.runs_dir() / "primary_scaffold" / "oof.npz")
    units = info["units"]
    g = TG.gdsc_auc(units)
    idx = [i for i, u in enumerate(units) if u in g.index]
    ga = np.array([g[units[i]] for i in idx])
    from scipy.stats import spearmanr
    out["gdsc2_a549_auc_vs_oof"] = {a: dict(n=len(idx), spearman=float(spearmanr(z[a].mean(0)[idx], ga).statistic))
                                    for a in ("BASE", "M", "T", "F", "GBM_F")}
    out["gdsc2_a549_auc_vs_prism_primary"] = dict(n=len(idx), spearman=float(spearmanr(prob.y[idx], ga).statistic))
    tp = info["two_plate"]
    yy = prob.y[tp]
    out["two_plate_scoring_of_primary_oof"] = {a: float(E.r2(z[a][:, tp], yy).mean()) for a in
                                               ("BASE", "M", "T", "F", "GBM_F", "NN_F") if a in z.files}
    out["two_plate_scoring_of_primary_oof"]["n"] = int(tp.sum())
    pl = A.plans(info["scaffold_group"], info["compound_group"], prob.moa, dict(plan_cfg(cs), schemes=["scaffold"]),
                 cs["seed"])["scaffold"]
    arm = E.Arm("BASE_desc", moa=True, chem=True, nuis=True, extra=("desc",))
    r = E.run_arm(E.Featurizer(prob), arm, pl)
    out["base_plus_rdkit_descriptors_R2"] = float(E.r2(r["oof"], prob.y).mean())
    # which nuisance covariate carries B3, and do the modalities add to MoA + chemistry WITHOUT the nuisance block?
    feat = E.Featurizer(prob)
    single = {}
    for j, c in enumerate(info["nuis_cols"]):
        q = E.Problem(prob.y, prob.moa, prob.nuis[:, [j]], prob.bits, {})
        single[c] = float(E.r2(E.run_arm(E.Featurizer(q), E.Arm(c, nuis=True), pl)["oof"], prob.y).mean())
    out["nuisance_single_covariate_R2"] = single
    nob = {}
    for nm, blocks in (("MoA+chem", ()), ("MoA+chem+M", ("morph",)), ("MoA+chem+T", ("tx",)),
                       ("MoA+chem+M+T", ("morph", "tx"))):
        nob[nm] = float(E.r2(E.run_arm(feat, E.Arm(nm, moa=True, chem=True, blocks=blocks, k=5), pl)["oof"],
                             prob.y).mean())
    out["without_nuisance_block_R2"] = nob
    log(f"exploratory: {out}")
    save(name, out)


def main():
    t0 = time.time()
    cp, cs = cfg_load("primary.yaml"), cfg_load("secondary.yaml")
    stage_downloads()
    stage_features()
    m = manifest(cp, cs)
    log(f"run_all start git={m['git_sha']} quick={ARGS.quick}")
    if ARGS.only in (None, "primary"):
        stage_primary(cp)
    if ARGS.only in (None, "secondary"):
        stage_secondary(cs, cp)
    if ARGS.only in (None, "exploratory"):
        stage_exploratory(cs, cp)
    if ARGS.only in (None, "report"):
        report.build(log=log)
    log(f"run_all finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
