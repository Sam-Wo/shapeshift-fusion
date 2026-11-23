"""Aggregate tables (results/*.csv) and figures (figures/*.png) from runs/<name>/summary.json.

Only summary statistics are written: no per-compound rows, no Repurposing Hub columns, no GDSC values beyond one
aggregate correlation.
"""
import json

import numpy as np
import pandas as pd

from . import paths

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
ARM_ORDER = ["B0", "B1", "B2_svd", "B2_knn", "B3", "BASE", "M", "T", "F", "F_late_avg", "F_late_stack", "M_only",
             "T_only", "GBM_BASE", "GBM_M", "GBM_T", "GBM_F", "NN_F", "NN_F_zt0", "NN_F_zm0"]


def _runs():
    out = {}
    for d in sorted(paths.runs_dir().glob("*/summary.json")):
        if d.parent.name == "quick":
            continue
        out[d.parent.name] = json.loads(d.read_text())
    return out


def _delta(s, c):
    return next((d for d in s.get("deltas", []) if d["contrast"] == c), None)


def tables(runs):
    R = paths.results_dir()
    arm_rows, dl_rows, summ = [], [], []
    for name, s in runs.items():
        if "arms" not in s:
            continue
        for a in s["arms"]:
            arm_rows.append(dict(analysis=name, n=s["n"], **a))
        for d in s["deltas"]:
            dl_rows.append(dict(analysis=name, n=s["n"], **d))
        dT, dM = _delta(s, "F - M"), _delta(s, "F - T")
        gT, gM = _delta(s, "GBM_F - GBM_M"), _delta(s, "GBM_F - GBM_T")
        r2 = {a["arm"]: a["R2"] for a in s["arms"]}
        fl = s.get("fl", {})
        summ.append(dict(
            analysis=name, kind=s.get("kind", "primary"), tier=s.get("tier"), target=s.get("target"),
            variant=s.get("variant", s.get("scheme")), n=s["n"],
            R2_BASE=r2.get("BASE"), R2_M=r2.get("M"), R2_T=r2.get("T"), R2_F=r2.get("F"),
            dR2_T=dT["dR2"], dR2_T_lo=dT["ci_lo"], dR2_T_hi=dT["ci_hi"],
            dR2_M=dM["dR2"], dR2_M_lo=dM["ci_lo"], dR2_M_hi=dM["ci_hi"],
            min_dR2_ub95=s.get("min_delta_ub95_one_sided"),
            fl_p_tx=fl.get("tx", {}).get("p"), fl_p_morph=fl.get("morph", {}).get("p"),
            gbm_dR2_T=gT["dR2"] if gT else None, gbm_dR2_M=gM["dR2"] if gM else None,
            verdict=s.get("decision", {}).get("verdict"), git_sha=s.get("git_sha")))
    arms = pd.DataFrame(arm_rows)
    arms["_o"] = arms.arm.map({a: i for i, a in enumerate(ARM_ORDER)}).fillna(99)
    arms = arms.sort_values(["analysis", "_o"]).drop(columns="_o")
    arms.to_csv(R / "arms_all_analyses.csv", index=False, float_format="%.4f")
    pd.DataFrame(dl_rows).to_csv(R / "deltas_all_analyses.csv", index=False, float_format="%.4f")
    sm = pd.DataFrame(summ)
    sm.to_csv(R / "summary_by_analysis.csv", index=False, float_format="%.4f")
    if "primary_scaffold" in runs:
        p = runs["primary_scaffold"]
        dec = dict(p["decision"])
        dT, dM = _delta(p, "F - M"), _delta(p, "F - T")
        gT, gM = _delta(p, "GBM_F - GBM_M"), _delta(p, "GBM_F - GBM_T")
        dec.update(n=p["n"], dR2_T=dT["dR2"], dR2_T_ci_lo=dT["ci_lo"], dR2_T_ci_hi=dT["ci_hi"],
                   dR2_T_lb95=dT["lb95_one_sided"], dR2_M=dM["dR2"], dR2_M_ci_lo=dM["ci_lo"],
                   dR2_M_ci_hi=dM["ci_hi"], dR2_M_lb95=dM["lb95_one_sided"],
                   fl_p_tx=p["fl"]["tx"]["p"], fl_p_morph=p["fl"]["morph"]["p"], n_perm=len(p["fl_null"]["tx"]),
                   fl_n_permutable=p["fl"]["tx"]["n_permutable"], fl_regressor_rank_tx=p["fl"]["tx"]["regressor_rank"],
                   fl_regressor_rank_morph=p["fl"]["morph"]["regressor_rank"],
                   gbm_dR2_T=gT["dR2"], gbm_dR2_M=gM["dR2"], min_dR2_ub95=p["min_delta_ub95_one_sided"],
                   git_sha=p.get("git_sha"))
        pd.DataFrame([dec]).to_csv(R / "primary_decision.csv", index=False, float_format="%.4f")
        q = {b: np.quantile(p["fl_null"][b], [0.5, 0.95, 0.99]) for b in ("tx", "morph")}
        pd.DataFrame([dict(block=b, null_median=v[0], null_q95=v[1], null_q99=v[2], obs=p["fl"][b]["obs"],
                           p=p["fl"][b]["p"]) for b, v in q.items()]).to_csv(R / "primary_fl_null.csv", index=False,
                                                                           float_format="%.4f")
    tim = [dict(analysis=k, **{f"{a}_s": round(b, 1) for a, b in s.get("timing_s", {}).items()}) for k, s in runs.items()]
    pd.DataFrame(tim).to_csv(R / "timing.csv", index=False)
    if "exploratory" in runs:
        e = runs["exploratory"]
        rows = [dict(analysis="gdsc2_a549_auc_vs_prism_primary", n=e["gdsc2_a549_auc_vs_prism_primary"]["n"],
                     spearman=e["gdsc2_a549_auc_vs_prism_primary"]["spearman"])]
        rows += [dict(analysis=f"gdsc2_a549_auc_vs_oof_{a}", **v) for a, v in e["gdsc2_a549_auc_vs_oof"].items()]
        tp = dict(e["two_plate_scoring_of_primary_oof"])
        ntp = tp.pop("n")
        rows += [dict(analysis=f"two_plate_scoring_R2_{a}", n=ntp, R2=v) for a, v in tp.items()]
        n0 = runs.get("primary_scaffold", {}).get("n")
        rows.append(dict(analysis="BASE_plus_rdkit_descriptors_R2", n=n0, R2=e["base_plus_rdkit_descriptors_R2"]))
        rows += [dict(analysis=f"nuisance_only_{k}_R2", n=n0, R2=v)
                 for k, v in e.get("nuisance_single_covariate_R2", {}).items()]
        rows += [dict(analysis=f"no_nuisance_{k}_R2", n=n0, R2=v)
                 for k, v in e.get("without_nuisance_block_R2", {}).items()]
        pd.DataFrame(rows).to_csv(R / "exploratory.csv", index=False, float_format="%.4f")
    return sm


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def figures(runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = paths.figures_dir()
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK, "figure.facecolor": SURF})
    p = runs.get("primary_scaffold")
    if p:
        # fig 1: R2 per arm with bootstrap CIs
        arms = [a for a in ARM_ORDER if any(x["arm"] == a for x in p["arms"])]
        d = {x["arm"]: x for x in p["arms"]}
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        col = [C2 if a in ("M", "T", "F") else (C3 if a.startswith(("GBM", "NN")) else C1) for a in arms]
        y = np.arange(len(arms))[::-1]
        for yi, a, c in zip(y, arms, col):
            ax.plot([d[a]["R2_ci_lo"], d[a]["R2_ci_hi"]], [yi, yi], color=c, lw=2, solid_capstyle="round")
            ax.plot(d[a]["R2"], yi, "o", color=c, ms=7, mec=SURF, mew=2)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(y, arms)
        ax.set_xlabel("out-of-fold R² (scaffold 10-fold × 10, mean over repeats; 95% group-bootstrap CI)")
        ax.set_title(f"Primary: T1 A549, PRISM primary LFC (n = {p['n']})", loc="left", fontsize=11)
        _style(ax)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=c, marker="o", lw=2, label=lab) for c, lab in
                           ((C1, "baselines / BASE (ridge)"), (C2, "modality arms (ridge)"), (C3, "GBM / NN"))],
                  frameon=False, loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(F / "fig1_primary_r2_by_arm.png", dpi=160)
        plt.close(fig)
        # fig 2: observed dR2 vs Freedman-Lane null
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, b, lab in zip(axs, ("tx", "morph"), ("ΔR²_T = R²(F) − R²(M)", "ΔR²_M = R²(F) − R²(T)")):
            null = np.array(p["fl_null"][b])
            ax.hist(null, bins=40, color=C1, alpha=0.85, edgecolor=SURF, lw=0.5)
            ax.axvline(p["fl"][b]["obs"], color=C2, lw=2.5)
            ax.axvline(0.03, color=INK2, lw=1, ls="--")
            ax.set_title(f"{lab}\nobserved {p['fl'][b]['obs']:+.3f}, within-MoA FL p = {p['fl'][b]['p']:.3f}",
                         loc="left", fontsize=10)
            ax.set_xlabel("ΔR² under the permuted block (orange: observed; dashed: 0.03 threshold)", fontsize=8)
            _style(ax)
        axs[0].set_ylabel("permutations")
        fig.tight_layout()
        fig.savefig(F / "fig2_primary_permutation_null.png", dpi=160)
        plt.close(fig)
    # fig 3: forest of dR2_T / dR2_M across analyses
    rows = [(k, s) for k, s in runs.items() if "deltas" in s]
    if rows:
        fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.4))
        y = np.arange(len(rows))[::-1]
        for yi, (k, s) in zip(y, rows):
            for c, off, con in ((C2, 0.13, "F - M"), (C1, -0.13, "F - T")):
                dd = _delta(s, con)
                ax.plot([dd["ci_lo"], dd["ci_hi"]], [yi + off] * 2, color=c, lw=2, solid_capstyle="round")
                ax.plot(dd["dR2"], yi + off, "o", color=c, ms=6, mec=SURF, mew=1.5)
        ax.axvline(0, color=INK2, lw=1)
        ax.axvline(0.03, color=INK2, lw=1, ls="--")
        ax.set_yticks(y, [f"{k.replace('secondary_', '')} (n={s['n']})" for k, s in rows], fontsize=8)
        ax.set_xlabel("ΔR² with 95% group-bootstrap CI (dashed: pre-registered 0.03)")
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=C2, marker="o", lw=2, label="ΔR²_T: transcriptome added to BASE+M"),
                           Line2D([], [], color=C1, marker="o", lw=2, label="ΔR²_M: morphology added to BASE+T")],
                  frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / len(rows)), ncol=2, fontsize=8)
        _style(ax)
        fig.tight_layout()
        fig.savefig(F / "fig3_delta_r2_all_analyses.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def build(log=print):
    runs = _runs()
    sm = tables(runs)
    figures(runs)
    log(f"report: {len(runs)} analyses -> results/, figures/")
    return sm
