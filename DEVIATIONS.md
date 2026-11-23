# Deviations and clarifications

DESIGN.md (commit `31cd6c7`, pushed before any model was fit) is the pre-registration. This file records every
place where the implementation departs from it, or where it fixes a detail the design left open. Each entry is
dated, and all entries in the first block were written **before any real target was loaded into a model**. Later
entries say whether results had been seen.

None of these changes the primary comparison (T1 × PRISM 20Q2 primary A549, 183 units), the decision rule, the
splits, the feature construction or the arms.

## Before freezing (2026-11-23; no real target in any model yet)

**D-01 Code reuse and data root.** Structure matching (`overlap.std_key`, `overlap.std_mol`) and plate-centring
(`compare_tiers.Tahoe().sig(..., centered=True)`) and morphology loading (`compare_tiers.load_morph`) are imported
from the public ShapeShift code (`SHAPESHIFT_CODE`, default `../shapeshift-repo`). `compare_tiers` derives its
data root from its own location, so `fusion/shapeshift.py` sets its module globals `ROOT` and `S2` to
`SHAPESHIFT_DATA` (default `../shapeshift`) after import. Nothing is copied. The two checkouts' `overlap.py` and
`compare_tiers.py` are byte-identical. The SHA-256 of every Step-2 file read is recorded in the feature cache and
in `results/run_manifest.json` (design risk 10). The design's "pinned commit / submodule" is replaced by recording
the code checkout's git commit.

**D-02 GBM library.** LightGBM 4.6.0 is installed, so the GBM is LightGBM (the design: "Proposed for v1: LightGBM
or XGBoost"), not sklearn HGB. The 32-point grid maps as `max_depth` {2, 3} with `num_leaves` = 2^depth,
`learning_rate` {0.03, 0.1}, `n_estimators` {100, 300} (= HGB `max_iter`), `min_child_samples` {10, 20}
(= `min_samples_leaf`), `reg_lambda` {0, 1} (= `l2_regularization`); no early stopping; `max_bin` 63,
`min_data_in_bin` 1 (n ≈ 150 training rows); `n_jobs` 1, deterministic. The 100-tree model is read off the first
100 trees of the 300-tree fit (identical by construction). GBM BASE uses a fixed Morgan SVD k = 16 (the ridge arms
choose 16 or 32 by inner CV).

**D-03 Scaffold groups.** Generic Bemis–Murcko scaffolds (`MurckoScaffold.MakeScaffoldGeneric` of
`GetScaffoldForMol` on `overlap.std_mol`); Butina clustering is applied to the *distinct generic scaffolds*, on
Morgan r = 2, 2,048-bit fingerprints of the scaffold molecules, at Tanimoto ≥ 0.6 (distance cutoff 0.4). Acyclic
compounds (empty scaffold) are each their own cluster. Split groups are the connected components of
{same scaffold cluster} ∪ {same compound group}. T1: 77 groups for 187 units (largest 19). Folds are formed by
visiting groups in random order and assigning each to the fold with the fewest compounds; each repeat uses a
different seed; inner folds use the same rule inside the outer-training set.

**D-04 Compound groups.** Units are already stereo-flat (the 7 T1 `stereo_flag` units are single units), so the
only merge is units that share a dataset compound ID: ixazomib / ixazomib citrate share one LINCS BRD ID in T1.
T1 has 186 compound groups for 187 units; T2 and T3 have none to merge.

**D-05 T3 morphology activity covariate.** `step2/morph_activity.csv` (JUMP) has no `norm_z` column. For T3 the
morphology-activity nuisance covariate is the log L2 norm of the 10 µM consensus profile. T1 and T2 use `norm_z`
at the tier dose, as designed. T1 also carries `block_reliable` (LINCS 3.33 µM).

**D-06 Nuisance covariates, made explicit.**
- `tx_magnitude` is the L2 norm of the plate-centred 5 µM signature that enters the model: the line signature for
  T1/T2 and the consensus for T3.
- `tx_log2_cellratio` is log2(`n_cells_trt` / the median over the first plate × dose of that line), as in
  `scripts/probe_targets.py`; for T3 it is the mean over all lines.
- `tx_two_plate` is `n_plates` ≥ 2 at 5 µM.
- `morph_n_wells` is the number of wells at the tier dose.
- Missing values are median-imputed in-fold for ridge and left missing for LightGBM.

**D-07 Verdict mapping.** The design defines only whether "jointly informative" is declared. The verdict is
mapped as follows, fixed before any result:
- **POSITIVE:** all three §4.4 conditions hold.
- **INCONCLUSIVE:** condition 1 holds (both ΔR² ≥ 0.03), but condition 2 (both within-MoA Freedman–Lane p < 0.05)
  or condition 3 (the GBM Δs share the ridge sign) fails.
- **NULL:** condition 1 fails. It is qualified as *bounded* if the one-sided 95% paired-bootstrap upper bound of
  min(ΔR²_T, ΔR²_M) is below 0.03, and *not bounded* otherwise.

**D-08 Freedman–Lane permutation, implementation.** Regressing the raw block (5,000 genes or 418 features) on the
raw [BASE, other block] is degenerate at n = 183: there are more regressors than rows, so the residuals vanish.
- **The block stays raw**, so in-fold PCA is rerun on every permuted block.
- **The regressors are low-rank and unsupervised**, computed on all 183 units with no target: an intercept, the MoA
  one-hot, the global Morgan SVD (k = 16), the median-imputed nuisance covariates, and the top 5 global PCs of the
  other block. The permuted block is X* = HX + P(I − H)X, where H is the hat matrix and P permutes rows within
  Tahoe `moa-fine` classes.
- **Strata.** "unclear" is one stratum, as it is one level of the MoA one-hot (§2.3). Singleton classes stay
  fixed.
- **Only F-early is refitted.** BASE and the reduced arm are unchanged under the permutation.
- **Efficiency.** It is computed exactly through Gram matrices (`engine.fl_gram`); a test checks this against the
  explicitly permuted matrix. The identity permutation reproduces the observed ΔR² exactly (test).
- **p-value:** p = (1 + #{null ≥ obs}) / (1 + B).

**D-09 Arms, made explicit.**
- **B2** is reported as two arms: `B2_svd` (ridge on in-fold Morgan SVD, k ∈ {16, 32}) and `B2_knn` (Tanimoto
  kNN, k ∈ {3, 5, 10}). BASE uses the SVD view only, per §3.1.
- **B1 and the MoA one-hot** treat "unclear" as its own level (§2.3).
- **Ridge grid:** α ∈ logspace(0, 4, 8) × Morgan SVD k ∈ {16, 32}, which gives 16 grid points. Every column is
  standardised in-fold. PCA scores are not whitened beyond that standardisation.
- **F-late-stack** is `Ridge(alpha = 1, positive = True)` on the inner-CV out-of-fold predictions of M and T at
  their selected hyperparameters.
- **Morgan count fingerprints and the ~200 RDKit descriptors** are computed. No pre-registered arm uses them; the
  descriptors appear only in an exploratory arm.
- **Target-permutation tests** (1,000 permutations, free shuffle) are run for B1 and every ridge arm of the primary
  analysis, vectorised over permutations. They are not run for kNN or the late-fusion arms.
- **Binary readout (LFC < −1):** AUROC/AUPRC are computed on the regression out-of-fold predictions (negated), not
  from a separate classifier.
- **6-dose LINCS stack:** each dose's 418 features get their own in-fold PCA (k = 5), giving 30 columns.
- **Inner-CV k** is one k from {4, 8, 16, 32} shared by both blocks.
- **Banded ridge** scales each modality block's penalty by c ∈ {0.25, 1, 4}.

**D-10 PyTorch model, details.**
- **Grid.** The 4-configuration grid is weight decay {1e-3, 1e-2} × block k {5, 16}.
- **BASE vector:** MoA one-hot + Morgan SVD 16 + nuisance (about 40 dims).
- **Early stopping** uses a grouped validation split of about 1/6 of the outer-training groups.
- **Threads.** Each worker process uses `torch.set_num_threads(1)` rather than 4, because folds run in parallel
  processes.
- **Scope.** The model is run on the primary tier and the scaffold split only (10 × 10), as costed in §5.

**D-11 Pre-freeze check cost control.** The design requires the re-simulation and the type-I check to be run on the
real feature dimensions with scaffold folds. For cost, each simulated dataset uses one repeat of the 10-fold
scaffold split (the analysis uses 10), with 99 permutations per dataset as in `power_sim_rule.py`. There are 200
datasets per null cell and 100 per power cell. Results are in `results/prefreeze_check.json` and PREREG.md.

**D-12 Target n differs slightly from the design counts. These are availability counts, not values.**
- **T1 primary: 183** as designed. Of these, 179 are HTS-only, 3 HTS + MTS004 and 1 MTS004-only, and 24 have more
  than one column (22 within the HTS columns used).
- **T1 key secondary (official `auc`): 93, not 94.** Four of the 97 structure-matched units have no A549
  curve-fit row in the 20Q2 curve-parameter file. The design's 94 counted units with ≥ 6 of 8 LFC doses.
- **T3 primary (U2OS): 135, not 136.** One unit's only U2OS primary column is at 0.94 µM, not the nominal 2.5 µM.
  The primary target is pinned to 2.5 µM (§1.5), so every primary-screen column is filtered to
  |log(dose/2.5)| < 0.2. This drops that unit and nothing in T1 or T2.
- **T2 primary: 156** as designed.
- **Dose-matched secondary LFC (T1):** 96 units at 2.5 µM and 95 at 10 µM (MTS010 preferred).

**D-13 Downloads.** The only new downloads are the two the design lists (§5): the PRISM 20Q2 secondary
curve-parameter file (290.2 MB, streamed once; only the A549/HepG2/U2OS rows are kept and the Repurposing Hub
columns are dropped) and the primary replicate-collapsed LFC matrix (46.7 MB, reduced to per-column medians over
the 578 lines). That is 337 MB in total, under the 0.5 GB budget. Both are recorded in `data/MANIFEST.json`. No
Step-2 input was re-streamed.

**D-14 CI.** The design's CI runs `ruff` as well as pytest. ruff is not installed here and was not added, so the
workflow runs pytest only, on ubuntu and windows with Python 3.13 and CPU torch. The guard job checks for tracked
data files and GDSC/Hub column names in `results/`.

**D-15 Two-plate subsets.** The pre-registered two-plate analysis (§3.6.3, descriptive) refits every arm on the
two-plate units only, with the scaffold 10 × 10 split. As an additional exploratory view, the primary out-of-fold
predictions are also scored on that subset.

**D-16 Compute budget.** Block permutations are pre-registered for the primary analysis only (§5): 2 × 1,000.
For the secondary targets and variants, the Freedman–Lane p-values are computed descriptively with 200
permutations each. They are not decision tests. The GBM is run on the primary analysis (all three split schemes),
the secondary targets and the two-plate subset, but not on the other T1 feature variants, which are ridge-only as
in §3.6. The PyTorch model runs on the primary scaffold split only (D-10). This keeps the total budget near the
design's 10–25 CPU-hours.
