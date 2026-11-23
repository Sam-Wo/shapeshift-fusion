# Results

The pre-registration is DESIGN.md, frozen in PREREG.md (tag `prereg-freeze`). Deviations and clarifications are
in DEVIATIONS.md; D-01 to D-16 were written before any target value entered a model. The analysis code is commit
`979205f`; its primary and secondary code paths are identical to the freeze commit `acb1934`/`cebc76d`. Every number
below is from `results/*.csv`, which hold aggregates only.

## Primary result: NULL

**The pre-registered rule is not met. Morphology plus transcriptome is *not* jointly informative about A549
viability beyond chemistry, MoA and cytotoxicity covariates.**

- **Setup:** T1 (LINCS A549 morphology at 3.33 µM × plate-centred Tahoe-A549 at 5 µM) × PRISM 20Q2 primary A549
  LFC, n = 183. Ridge with 5 in-fold PCs per block, generic-scaffold/Butina 10-fold × 10 repeats, nested 5-fold
  inner CV.
- **CIs** are 95% bootstrap intervals over compound groups (2,000 resamples, paired).

| Quantity | Estimate | 95% CI | One-sided 95% bound | Within-MoA Freedman–Lane p (1,000 perms) | GBM (LightGBM) Δ |
|---|---|---|---|---|---|
| ΔR²_T = R²(F) − R²(M), transcriptome added to BASE + morphology | **−0.027** | −0.064 to 0.010 | upper 0.004 | 0.445 | −0.008 |
| ΔR²_M = R²(F) − R²(T), morphology added to BASE + transcriptome | **+0.007** | −0.036 to 0.062 | upper 0.053 | 0.078 | −0.040 |
| min(ΔR²_T, ΔR²_M) | −0.027 | −0.064 to 0.005 | **upper −0.001** | | |

**Rule (§4.4):** all three conditions fail.
1. min(ΔR²) ≥ 0.03: **no.**
2. Both permutation p-values < 0.05: **no** (0.445, 0.078).
3. The GBM Δs share the ridge sign: **no** (GBM ΔR²_M is negative).

**Verdict** (mapping D-07): **NULL, bounded.** The one-sided 95% bootstrap upper bound of the joint gain,
min(ΔR²_T, ΔR²_M), is −0.001, below the 0.03 "meaningful" threshold.
- Per block, a transcriptome gain over BASE + morphology above 0.004 R² is excluded at 95% (one-sided).
- A morphology gain over BASE + transcriptome of up to about 0.05 is not excluded.
- Caveat on power: the pre-freeze check (PREREG.md) found the one-block rule has only about 0.45–0.51 power at a
  true δ = 0.08 on these features, so the bootstrap bound, not the test, is the informative part.

**R² by arm** (scaffold split; `results/arms_all_analyses.csv`; figure 1). Target-permutation p ≤ 0.001
(1,000 permutations) for every arm marked *.

| Arm | R² | 95% CI | Spearman ρ | AUROC (LFC < −1, 21 positives) |
|---|---|---|---|---|
| B0 mean | −0.012 | | | |
| B1 MoA-class mean* | 0.172 | −0.20 to 0.40 | 0.28 | 0.81 |
| B2 Morgan SVD ridge | −0.024 | | −0.10 | 0.39 |
| B2 Tanimoto kNN | −0.057 | | 0.07 | 0.57 |
| **B3 nuisance only*** | **0.517** | 0.29 to 0.66 | 0.32 | 0.93 |
| **BASE*** (MoA + chemistry + nuisance) | **0.512** | 0.32 to 0.61 | 0.41 | 0.94 |
| M = BASE + morphology* | 0.526 | 0.36 to 0.61 | 0.38 | 0.92 |
| T = BASE + transcriptome* | 0.492 | 0.31 to 0.59 | 0.43 | 0.93 |
| **F-early = BASE + both*** | **0.499** | 0.34 to 0.59 | 0.41 | 0.91 |
| F-late-avg / F-late-stack | 0.520 / 0.507 | | | |
| M-only* / T-only* (no BASE) | 0.342 / 0.262 | | 0.29 / 0.21 | 0.94 / 0.71 |
| GBM: BASE / M / T / F | 0.491 / 0.437 / 0.469 / 0.429 | | | |
| PyTorch two-encoder: F / z_t zeroed / z_m zeroed | 0.488 / 0.499 / 0.461 | | | |

- **Nothing beats BASE.** No fusion variant does: not early, late-average, stacking, GBM or the neural network.
  GBM, "the baseline to beat", does not beat ridge BASE either.
- **The NN expectation held.** As pre-registered, the NN does not beat ridge. Its built-in ablation shows a small
  transcriptome contribution inside the NN (0.488 vs 0.461 with z_m zeroed, Δ 0.027, CI 0.007–0.047), but the NN
  as a whole is below BASE.
- **The other split schemes agree** (§4.6: all three are reported).
  - LOCO: ΔR²_T −0.040 (−0.088 to 0.011), ΔR²_M −0.019 (−0.081 to 0.058).
  - Leave-one-MoA-class-out: ΔR²_T −0.012 (−0.038 to 0.016), ΔR²_M +0.013 (−0.052 to 0.099).
- **Within-MoA Freedman–Lane permutation:** 179 of 183 units are in permutable strata. The regressor rank is 53.
  The null ΔR² medians are −0.029 (T) and −0.017 (M), which shows the out-of-fold cost of adding a near-null
  block. Null distributions are in `results/primary_fl_null.csv` and figure 2.

## What the null result means (design §6, "What a null result would mean")

The pattern is "BASE ≈ M ≈ T ≈ F". Most of BASE's R² comes from the **nuisance covariates** (B3 alone: 0.517).
Two exploratory single-covariate fits make this concrete:
- morphology norm z alone: R² 0.45;
- Tahoe signature magnitude alone: R² 0.32.

In other words, both assays mainly see "how hard the drug hit the cells" (design risk 1). Once that magnitude is
in the model, neither the morphology PCs nor the transcriptome PCs add viability information, alone or together.
Without the nuisance block (exploratory), MoA + chemistry gives 0.18, + morphology 0.43, + transcriptome 0.31, and
+ both 0.46. The modalities do carry viability information, but it is the same information the activity
magnitudes carry.

## Secondary analyses (pre-registered; all descriptive)

All use the scaffold 10-fold × 10 split. Freedman–Lane p-values use 200 permutations here (D-16) and are not
decision tests. 95% bootstrap CIs are in brackets.

| Analysis | n | R² BASE / M / T / F | ΔR²_T | ΔR²_M | FL p (T, M) | GBM Δ_T, Δ_M |
|---|---|---|---|---|---|---|
| **Key secondary:** T1 × PRISM secondary AUC (official `auc`) | 93 | 0.516 / 0.506 / 0.543 / 0.526 | +0.020 [−0.011, 0.050] | −0.017 [−0.043, 0.010] | 0.040, 0.567 | −0.014, +0.013 |
| T2 HepG2 (EU-OS × Tahoe-C3A) × PRISM primary | 156 | 0.187 / 0.256 / 0.210 / 0.272 | +0.016 [−0.017, 0.045] | **+0.062 [0.021, 0.098]** | 0.637, 0.110 | −0.010, +0.227 |
| T3 U2OS (JUMP × Tahoe consensus) × PRISM primary | 135 | 0.071 / 0.128 / 0.096 / 0.144 | +0.016 [−0.010, 0.039] | **+0.048 [0.004, 0.092]** | 0.204, 0.657 | +0.017, −0.005 |
| T1 line-selective residual (A549 − median of 578 lines) | 183 | 0.182 / 0.132 / 0.117 / 0.100 | −0.032 [−0.073, 0.005] | −0.017 [−0.070, 0.022] | 0.294, 0.537 | −0.011, −0.080 |
| T1 dose-matched secondary LFC at 2.5 µM | 96 | 0.496 / 0.559 / 0.482 / 0.545 | −0.014 [−0.046, 0.020] | **+0.063 [0.012, 0.122]** | 0.154, 0.010 | −0.023, −0.050 |
| T1 dose-matched secondary LFC at 10 µM | 95 | 0.224 / 0.267 / 0.220 / 0.242 | −0.025 [−0.055, −0.002] | +0.023 [−0.004, 0.045] | 0.532, 0.284 | −0.041, −0.029 |
| T1 residualised PCs (Tahoe on magnitude + cell ratio; morphology on norm z) | 183 | 0.512 / 0.509 / 0.472 / 0.464 | −0.045 [−0.082, 0.003] | −0.008 [−0.052, 0.043] | 0.607, 0.224 | |
| T1 Tahoe direction only (L2-normalised) | 183 | 0.512 / 0.526 / 0.502 / 0.511 | −0.015 [−0.032, −0.000] | +0.009 [−0.040, 0.072] | 0.632, 0.294 | |
| T1 LINCS 6-dose morphology stack | 183 | 0.512 / 0.469 / 0.492 / 0.462 | −0.007 [−0.041, 0.043] | −0.030 [−0.128, 0.077] | 0.338, n/a | |
| T1 inner-CV k ∈ {4, 8, 16, 32} | 183 | 0.512 / 0.456 / 0.430 / 0.458 | +0.002 [−0.058, 0.058] | +0.028 [−0.043, 0.100] | | |
| T1 banded ridge (per-block penalty) | 183 | 0.512 / 0.485 / 0.506 / 0.472 | −0.012 [−0.045, 0.019] | −0.034 [−0.089, 0.034] | | |
| T1 interaction (F + 25 PC products) | 183 | 0.512 / 0.526 / 0.492 / 0.388 | −0.138 [−0.293, 0.009] | −0.104 [−0.269, 0.054] | | |
| T1 two-plate Tahoe units only | 61 | 0.761 / 0.652 / 0.733 / 0.639 | −0.013 [−0.061, 0.056] | −0.094 [−0.185, 0.011] | 0.353, 0.761 | +0.031, −0.008 |

**No secondary analysis shows joint information:** in none of them are both ΔR² ≥ 0.03. The consistent secondary
pattern is that **one modality dominates: morphology.** This is the design's "only one modality adds; not joint"
outcome.
- **Where morphology adds over BASE + transcriptome:**
  - T2 HepG2, 10 µM, 24 h: +0.062. Its GBM version is large: GBM_M 0.514 vs GBM_BASE 0.270.
  - T3 U2OS: +0.048.
  - T1 against the dose-matched 2.5 µM secondary LFC: +0.063, FL p = 0.010.
- **Where the transcriptome adds over BASE + morphology:** only on the key secondary AUC target (+0.020, CI
  includes 0, FL p = 0.04). It is below the 0.03 threshold.
- **Line-selective target.** On the line-selective residual target, which removes pan-line toxicity, BASE draws on
  MoA (B1 0.29), and neither modality adds anything (both Δ < 0). This fits the reading that what the modalities
  carry is generic toxicity.
- **Two-plate subset.** In the units where Step 2 found its agreement signal, there is no sign of joint
  information either.
- **Binary readout** (LFC < −1; descriptive, 21–22 positives). The AUROC of BASE / M / T / F is
  0.94 / 0.92 / 0.93 / 0.91 in T1, 0.79 / 0.85 / 0.81 / 0.85 in T2 and 0.78 / 0.80 / 0.78 / 0.80 in T3. With about
  21 positives the CI is about ±0.13 (§3.5).

These are secondary and uncorrected. They are hypothesis-generating only, especially T2's morphology gain, which
appears at the time-matched 24 h exposure.

## Exploratory (not pre-registered)

From `results/exploratory.csv`:
- **GDSC2 A549 AUC as an external check** (n = 28, aggregate only; GDSC is not redistributed).
  - Spearman ρ of out-of-fold predictions with GDSC AUC: BASE 0.59, M 0.56, T 0.61, F 0.61, GBM-F 0.55.
  - The PRISM primary target itself correlates with GDSC AUC at 0.44.
  - The models' predictions transfer to an independent assay about as well as the PRISM measurement does, and
    fusion is no better than BASE.
- **Primary out-of-fold predictions scored on the 61 two-plate units:** BASE 0.53, M 0.50, T 0.50, F 0.46.
- **Single nuisance covariates:**
  - morphology norm z: R² 0.45;
  - Tahoe magnitude: 0.32;
  - the cell-count ratio, two-plate flag, n_wells and block-reliable flag: about 0 each.
- **Without the nuisance block:** MoA + chemistry 0.18, + morphology 0.43, + transcriptome 0.31, + both 0.46.
- **BASE + ~200 RDKit descriptors:** 0.34. That is worse than BASE, because the extra columns cost shrinkage.

## Compute

- **Hardware:** a Ryzen 7 5800X3D (8 cores, 16 threads), CPU only, with 15 worker processes.
- **Pre-freeze check:** 27 min wall time.
- **`scripts/run_all.py`:** 101.6 min wall time. The primary scaffold analysis took 36 min of that: GBM 5 min, NN
  20 min, block permutations 10 min and target permutations 1.6 min. Per-stage times are in
  `results/timing.csv`.
- **Total CPU:** at most about 24 CPU-hours for `run_all` (worker-slots × wall time of the parallel stages) plus
  about 7 for the pre-freeze check. That is within the design's 10–25 CPU-hour estimate plus the check.
- **Permutations:** the primary analysis ran its full budget (2 × 1,000 block permutations, 1,000 target
  permutations, 2,000 bootstrap resamples). The secondaries used 200 block permutations each (D-16).

## Limitations

- **n.** The study has n = 183, and the real-feature power of the decision rule is about 0.5 per block at
  δ = 0.08. Small joint gains (0.03–0.05) could be missed, although the bootstrap bound argues against them here.
- **PRISM primary is a single-dose, noisy target.** The bootstrap CIs on R² are wide (about ±0.15).
- **MoA control is partial:** 102 of 183 units are "unclear", and the bootstrap does not correct for MoA
  clustering (§4.2).
- **The decisive driver is cytotoxicity** (activity magnitude) seen at 24–48 h. This is exactly the trivial
  pathway the design anticipated.
