# ShapeShift-Fusion: design doc

*Draft 2026-11-22. Design only: no model has been trained. The only computation so far is metadata matching, a few target probes, and a power simulation on synthetic data.*

**Question.** ShapeShift Step 2 found that Cell Painting morphology and Tahoe-100M transcriptomic signatures rank compounds very differently: the all-unit Mantel r is 0.02–0.05 after plate-centring (`../shapeshift/STEP2_REPORT.md`). Little *shared* structure does not mean the two are *jointly useless*. They could still carry complementary information about a phenotype neither predicts well alone. This project tests one such phenotype, drug sensitivity (cell viability), on the compounds that have both modalities.

**How numbers are marked**
- **[V]** verified. The source is linked, or it is a local file produced by the scripts in `scripts/` from data downloaded on 2026-11-22.
- **[E]** an estimate, an assumption or a simulation.
- **[S2]** carried over from the ShapeShift Step-2 report, which was not re-verified here.

---

## TL;DR

1. **The honest n is about 140–190 compounds per tier, and the target that delivers it is PRISM's single-dose primary screen.** The PRISM source of record is the 20Q2 release; the 24Q2 release was not probed.

   | Tier | Viability target | Matched compounds [V] |
   |---|---|---|
   | T1: A549 × LINCS × Tahoe-A549 | PRISM primary (2.5 µM, 5 days) | **186** (183 by structure = primary n) |
   | T2: HepG2 × EU-OS × Tahoe-HepG2/C3A | PRISM primary | **156** |
   | T3: U2OS × JUMP × Tahoe consensus | PRISM primary | **140** |

   - **Dose-response targets give much smaller sets:** PRISM secondary has 98 / 32 / 69 matched compounds [V].
   - **GDSC2 is too small to use as a target:** 29 / 25 / 14 compounds [V]. For C3A and U2OS, only 4–7% of the matched drugs reach an IC50 inside the tested range.
   - **The replicated-Tahoe subset, where Step 2 found its signal, is too small** for anything but exploration: 61 / 50 / 38 compounds on the primary target [V].
2. **Three corrections to the brief, all [V]:**
   - U2OS *is* in GDSC2 (as U-2-OS, 294 drugs) and in PRISM.
   - GDSC2 has HepG2/C3A (Tahoe's exact line) but not the HepG2 parent.
   - PRISM has the HepG2 parent. Its secondary dose-response data exist only in the 20Q2 release, not 19Q4.
3. **Licences:**
   - PRISM 19Q4 and 20Q2 are **CC BY 4.0** [V].
   - GDSC (now distributed through Sanger Cell Model Passports) falls under the Sanger DepMap Data Usage Policy [V]. That grants a "non-exclusive, non-transferable right to use data files for internal proprietary research and educational purposes", excludes resale and commercial services, and requires prior consent for commercial use. The policy is **silent on redistributing derived data**, so the repo treats GDSC as do-not-redistribute.
   - PRISM's MoA/target columns are Broad Repurposing Hub annotations, which carry a non-commercial notice [V, via `../shapeshift/docs_staging/DATA_LICENSES.md`].
4. **Baselines are strong, and the features leak viability.**
   - MoA class alone predicts PRISM A549 response at leave-one-out R² 0.32–0.38 [V, probe].
   - Tahoe signature magnitude correlates with PRISM response at ρ = −0.16 to −0.26 [V, probe].
   - Every comparison therefore puts chemistry, MoA and cytotoxicity nuisance covariates in *both* arms.
5. **Pre-registered primary test:** on T1 × PRISM-primary-A549, does morphology + transcriptome beat *each* single modality, on top of the covariate base? The test is an intersection–union test on out-of-fold ΔR², using scaffold-grouped CV, ridge with 5 in-fold PCs per modality, and a within-MoA conditional permutation.
   - The minimum effect that counts as "jointly informative" is ΔR² ≥ 0.03.
   - **Simulated power is modest, and the simulation is optimistic** [E]. For one block, the decision rule has 80% power at a true ΔR² ≈ 0.08, 0.44 at 0.05, and well below that at 0.03 (not simulated). Both blocks must pass, so joint power is ≈ 0.7 at 0.08.
   - A null result therefore bounds the joint gain; it does not rule out small gains.
6. **Cost:** about 10–25 CPU-hours on the existing laptop. No GPU is needed, the cloud cost is **$0**, and further downloads are under 0.5 GB [E].

---

## 1. Targets

### 1.1 What the primary sources say

**GDSC2** (Genomics of Drug Sensitivity in Cancer, release 8.5, file dated 27 Oct 2023)

- **Where it lives now.** `cancerrxgene.org/downloads/bulk_download` now only says that GDSC "has been integrated into the Sanger DepMap" and links to the Cell Model Passports downloads page [V: https://www.cancerrxgene.org/downloads/bulk_download, fetched 2026-11-22]. The old Sanger FTP release directory returns HTTP 410 [V: https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/].
- **Files used** [V: https://cellmodelpassports.sanger.ac.uk/downloads]:
  - `GDSC2_fitted_dose_response_27Oct23.xlsx` (22.1 MB): https://cmp.cog.sanger.ac.uk/download/GDSC2_fitted_dose_response_27Oct23.xlsx
  - `screened_compounds_rel_8.5_inchi.csv` (61 kB), with InChIKeys and ChEMBL IDs but no SMILES: https://cog.sanger.ac.uk/cmp/download/screened_compounds_rel_8.5_inchi.csv
  - `model_list_20261013.csv` (0.95 MB)
- **Size of the file:** 242,036 rows, 969 cell lines, 295 `DRUG_ID`s and 286 drug names [V, local]. The Sanger docs page says 297 compounds [V: https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/].
- **Response metrics per (line, drug)** [V, local columns]:
  - `LN_IC50`, the natural log of the fitted IC50 in µM; it is extrapolated when the IC50 is above the highest tested dose;
  - `AUC`, the fraction of the area between the lowest and highest screened concentration;
  - `Z_SCORE`, which is LN_IC50 standardized across lines;
  - `RMSE`, plus `MIN_CONC` and `MAX_CONC`.
- **Assay** [V: the same docs page]: CellTiter-Glo read 72 h after drugging, 7 concentrations, a 1000- or 1024-fold range.

| Line | Sanger model ID | In GDSC2? | Drugs measured [V, local] |
|---|---|---|---|
| A549 | SIDM00903 | yes | 281 |
| Hep-G2 (parent) | SIDM00904 | in the model list, **no GDSC2 rows** | 0 |
| C3A (= Hep-G2/C3A, Tahoe's line CVCL_1098) | SIDM01237 | yes | 279 |
| U-2-OS | SIDM01191 | **yes** (the brief assumed not) | 294 |

**PRISM Repurposing**

- **Assay** (Corsello et al., *Nat Cancer* 2020 [V: https://pmc.ncbi.nlm.nih.gov/articles/PMC7328899/]): pooled, barcoded cell lines.
  - Primary screen: one dose, 2.5 µM.
  - Secondary screen: 8 points, 4-fold dilution from 10 µM ("10 μM to 600 pM").
  - Both use a 5-day incubation ("After a five-day incubation, cells were lysed").
  - Response is the log2 fold change against DMSO after ComBat correction (readmes, [V, local]).
- **19Q4 release** [V: https://api.figshare.com/v2/articles/9393293; DOI 10.6084/m9.figshare.9393293.v4; licence field "CC BY 4.0"]: 4,518 compounds × 578 lines in the primary screen, and 1,448 compounds × 489 lines in the secondary.
- **20Q2 release** [V: https://api.figshare.com/v2/articles/20564034; DOI 10.6084/m9.figshare.20564034.v1; "CC BY 4.0"]: primary 5,274 compounds per the figshare description, or 5,101 per its own readme [V, local]. The secondary screen adds a PR300 cell pool.
  - **Source of record:** 20Q2. 19Q4 is reported in parentheses where it differs.
- **24Q2 "Repurposing-1M / Repurposing-300":** 1,522 compounds at 2.5 µM over 5 days × 906 lines [V: https://forum.depmap.org/t/announcing-the-24q2-release/3312]. It is distributed through the DepMap portal, not figshare, and was **not probed**. It is a candidate extension for v2.

| Line | DepMap ID | Primary 19Q4 / 20Q2 | Secondary 19Q4 | Secondary 20Q2 |
|---|---|---|---|---|
| A549 | ACH-000681 | data row present / present | present | present (PR500 pool) |
| HepG2 (parent) | ACH-000739 | present / present | **listed in cell-line-info, but no data row**. It was pooled only in the primary-screen pools (HTS, MTS004), not in PR500 [V, `secondary-screen-pooling-info.csv`] | **present (PR300 pool, MTS010)** |
| HepG2/C3A | n/a | not in PRISM | not in PRISM | not in PRISM |
| U2OS | ACH-000364 | present / present | present | present (PR500 and PR300) |

"Present" means the row was fetched and has non-missing values [V, local `data_probe/prism*/…_LFC_*.csv`].

### 1.2 Licences [V]

| Source | Licence | Consequence for a public repo |
|---|---|---|
| PRISM 19Q4 / 20Q2 (figshare) | CC BY 4.0 (figshare API `license` field, links above) | Derived targets can be redistributed with attribution and a note of changes. Cite Corsello et al. 2020 and the figshare DOI. |
| PRISM columns `moa`, `target`, `indication`, `phase` | These are Broad Drug Repurposing Hub annotations, which carry "provided for non-commercial use only" | Do **not** redistribute these columns. Use Tahoe's own `moa-fine` for MoA instead (Tahoe-100M is CC0). Details: `../shapeshift/docs_staging/DATA_LICENSES.md`, flag 1. |
| GDSC2 via Cell Model Passports / Sanger DepMap | "non-exclusive, non-transferable right to use data files for internal proprietary research and educational purposes"; resale and commercial services are excluded; "Commercial use … is not permitted without prior consent" [V: https://depmap.sanger.ac.uk/documentation/data-usage-policy/]. The page is silent on redistributing derived data. | Do **not** commit GDSC tables or derived per-compound values. Ship a download script and aggregate metrics only. For anything more, ask depmap@sanger.ac.uk. |
| Tahoe-100M | CC0 1.0 [S2 / DATA_LICENSES.md row 1] | Free to use |
| Cell Painting Gallery (LINCS cpg0004, JUMP cpg0016, EU-OS cpg0036) | CC0 1.0; the EU-OS Zenodo aggregated profiles are CC BY 4.0 [DATA_LICENSES.md rows 2 and 5] | Attribution for EU-OS |
| PubChem (used to turn GDSC InChIKeys into SMILES) | NCBI places no restrictions [DATA_LICENSES.md row 7] | Fine |

### 1.3 Actual compound overlap [V]

**How the overlap was matched**
- **Script:** `scripts/count_overlap.py`, which imports `../shapeshift/overlap.py` for `std_key`, `std_mol`, `norm_name` and `_pc_get`; nothing was reimplemented.
- **Unit set:** the Step-2 units of tiers T1, T2 and T3 that have a Tahoe signature at 5 µM, from `step2/units.csv` plus the Tahoe parquet files. That is 187, 171 and 175 units.
- **PRISM:** the per-batch SMILES string goes through `std_key` and is compared with the unit, which is itself a `std_key` of Tahoe's SMILES.
- **GDSC:** GDSC ships InChIKeys, not SMILES. Each InChIKey goes to PubChem for SMILES, then through `std_key`.
  - 501 of 621 compound rows have an InChIKey. All 417 unique keys resolved, with no transient failures [V, `data_probe/pubchem_cache_fusion.json`].
  - The 120 rows without one are mostly in-house tool compounds.
- **Name fallback:** `norm_name` on drug names and synonyms against Tahoe and dataset names, used only when there is no structure match. It is flagged and excluded from the primary analysis.
- **Names matched only by fallback** (all are flagged): rapamycin (GDSC's InChIKey does not std_key-match Tahoe's sirolimus), doxycycline, piroxicam, meloxicam, methylprednisolone, abiraterone and medroxyprogesterone. The last two may be ester prodrugs, so they need a hand check.
- **Counting rule:** a compound counts as "measured" in a line if that line's row has a value. For the secondary screen it must have ≥6 of the 8 doses, and MTS010 is preferred where present (readme recommendation).

**Overlap per tier and target.** The source of record is PRISM 20Q2; 19Q4 values are in parentheses where they differ.

| Route | Units with morph + Tahoe 5 µM | GDSC2 | PRISM primary (2.5 µM) | PRISM secondary (dose-response) | GDSC2 ∪ PRISM-sec |
|---|---|---|---|---|---|
| **T1** A549 × LINCS × Tahoe-A549 | 187 | **29** (1 by name) | **186** (3 by name) | **98** (97) (4 by name) | 99 |
| ↳ of which two-plate Tahoe units at 5 µM | 61 | 10 | 61 | 31 | 32 |
| ↳ annotated MoA / MoA classes with ≥3 members / Murcko scaffolds | | 25 / 2 / 27 | 84 / 13 / 159 | 63 / 9 / 87 | |
| **T2** HepG2 × EU-OS × Tahoe-HepG2/C3A | 171 | **25** (C3A) | **156** (HepG2 parent) | **32** (19Q4: **0**) | 42 |
| ↳ two-plate units | 56 | 8 | 50 | 5 | 11 |
| ↳ annotated / classes ≥3 / scaffolds | | 23 / 3 / 25 | 74 / 13 / 140 | 29 / 4 / 32 | |
| **T3** U2OS × JUMP × Tahoe consensus → U2OS target | 175 | **14** | **140** (135) | **69** (64) | 70 |
| ↳ two-plate units | 45 | 6 | 38 | 19 | 19 |
| T3 → A549 target (morphology line ≠ target line) | 175 | 15 | 140 | 70 | 71 |
| T3 → HepG2(/C3A) target | 175 | 15 | 135 | 25 | 30 |

Sources: `data_probe/overlap_counts_prism20q2.csv` and `data_probe/overlap_counts.csv` (19Q4). Tier-level two-plate counts (61 / 56 / 45) are [S2]. Two-plate counts come from `step2/tahoe_consistency.csv` (`n_plates ≥ 2` at 5 µM).

**What the counts mean, plainly**
- **GDSC2 is too small on every route** (14–29 compounds).
  - Matched GDSC drugs are mostly inactive in C3A and U2OS. Only 4% (C3A) and 7% (U2OS) have IC50 below the highest dose, and the AUC SD is 0.06–0.07.
  - A549 is better: 52% have IC50 in range, and the AUC SD is 0.14.
  - GDSC2 is used only as an external sanity check on A549.
- **PRISM secondary for HepG2 (32) is too small**, and the two-plate subsets of every secondary route (5–31) are too small as well.
- **PRISM primary is the only target with n ≥ 140 on every tier.** It is single-dose, so it is noisy and mostly inactive: only 12–15% of matched compounds have LFC < −1 at 2.5 µM.
- **T1 × PRISM secondary (n = 98) is the only dose-response target of usable size.** Its compounds were selected for primary-screen activity, so 60% reach LFC < −1 at some dose.

### 1.4 Target properties (probe, [V] local, `scripts/probe_targets.py` → `data_probe/probe_targets.json`)

The probes ran on PRISM 19Q4 over the union of all 300 Step-2 units. The secondary AUC used here is **self-computed** by the trapezoid rule on viability = 2^LFC over log10 dose, so it is an estimate. The project will use the official `auc` column from `secondary-screen-dose-response-curve-parameters.csv` (290 MB for 20Q2; not downloaded).

| Probe | Result |
|---|---|
| A549: PRISM secondary AUC vs PRISM primary LFC (same lab, different screen) | ρ = 0.65 (n = 134) |
| A549: GDSC2 AUC vs PRISM secondary AUC (different lab and assay) | ρ = 0.51 (n = 34) |
| A549: GDSC2 AUC vs PRISM primary LFC | ρ = 0.48 (n = 36) |
| Cross-line, PRISM primary: A549 vs HepG2 / A549 vs U2OS | ρ = 0.36 / 0.51 (n = 252) |
| Cross-line, PRISM secondary AUC: A549 vs U2OS | ρ = 0.72 (n = 128) |
| MoA-class mean, leave-one-out R²: A549 primary LFC / A549 secondary AUC | **0.32** (n = 116 annotated) / **0.38** (n = 84) |
| MoA-class mean, LOO R²: HepG2 primary / U2OS primary / U2OS secondary | 0.10 / 0.22 / 0.07 |
| Leakage: A549 primary LFC vs Tahoe-A549 5 µM signature L2 norm (raw) | ρ = −0.19 (p = 0.002, n = 252) |
| Leakage: A549 secondary AUC vs Tahoe-A549 norm / vs centred consensus magnitude | ρ = −0.26 / −0.23 (n = 134) |
| Leakage: HepG2 primary vs Tahoe-C3A norm | ρ = −0.16 (n = 250) |
| Leakage: PRISM vs Tahoe log2 cell-count ratio (n_cells_trt / plate × line median) | ρ = −0.04 to +0.11, n.s. |

**What this means for the design**
- **Ceiling.** Target reliability limits what any model can reach. Treating ρ ≈ 0.5–0.65 between assays as a reliability proxy puts the attainable R² against a *true* sensitivity value at roughly 0.25–0.42 [E].
- **Baselines are not trivial.** MoA alone already gets about 0.3 in A549.
- **Viability is partly pan-line toxicity.** That is why a line-selective residual target is a secondary analysis (§3.6).
- **The Tahoe magnitude carries viability information** that is not specific to any mechanism. The Tahoe cell-count ratio is a weak proxy here, unlike the within-A549 norm ↔ cell count ρ = −0.70 [S2].

### 1.5 Target decision (pre-registered)

| Role | Tier × target | n | Metric |
|---|---|---|---|
| **PRIMARY** | T1 × PRISM 20Q2 primary, A549, replicate-collapsed LFC at nominal 2.5 µM (actual 2.37–2.59 µM for the matched T1 columns [V, local]). **Screen pinned:** HTS. 179 of the 183 units are HTS-only, 3 are HTS + MTS004 and 1 is MTS004-only [V, local]; MTS004 is used only where HTS is missing. 24 units have more than one column (several broad_ids or batches; median within-unit LFC range 0.49 [V]); use the **median** over the unit's HTS columns. | 186 (183 by structure; the 3 name matches are dropped → **183**) | out-of-fold R²; ΔR² test (§4.4) |
| Key secondary | T1 × PRISM 20Q2 secondary, A549, official `auc` (MTS010 preferred) | 94 by structure | R², Spearman |
| Secondary | T2 × PRISM primary, HepG2 parent | 156 | same |
| Secondary | T3 × PRISM primary, U2OS (morphology line matched, transcriptome = Tahoe consensus) | 136 by structure | same |
| Secondary | T1 × PRISM primary, line-selective residual: A549 LFC minus the median LFC over all PRISM lines (needs the full 46.7 MB primary matrix) | 183 | same |
| Exploratory | two-plate Tahoe subsets (38–61); GDSC2 A549 AUC (29) as an external check of predictions | small | descriptive only |
| Binary readout (secondary) | sensitive = PRISM primary LFC < −1 (viability < 50%) | about 22 positives in T1 [E: 12% × 183] | AUROC, AUPRC |

---

## 2. Inputs

All inputs are existing ShapeShift Step-2 outputs, loaded through Step-2 code; no pipeline is rewritten.

### 2.1 Morphology (per tier, never merged across tiers)

| Tier | File | Dimensions [V] | Dose used |
|---|---|---|---|
| T1 | `step2/morph_lincs.parquet` (A549, 48 h) | 418 sphered features | **3.33 µM** (closest to PRISM 2.5 µM and Tahoe 5 µM). Sensitivity: the 6-dose stack (0.041–10 µM) as a "morphological dose-response" block, built from per-dose PCA scores |
| T2 | `step2/morph_euos.parquet` (HepG2, 24 h, 4 sites) | 530 | 10 µM (the only dose) |
| T3 | `morph_profiles_high.parquet` (JUMP U2OS, 48 h) | 532 | 10 µM. JCP → unit mapping as in `compare_tiers.load_morph` |

- **Reuse.** Profiles are already robust-z'd per plate, feature-selected and PCA-whitened on controls [S2, `step2/MORPH_SPEC.md`]. Load them with `compare_tiers.load_morph(tier, units)`.
- **Nuisance covariates from morphology:** consensus norm z (`norm_z` in the activity CSVs), `n_wells`, and for T1 the "block-reliable" flag from `lincs_notes.md`.

### 2.2 Transcriptome (Tahoe-100M, 24 h)

- **Dose: 5 µM only.** It is the only Tahoe dose whose similarity structure replicates across plates (split-plate reliability 0.23–0.28 centred; ≈0 at 0.05 and 0.5 µM) [S2].
- **Signature:** the matched line's signature for T1 (A549, CVCL_0023) and T2 (HepG2/C3A, CVCL_1098), and the cross-line consensus for T3. There are 5,000 genes.
- **Plate-centred**, exactly as `compare_tiers.py` does it: `Tahoe().sig(which, 5.0, centered=True)`, which subtracts the mean of `step2/tahoe_plate_line_mean.parquet` (plate × dose × line) over the unit's own plates. Import the class; do not copy it.
- **Covariates kept alongside:** `n_plates` (1 vs ≥2), the centred magnitude, and the log2 cell-count ratio (`n_cells_trt` over the plate × line median, as in `scripts/probe_targets.py`).
- **Sensitivity variant:** use direction only (L2-normalised signature), which removes the magnitude that tracks cytotoxicity.

### 2.3 Chemistry and annotation

- **Morgan fingerprints:** radius 2, 2,048 bits, plus counts, computed on `overlap.std_mol(canonical_smiles)` from Tahoe `drug_metadata.parquet`.
- **Second chemistry view:** about 200 RDKit 2-D descriptors.
- **MoA:** Tahoe `moa-fine`, one-hot. "unclear" (~50% of units [S2]) is its own level. The Repurposing Hub MoA is *not* used, for licensing reasons (§1.2).

### 2.4 Dimensionality reduction: inside the folds only

- **Fitting.** PCA per modality is fitted on the outer-training fold only, and the held-out compound is projected.
- **Number of components.** The primary analysis fixes k_morph = k_tx = 5 (pre-registered; see the power simulation in §4.5, where larger blocks cost power). As a secondary analysis, k is chosen from {4, 8, 16, 32} by inner CV.
- **Morgan bits:** TruncatedSVD (k ∈ {16, 32}) fitted in-fold for the linear models; raw bits are fed to the GBM.
- **No supervised step outside the folds:** no gene selection by target correlation, no PLS fitted on all data.
- **Standardisation and imputation** are fitted in-fold as well.
- **Enforcement.** Everything sits in one sklearn `Pipeline` per arm, so leakage is impossible by construction. A unit test checks it: shuffling the test-fold rows must not change the training-fold transforms.

---

## 3. Models

### 3.1 Feature arms

Each arm is fitted with the same learner and the same folds.

| Arm | Features |
|---|---|
| B0 | target mean of the training fold (null model) |
| B1 | MoA-class mean of the training fold (falls back to the global mean) |
| B2 | chemistry only: Morgan fingerprints → ridge on SVD, plus Tanimoto kNN (k ∈ {3, 5, 10}) |
| B3 | nuisance only: Tahoe magnitude, cell-count ratio, n_plates, morphology norm z, n_wells |
| **BASE** | B1 + B2 + B3 combined (MoA one-hot + chemistry SVD + nuisance) |
| M | BASE + morphology PCs |
| T | BASE + transcriptome PCs |
| **F-early** | BASE + morphology PCs + transcriptome PCs (concatenation) |
| F-late-avg | mean of the M and T predictions |
| F-late-stack | ridge meta-learner (non-negative weights) on the inner-CV out-of-fold predictions of M and T, fitted inside each outer-training fold |
| M-only / T-only | morphology or transcriptome without BASE (descriptive; shows raw modality signal) |

### 3.2 Learners

- **Ridge** is the primary learner. Its alpha is chosen in the inner CV.
- **GBM: "the baseline to beat".** It is a gradient-boosted tree model on the concatenated features [BASE + morphology PCs + transcriptome PCs + raw Morgan bits].
  - Implementation now: sklearn `HistGradientBoostingRegressor`.
  - Proposed for v1: LightGBM or XGBoost, for speed and monotone constraints; not installed now, so nothing is pip-installed.
  - Grid: depth {2, 3}, learning rate {0.03, 0.1}, `max_iter` {100, 300}, `min_samples_leaf` {10, 20}, l2 {0, 1}; early stopping off, because inner CV handles it.
  - Every fusion variant (stacking, PyTorch) must beat this GBM before it can be called *better fusion*. Beating BASE is what makes the modalities count as *jointly informative*.
- **Why ridge is primary.** At n ≈ 183, ridge on a few in-fold PCs has much lower variance than trees. The scientific question is about information content, not model capacity, so the GBM serves as a pre-registered robustness check: it must agree in sign.

### 3.3 Small PyTorch fusion model (secondary, CPU)

- **Inputs:** in-fold PCA scores, morphology (k_m ≤ 32) and transcriptome (k_t ≤ 32), plus the BASE vector (about 50 dims after SVD).
- **Encoders:** `Linear(k, 64) → GELU → Dropout(0.3) → Linear(64, 16)`, one per modality. LayerNorm on the input.
- **Head:** `concat[z_m, z_t, BASE] → Linear(·, 32) → GELU → Dropout(0.3) → Linear(32, 1)`. A skip connection carries a linear BASE term, so the network learns a residual over the ridge-like base.
- **Size:** about 7–9 k parameters [E: 2 × (32·64 + 64·16) + (82·32 + 32) ≈ 8.9 k].
- **Regularisation:**
  - AdamW with weight decay ∈ {1e-3, 1e-2};
  - dropout 0.3;
  - **modality dropout**: with p = 0.25 a whole encoder output is zeroed per sample, which makes the head robust to a weak modality and gives single-modality ablations for free at test time;
  - Gaussian input noise σ = 0.1;
  - Huber loss.
- **Training:**
  - batch 32, learning rate 1e-3 with cosine decay;
  - at most 300 epochs, early stopping on an inner-validation split (patience 30);
  - average of 5 seeds;
  - deterministic flags; `torch.set_num_threads(4)`.
  - One fit takes about 2–5 s on CPU [E].
- **Evaluation:** the same outer folds as ridge and GBM. The hyperparameter grid has 4 configurations, chosen in a 5-fold inner CV.
- **Pre-registered expectation:** at this n it will *not* beat ridge or GBM. It is included for completeness and for code quality, and its result is not part of the primary claim.

### 3.4 Splits

- **Unit of splitting: a "compound group".** It is the Step-2 unit (which already merges salts and hydrates), further merged with:
  - stereo/racemate-flagged pairs (`units.stereo_flag`);
  - name-bridge prodrug pairs, e.g. ixazomib / ixazomib citrate, which share LINCS wells.
- **Leave-one-compound-group-out (LOCO).** Reported, but not primary, for two reasons:
  - LOO R² is negatively biased for weak signals, because the training mean shifts against the held-out y;
  - close analogues leak through chemistry.
- **Scaffold split (PRIMARY):** 10-fold grouped CV.
  - Groups are generic Bemis–Murcko scaffolds (RDKit `MurckoScaffold`, made generic), merged by Butina clustering at Tanimoto ≥ 0.6 on Morgan bits so that near-identical scaffolds share a fold.
  - It is repeated 10 times with different group-to-fold assignments. 159 scaffolds for 186 compounds [V] means a plain scaffold split is nearly LOCO, hence the clustering.
- **MoA-grouped split:** leave-one-MoA-class-out over annotated classes. Each "unclear" compound is its own group.
  - In this split B1 collapses to the global mean by construction.
  - The question it answers is whether the features generalise *beyond* known classes.
- **Nested CV:** every hyperparameter (alpha, k, GBM grid, NN grid, stacking weights) is chosen in a 5-fold inner CV that uses the same grouping rule inside each outer-training set.

### 3.5 Metrics and inference

- **R²:** out-of-fold, R² = 1 − SSE / SST, with SST taken over the evaluated compounds against the *pooled* mean. It is averaged over the 10 split repeats.
- **Spearman ρ** of the out-of-fold predictions.
- **For the binary readout:** AUROC and AUPRC.
  - With about 22 positives among 183, an AUROC of 0.70 has a 95% CI of about ±0.13 [E, Hanley–McNeil].
  - So AUROC is descriptive only.
- **CIs:** bootstrap over compound groups (2,000 resamples) of the fixed out-of-fold predictions. Differences between arms use a *paired* bootstrap, with the same resample applied to both arms.
- **Permutation tests:**
  - *Target permutation* (y shuffled, whole nested pipeline rerun, 1,000×) asks whether each arm is better than chance.
  - *Conditional block permutation* is the test for joint information. The *residual* of the transcriptome block (or the morphology block), after regression on BASE and the other block, is shuffled across compounds *within MoA class* and the fitted part is added back (Freedman–Lane style). Then the pipeline is rerun 1,000×. This keeps the block's correlation with BASE (e.g. Tahoe magnitude) intact.
  - Shuffling within MoA keeps the block's MoA-level information, so the test asks for information *beyond* MoA.

### 3.6 Secondary analyses (all labelled secondary)

1. The key secondary target, the other tiers, and the line-selective residual target (§1.5).
2. **Residualised features:** the Tahoe PCs are regressed in-fold on (magnitude, log cell-count ratio), and the morphology PCs on norm z. This answers whether any gain survives once "how hard the drug hit the cells at 24–48 h" is removed.
3. **Two-plate Tahoe units** (61 in T1), descriptive. Step 2's agreement signal lived there.
4. **Dose-matched secondary target:** PRISM secondary LFC at 2.5 µM (A549), and at 10 µM to match the T2/T3 morphology dose.
5. **Interaction:** F-early with pairwise morphology × transcriptome PC products, ridge only. Does the combination matter beyond additivity?

---

## 4. Small-n honesty

### 4.1 n, stated up front

The primary analysis has **183** compounds (T1, structure-matched, PRISM primary A549) [V]. It is *not* the ~100–200 originally hoped for in dose-response form:
- dose-response targets give 94 (T1), 32 (T2) and 66 (T3) by structure;
- GDSC2 gives 14–29.

### 4.2 Effective sample size is smaller than n [E]

- **MoA classes.** 84 of 186 T1 compounds are annotated, in 13 classes with ≥3 members. Same-class compounds are not independent draws.
- **Plates.** Tahoe plates hold about 80 drugs, and same-MoA pairs share a plate 40% of the time against 30% otherwise [S2]. Plate-centring removes the shared-control component, but not plate-level batch interactions.
- **Bootstrap level.** The bootstrap resamples compound *groups*, but no correction is made for MoA clustering. CIs are therefore likely somewhat too narrow; the MoA-grouped split is the check.

### 4.3 Single-modality detectability (analytic, Fisher z, two-sided α = 0.05, power 0.8) [E]

| n | Smallest detectable ρ (R² ≈ ρ²) |
|---|---|
| 183 | 0.21 (0.043) |
| 136–156 | 0.22–0.24 |
| 94–98 | 0.28 (0.08) |
| 61 | 0.35 (0.12) |
| 29 | 0.50 (0.25) |

### 4.4 PRE-REGISTERED primary comparison

- **Data:** T1, PRISM 20Q2 primary A549 LFC at 2.5 µM, 183 structure-matched units.
- **Features:** LINCS 3.33 µM morphology and plate-centred Tahoe-A549 5 µM.
- **Learner:** ridge with in-fold PCA (k = 5 per modality block), using the scaffold-cluster 10-fold × 10-repeat CV of §3.4.
- **Statistics:**
  - ΔR²_T = R²(F-early) − R²(M), the transcriptome's added value over BASE + morphology;
  - ΔR²_M = R²(F-early) − R²(T), the morphology's added value over BASE + transcriptome.
- **"Jointly informative" is declared only if all of the following hold.** This is an intersection–union test, so the α of 0.05 needs no further correction.
  1. min(ΔR²_T, ΔR²_M) ≥ **0.03** (point estimates).
  2. The within-MoA *residual* (Freedman–Lane-style) block permutation (1,000 permutations; §4.5 item 3) gives p < 0.05 for *both* blocks. This is the decision test.
  3. The GBM-on-concatenation version of both Δ has the same sign.
- **Reported, but not a gate:** the one-sided 95% paired-bootstrap lower bounds of ΔR²_T and ΔR²_M. The bootstrap is too conservative to serve as the decision test (§4.5).
- **Why 0.03.**
  - The target-reliability ceiling is R² ≈ 0.25–0.42 (§1.4), and BASE is expected to take about 0.3 of that. That leaves an explainable headroom of roughly 0.05–0.15 [E].
  - 0.03 is about a quarter of that headroom, the smallest gain that would matter for prioritising compounds.
  - It is **not** reliably detectable at this n: the 80%-power detectable effect is a true ΔR² ≈ 0.08 per block (§4.5). The threshold defines *meaningful*; the power analysis defines what the study can actually *see*.
- **Weaker outcome, reported but not called "joint":** the transcriptome adds to BASE + morphology but not the reverse. This means one modality dominates; it is not evidence of joint information.
- **Freezing:** before any real target is loaded into a model, this section is copied to `PREREG.md` and its SHA-256 is committed. Before freezing, two checks are run:
  - `power_sim_rule.py` is rerun with the *real* feature dimensions (BASE about 50 columns, k = 5) and scaffold-grouped folds, using random targets drawn independently of the real ones;
  - the residual-permutation type-I rate must be ≤ 0.06 at δ = 0.

### 4.5 Power of the primary test (simulation, [E], `scripts/power_sim.py` → `data_probe/power_sim.json`, `power_sim_k5.json`)

**Synthetic setup**
- BASE has 10 features with true R² = 0.30.
- Morphology has 20 PCs, with a unique true R² of 0.08 spread over 3 PCs.
- The transcriptome has 20 PCs, with a unique true R² of δ spread over 3 PCs.
- The test: ridge (RidgeCV), 10-fold out-of-fold predictions, and a one-sided paired bootstrap of ΔR²_T; 200 simulated datasets per cell.

**Power of the *bootstrap* gate.** Not the decision rule (see the next table). It is the fraction of 200 simulated datasets in which the one-sided 95% paired-bootstrap lower bound of ΔR²_T is > 0.

| n | k = 20 PCs per block: δ = 0 / 0.03 / 0.05 / 0.08 / 0.12 | k = 5 PCs per block: δ = 0 / 0.03 / 0.05 / 0.08 / 0.12 |
|---|---|---|
| 61 | 0.005 / 0.04 / 0.05 / 0.07 / 0.13 | n/a |
| 98 | 0.005 / 0.04 / 0.05 / 0.08 / 0.21 | 0.005 / 0.10 / 0.16 / 0.38 / 0.66 |
| 140 | 0.01 / 0.02 / 0.06 / 0.14 / 0.46 | 0.005 / 0.09 / 0.25 / 0.51 / 0.82 |
| 156 | 0.005 / 0.02 / 0.05 / 0.25 / 0.45 | n/a |
| **186** | 0.00 / 0.02 / 0.08 / 0.26 / 0.60 | **0.005 / 0.14 / 0.34 / 0.65 / 0.975** |

The mean observed ΔR² at δ = 0 is −0.048 with k = 20 and −0.016 with k = 5, at n = 186. Every extra block of near-null PCs costs out-of-fold R², and ridge with a single shared α cannot avoid that cost.

**Power of the ACTUAL decision rule** (`scripts/power_sim_rule.py` → `data_probe/power_sim_rule.json`, [E])
- **Setup:** n = 186, k = 5, and a BASE feature correlated with the transcriptome block (r = 0.6, standing in for Tahoe magnitude). 100 simulated datasets per cell, 99 permutations each.
- **Rule simulated:** the rule for one block (transcriptome), i.e. observed ΔR²_T ≥ 0.03 **and** permutation p < 0.05.

| true δ | mean observed ΔR²_T | P(observed ≥ 0.03) | P(perm p < 0.05), naive row shuffle | P(perm p < 0.05), residual (Freedman–Lane) shuffle | **P(rule), residual** |
|---|---|---|---|---|---|
| 0 | −0.015 | 0.01 | 0.06 | 0.05 | **0.01** (type I) |
| 0.05 | 0.031 | 0.44 | not run | 0.70 | **0.44** |
| 0.08 | 0.059 | 0.83 | not run | 0.94 | **0.83** |
| 0.12 | 0.095 | 0.96 | not run | 1.00 | **0.96** |

Monte-Carlo SE ≈ ±0.05 per cell.

**Conclusions, built into the design**
1. **What this n can detect.**
   - Under the decision rule, the smallest *true* increment detectable with 80% power for one block is **δ ≈ 0.08** at n ≈ 183 with k = 5.
   - The IUT needs both blocks. If both have δ = 0.08 and the tests are roughly independent, joint power is ≈ 0.83² ≈ 0.69 [E].
   - At δ = 0.05 the power is 0.44. δ = 0.03 was not simulated under the rule; its power lies between 0.01 (δ = 0) and 0.44.
   - With k = 20 everything is worse: the bootstrap table shows the shrinkage cost doubling. Hence k = 5 is fixed for the primary analysis, and inner-CV k and banded ridge (one α per block) are secondary.
2. **The observed-value gate is the binding part.** Out-of-fold ΔR² runs about 0.015–0.025 *below* the true δ, because of ridge shrinkage on the extra block. The gate "observed ≥ 0.03" therefore corresponds to a true δ of about 0.05. This is intended: it is the effect size we would call meaningful.
3. **Gate and permutation scheme.**
   - The paired bootstrap is very conservative (type I 0.005), so it supplies CIs only.
   - The **pre-registered permutation is the residual (Freedman–Lane-style) block permutation:** regress the block on [BASE, other block] inside the analysis, shuffle the residual rows within MoA class, and add the fitted part back. It keeps the block's correlation with BASE (e.g. Tahoe magnitude).
   - In the simulation, the naive row shuffle was only marginally liberal (0.06). The residual scheme is still preferred because the real BASE–block correlation may be stronger than r = 0.6.
4. **The simulation is optimistic** in four ways:
   - its BASE has 10 features, whereas the real one has about 50 (§3.3), which costs more shrinkage;
   - it uses random rather than scaffold-grouped folds;
   - its signal sits in the top unsupervised PCs, and real signal may not;
   - its errors are Gaussian, whereas PRISM primary is zero-inflated.

   Real power is therefore lower. The re-simulation on the real feature dimensions under "Freezing" (§4.4) must be done before freezing.
5. **Meaningful versus detectable.**
   - The 0.03 observed-ΔR² floor defines *meaningful*.
   - The study can reliably *see* only true joint gains of roughly ≥ 0.08 per block.
   - A null result rules out large complementarity, not small.
   - The secondary targets (n = 94 and 136–156) have less power still.
6. **Consequence for scope.** If Step 2's picture holds (weak, noisy Tahoe signal), the realistic expectation is a null result with an informative upper bound. A positive result needs a large complementary effect.

### 4.6 What the analysis will not do

- No target or feature choices after seeing results: the arms, the grid and k are fixed above.
- No reporting of the best of the LOCO, scaffold and MoA splits; all three are reported, and the scaffold split is primary.
- No pooling of tiers into one model. Morphology features are dataset-specific [S2].
- No use of Tahoe p-values; the DESeq replicate scheme is unconfirmed [S2].

---

## 5. Compute estimate [E]

| Job | Fits | CPU time |
|---|---|---|
| Ridge, 11 arms × 3 split schemes × 10 repeats, nested (5 inner folds × ~16 grid points) | ~3 × 10⁵ small fits | ~0.5 h |
| Conditional block permutations, 2 blocks × 1,000 × ridge pipeline (primary only) | 2 × 10³ pipelines | 2–4 h |
| Target permutations, 1,000 per primary arm (5 arms) | 5 × 10³ pipelines | 2–4 h |
| GBM (sklearn HGB), 3 split schemes, 32-point grid nested | ~5 × 10⁴ fits at ~0.2 s | ~3 h |
| PyTorch fusion, 100 outer folds × 5 inner × 4 configurations × 5 seeds, at ~3 s per fit | 10⁴ fits | ~8 h (4 threads: ~2–3 h wall) |
| Secondary tiers and targets (about 4× the primary ridge and GBM, no NN permutations) | | ~4 h |
| **Total** | | **≈ 10–25 CPU-hours** on the current Windows laptop; a few hours of wall time if parallelised with joblib |

- **No GPU is needed.**
- **Cost: $0.** A cloud CPU box (e.g. 16 vCPU at about $0.5–0.8/h) would only shorten the wall time, for **< $20** total.
- **Additional downloads for the real run:**
  - PRISM 20Q2 secondary curve parameters (290 MB);
  - the full 20Q2 primary collapsed LFC (46.7 MB), for the line-selective residual target;
  - optionally the 24Q2 Repurposing data.
  - Total < 0.5 GB. Step-2 features are already on disk.

---

## 6. Risks

1. **Cytotoxicity leaks into the features.** This is the biggest risk.
   - Tahoe signature magnitude tracks treated-cell count (ρ = −0.70 within A549 [S2]). The same magnitude correlates with the PRISM target (ρ = −0.19 to −0.26, §1.4).
   - Morphology activity (norm z) and CellProfiler cell-count features carry the same information.
   - A "joint" gain could therefore just be that two assays both saw cells dying, at 24–48 h, which then predicts death at 5 days. That would be true but trivial.
   - *Mitigation:* nuisance covariates are in BASE in every arm; the residualised-feature and direction-only variants (§3.6); and the line-selective residual target, which removes pan-line toxicity (cross-line ρ is 0.36–0.72).
2. **MoA confounding.**
   - MoA class alone explains about 0.3 of the target variance in A549. Both modalities retrieve MoA (morphology mAP 0.26–0.37, transcriptome 0.17–0.19 [S2]), so either can look predictive just by recovering MoA.
   - Tahoe plates are partly MoA-structured [S2].
   - *Mitigation:* MoA one-hot in BASE; the within-MoA conditional permutation; the MoA-grouped split; plate-centring.
   - *Residual risk:* about 55% of T1 units are "unclear" (85 of 187 annotated [S2]), and more sit in tiny classes, so MoA is only partly controlled.
3. **Dose and time mismatch.**

   | Modality | Dose | Exposure |
   |---|---|---|
   | PRISM target [V, Corsello 2020] | 2.5 µM (primary) or 0.6 nM–10 µM (secondary) | 5 days |
   | GDSC target [V, Sanger docs] | 7 doses | 72 h |
   | Tahoe [S2] | 5 µM | 24 h |
   | LINCS morphology [S2] | 3.33 µM | 48 h |
   | EU-OS morphology [S2] | 10 µM | 24 h |
   | JUMP morphology [S2] | 10 µM | 48 h |

   - The nominal µM values are also not equivalent across media, plate formats and cell densities.
   - *Mitigation:* the dose-matched secondary target (§3.6), and reporting the LINCS 6-dose stack.
   - Below 5 µM, Tahoe has no reliable signal [S2], so lower Tahoe doses cannot be used.
4. **Cell-line mismatch.**
   - **T2:** morphology is HepG2 parent (EU-OS), the transcriptome is HepG2/C3A (Tahoe), and PRISM is HepG2 parent. GDSC is C3A, but it is too small to use.
   - **T3:** morphology and target are U2OS, but U2OS is not a Tahoe line, so the transcriptome is the cross-line consensus.
   - **T1** is fully matched (A549 everywhere).
   - Step 2 found the line match does not raise agreement [S2]. For viability it may matter more. T1 is primary partly for this reason.
   - PRISM's A549 is a barcoded, pooled derivative grown in RPMI, which may differ from LINCS's and Tahoe's A549 stocks [E].
5. **Noisy targets.**
   - PRISM primary is single-dose and ComBat-corrected, and 85–88% of matched compounds have LFC ≥ −1, i.e. viability above 50% [V, §1.3].
   - The cross-screen ρ of 0.65 means a ceiling well below 1.
   - *Mitigation:* a Huber loss for the NN, R² reported alongside Spearman, and the secondary AUC target.
6. **Selection bias.** PRISM secondary compounds were chosen *because* they were active in the primary screen, and Tahoe and the Cell Painting libraries are oncology-heavy. Results do not generalise to random chemical space.
7. **Matching errors.**
   - Name-fallback matches (1–4 per route) are excluded from the primary analysis.
   - Stereo-flat units collapse enantiomers, e.g. quinine/quinidine and doxorubicin/epirubicin [S2].
   - Ester prodrugs found by name need a hand check (abiraterone vs abiraterone acetate).
8. **Data licensing.**
   - GDSC cannot be redistributed and is non-commercial (§1.2). Keeping GDSC out of the primary analysis removes most of that exposure.
   - PRISM's Repurposing Hub columns are non-commercial; Tahoe `moa-fine` is used instead.
   - The public repo ships code, frozen counts and aggregate results only; raw tables come from download scripts.
9. **Forking paths.** The many arms, splits and tiers are handled by one pre-registered primary test (§4.4) and a fixed list of secondaries. Everything else is labelled exploratory.
10. **Dependency on the Step-2 artefacts** (about 2.8 GB, not public).
    - *Mitigation:* the new repo imports `shapeshift` at a pinned commit and records the SHA-256 of every Step-2 parquet it reads.

### What a null result would mean

| Outcome | Interpretation |
|---|---|
| BASE ≈ M ≈ T ≈ F (no modality adds to chemistry + MoA + nuisance) | At these doses and times, neither Cell Painting nor Tahoe carries viability information beyond what class membership and generic "activity" already give. This is a strong negative for both modalities as viability predictors, within the limits of n ≈ 183: a per-block increment of about ≥ 0.08 R² would probably (≥ 80%, optimistic simulation) have been seen; one of 0.03–0.05 could easily have been missed (§4.5). |
| M > BASE and T > BASE, but F ≈ max(M, T) | Each modality is informative, but the information is **redundant**. That fits Step 2: the shared part is mostly MoA and activity, and the non-shared part is noise. |
| Only one modality adds | Not joint. Report which one. Step 2 makes morphology the likelier winner (more reliable at the matched dose). |
| Primary null, but residual or two-plate secondaries positive | Hypothesis-generating only. It points to Tahoe noise as the limit, as in Step 2, and argues for replicate-well transcriptomics rather than larger models. |

A null result is **not** evidence that the modalities are complementary-but-undetectable, and it does not say either modality is useless for other tasks (MoA retrieval works in both [S2]). Its upper bound is the most useful thing to report: "joint gain < X R² at 95% confidence".

---

## 7. Repository layout (future public repo) and CI

```
shapeshift-fusion/
  README.md                 question, headline figure, how to reproduce, limitations
  DESIGN.md                 this file
  PREREG.md                 frozen copy of §4.4 + SHA-256, committed before any model sees targets
  DATA_LICENSES.md          per-source licence table (extends shapeshift/docs_staging/DATA_LICENSES.md)
  LICENSE                   MIT (code only)
  pyproject.toml            pinned deps; extras: [gbm] lightgbm, [nn] torch (CPU wheel)
  configs/
    primary.yaml            tier, target, arms, splits, grids, seeds (the pre-registered run)
    secondary_*.yaml
  src/fusion/
    units.py                load Step-2 units; group definition (unit + stereo/prodrug merges)
    targets/prism.py        range-read / download PRISM rows; LFC, official AUC, residual target
    targets/gdsc.py         GDSC2 via Cell Model Passports (never committed)
    features/morph.py       thin wrapper on shapeshift compare_tiers.load_morph
    features/tahoe.py       thin wrapper on compare_tiers.Tahoe(...).sig(..., centered=True)
    features/chem.py        Morgan / descriptors on overlap.std_mol; Murcko + Butina groups
    features/nuisance.py
    splits.py               LOCO, scaffold-cluster, MoA-grouped; nested inner splitters
    pipelines.py            sklearn Pipelines per arm (in-fold scaler/PCA/SVD)
    models/baselines.py     B0-B3, kNN-Tanimoto
    models/gbm.py           HGB now, LightGBM optional
    models/torch_fusion.py  encoders + head, trainer, modality dropout
    stats.py                paired bootstrap, conditional block permutation, IUT decision
    report.py               tables + figures
  scripts/
    fetch_targets.py        (from scripts/fetch_prism*_rows.py, count_overlap.py here)
    run.py --config configs/primary.yaml
    power_sim.py
  tests/
    fixtures/               synthetic 60-compound dataset (no real data)
    test_no_leakage.py      transforms fitted on train only; group integrity across folds
    test_splits.py          stereo/prodrug pairs never straddle folds
    test_stats.py           bootstrap/permutation calibration on null data (type-I ≈ 0.05)
    test_torch_smoke.py     2-epoch CPU fit, deterministic output
    test_overlap_counts.py  (marked `network`) recompute §1.3 counts, compare to frozen CSV
  results/                  aggregate metrics only (no GDSC values, no Hub columns)
  data/                     .gitignored; MANIFEST.json with URL, bytes, SHA-256 per file
```

- **Reuse, not copies.** `shapeshift` (`overlap.py`, `step2/compare_tiers.py`) is a dependency pinned to a commit, either as a git submodule or as a small installable package. Matching and plate-centring are not copied.

**CI (GitHub Actions)**
- **On every push and PR:** a matrix of ubuntu-latest and windows-latest on Python 3.13. It runs `ruff check` and `ruff format --check`, then `pytest -m "not network"` on the synthetic fixtures.
  - That includes the CPU torch smoke test, installed with `--index-url https://download.pytorch.org/whl/cpu`.
  - Target runtime is under 5 minutes.
- **Guard job:** it fails if any file under `data/` is tracked, if a GDSC-derived column name appears in `results/`, or if `PREREG.md` changes after its tagged commit.
- **Weekly scheduled job, allowed to fail:** `pytest -m network` re-downloads the small metadata (< 20 MB, range reads), recomputes the overlap counts, and diffs them against `results/overlap_counts.csv`, so upstream changes to PRISM or GDSC releases are caught.
- **No CI job touches the Step-2 parquets or trains on real data.** Real runs are local and logged, with the config hash, git SHA and data-manifest hash in each results file.

---

## Appendix A: what was downloaded for this design (all in `data_probe/`) [V]

| What | Where from | Bytes transferred |
|---|---|---|
| GDSC2 fitted dose-response xlsx, compound annotation (InChIKeys), CMP model list | cmp.cog.sanger.ac.uk / cog.sanger.ac.uk | 23.1 MB |
| PRISM 19Q4: readmes, cell-line info, pooling info, replicate-collapsed treatment info (primary + secondary) | figshare 9393293 | 5.5 MB |
| PRISM 19Q4: A549 / HepG2 / U2OS rows of the primary (41 MB) and secondary (92 MB) collapsed-LFC matrices, by **HTTP range reads** with binary search (the files are sorted by DepMap ID), plus layout probes and a peek at the curve-parameter file | figshare 9393293 | 27.3 MB (range reads, not whole files) |
| PRISM 20Q2: the same metadata files | figshare 20564034 | 6.1 MB |
| PRISM 20Q2: primary collapsed LFC (unsorted, so **streamed once**; only the 3 target rows kept) | figshare 20564034 | 46.7 MB |
| PRISM 20Q2: secondary collapsed-LFC rows (PR500 / PR300 blocks, range reads) and layout probes | figshare 20564034 | 11.7 MB |
| PubChem PUG-REST: InChIKey → SMILES for 417 GDSC keys (cached in `pubchem_cache_fusion.json`) | pubchem.ncbi.nlm.nih.gov | < 1 MB |
| **Total** | | **≈ 120 MB transferred; 40 MB on disk** |

**Scripts** (all in `scripts/`; run with `py -3.13 -I -X utf8`)
- `range_rows.py`: the range-read helper.
- `fetch_prism_rows.py`: 19Q4 rows.
- `fetch_prism20q2_rows.py`: 20Q2 rows.
- `count_overlap.py [prism|prism20q2]`: writes `overlap_matches*.csv`, `overlap_counts*.csv` and `overlap_summary*.json`.
- `probe_targets.py`: writes `probe_targets.json`.
- `power_sim.py [K]`: writes `power_sim.json` (K = 20) and `power_sim_k5.json` (K = 5); synthetic data only.
- `power_sim_rule.py`: writes `power_sim_rule.json`. It simulates the actual decision rule, including the permutation, on synthetic data.

Not downloaded: the PRISM secondary dose-response curve-parameter files (264 / 290 MB) and the full PRISM replicate-level matrices.
