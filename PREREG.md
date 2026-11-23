# PREREG: frozen primary comparison

This file freezes DESIGN.md §4.4 before any real target enters a model, as §4.4 ("Freezing") requires.

- **Design commit:** `31cd6c78550c6a00776812ad71717c24a751d9aa`. It was pushed to github.com/Sam-Wo/shapeshift-fusion before any model was fit.
- **SHA-256 of §4.4 below** (LF line endings, from the committed blob): `e40bc012382e0e535876625253fecb1159773b279a044b8f1a16e309c486af66`
- **Verify:** `git show 31cd6c7:DESIGN.md | awk '/^### 4.4 PRE-REGISTERED/{f=1} /^### 4.5 /{f=0} f' | sha256sum`
- **Implementation clarifications** are in DEVIATIONS.md, D-01 to D-16. They were all written before freezing, and none changes the comparison, the rule, the splits, the features or the arms.
- **Analysis code at freeze:** commit `acb1934`. `configs/primary.yaml` holds the run parameters.

---

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


---

## Pre-freeze checks (run on real T1 feature matrices with synthetic targets; `scripts/prefreeze_check.py`)

The setup is n = 183 T1 units and the real feature matrices: BASE = MoA one-hot + Morgan SVD + nuisance
(48 columns before the SVD-k choice), k = 5 in-fold PCs per block, and the nested 5-fold grouped inner CV. Each
dataset uses one repeat of the 10-fold scaffold split (D-11) and 99 within-MoA Freedman–Lane permutations. The
targets are synthetic: y = BASE signal (R² 0.30) + the other block's unique signal (0.08) + δ × the tested
block's unique signal + noise.

| Tested block | δ | Datasets | Mean observed ΔR² | P(observed ≥ 0.03) | P(FL p < 0.05) | P(rule) |
|---|---|---|---|---|---|---|
| transcriptome | 0 | 200 | −0.006 | 0.095 | **0.040** | 0.020 |
| morphology | 0 | 200 | −0.005 | 0.040 | **0.025** | 0.015 |
| transcriptome | 0.08 | 100 | 0.045 | 0.52 | 0.54 | 0.45 |
| morphology | 0.08 | 100 | 0.039 | 0.57 | 0.58 | 0.51 |

- **Type-I check: passed.** The residual-permutation type-I rate is ≤ 0.06 at δ = 0 for both blocks (0.040 and
  0.025; Monte-Carlo SE about 0.01–0.014).
- **Power is lower than the design's simulation**, as §4.5 warned it would be. On the real feature dimensions
  with scaffold folds, the one-block rule has power of about 0.45–0.51 at a true δ = 0.08 per block (the design
  simulation gave 0.83). The joint power of both blocks at δ = 0.08 is therefore roughly 0.2–0.25. A null result
  will bound only large joint gains.
- **Where the numbers are:** `results/prefreeze_check.json`.
