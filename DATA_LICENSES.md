# Data licences

This repository contains code (MIT, see LICENSE) and aggregate results only. It contains no third-party data
tables. The inputs are downloaded or produced locally (see README, "Reproduce").

| Source | Licence | What this repo does with it |
|---|---|---|
| PRISM Repurposing 19Q4 / 20Q2 (Corsello et al., *Nat Cancer* 2020; figshare 10.6084/m9.figshare.9393293, 10.6084/m9.figshare.20564034) | CC BY 4.0 | Targets: viability LFC and curve AUC. Only aggregate metrics are committed. Changes: replicate-collapsed values median-collapsed per structure-matched compound; a line-selective residual (minus the median over all lines). |
| PRISM columns `name`, `moa`, `target`, `indication`, `phase` (Broad Drug Repurposing Hub annotations) | "provided for non-commercial use only" | Not used for modelling and never written. MoA comes from Tahoe `moa-fine` instead. The curve-parameter download drops these columns on the fly. |
| GDSC2 via Sanger Cell Model Passports | Sanger DepMap Data Usage Policy (internal research and education; no redistribution terms) | **Not redistributed.** It is used only for one exploratory aggregate correlation (A549 AUC vs predictions, n ≈ 28). No per-compound GDSC value is committed. |
| Tahoe-100M (Zhang et al. 2025) | CC0 1.0 | Transcriptomic signatures (via ShapeShift Step 2); `moa-fine` labels. |
| Cell Painting Gallery: LINCS cpg0004, JUMP cpg0016 (Weisbart et al. 2024) | CC0 1.0 | Morphology profiles (via ShapeShift Step 2). |
| EU-OPENSCREEN bioactives, cpg0036 / Zenodo 10.5281/zenodo.19347244 (Wolff et al. 2025) | CC BY 4.0 | T2 morphology profiles (via ShapeShift Step 2). Attribution: EU-OPENSCREEN ERIC. |
| PubChem (InChIKey → SMILES for GDSC matching) | No restrictions (NCBI) | Design-stage matching only. |
