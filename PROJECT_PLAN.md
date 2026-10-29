# Project Objectives, Comparison, and Methodology

## Proposed project title

**Explainable and Uncertainty-Aware Drug Sensitivity Prediction with Evidence-Weighted Pathway Diffusion and Double Cold-Start Evaluation**

This is a proposed extension of PathDSP, not a claim of worldwide research novelty. The current app is a completed PathDSP-style baseline. The additional experiments below are planned work and must be reported as completed only after they are implemented and evaluated.

## Project objectives

1. Reproduce a transparent PathDSP baseline using the supplied GDSC/CCLE feature tables, the five established feature groups, and regression on `-log10(IC50 in μM)`.
2. Evaluate a harder deployment scenario in which both the test drugs and test cell lines are unseen during training. Report performance overall and by cancer lineage, alongside ordinary random-pair, leave-drug-out, leave-cell-line-out, and cross-dataset comparisons.
3. Test whether mutation and copy-number evidence improves pathway features when used as weighted RWR restart signals, with the exact seed definitions and network/pathway versions recorded.
4. Add prediction intervals with split conformal calibration, then report empirical coverage and interval width overall and by evaluation scenario.
5. Assess whether SHAP explanations are stable across folds/seeds and whether highlighted pathways agree with independent CRISPR dependency evidence after cell-line identifiers are matched.
6. Deliver the methods, metrics, plots, IC50 tables, uncertainty intervals, and plain-language explanation reports through the PathDSP Lab app.

## Paper versus current project versus proposed extension

| Topic | PathDSP paper / author implementation | Current project implementation | Proposed extension |
|---|---|---|---|
| Response target | Regression on transformed drug response (`-log10(IC50)`). | Same response scale; tables include actual and predicted transformed values and IC50 in μM. | Keep the same primary endpoint for a like-for-like baseline. State units and source assay limits. |
| Input representation | Morgan chemical fingerprints plus drug-target and cell-line pathway features for expression, mutation, and CNV. | Uses the supplied CHEM, DGNet, EXP, MUT-Net, and CNV-Net matrices (1,031 joined features). | Add evidence-specific mutation/CNV feature variants only after checking source files, identifiers, and data versions. |
| RWR / pathway processing | Uses network diffusion / pathway enrichment in the feature-generation workflow. | Consumes the supplied RWR-derived pathway matrices and includes a reusable RWR helper; it does not reproduce the complete upstream NetPEA permutation/z-score workflow from raw genes. | Compare binary, damaging, and hotspot mutation seeds; separate CNV gains/losses where signed data support it; compare a paper-style network/pathway baseline with versioned STRING/MSigDB inputs. |
| Prediction model | Fully connected neural network (FNN) baseline. | PathDSP-style FNN with training-only scaling; final model is saved for app inference. | Preserve the FNN as the reference. Test alternatives only as controlled comparisons on the same split and preprocessing. |
| Evaluation | Repeated cross-validation and separate unseen-drug, unseen-cell-line, tissue, and CCLE evaluations in the paper/code. | One 80/20 random drug-cell pair split plus a separate CCLE evaluation have been run. | Add repeated seeds/folds, leave-drug-out, leave-cell-line-out, lineage-stratified reporting, and a strict double cold-start test. Deduplicate/track overlapping response pairs before any split. |
| Explainability | SHAP used to interpret feature contributions. | Exact SHAP over five feature groups; sampled feature-level Gradient SHAP; charts and CSV exports. | Report local and global explanations in readable prose; measure rank/top-k stability across folds and compare pathway findings with CRISPR evidence. Treat both as model associations, not causal proof. |
| Uncertainty | Point predictions. | Point predictions only. | Add split conformal prediction intervals on the transformed response scale; report coverage and width, including after conversion to IC50 units. |
| User interface | Research code and analysis outputs. | Streamlit app for prediction, batch upload, evaluation, methods, plots, and SHAP. | Add prediction intervals and persistent downloadable prediction/explanation reports; publish only after hosted-app validation. |

The current single random-pair split is not equivalent to the paper's repeated validation. A random split can place the same drug and cell line in both train and test pairs. If the two supplied GDSC response sources contain the exact pair overlap identified in the preliminary audit, confirm it against the actual files and preserve a canonical drug-cell key so overlapping observations are never treated as independent validation.

## Proposed methodology

### 1. Data audit and provenance

- Record each file's source, release/version, license, checksum, response units, and preprocessing history.
- Normalize drug and cell-line identifiers using explicit mapping tables. Keep unmapped and ambiguous identifiers in an audit log rather than silently merging them.
- Build a canonical `(dataset, drug_id, cell_line_id, assay/source)` key. Compare response sources for duplicate pairs and reconcile duplicates before splitting. Do not treat overlapping releases as independent external validation.
- Inspect mutation files to confirm whether damaging/hotspot columns are binary calls or graded evidence. Inspect CNV files to confirm whether signed gain/loss information is available. Do not infer unavailable variant confidence or germline status from somatic matrices.
- Record gene-set and network versions, including the exact MSigDB collection and STRING release used in each RWR comparison.

### 2. Baseline and evaluation design

- Re-run the PathDSP-style FNN with the current five features, the same target transform, and training-only imputation/scaling. Save seeds, split assignments, fitted preprocessing values, code version, metrics, and per-pair predictions.
- Report an ordinary random-pair evaluation only as a familiar baseline. Also run leave-drug-out and leave-cell-line-out scenarios with repeated seeds or grouped folds.
- For **double cold-start**, partition drug IDs into train/test sets and cell-line IDs into train/test sets. Train only on train-drug × train-cell pairs, and score only test-drug × test-cell pairs. Neither held-out drugs nor held-out cells may occur in training. Keep validation groups disjoint from training groups when selecting epochs/hyperparameters.
- Report metrics overall and by cancer lineage (sample count, MAE, RMSE, R², Pearson and Spearman correlation). Explain missing or small lineage groups; do not rank groups with too few observations.
- Evaluate GDSC-to-CCLE separately. Treat it as dataset-shift evaluation, not automatically independent validation, until overlap and assay compatibility have been audited.
- Use regression metrics as primary results. ROC/AUC and confusion matrices may be secondary exploratory displays only if a threshold is defined and justified in advance.

### 3. Evidence-weighted mutation/CNV RWR

- Preserve the supplied pathway matrices as the baseline.
- Create mutation seed variants from the available somatic files: all observed mutations, damaging mutations, hotspot mutations, and a combined evidence-weighted variant only when the file fields support a defensible weighting rule. If only binary indicators exist, compare seed sets rather than inventing numeric confidence weights.
- Where signed CNV values exist, run separate gain and loss seeds; document thresholding and normalization. Do not split gain/loss if the source contains only unsigned calls.
- Apply RWR with recorded restart probability, convergence tolerance, network release, node mapping, and seed weights. Run the same pipeline with paper-style and updated pathway/network collections where data licensing and identifier coverage permit.
- Compare predictive performance and pathway-rank stability against the original supplied features. Use ablations to identify which evidence source changes the result.

### 4. Uncertainty intervals

- Hold out a calibration set that is not used for model fitting or hyperparameter selection. Compute absolute residuals on `-log10(IC50 in μM)` and use a prespecified conformal quantile to form prediction intervals.
- Evaluate empirical coverage and mean interval width on untouched test data. Repeat by split scenario and lineage. For double cold-start, calibrate using groups that reflect the intended unseen-drug/unseen-cell setting where feasible; state when exchangeability assumptions may be weakened by dataset shift.
- If the transformed-response interval is `[L, U]`, report the corresponding IC50 range as `[10^(-U), 10^(-L)] μM` (reversed endpoints because the transform is decreasing). Describe these as prediction intervals, not individual-patient confidence probabilities.

### 5. Explanation and biological cross-check

- Keep exact five-group SHAP and local feature-level Gradient SHAP for model interpretation.
- Calculate global feature/pathway summaries separately on each held-out fold/seed. Report top-k overlap and rank correlation across runs; do not use a final model trained on all rows to claim held-out explanatory performance.
- Match cell-line identifiers before comparing pathway importance with CRISPR dependency evidence. Report the overlap and direction of association, including unmatched IDs and coverage.
- Generate a human-readable report with the prediction scale, baseline, local contributions, global importance, sample/model scope, and interpretation limits. Fingerprint bits require a valid Morgan-bit-to-substructure mapping before naming chemical fragments.
- State that SHAP describes model attribution and CRISPR comparisons provide supporting biological evidence; neither alone establishes a causal mechanism.

## Implementation phases

| Phase | Work | Status |
|---|---|---|
| 1. Source and data audit | Unpack the author archive, inspect paper/source files and processed matrices, identify target scale and required features. | **Complete for the current baseline.** Duplicate-response audit and version manifest remain part of the extension. |
| 2. Baseline implementation | PathDSP-style FNN, five feature groups, app, prediction CSVs, regression metrics, figures, group and feature SHAP. | **Complete locally.** |
| 3. Baseline reproducibility | Record config/seed, run repeated baseline folds and drug/cell-held-out tests, preserve split-level predictions. | **Next.** The current one-seed pair split and CCLE result are preliminary baseline evidence only. |
| 4. Hard generalization and lineage report | Add strict double cold-start and lineage-stratified evaluation with overlap control. | **Planned.** |
| 5. RWR evidence variants | Build mutation/CNV seed ablations and compare versioned pathway/network choices. | **Planned.** First confirm field semantics and signed CNV support. |
| 6. Uncertainty and explanation stability | Add conformal intervals, SHAP stability, and CRISPR cross-check; generate reports. | **Partly complete.** The readable SHAP report is available locally; intervals and stability experiments remain planned. |
| 7. Final packaging and deployment | Finalize documentation, rerun reproducible results, validate accessibility and batch workflows, then publish the app. | **Local app complete; public deployment pending.** |

## Practical scope and claims

The recommended feasible extension is phases 3–6, prioritizing double cold-start evaluation, evidence-aware RWR ablations, conformal intervals, and explanation stability. Drug-induced expression, Transformers/cross-attention, combination response, and multi-task prediction can be considered later; they require additional data and validation and should not be promised as part of the current implementation. These proposed changes may make the project more rigorous, but do not by themselves establish novelty across the literature.

## References

- Tang and Gottlieb, “Explainable drug sensitivity prediction through cancer pathway enrichment,” *Scientific Reports* (2021): [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7862690/); [author repository](https://github.com/TangYiChing/PathDSP).
- A. Hernandez et al., conformal prediction for cancer drug response, *Proceedings of Machine Learning Research* (2022): [paper](https://proceedings.mlr.press/v179/hernandez-hernandez22a.html).
- DepMap/Sanger drug sensitivity documentation: [documentation](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/).
- DepMap mutation documentation: [documentation](https://depmap.sanger.ac.uk/documentation/datasets/mutation/).
- “The Specification Game,” *Nature Methods* (2025): [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11907791/).
