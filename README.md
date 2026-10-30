# PathDSP Lab

PathDSP Lab is a deployable research dashboard for cancer cell-line drug-response prediction. It adapts the supplied PathDSP research code and processed GDSC/CCLE feature tables into a reproducible training pipeline, an interactive Streamlit app, and downloadable actual-versus-predicted outputs.

## Project status

- **Phase 1 — source and data check: complete.** `PathDSP-main.zip` has been unpacked as `PathDSP-main/` and its 13 files match the extracted folder. The 27-table `processed_data.zip` was unpacked locally. A compact 12-table bundle is in `app_data/pathdsp_core.zip` for deployment.
- **Phase 2 — model and app implementation: complete.** The app uses the supplied five PathDSP feature groups and the author FNN architecture. It includes response prediction, holdout and CCLE evaluation, SHAP explanations, RWR method notes, model metrics, diagnostic plots, and batch feature-file upload.
- **Phase 3 — local training/output generation: complete.** The current run produced `artifacts/` and `outputs/`. The deployable `artifacts/` are intentionally kept separate from the evaluation predictions, which are generated from a held-out split.
- **Phase 4 — public deployment: not yet published.** Follow the steps below after training and reviewing the outputs. Publishing requires your GitHub/hosting account and credentials.

## Model and response scale

PathDSP combines 256 Morgan fingerprint bits with drug-target DG-Net, and cell-line EXP/ssGSEA, MUT-Net and CNV-Net pathway features. These are already supplied in the processed bundle. The mutation, copy-number and drug-target NetPEA features use the paper's Random Walk with Restart (RWR) network diffusion step; the package also includes a reusable RWR helper. The app does not claim to regenerate the complete upstream NetPEA permutation-based pathway scores from raw gene lists.

The response matrices are named `RESP.negLog10IC50`. This implementation treats their target as `-log10(IC50 in μM)` and converts with `IC50_μM = 10 ** (-target)`. The CCLE target floor in the supplied data is approximately `-log10(8 μM)`, consistent with the paper's reported CCLE 8 μM assay cap. Every exported row keeps the original target scale as well as the converted IC50 values.

The active model retains the supplied FNN dimensions (1000, 800, 500, 100), ELU activations and 0.1 dropout. The new training recipe uses Xavier initialization, AdamW and an MSE objective. Features are standardized from fitting rows only. `scripts/train_mse.py` fits on 27,166 pairs, selects the checkpoint on 3,019 validation pairs and evaluates on 7,547 held-out pairs. That evaluated checkpoint is served directly; there is no final refit on held-out rows. The legacy `train.py` workflow is a separate experiment.

### Current run results

The current run uses split seed 42, training seed 2026, 60 maximum epochs and early stopping after 51 epochs (best validation checkpoint at epoch 41). GDSC held-out MAE is **0.310**, RMSE **0.420**, R² **0.880**. The preceding FNN scored MAE 0.321 and RMSE 0.432 on the same historical split. This modest improvement is exploratory because the holdout has been reused across model comparisons; it is not independent final certification.

External CCLE MAE is **0.886**, RMSE **1.085**, R² **−0.524**. MAE is slightly worse than the preceding deployment model's 0.866; transfer remains unreliable. No claim of improvement on both datasets is made. The screenshot's 5-Fluorouracil × SIDM00003 example now predicts about **190 μM** versus **217.8 μM** observed, rather than 16.51 μM, but this pair was in the fitting set and is not independent validation. These are this implementation's run results, not the paper's scores or repeated cross-validation.

ROC/AUC and confusion-matrix charts are shown as **exploratory visualizations** only. Their response label uses the GDSC training median, not a clinical sensitivity threshold. The paper uses regression because no universal sensitive/resistant cutoff is established.

## Run locally (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The saved model is included; retraining is unnecessary to run the app. Reproduce the active candidate with `python scripts/train_mse.py --output .local_training/mse_candidate`; this creates a candidate bundle without replacing the serving model. In-app training from **Methods & deploy** is disabled unless `PATHDSP_ENABLE_TRAINING=1` is explicitly set for local research. For stricter generalization checks, run `python train.py --split cell` or `python train.py --split drug`; these evaluate on cell lines or drugs not seen during training and are more challenging than a random pair split.

## Outputs

Training writes:

- `artifacts/model.pt`, `artifacts/metadata.json`, and `artifacts/manifest.json` — the deployment model, ordered feature list and training-only imputation/scaling values.
- `outputs/holdout_predictions.csv` — observed and predicted `-log10(IC50)` and converted IC50 for each held-out drug-cell pair.
- `outputs/ccle_external_predictions.csv` — GDSC-trained model evaluated on available CCLE pairs.
- `outputs/metrics.json`, `outputs/training_history.csv`, `outputs/global_group_shap.csv`, `outputs/global_feature_shap.csv` and `outputs/sampled_feature_shap_values.csv`.
- `outputs/figures/` — response distribution, actual-vs-predicted, residual diagnostics, training curve, ROC curve, confusion matrix, feature correlation heatmap and global group-SHAP chart, plus a global feature-SHAP chart when available.
- `PROJECT_PLAN.md` — objectives, comparison with PathDSP, proposed methodology, and phased extension plan. `SHAP_INTERPRETATION_GUIDE.md` explains how to interpret the SHAP reports.

The app always calculates exact Shapley values over the five feature groups. `shap` (included in `requirements.txt`) enables feature-level Gradient SHAP (expected gradients) attributions for a selected pair. The Explainability page displays and downloads local and global Markdown interpretation reports as well as the CSV and chart outputs.

To regenerate the sampled feature-level SHAP output for the saved final model without retraining, run `python explain_outputs.py` after training.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository for this project. Include `app.py`, `pathdsp/`, `ui/`, `train.py`, `explain_outputs.py`, `requirements.txt`, `.streamlit/`, `Dockerfile`, `app_data/pathdsp_core.zip`, `PROJECT_PLAN.md`, `SHAP_INTERPRETATION_GUIDE.md`, and the trained `artifacts/` folder.
2. Do not upload the full `processed_data/`, raw omics files, the 297 MB PPI pickle, or the original multi-hundred-megabyte input archives. `.gitignore` excludes these local-only files.
3. In Streamlit Community Cloud, choose the repository and set the app file to `app.py`.
4. Open the deployed page and check one known GDSC drug-cell pair, one CCLE pair, the holdout metrics, and a batch CSV before sharing the URL.

The included `Dockerfile` can also run the app on a container platform using port 8501. Do not put secrets in source code; use your hosting provider's secret manager if you add any later.

## Intended use and limits

This is an educational and research prototype built from public cell-line data. Predictions and SHAP values describe model behavior and associations; they do not establish causality, patient benefit, or a treatment recommendation. Do not upload patient-identifiable or clinical data.

## Attribution

The scientific method and supplied source implementation are by Tang and Gottlieb, “Explainable drug sensitivity prediction through cancer pathway enrichment,” *Scientific Reports* 11, 3128 (2021). The author's repository and MIT license are preserved in `PathDSP-main/`. See its README for data-processing references and upstream citations.


## Interface and repository-only setup

The dashboard runs using the committed `app_data/pathdsp_core.zip`, `artifacts/`,
`pathdsp/`, and `ui/` files. Neither `input.zip` nor `PathDSP-main.zip` is needed
for prediction, explanations, or training with the bundled processed features.
Keep any original archives as optional backups outside the repository.

The Overview opens with an interactive, locally rendered molecular illustration.
It is decorative, not a chemical structure for the selected drug. Use **Pause
motion** to stop rotation; the illustration also respects reduced-motion settings.
Choose **Enter the prediction lab** to begin. In Predict and Explainability, type
part of a drug or cell-line name in the search fields, then choose a matching
identifier. Search ignores capitalization; a unique match is selected automatically.
Only identifiers from the selected dataset can be submitted.

To update an existing Windows Git checkout, first back up or commit local changes,
then open PowerShell in the project directory:

```powershell
git pull --ff-only origin main
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

For a folder downloaded as a ZIP, download the updated repository ZIP from GitHub
and extract it to a new folder to preserve local files. Create a virtual environment
and install requirements as described above; retraining is unnecessary because a
saved model is included. Interface checks: `python -m unittest discover -s tests`.

### Prediction accuracy audit

`outputs/deployment_model_audit.json` audits all bundled pairs. GDSC all-row MAE is 0.195 and RMSE 0.275; this mixes fitting, validation and holdout rows and is not a generalization score. External CCLE MAE is 0.886, RMSE 1.085 and R² −0.524. Median concentration error is about 1.40× on GDSC and 6.22× on CCLE. Use the explicit held-out metrics above when assessing generalization. `outputs/model_split_membership.csv` records each GDSC pair's role.

Feature ordering, scaling and all processed tables were audited. No duplicate, infinite, nonnumeric or missing feature values required deleting rows. Observed responses are never substituted for model estimates. A one-unit log error means a tenfold concentration error; exact agreement cannot be promised. The active model was retrained, and every prediction table, SHAP summary and evaluation figure was regenerated from its evaluated checkpoint.

Use the top **Dark mode** switch to choose a light or dark workspace. The
choice remains active while navigating the current session. Reproduce the saved
model audit with `python scripts/audit_model.py`; it updates only the audit JSON.

## Immersive research workspace and validation

The landing experience now fills a large responsive scene with interactive
molecule, cell-landscape and pathway-network views. Every section has its own
animated banner. Light and dark palettes, mobile navigation, pause controls and
OS/top-bar reduced-motion options are supported. The art is illustrative.

Prediction uploads now have explicit size, numeric-value and missingness checks.
Artifacts have consistency hashes; inference concurrency and batch sizes are
bounded. In-app training is disabled by default to protect the serving model.
The optional `api.py` service provides research inference and functional health
checks. No uploaded raw ZIP archives are used.

See [release evidence, accuracy limits and deployment instructions](docs/READINESS.md).
The current CCLE transfer performance remains poor. **This is not a clinically
validated or universally production-ready model.** Three tree-model trials failed to improve the earlier FNN. The subsequent MSE-FNN trial improved GDSC held-out scores and was promoted, with the CCLE limitation retained explicitly.

## Evaluation reports, populated samples and pathway comparisons

Use the horizontal navigation above the workspace. Overview presents the workflow;
the GDSC response distribution and validation evidence are in **Evaluation**.
Predict now downloads populated CSV and TSV examples containing five real rows and
all model features; these can be uploaded directly. Feature-level SHAP downloads
include all features, even though plots display only the largest contributions.

Make a prediction, then open Evaluation: its reference drug and cell line follow
your last prediction. You can also choose a different pair there. The app compares
all eligible drugs for that cell line, reports measured and predicted IC50 values
separately, and explains the selected pair with exact group SHAP and approximate
feature/pathway SHAP. Lower model-estimated IC50 is a **research ranking**, not a
clinical recommendation; toxicity, achievable dose and patient outcomes are absent.
No one-to-one drug–pathway assignment is supplied. API keys do not improve model
accuracy by themselves. The FNN remains a baseline with known CCLE transfer limits.

Choose **Prepare complete report and evidence bundle** in Evaluation. The standalone
HTML report includes every evaluation figure, reference-drug comparisons, complete
embedded CSV data with table previews, SHAP graphs, metric/audit records and model identity.
The ZIP includes the same HTML plus every table as CSV and every figure as PNG.
On-screen previews and filters do not truncate the exported data. Open the HTML
in a browser to read or print it. No patient health records exist in these inputs;
tables explicitly mark patient health status as unavailable instead of inventing it.

### Windows artifact-integrity error

JSON metadata hashing now tolerates Git's LF→CRLF line-ending conversion on Windows;
model binaries and actual metadata content remain checked. `.gitattributes` also
keeps JSON files in LF form. Stop Streamlit, pull the latest commit and restart.
If the error persists because your local bundle actually differs, preserve and
restore it with:

```powershell
.\.venv\Scripts\python.exe scripts\restore_artifacts.py
```

This checks the committed bundle, backs up existing artifacts to a verified ZIP
under `.local_backups/`, then restores the three matching files from local Git
HEAD. Restart Streamlit afterwards. It does not retrain or silently bless altered
scalers. Keep the backup if you previously trained a local model.


## IC50 concentration scale, clinical limits and model retraining

Predict and Evaluation show a logarithmic concentration ruler, a conversion table
and the same drug's measured cell-line distribution. The report and evidence ZIP
include those visualizations and underlying tables. A comparison calculator makes
ratios explicit: 16.5 μM / 9 μM = 1.83×, with a target-scale difference of about
0.263. This is a concentration comparison under matched assay conditions, not a
patient-progress score. The previous green prediction-error arrow was removed.

IC50 does not supply a safe dose, treatment duration, patient response, death risk
or treatment cost. The interface and report state the unavailable clinical/price
inputs instead of inventing thresholds, regimens or prices. The μM conversion
still depends on the source-unit assumption. No dose–response curve is invented
from a single IC50 measurement.

To reproduce a candidate and prepare its matching release outputs offline:

```bash
python scripts/train_mse.py --output .local_training/mse_candidate
python scripts/prepare_mse_release.py --candidate .local_training/mse_candidate --results .local_training/mse_candidate/mse_candidate_results.json --history .local_training/mse_candidate/mse_training_history.csv --output .local_training/mse_release
```

Use a new output directory each time. Review the candidate, its external results
and all regenerated files before a release; these commands do not publish or
replace the serving bundle. Restart serving processes after deploying a complete
reviewed bundle. `outputs/mse_training_comparison.json` records the current trial,
the prior baseline, the screenshot example's fitting-set role and the CCLE trade-off.

## Hover coordinates and complete workspace reports

Evaluation is now a compact tables-and-charts workspace. Data plots use the same
interactive chart definitions in the app and the downloadable HTML report.
Move within a plotting area to read cursor coordinates in axis units, including
physical concentration values on logarithmic axes. Hover data marks for the exact
recorded values, identifiers, bin counts, SHAP contributions or correlations.
A cursor position in empty plot space is not an additional measurement.

The report embeds the chart library and works offline in a normal JavaScript-enabled
browser. PDF/print and PNG copies are static; use the HTML for hover, zoom and pan.
The evidence ZIP retains the original PNGs and adds interactive chart definitions.
All CSV rows remain included. Large HTML tables show small previews and embed a
complete downloadable CSV to avoid constructing millions of browser table cells.

The report covers all five workspace sections: overview/scientific illustration,
generated Predict results and batches, Evaluation, Explainability, and scientific
methods/model metadata. Deployment/run instructions are excluded. Results generated
in the current browser session are recorded with their original pair/model context;
analyses not run are explicitly marked as not generated. Closing/resetting the
session does not reconstruct previous unsaved session outputs. Download a report
before closing if you need those records. No patient health data or treatment
outcomes are invented; clinical limitations remain visible as table rows.

After applying this update, install requirements to add Plotly, then restart the
app. No model retraining is needed for these presentation/export changes.

Optional browser regression checks (install Playwright and Chromium separately):

```bash
python scripts/check_chart_hover.py --url http://127.0.0.1:8501
```

The check uses an isolated browser session and a temporary loopback report server.
It checks axis coordinates, full exported data, generated batch/SHAP snapshots,
light/dark rendering and offline report interaction. A system Chromium path can
be supplied with `--chromium-path`; otherwise Playwright's installed browser is used.


### Separate Evaluation display from the complete report

Evaluation displays exactly three coordinate-enabled charts: the filtered held-out
scatter, GDSC held-out hexagonal-density agreement, and CCLE external hexagonal-density
agreement. The two density views keep their fixed axes and expose bin-center
coordinates and pair counts on hover. All Evaluation tables remain available.

The report is intentionally broader than the page: it includes those three graphs,
all additional diagnostics/IC50/SHAP/training charts, every Evaluation table and key
statistic, generated session outputs, shared Overview and Methods scientific text,
and the complete SHAP interpretation guide. Deployment instructions are excluded.
A table coverage inventory and clickable contents make missing exports detectable
and the complete report navigable. No model retraining or accuracy claims are made
by this display/export update.
