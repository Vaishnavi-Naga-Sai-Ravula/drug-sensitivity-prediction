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

The model follows the author FNN dimensions (1000, 800, 500, 100), ELU activations and 0.1 dropout. Features are z-scored using training data only. `train.py` evaluates an untouched holdout set, optionally holds out entire drugs or cell lines, separately evaluates on CCLE, and then fits a final model on all GDSC rows for interactive inference.

### Current run results

The checked-in outputs are from one 20-epoch, seed-42, random drug-cell-pair 80/20 split: GDSC holdout MAE 0.328 and RMSE 0.440 on the `-log10(IC50 in μM)` scale (R² 0.868). The separate CCLE evaluation has MAE 0.857 and RMSE 1.079 (R² -0.506), showing that transfer to CCLE is substantially less reliable. These are this implementation's run results, not the paper's scores; they are not a repeated 10-fold cross-validation result. Run `python train.py --split cell` and `python train.py --split drug` for harder generalization checks before making broad performance claims.

ROC/AUC and confusion-matrix charts are shown as **exploratory visualizations** only. Their response label uses the GDSC training median, not a clinical sensitivity threshold. The paper uses regression because no universal sensitive/resistant cutoff is established.

## Run locally (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py --split pair --epochs 30
streamlit run app.py
```

The app can also train from **Methods & deploy**. For stricter generalization checks, run `python train.py --split cell` or `python train.py --split drug`; these evaluate on cell lines or drugs not seen during training and are more challenging than a random pair split.

## Outputs

Training writes:

- `artifacts/model.pt` and `artifacts/metadata.json` — the final model, ordered feature list and training-only imputation/scaling values.
- `outputs/holdout_predictions.csv` — observed and predicted `-log10(IC50)` and converted IC50 for each held-out drug-cell pair.
- `outputs/ccle_external_predictions.csv` — GDSC-trained model evaluated on available CCLE pairs.
- `outputs/metrics.json`, `outputs/training_history.csv`, `outputs/global_group_shap.csv`, `outputs/global_feature_shap.csv` and `outputs/sampled_feature_shap_values.csv`.
- `outputs/figures/` — response distribution, actual-vs-predicted, residual diagnostics, training curve, ROC curve, confusion matrix, feature correlation heatmap and global group-SHAP chart, plus a global feature-SHAP chart when available.
- `PROJECT_PLAN.md` — objectives, comparison with PathDSP, proposed methodology, and phased extension plan. `SHAP_INTERPRETATION_GUIDE.md` explains how to interpret the SHAP reports.

The app always calculates exact Shapley values over the five feature groups. `shap` (included in `requirements.txt`) enables feature-level Gradient SHAP (expected gradients) attributions for a selected pair. The Explainability page displays and downloads local and global Markdown interpretation reports as well as the CSV and chart outputs.

To regenerate the sampled feature-level SHAP output for the saved final model without retraining, run `python explain_outputs.py` after training.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository for this project. Include `app.py`, `pathdsp/`, `train.py`, `explain_outputs.py`, `requirements.txt`, `.streamlit/`, `Dockerfile`, `app_data/pathdsp_core.zip`, `PROJECT_PLAN.md`, `SHAP_INTERPRETATION_GUIDE.md`, and the trained `artifacts/` folder.
2. Do not upload the full `processed_data/`, raw omics files, the 297 MB PPI pickle, or the original multi-hundred-megabyte input archives. `.gitignore` excludes these local-only files.
3. In Streamlit Community Cloud, choose the repository and set the app file to `app.py`.
4. Open the deployed page and check one known GDSC drug-cell pair, one CCLE pair, the holdout metrics, and a batch CSV before sharing the URL.

The included `Dockerfile` can also run the app on a container platform using port 8501. Do not put secrets in source code; use your hosting provider's secret manager if you add any later.

## Intended use and limits

This is an educational and research prototype built from public cell-line data. Predictions and SHAP values describe model behavior and associations; they do not establish causality, patient benefit, or a treatment recommendation. Do not upload patient-identifiable or clinical data.

## Attribution

The scientific method and supplied source implementation are by Tang and Gottlieb, “Explainable drug sensitivity prediction through cancer pathway enrichment,” *Scientific Reports* 11, 3128 (2021). The author's repository and MIT license are preserved in `PathDSP-main/`. See its README for data-processing references and upstream citations.

