# SHAP Interpretation Guide

## What SHAP means

SHAP (SHapley Additive exPlanations) assigns each feature or feature group a contribution to one model prediction relative to a baseline. For a prediction `f(x)`, the additive explanation is written as:

`f(x) = baseline + sum(SHAP contributions)`

The contribution for feature `i` is an average of its change in model output over possible feature coalitions. A positive contribution raises the model output relative to the baseline; a negative contribution lowers it. The size indicates contribution magnitude in the model's output units.

## How this project calculates it

- **Feature-group SHAP:** the app computes exact Shapley values for the five supplied input groups: CHEM, DGNet, EXP, MUT-Net, and CNV-Net. Five groups require only 32 coalitions. The baseline is a zero vector after training-mean z-scoring, representing the training means for the standardized features.
- **Feature-level SHAP:** when requested, the app uses SHAP `GradientExplainer` (expected gradients) with a sampled background set. This is an approximation and depends on the selected background examples.
- **Global importance:** mean absolute SHAP summarizes average magnitude and discards direction. Mean signed SHAP summarizes average direction but can hide mixed effects that cancel across examples.

## Reading the drug-response direction

The model output is `-log10(IC50 in μM)`. Therefore:

- Positive SHAP contribution → higher predicted `-log10(IC50)` → lower predicted IC50.
- Negative SHAP contribution → lower predicted `-log10(IC50)` → higher predicted IC50.

This is a direction in the model's transformed response, not a clinical sensitive/resistant label. Converting a point estimate back to concentration uses `IC50_μM = 10^(-prediction)`.

## Important limits

SHAP explains how this fitted model used its input values. It does not establish that a feature or pathway causes drug response. Correlated features can share or shift attribution. A Morgan fingerprint label such as `CHEM__mBit_11` is not a named chemical fragment unless the bit is mapped back to the molecular structure with the appropriate fingerprint settings. Small sampled summaries are exploratory and should not be presented as stable biological discoveries without repeated-fold/seed analysis.

The app's local report includes the selected drug-cell prediction, available observed response, exact feature-group contributions, and—after the user requests it—sampled local feature-level contributions. It can be downloaded as Markdown for a report or project presentation.
