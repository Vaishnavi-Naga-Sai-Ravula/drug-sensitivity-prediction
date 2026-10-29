from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pathdsp.data import build_dataset
from pathdsp.explain import exact_group_shap, shap_feature_values
from pathdsp.model import (
    fit_and_evaluate,
    fit_final_model,
    load_model_bundle,
    predict_scaled,
    save_model_bundle,
    standardize_transform,
)
from pathdsp.visualizations import (
    PALETTE,
    save_evaluation_figures,
    save_group_shap_figure,
    save_target_distribution,
    save_training_history_figure,
)
from train import save_feature_shap_outputs


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs"

st.set_page_config(page_title="PathDSP Lab", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2rem; max-width: 1320px;}
.hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(115deg,#102a43,#176b87); color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0 0 .25rem 0; font-size: 2.2rem;}
.hero p {margin: 0; color: #e4f3f7; font-size: 1rem;}
.notice {padding: .85rem 1rem; border-left: 4px solid #f2994a; background: #fff8ed; border-radius: 7px; color: #333;}
div[data-testid="stMetric"] {background: #f5f8fb; padding: .8rem 1rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading the supplied PathDSP matrices…")
def get_dataset(name: str):
    # Prediction/overview only need the compact drug and cell tables; avoid
    # expanding every measured drug-cell pair into a 155 MB feature matrix.
    return build_dataset(name, include_examples=False)


@st.cache_data(show_spinner="Preparing GDSC examples for training…")
def get_training_dataset():
    return build_dataset("GDSC", include_examples=True)


@st.cache_resource(show_spinner=False)
def get_deploy_model(feature_names: tuple[str, ...]):
    return load_model_bundle(ARTIFACT_DIR, list(feature_names))


def ic50_from_target(value: float) -> float:
    return float(np.power(10.0, -np.clip(value, -20, 20)))


def shap_direction(value: float) -> str:
    if value > 0:
        return "pushes the model toward a higher −log10(IC50), and therefore a lower predicted IC50"
    if value < 0:
        return "pushes the model toward a lower −log10(IC50), and therefore a higher predicted IC50"
    return "has approximately no signed contribution for this prediction"


def build_pair_shap_report(
    drug: str,
    cell: str,
    prediction: float,
    baseline: float,
    actual: float | None,
    group_values: dict[str, float],
    feature_frame: pd.DataFrame | None = None,
) -> str:
    """Create a plain-language, downloadable report for one drug-cell pair."""
    group_rows = sorted(group_values.items(), key=lambda item: abs(item[1]), reverse=True)
    lines = [
        f"### SHAP interpretation for {drug} × {cell}",
        "",
        f"- **Predicted response:** {prediction:.4f} on the −log10(IC50 in μM) scale "
        f"(approximately {ic50_from_target(prediction):.4g} μM).",
        f"- **Zero-standardized baseline:** {baseline:.4f} on the same scale "
        f"(approximately {ic50_from_target(baseline):.4g} μM).",
    ]
    if actual is not None:
        lines.append(
            f"- **Observed response in the supplied GDSC matrix:** {actual:.4f} on the same scale "
            f"(approximately {ic50_from_target(actual):.4g} μM)."
        )
    else:
        lines.append("- **Observed response:** not present for this pair in the supplied GDSC response matrix.")
    group_sum = float(sum(group_values.values()))
    additivity_error = (prediction - baseline) - group_sum
    lines.extend([
        "",
        "The exact five-group Shapley contributions sum to the prediction minus the zero-standardized baseline "
        f"(additivity residual: {additivity_error:.2e}). A positive contribution {shap_direction(1.0)}; "
        "a negative contribution moves it in the opposite direction.",
        "",
        "**Largest exact group contributions** (units are the model's transformed response):",
    ])
    for name, value in group_rows[:5]:
        lines.append(f"- **{name}: {value:+.4f}** — {shap_direction(value)}.")

    if feature_frame is not None and not feature_frame.empty:
        top_features = feature_frame.assign(
            absolute_shap=feature_frame["shap_value"].abs()
        ).sort_values("absolute_shap", ascending=False).head(8)
        lines.extend([
            "",
            "**Largest local feature-level contributions** (estimated with Gradient SHAP / expected gradients):",
        ])
        for row in top_features.itertuples(index=False):
            lines.append(f"- **{row.feature}: {row.shap_value:+.4f}** — {shap_direction(float(row.shap_value))}.")
        lines.extend([
            "",
            "Feature-level values are approximate and depend on the sampled background pairs. Morgan fingerprint "
            "labels such as `CHEM__mBit_11` are bit indices, not named chemical fragments.",
        ])
    lines.extend([
        "",
        "**Interpretation limit:** this describes how the fitted model used these inputs. It does not establish "
        "that a feature or pathway causes drug response and does not define a clinical sensitivity category.",
    ])
    return "\n".join(lines)


def show_metric_cards(metrics: dict) -> None:
    cols = st.columns(5)
    cards = [
        ("Holdout MAE", f"{metrics.get('mae_neglog10_ic50', float('nan')):.3f}"),
        ("Holdout RMSE", f"{metrics.get('rmse_neglog10_ic50', float('nan')):.3f}"),
        ("R²", f"{metrics.get('r2', float('nan')):.3f}"),
        ("Pearson r", f"{metrics.get('pearson_r', float('nan')):.3f}"),
        ("Exploratory AUC", f"{metrics.get('exploratory_auc', float('nan')):.3f}"),
    ]
    for col, (label, value) in zip(cols, cards):
        col.metric(label, value)


def show_image(name: str, caption: str) -> None:
    path = OUTPUT_DIR / "figures" / name
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")


def get_saved_metrics() -> dict | None:
    path = OUTPUT_DIR / "metrics.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def train_from_app(bundle, max_epochs: int, split_mode: str, status) -> dict:
    status.update(label="Training and evaluating the holdout model…", state="running")
    result = fit_and_evaluate(
        bundle.X, bundle.y, bundle.meta, split_mode=split_mode, max_epochs=max_epochs,
    )
    status.update(label="Fitting the deployable model on the full GDSC set…", state="running")
    final = fit_final_model(bundle.X, bundle.y, max_epochs=max_epochs, seed=43)
    save_model_bundle(final["model"], bundle.feature_names, bundle.feature_groups,
                      final["medians"], final["means"], final["scales"], float(np.median(bundle.y)),
                      ARTIFACT_DIR, training_metrics=result["metrics"])
    predictions = result["predictions"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)
    pd.DataFrame(result["history"]).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    save_target_distribution(OUTPUT_DIR / "figures", bundle.y)
    save_evaluation_figures(
        OUTPUT_DIR / "figures", predictions, result["metrics"],
        X_test=bundle.X[result["test_idx"]], feature_names=bundle.feature_names,
    )
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(result["test_idx"]), size=min(64, len(result["test_idx"])), replace=False)
    group_rows = [exact_group_shap(
        result["model"], result["X_scaled"][result["test_idx"][i]], bundle.feature_groups
    ) for i in chosen]
    group_frame = pd.DataFrame(group_rows)
    group_summary = pd.DataFrame({
        "feature_group": group_frame.columns,
        "mean_absolute_group_shap": [float(group_frame[col].abs().mean()) for col in group_frame.columns],
        "mean_signed_group_shap": [float(group_frame[col].mean()) for col in group_frame.columns],
    }).sort_values("mean_absolute_group_shap", ascending=False)
    group_frame.to_csv(OUTPUT_DIR / "holdout_group_shap_values.csv", index=False)
    group_summary.to_csv(OUTPUT_DIR / "global_group_shap.csv", index=False)
    save_training_history_figure(OUTPUT_DIR / "figures", pd.DataFrame(result["history"]))
    save_group_shap_figure(OUTPUT_DIR / "figures", group_summary)
    feature_shap_examples = 0
    if importlib.util.find_spec("shap"):
        rng = np.random.default_rng(49)
        background_idx = rng.choice(result["train_idx"], size=min(16, len(result["train_idx"])), replace=False)
        feature_idx = rng.choice(result["test_idx"], size=min(12, len(result["test_idx"])), replace=False)
        summary = save_feature_shap_outputs(
            result["model"], result["X_scaled"][background_idx], result["X_scaled"][feature_idx],
            bundle.meta.iloc[feature_idx].reset_index(drop=True), bundle.feature_names, OUTPUT_DIR, nsamples=32,
        )
        feature_shap_examples = len(feature_idx)
    metrics = {
        "dataset": "GDSC", "examples": int(len(bundle.y)), "feature_count": len(bundle.feature_names),
        "split_mode": split_mode, "holdout": result["metrics"],
        "feature_shap_examples": feature_shap_examples,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    status.update(label="Model ready", state="complete")
    get_deploy_model.clear()
    return metrics


st.sidebar.markdown("## 🧬 PathDSP Lab")
st.sidebar.caption("Pathway-based drug response research dashboard")
page = st.sidebar.radio("Open section", ["Overview", "Predict", "Evaluation", "Explainability", "Methods & deploy"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Research use only**")
st.sidebar.caption("Cell-line model for research and education. Not a clinical test or treatment recommendation.")


if page == "Overview":
    st.markdown('<section class="hero"><h1>PathDSP Lab</h1><p>Explore pathway-informed drug response predictions with model metrics and explanations.</p></section>', unsafe_allow_html=True)
    st.markdown('<div class="notice"><b>Interpretation:</b> the model predicts the supplied response target, −log10(IC50). Higher values correspond to lower IC50 under the source transformation. Values shown in μM are calculated as 10<sup>−target</sup> and inherit the source concentration unit.</div>', unsafe_allow_html=True)
    try:
        gdsc = get_dataset("GDSC")
        metrics_file = get_saved_metrics()
        st.subheader("Project snapshot")
        a, b, c, d = st.columns(4)
        a.metric("Drug-cell response pairs", f"{len(gdsc.y):,}")
        b.metric("Drugs with complete drug features", f"{len(gdsc.drug_ids):,}")
        c.metric("Cell lines with complete cell features", f"{len(gdsc.cell_ids):,}")
        d.metric("Model input features", f"{len(gdsc.feature_names):,}")
        st.caption("Inputs: Morgan fingerprints, DG-Net, EXP/ssGSEA, MUT-Net and CNV-Net pathway features.")
        if metrics_file:
            st.subheader("Latest holdout performance")
            show_metric_cards(metrics_file.get("holdout", {}))
            st.caption(f"Validation split: {metrics_file.get('split_mode', 'pair')} holdout. The final model is then refit on all GDSC pairs for the prediction page.")
        else:
            st.info("No trained model is saved yet. Open **Methods & deploy** and choose Train model.")
        show_image("target_distribution.png", "Distribution of measured GDSC drug-cell line responses")
    except Exception as exc:
        st.error(f"Could not load the PathDSP data: {exc}")
        st.info("Expand processed_data.zip in the project folder, or add app_data/pathdsp_core.zip.")

elif page == "Predict":
    st.title("Predict drug response")
    st.caption("Choose a measured drug and cell line from the supplied GDSC or CCLE feature matrices, or upload PathDSP-ready feature rows.")
    source = st.radio("Feature source", ["GDSC", "CCLE"], horizontal=True)
    try:
        bundle = get_dataset(source)
        model, metadata = get_deploy_model(tuple(bundle.feature_names))
    except Exception as exc:
        st.error(f"Prediction model or features are unavailable: {exc}")
        st.stop()

    with st.form("pair_prediction"):
        c1, c2 = st.columns(2)
        drug = c1.selectbox("Drug", sorted(bundle.drug_ids, key=str.casefold))
        cell = c2.selectbox("Cancer cell line", sorted(bundle.cell_ids, key=str.casefold))
        submitted = st.form_submit_button("Predict response", type="primary")
    if submitted:
        raw_x = bundle.pair_features(drug, cell).reshape(1, -1)
        x_scaled = standardize_transform(
            raw_x, np.asarray(metadata["imputation_medians"], dtype=np.float32),
            np.asarray(metadata["scaler_means"], dtype=np.float32),
            np.asarray(metadata["scaler_scales"], dtype=np.float32),
        )
        predicted = float(predict_scaled(model, x_scaled)[0])
        actual = bundle.observed_response(drug, cell)
        p1, p2, p3 = st.columns(3)
        p1.metric("Predicted −log10(IC50)", f"{predicted:.3f}")
        p2.metric("Predicted IC50 (μM*)", f"{ic50_from_target(predicted):.4g}")
        p3.metric("Observed −log10(IC50)", "Not available" if actual is None else f"{actual:.3f}",
                  delta=None if actual is None else f"{predicted - actual:+.3f} predicted − observed")
        st.caption("*The μM conversion assumes the supplied target was formed from IC50 measured in μM. Always check the source assay metadata before interpreting concentration values.")
        if actual is not None:
            st.write(f"Observed IC50 (μM, converted): **{ic50_from_target(actual):.4g}**")
            if source == "GDSC":
                st.caption("This pair is part of the GDSC reference table used to fit the deployable model. Treat this row as an in-sample illustration; use the held-out table and metrics for performance claims.")
            result = pd.DataFrame([{
                "dataset": source, "drug": drug, "cell": cell,
                "actual_neglog10_ic50": actual, "predicted_neglog10_ic50": predicted,
                "actual_ic50_uM": ic50_from_target(actual), "predicted_ic50_uM": ic50_from_target(predicted),
            }])
            st.download_button("Download this prediction", result.to_csv(index=False), "pathdsp_prediction.csv", "text/csv")
        group_values = exact_group_shap(model, x_scaled[0], metadata["feature_groups"])
        shap_frame = pd.DataFrame({"Feature group": list(group_values), "SHAP value": list(group_values.values())})
        shap_frame["Absolute contribution"] = shap_frame["SHAP value"].abs()
        shap_frame = shap_frame.sort_values("Absolute contribution", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.3))
        ax.barh(shap_frame["Feature group"], shap_frame["SHAP value"],
                color=[PALETTE["teal"] if value >= 0 else PALETTE["rose"] for value in shap_frame["SHAP value"]])
        ax.axvline(0, color="#52606d", linewidth=0.8)
        ax.set_xlabel("Contribution to predicted −log10(IC50)")
        ax.set_title("Exact SHAP values by PathDSP feature group")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.caption("Positive SHAP contribution raises predicted −log10(IC50), corresponding to lower IC50 under the source transform. These are model associations, not causal effects.")

    st.markdown("---")
    st.subheader("Batch predictions from a PathDSP-ready CSV/TSV")
    st.caption("The file must contain all model feature columns with the same definitions and units. Missing values are imputed with training medians.")
    st.download_button("Download feature-column template", pd.DataFrame(columns=metadata["feature_names"]).to_csv(index=False),
                       "pathdsp_feature_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload feature rows", type=["csv", "tsv", "txt"], key="batch_upload")
    if uploaded is not None:
        sep = "\t" if uploaded.name.lower().endswith((".tsv", ".txt")) else ","
        input_frame = pd.read_csv(uploaded, sep=sep)
        missing = sorted(set(metadata["feature_names"]) - set(input_frame.columns))
        if missing:
            st.error(f"The upload is missing {len(missing)} required feature columns. Example: {', '.join(missing[:6])}")
        else:
            raw = input_frame[metadata["feature_names"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            medians = np.asarray(metadata["imputation_medians"], dtype=np.float32)
            means = np.asarray(metadata["scaler_means"], dtype=np.float32)
            scales = np.asarray(metadata["scaler_scales"], dtype=np.float32)
            scaled = standardize_transform(raw, medians, means, scales)
            batch_pred = predict_scaled(model, scaled)
            out = input_frame.copy()
            out["predicted_neglog10_ic50"] = batch_pred
            out["predicted_ic50_uM_assuming_source_unit"] = np.power(10.0, -np.clip(batch_pred, -20, 20))
            st.dataframe(out.head(100), width="stretch")
            st.download_button("Download all predictions", out.to_csv(index=False), "pathdsp_batch_predictions.csv", "text/csv")

elif page == "Evaluation":
    st.title("Model evaluation")
    metrics_file = get_saved_metrics()
    if not metrics_file:
        st.info("Train the model first to generate holdout predictions and evaluation plots.")
        st.stop()
    holdout = metrics_file.get("holdout", {})
    show_metric_cards(holdout)
    st.caption("Regression metrics are calculated on a held-out set. The ROC/AUC and confusion matrix below are exploratory only: labels use the training-set median response as a visualization threshold, not a validated clinical cutoff.")
    binary = {key.removeprefix("exploratory_"): holdout[key] for key in
              ["exploratory_accuracy", "exploratory_precision", "exploratory_recall",
               "exploratory_specificity", "exploratory_f1", "exploratory_auc"] if key in holdout}
    if binary:
        st.dataframe(pd.DataFrame([{"Exploratory median-split metric": key, "Value": value}
                                   for key, value in binary.items()]), width="stretch", hide_index=True)
    left, right = st.columns(2)
    with left:
        show_image("actual_vs_predicted.png", "Held-out observed versus predicted response")
        show_image("roc_curve.png", "Exploratory ROC curve using the training median as a threshold")
        show_image("correlation_heatmap.png", "Correlation matrix for the most variable test-set features and target")
    with right:
        show_image("residual_diagnostics.png", "Held-out residual diagnostics")
        show_image("confusion_matrix.png", "Exploratory confusion matrix using the training median")
        show_image("target_distribution.png", "Distribution of measured GDSC responses")
        show_image("training_curve.png", "Training and validation RMSE over epochs")
        show_image("global_group_shap.png", "Global mean absolute SHAP contribution by feature group")
    if "ccle_external" in metrics_file:
        st.subheader("Independent CCLE evaluation")
        external = metrics_file["ccle_external"]
        cols = st.columns(4)
        for col, (label, key) in zip(cols, [("External MAE", "mae_neglog10_ic50"),
                                             ("External RMSE", "rmse_neglog10_ic50"),
                                             ("External R²", "r2"), ("Pearson r", "pearson_r")]):
            col.metric(label, f"{external.get(key, float('nan')):.3f}")
        st.caption("The model was trained on GDSC and evaluated on CCLE. Assay ranges differ between datasets, so interpret the external error separately.")
        file = OUTPUT_DIR / "ccle_external_predictions.csv"
        if file.is_file():
            st.download_button("Download CCLE actual/predicted table", file.read_bytes(), "ccle_external_predictions.csv", "text/csv")
    predictions_path = OUTPUT_DIR / "holdout_predictions.csv"
    if predictions_path.is_file():
        st.download_button("Download GDSC holdout actual/predicted table", predictions_path.read_bytes(), "holdout_predictions.csv", "text/csv")
    st.json(metrics_file)

elif page == "Explainability":
    st.title("Model explanations")
    guide_path = ROOT / "SHAP_INTERPRETATION_GUIDE.md"
    if guide_path.is_file():
        with st.expander("SHAP theory: what the values mean and how to read them", expanded=True):
            st.markdown(guide_path.read_text(encoding="utf-8"))
    else:
        st.info("SHAP values are model attributions relative to a baseline. Positive values raise predicted −log10(IC50), corresponding to lower predicted IC50; they are not causal effects.")
    try:
        bundle = get_dataset("GDSC")
        model, metadata = get_deploy_model(tuple(bundle.feature_names))
    except Exception as exc:
        st.error(f"Model or feature data are unavailable: {exc}")
        st.stop()
    drug = st.selectbox("Drug for local explanation", sorted(bundle.drug_ids, key=str.casefold), key="explain_drug")
    cell = st.selectbox("Cell line for local explanation", sorted(bundle.cell_ids, key=str.casefold), key="explain_cell")
    raw_x = bundle.pair_features(drug, cell).reshape(1, -1)
    medians = np.asarray(metadata["imputation_medians"], dtype=np.float32)
    means = np.asarray(metadata["scaler_means"], dtype=np.float32)
    scales = np.asarray(metadata["scaler_scales"], dtype=np.float32)
    x_scaled = standardize_transform(raw_x, medians, means, scales)
    predicted_target = float(predict_scaled(model, x_scaled)[0])
    baseline_target = float(predict_scaled(model, np.zeros_like(x_scaled))[0])
    observed_target = bundle.observed_response(drug, cell)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Predicted −log10(IC50)", f"{predicted_target:.3f}")
    metric_cols[1].metric("Predicted IC50 (μM*)", f"{ic50_from_target(predicted_target):.4g}")
    if observed_target is not None:
        metric_cols[2].metric("Observed IC50 (μM*)", f"{ic50_from_target(observed_target):.4g}")
    else:
        metric_cols[2].metric("Observed IC50", "Not available")
    st.caption("*Conversion assumes the supplied transformed target used μM. These cell-line model results are research outputs, not treatment advice.")
    group_values = exact_group_shap(model, x_scaled[0], metadata["feature_groups"])
    group_frame = pd.DataFrame({"feature_group": group_values.keys(), "shap_value": group_values.values()})
    group_frame["absolute_shap"] = group_frame["shap_value"].abs()
    st.dataframe(group_frame.sort_values("absolute_shap", ascending=False), width="stretch", hide_index=True)
    group_report = build_pair_shap_report(
        drug, cell, predicted_target, baseline_target, observed_target, group_values
    )
    st.markdown(group_report)
    st.download_button("Download local SHAP interpretation report (.md)", group_report,
                       "local_shap_interpretation_report.md", "text/markdown", key="download_group_shap_report")

    if importlib.util.find_spec("shap"):
        if st.button("Calculate feature-level SHAP values for this pair", type="primary"):
            rng = np.random.default_rng(17)
            background_rows = rng.choice(len(bundle.meta), size=min(32, len(bundle.meta)), replace=False)
            background_raw = np.stack([
                bundle.pair_features(bundle.meta.iloc[i]["drug"], bundle.meta.iloc[i]["cell"])
                for i in background_rows
            ])
            background = standardize_transform(background_raw, medians, means, scales)
            try:
                values = shap_feature_values(model, background, x_scaled[0], nsamples=64)
                top_ids = np.argsort(np.abs(values))[-25:]
                feature_frame = pd.DataFrame({"feature": np.asarray(metadata["feature_names"])[top_ids],
                                              "shap_value": values[top_ids]})
                st.session_state["local_feature_shap_pair"] = (drug, cell)
                st.session_state["local_feature_shap_frame"] = feature_frame
            except Exception as exc:
                st.error(f"Feature-level SHAP could not run: {exc}")
    else:
        st.info("Exact five-group SHAP is available now. Install the `shap` package to calculate feature-level Gradient SHAP values.")
    if st.session_state.get("local_feature_shap_pair") == (drug, cell):
        feature_frame = st.session_state["local_feature_shap_frame"]
        local_report = build_pair_shap_report(
            drug, cell, predicted_target, baseline_target, observed_target, group_values, feature_frame
        )
        st.subheader("Feature-level local explanation report")
        st.markdown(local_report)
        display_frame = feature_frame.sort_values("shap_value")
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(display_frame["feature"], display_frame["shap_value"],
                color=[PALETTE["teal"] if v >= 0 else PALETTE["rose"] for v in display_frame["shap_value"]])
        ax.axvline(0, color="#52606d", linewidth=0.8)
        ax.set_title("Feature-level SHAP values (top 25 by absolute contribution)")
        ax.set_xlabel("Contribution to predicted −log10(IC50)")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.download_button("Download local feature SHAP values (.csv)", feature_frame.to_csv(index=False),
                           "feature_shap_values.csv", "text/csv", key="download_local_feature_shap")
        st.download_button("Download feature-level interpretation report (.md)", local_report,
                           "local_feature_shap_interpretation_report.md", "text/markdown", key="download_local_feature_report")

    feature_summary_path = OUTPUT_DIR / "global_feature_shap.csv"
    feature_figure_path = OUTPUT_DIR / "figures" / "feature_shap_global.png"
    if feature_summary_path.is_file():
        st.subheader("Global feature-level SHAP on sampled GDSC pairs")
        feature_summary = pd.read_csv(feature_summary_path)
        if feature_figure_path.is_file():
            st.image(str(feature_figure_path), caption="Top pathway and fingerprint features by mean absolute SHAP value", width="stretch")
        st.dataframe(feature_summary.head(10), width="stretch", hide_index=True)
        st.download_button("Download global feature SHAP summary", feature_summary_path.read_bytes(),
                           "global_feature_shap.csv", "text/csv")
    global_path = OUTPUT_DIR / "global_group_shap.csv"
    if global_path.is_file():
        st.subheader("Global group importance on held-out examples")
        global_frame = pd.read_csv(global_path).sort_values("mean_absolute_group_shap")
        metrics_path = OUTPUT_DIR / "metrics.json"
        shap_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        feature_summary = pd.read_csv(feature_summary_path) if feature_summary_path.is_file() else pd.DataFrame()
        top_group = global_frame.sort_values("mean_absolute_group_shap", ascending=False).iloc[0]
        report_lines = [
            "## Global SHAP summary report",
            "",
            "SHAP values decompose model output relative to a baseline. Mean absolute values describe average contribution magnitude, while signed averages describe net direction and may hide mixed effects.",
            "",
            f"- The largest mean absolute **group-level** contribution in the saved holdout summary is **{top_group['feature_group']}** ({top_group['mean_absolute_group_shap']:.4f}).",
            "- The saved group summary is calculated on held-out examples with exact five-group Shapley values.",
        ]
        if not feature_summary.empty:
            top_features = feature_summary.head(5)
            sample_count = shap_metrics.get("feature_shap_examples", "the recorded sample")
            background_count = shap_metrics.get("feature_shap_background_examples", "the recorded background")
            method = shap_metrics.get("feature_shap_method", "Gradient SHAP / expected gradients")
            model_scope = shap_metrics.get("feature_shap_model", "see metrics.json for model scope")
            report_lines.extend([
                f"- Feature-level summary: **{method}**, using {sample_count} sampled GDSC pairs and {background_count} background pairs; model scope: {model_scope}.",
                "- Top feature labels by mean absolute SHAP in this sample: " + ", ".join(
                    f"`{row.feature}` ({row.mean_absolute_shap:.4f})" for row in top_features.itertuples(index=False)
                ) + ".",
                "- A feature's mean absolute SHAP is an importance magnitude, not a signed biological effect. Fingerprint bit labels require molecular mapping before they can be described as chemical fragments.",
            ])
        report_lines.extend([
            "",
            "These are model attributions, not causal findings. The sampled feature-level summary is exploratory and should be checked for stability across held-out folds and random seeds.",
        ])
        global_report = "\n".join(report_lines)
        st.markdown(global_report)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(global_frame["feature_group"], global_frame["mean_absolute_group_shap"], color=PALETTE["blue"])
        ax.set_xlabel("Mean absolute group SHAP value")
        ax.set_title("Global contribution by feature group")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.download_button("Download global group SHAP summary", global_path.read_bytes(),
                           "global_group_shap.csv", "text/csv")
        st.download_button("Download global SHAP interpretation report (.md)", global_report,
                           "global_shap_interpretation_report.md", "text/markdown")

elif page == "Methods & deploy":
    st.title("Methods, training and deployment")
    st.markdown("### What the app implements")
    st.markdown("""
- **PathDSP features:** 256 Morgan fingerprint bits plus drug-target DG-Net and cell-line EXP, MUT-Net and CNV-Net pathway features.
- **Response:** regression on the supplied `negLog10IC50` matrix. For display, IC50 is calculated as `10 ** (-target)` in μM, assuming the source file used μM before transformation.
- **Model:** the author's fully connected network dimensions (1000, 800, 500, 100), ELU activations and dropout, with training-only feature z-scoring, early stopping and held-out evaluation.
- **RWR:** Random Walk with Restart is the method used by NetPEA to make the supplied DG-Net, MUT-Net and CNV-Net features. The upstream processed matrices are included, and `pathdsp/rwr.py` contains a reusable diffusion implementation. The app does not claim to rerun NetPEA's 1,000-permutation pathway normalization.
- **SHAP:** exact Shapley values across the five feature groups are always available. Feature-level Gradient SHAP (expected gradients) is available when `shap` is installed.
- **Classification-style charts:** ROC/AUC and confusion matrix use the GDSC training median as an exploratory threshold. The research paper uses regression because there is no universal sensitive/resistant cutoff.
""")
    st.markdown("### Train or retrain")
    st.caption("Training takes place locally or in the app process. It uses the supplied GDSC matrices and writes `artifacts/model.pt`, `artifacts/metadata.json` and evaluation outputs.")
    col1, col2 = st.columns(2)
    epochs = col1.slider("Maximum epochs", 5, 100, 30, 5)
    split_mode = col2.selectbox("Holdout strategy", ["pair", "cell", "drug"], index=0,
                                help="Pair split matches the paper's random pair evaluation. Cell/drug holdouts probe harder generalization settings.")
    if st.button("Train model and generate evaluation outputs", type="primary"):
        with st.status("Starting model run…", expanded=True) as status:
            try:
                gdsc = get_training_dataset()
                result = train_from_app(gdsc, epochs, split_mode, status)
                st.success(f"Training finished on {result['examples']:,} pairs and {result['feature_count']:,} features.")
                st.json(result)
            except Exception as exc:
                status.update(label="Training failed", state="error")
                st.exception(exc)
    st.markdown("### Run locally")
    st.code("python -m venv .venv\n.venv\\Scripts\\Activate.ps1  # Windows\npip install -r requirements.txt\npython train.py --split pair --epochs 30\nstreamlit run app.py", language="powershell")
    st.markdown("### Deploy")
    st.markdown("1. Push the app files, `app_data/pathdsp_core.zip` and the trained `artifacts/` folder to a GitHub repository. Keep raw omics files and the full PPI bundle out of the deployment repo.\n2. In Streamlit Community Cloud, create an app from that repo and set the entry point to `app.py`.\n3. For a container deployment, build with the included `Dockerfile`; the app listens on port 8501.\n4. Share the published URL after checking it with the included sample drug and cell line.")
    st.markdown("### Research paper and upstream code")
    st.markdown("Tang & Gottlieb, *Explainable drug sensitivity prediction through cancer pathway enrichment*, Scientific Reports (2021). The upstream code and MIT license are in `PathDSP-main/`.")
    st.markdown("This dashboard is a research prototype. Cell-line results do not establish patient benefit, causation or treatment choice. Patient data should not be uploaded.")

