from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pathdsp.data import build_dataset
from pathdsp.explain import exact_group_shap, shap_feature_values, summarize_feature_shap
from pathdsp.model import (
    fit_and_evaluate,
    fit_final_model,
    predict_scaled,
    regression_metrics,
    save_model_bundle,
    standardize_transform,
)
from pathdsp.visualizations import (
    save_evaluation_figures,
    save_group_shap_figure,
    save_target_distribution,
    save_feature_shap_figure,
    save_training_history_figure,
)


ROOT = Path(__file__).resolve().parent


def save_group_shap(model, X_scaled: np.ndarray, groups: dict[str, list[int]], output_dir: Path,
                    sample_count: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    selected = rng.choice(len(X_scaled), size=min(sample_count, len(X_scaled)), replace=False)
    rows = [exact_group_shap(model, X_scaled[i], groups) for i in selected]
    frame = pd.DataFrame(rows)
    summary = pd.DataFrame({
        "feature_group": frame.columns,
        "mean_absolute_group_shap": [float(frame[c].abs().mean()) for c in frame.columns],
        "mean_signed_group_shap": [float(frame[c].mean()) for c in frame.columns],
    }).sort_values("mean_absolute_group_shap", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "holdout_group_shap_values.csv", index=False)
    summary.to_csv(output_dir / "global_group_shap.csv", index=False)
    return summary


def save_feature_shap_outputs(model, background_scaled: np.ndarray, samples_scaled: np.ndarray,
                              sample_meta: pd.DataFrame, feature_names: list[str], output_dir: Path,
                              *, nsamples: int = 32) -> pd.DataFrame:
    values = shap_feature_values(model, background_scaled, samples_scaled, nsamples=nsamples)
    output_dir.mkdir(parents=True, exist_ok=True)
    value_frame = pd.DataFrame(values, columns=feature_names)
    value_frame.insert(0, "cell", sample_meta["cell"].astype(str).to_numpy())
    value_frame.insert(0, "drug", sample_meta["drug"].astype(str).to_numpy())
    value_frame.to_csv(output_dir / "sampled_feature_shap_values.csv", index=False)
    summary = summarize_feature_shap(values, feature_names)
    summary.to_csv(output_dir / "global_feature_shap.csv", index=False)
    save_feature_shap_figure(output_dir / "figures", summary)
    return summary


def run(*, split_mode: str = "pair", max_epochs: int = 30, test_size: float = 0.2, seed: int = 42,
        output_dir: Path | None = None, progress=None) -> dict:
    output_dir = output_dir or ROOT / "outputs"
    artifact_dir = ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_dataset("GDSC")
    evaluation = fit_and_evaluate(
        bundle.X, bundle.y, bundle.meta,
        test_size=test_size, split_mode=split_mode, max_epochs=max_epochs, seed=seed, progress=progress,
    )
    predictions = evaluation["predictions"]
    predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
    pd.DataFrame(evaluation["history"]).to_csv(output_dir / "training_history.csv", index=False)
    metrics: dict[str, object] = {
        "dataset": "GDSC",
        "examples": int(len(bundle.y)),
        "drugs_with_all_drug_features": int(len(bundle.drug_ids)),
        "cell_lines_with_all_cell_features": int(len(bundle.cell_ids)),
        "feature_count": int(len(bundle.feature_names)),
        "feature_groups": {name: len(indices) for name, indices in bundle.feature_groups.items()},
        "split_mode": split_mode,
        "test_size": test_size,
        "max_epochs": max_epochs,
        "epochs_used": len(evaluation["history"]),
        "seed": seed,
        "holdout": evaluation["metrics"],
    }

    # Independent CCLE evaluation follows the research paper's train-on-GDSC,
    # test-on-CCLE design. Results are separately labeled because assay ranges differ.
    external_metrics = None
    try:
        ccle = build_dataset("CCLE")
        if ccle.feature_names != bundle.feature_names:
            missing = sorted(set(bundle.feature_names) - set(ccle.feature_names))
            extra = sorted(set(ccle.feature_names) - set(bundle.feature_names))
            raise ValueError(f"GDSC/CCLE feature definitions differ (missing={len(missing)}, extra={len(extra)}).")
        ccle_scaled = standardize_transform(
            ccle.X, evaluation["medians"], evaluation["means"], evaluation["scales"]
        )
        ccle_pred = predict_scaled(evaluation["model"], ccle_scaled)
        # Do not apply the GDSC-derived binary threshold to CCLE: its capped
        # response range can put every pair in one class, making ROC/AUC meaningless.
        external_metrics = regression_metrics(ccle.y, ccle_pred, threshold=None)
        external = ccle.meta.reset_index(drop=True).copy()
        external["actual_neglog10_ic50"] = ccle.y
        external["predicted_neglog10_ic50"] = ccle_pred
        external["absolute_error"] = np.abs(ccle.y - ccle_pred)
        external["actual_ic50_uM"] = np.power(10.0, -np.clip(ccle.y, -20, 20))
        external["predicted_ic50_uM"] = np.power(10.0, -np.clip(ccle_pred, -20, 20))
        external.to_csv(output_dir / "ccle_external_predictions.csv", index=False)
        metrics["ccle_external"] = external_metrics
        metrics["ccle_external_examples"] = int(len(ccle.y))
    except (FileNotFoundError, ValueError) as exc:
        metrics["ccle_external_note"] = str(exc)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=True)

    figure_dir = output_dir / "figures"
    save_target_distribution(figure_dir, bundle.y)
    save_evaluation_figures(
        figure_dir, predictions, evaluation["metrics"],
        X_test=bundle.X[evaluation["test_idx"]], feature_names=bundle.feature_names,
    )
    group_summary = save_group_shap(
        evaluation["model"], evaluation["X_scaled"][evaluation["test_idx"]],
        bundle.feature_groups, output_dir,
    )
    save_training_history_figure(figure_dir, pd.DataFrame(evaluation["history"]))
    save_group_shap_figure(figure_dir, group_summary)

    feature_shap_note = None
    try:
        rng = np.random.default_rng(seed + 7)
        background_idx = rng.choice(evaluation["train_idx"], size=min(16, len(evaluation["train_idx"])), replace=False)
        feature_idx = rng.choice(evaluation["test_idx"], size=min(12, len(evaluation["test_idx"])), replace=False)
        feature_summary = save_feature_shap_outputs(
            evaluation["model"], evaluation["X_scaled"][background_idx],
            evaluation["X_scaled"][feature_idx], bundle.meta.iloc[feature_idx].reset_index(drop=True),
            bundle.feature_names, output_dir, nsamples=32,
        )
        metrics["feature_shap_examples"] = int(len(feature_idx))
        metrics["feature_shap_model"] = "holdout evaluation model"
        metrics["top_feature_shap"] = feature_summary.head(10).to_dict(orient="records")
    except RuntimeError as exc:
        feature_shap_note = str(exc)
    if feature_shap_note:
        metrics["feature_shap_note"] = feature_shap_note
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=True)

    # Fit a final deployable model on all supplied GDSC examples after measuring
    # generalization on the untouched holdout split above.
    final = fit_final_model(bundle.X, bundle.y, max_epochs=max_epochs, seed=seed + 1, progress=progress)
    save_model_bundle(
        final["model"], bundle.feature_names, bundle.feature_groups,
        final["medians"], final["means"], final["scales"], float(np.median(bundle.y)),
        artifact_dir, training_metrics=evaluation["metrics"],
    )
    return {"metrics": metrics, "group_shap": group_summary, "artifact_dir": artifact_dir,
            "output_dir": output_dir, "feature_count": len(bundle.feature_names),
            "examples": len(bundle.y), "epochs_used": len(evaluation["history"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the PathDSP research app model.")
    parser.add_argument("--split", choices=["pair", "drug", "cell"], default="pair",
                        help="hold out drug-cell pairs, whole drugs, or whole cell lines")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    result = run(split_mode=args.split, max_epochs=args.epochs, test_size=args.test_size,
                 seed=args.seed, output_dir=args.output_dir)
    print(json.dumps({"metrics": result["metrics"], "artifacts": str(result["artifact_dir"]),
                      "outputs": str(result["output_dir"]), "examples": result["examples"],
                      "features": result["feature_count"], "epochs_used": result["epochs_used"]}, indent=2))


if __name__ == "__main__":
    main()

