from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve


PALETTE = {"navy": "#102a43", "blue": "#2f80ed", "teal": "#18a999", "orange": "#f2994a", "rose": "#d65a72"}


def save_evaluation_figures(output_dir: Path, predictions: pd.DataFrame, metrics: dict, *,
                           X_test: np.ndarray | None = None, feature_names: list[str] | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    y = predictions["actual_neglog10_ic50"].to_numpy(dtype=float)
    pred = predictions["predicted_neglog10_ic50"].to_numpy(dtype=float)
    residual = y - pred
    threshold = float(metrics["exploratory_threshold_neglog10_ic50"])
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y, pred, s=18, alpha=0.42, color=PALETTE["blue"], edgecolors="none")
    lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=PALETTE["rose"], linewidth=1.8)
    ax.set(xlabel="Observed −log10(IC50)", ylabel="Predicted −log10(IC50)", title="Observed vs predicted response")
    ax.text(0.03, 0.96, f"MAE {metrics['mae_neglog10_ic50']:.3f}  |  RMSE {metrics['rmse_neglog10_ic50']:.3f}",
            transform=ax.transAxes, va="top", color=PALETTE["navy"])
    fig.tight_layout()
    path = output_dir / "actual_vs_predicted.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(residual, bins=35, kde=True, color=PALETTE["teal"], ax=axes[0])
    axes[0].axvline(0, color=PALETTE["rose"], linestyle="--")
    axes[0].set(title="Residual distribution", xlabel="Observed − predicted")
    axes[1].scatter(pred, residual, s=16, alpha=0.35, color=PALETTE["blue"], edgecolors="none")
    axes[1].axhline(0, color=PALETTE["rose"], linestyle="--")
    axes[1].set(title="Residuals vs predicted", xlabel="Predicted −log10(IC50)", ylabel="Residual")
    fig.tight_layout()
    path = output_dir / "residual_diagnostics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    truth_class = y >= threshold
    pred_class = pred >= threshold
    cm = confusion_matrix(truth_class, pred_class, labels=[False, True])
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Below median*", "Above median*"], yticklabels=["Below median*", "Above median*"], ax=ax)
    ax.set(xlabel="Predicted label", ylabel="Observed label", title="Exploratory confusion matrix")
    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    if len(np.unique(truth_class)) == 2:
        fpr, tpr, _ = roc_curve(truth_class, pred)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr, tpr, color=PALETTE["teal"], linewidth=2,
                label=f"AUC {metrics.get('exploratory_auc', float('nan')):.3f}")
        ax.plot([0, 1], [0, 1], color="#7b8794", linestyle="--")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Exploratory ROC curve", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="lower right")
        fig.tight_layout()
        path = output_dir / "roc_curve.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

    if X_test is not None and feature_names:
        rng = np.random.default_rng(42)
        rows = rng.choice(len(X_test), size=min(2500, len(X_test)), replace=False)
        sample = np.asarray(X_test[rows], dtype=np.float64)
        variability = np.nanvar(sample, axis=0)
        feature_ids = np.argsort(variability)[-20:]
        corr_data = np.column_stack([sample[:, feature_ids], y[rows]])
        corr = np.corrcoef(corr_data, rowvar=False)
        labels = [feature_names[i].replace("__", "\n", 1) for i in feature_ids] + ["target"]
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, xticklabels=labels, yticklabels=labels,
                    square=True, ax=ax, cbar_kws={"label": "Pearson correlation"})
        ax.set_title("Correlation heatmap: 20 most variable features + response")
        ax.tick_params(axis="x", labelrotation=70, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        path = output_dir / "correlation_heatmap.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

    return outputs


def save_target_distribution(output_dir: Path, targets: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(np.asarray(targets, dtype=float), bins=40, kde=True, color=PALETTE["blue"], ax=ax)
    ax.set(title="GDSC response distribution", xlabel="−log10(IC50)", ylabel="Drug-cell line pairs")
    fig.tight_layout()
    path = output_dir / "target_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_training_history_figure(output_dir: Path, history: pd.DataFrame) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["epoch"], history["train_rmse"], color=PALETTE["blue"], linewidth=2, label="Training")
    ax.plot(history["epoch"], history["valid_rmse"], color=PALETTE["teal"], linewidth=2, label="Validation")
    ax.set(title="FNN training history", xlabel="Epoch", ylabel="RMSE on −log10(IC50)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "training_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_group_shap_figure(output_dir: Path, summary: pd.DataFrame) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = summary.sort_values("mean_absolute_group_shap")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(frame["feature_group"], frame["mean_absolute_group_shap"], color=PALETTE["blue"])
    ax.set(title="Global SHAP by PathDSP feature group", xlabel="Mean absolute group SHAP value")
    fig.tight_layout()
    path = output_dir / "global_group_shap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_feature_shap_figure(output_dir: Path, summary: pd.DataFrame, top_n: int = 20) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = summary.nlargest(top_n, "mean_absolute_shap").sort_values("mean_absolute_shap")
    labels = frame["feature"].str.replace("__", " | ", n=1, regex=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(labels, frame["mean_absolute_shap"], color=PALETTE["teal"])
    ax.set(title=f"Top {len(frame)} pathway and fingerprint features by SHAP", xlabel="Mean absolute SHAP value")
    fig.tight_layout()
    path = output_dir / "feature_shap_global.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

