from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import hashlib
import tempfile
import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    confusion_matrix,
)
from scipy.stats import pearsonr, spearmanr
from .validation import validate_metadata


class PathDSPFNN(torch.nn.Module):
    """PathDSP fully connected neural network from the author implementation."""

    def __init__(self, n_inputs: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 1000),
            torch.nn.ELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(1000, 800),
            torch.nn.ELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(800, 500),
            torch.nn.ELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(500, 100),
            torch.nn.ELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(X, dtype=np.float32)
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    clean = np.where(np.isfinite(values), values, medians)
    means = clean.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = clean.std(axis=0, dtype=np.float64).astype(np.float32)
    scales = np.where(scales < 1e-8, 1.0, scales)
    return medians, means, scales


def standardize_transform(X: np.ndarray, medians: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float32)
    clean = np.where(np.isfinite(values), values, medians)
    return ((clean - means) / scales).astype(np.float32, copy=False)


def split_indices(meta: pd.DataFrame, test_size: float, seed: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(len(meta))
    if mode == "pair":
        return tuple(np.asarray(part, dtype=np.int64) for part in train_test_split(
            all_indices, test_size=test_size, random_state=seed, shuffle=True
        ))  # type: ignore[return-value]
    if mode not in {"drug", "cell"}:
        raise ValueError("split mode must be 'pair', 'drug', or 'cell'")
    groups = meta[mode].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(all_indices, groups=groups))
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def _init_weights(model: PathDSPFNN) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0.01)


def _fit_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    max_epochs: int = 30,
    batch_size: int = 256,
    patience: int = 6,
    learning_rate: float = 0.0004,
    seed: int = 42,
    progress=None,
) -> tuple[PathDSPFNN, list[dict[str, float]]]:
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = PathDSPFNN(X_train.shape[1])
    _init_weights(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    train_set = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.reshape(-1, 1)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    Xv = torch.from_numpy(X_valid)
    yv = torch.from_numpy(y_valid.reshape(-1, 1))
    best_state = deepcopy(model.state_dict())
    best_loss = math.inf
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        count = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = torch.sqrt(criterion(prediction, yb) + 1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)
            count += len(xb)

        model.eval()
        with torch.no_grad():
            val_prediction = model(Xv)
            val_rmse = float(torch.sqrt(criterion(val_prediction, yv)).item())
        train_rmse = total_loss / max(count, 1)
        history.append({"epoch": float(epoch + 1), "train_rmse": train_rmse, "valid_rmse": val_rmse})
        if progress is not None:
            progress(epoch + 1, max_epochs, train_rmse, val_rmse)
        if val_rmse < best_loss - 1e-5:
            best_loss = val_rmse
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_scaled(model: PathDSPFNN, X_scaled: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_scaled), batch_size):
            chunk = torch.from_numpy(np.asarray(X_scaled[start:start + batch_size], dtype=np.float32))
            predictions.append(model(chunk).squeeze(1).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float | None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    result = {
        "n_test": int(len(y_true)),
        "mae_neglog10_ic50": float(mean_absolute_error(y_true, y_pred)),
        "rmse_neglog10_ic50": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic) if np.std(y_true) and np.std(y_pred) else 0.0,
        "spearman_r": float(spearmanr(y_true, y_pred).statistic) if np.std(y_true) and np.std(y_pred) else 0.0,
    }
    if threshold is None:
        return result
    result["exploratory_threshold_neglog10_ic50"] = float(threshold)
    true_class = y_true >= threshold
    pred_class = y_pred >= threshold
    result["exploratory_auc"] = float(roc_auc_score(true_class, y_pred)) if len(np.unique(true_class)) == 2 else float("nan")
    result["exploratory_accuracy"] = float(np.mean(true_class == pred_class))
    tn, fp, fn, tp = confusion_matrix(true_class, pred_class, labels=[False, True]).ravel()
    result.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    result["exploratory_precision"] = float(tp / (tp + fp)) if tp + fp else 0.0
    result["exploratory_recall"] = float(tp / (tp + fn)) if tp + fn else 0.0
    result["exploratory_specificity"] = float(tn / (tn + fp)) if tn + fp else 0.0
    precision = result["exploratory_precision"]
    recall = result["exploratory_recall"]
    result["exploratory_f1"] = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return result


def fit_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    test_size: float = 0.2,
    split_mode: str = "pair",
    max_epochs: int = 30,
    seed: int = 42,
    progress=None,
) -> dict:
    train_idx, test_idx = split_indices(meta, test_size, seed, split_mode)
    inner_train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.1, random_state=seed, shuffle=True
    )
    medians, means, scales = standardize_fit(X[inner_train_idx])
    X_scaled = standardize_transform(X, medians, means, scales)
    model, history = _fit_network(
        X_scaled[inner_train_idx], y[inner_train_idx], X_scaled[valid_idx], y[valid_idx],
        max_epochs=max_epochs, seed=seed, progress=progress,
    )
    threshold = float(np.median(y[train_idx]))
    y_pred = predict_scaled(model, X_scaled[test_idx])
    metrics = regression_metrics(y[test_idx], y_pred, threshold)
    predictions = meta.iloc[test_idx].reset_index(drop=True).copy()
    predictions["actual_neglog10_ic50"] = y[test_idx]
    predictions["predicted_neglog10_ic50"] = y_pred
    predictions["absolute_error"] = np.abs(y[test_idx] - y_pred)
    predictions["actual_ic50_uM"] = np.power(10.0, -np.clip(y[test_idx], -20, 20))
    predictions["predicted_ic50_uM"] = np.power(10.0, -np.clip(y_pred, -20, 20))
    return {
        "model": model,
        "history": history,
        "medians": medians,
        "means": means,
        "scales": scales,
        "X_scaled": X_scaled,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "threshold": threshold,
        "predictions": predictions,
        "metrics": metrics,
    }


def fit_final_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_epochs: int = 30,
    seed: int = 42,
    progress=None,
) -> dict:
    train_idx, valid_idx = train_test_split(np.arange(len(y)), test_size=0.1, random_state=seed, shuffle=True)
    medians, means, scales = standardize_fit(X[train_idx])
    X_scaled = standardize_transform(X, medians, means, scales)
    model, history = _fit_network(
        X_scaled[train_idx], y[train_idx], X_scaled[valid_idx], y[valid_idx],
        max_epochs=max_epochs, seed=seed, progress=progress,
    )
    return {"model": model, "history": history, "medians": medians, "means": means, "scales": scales}


def artifact_digest(path: Path) -> str:
    """Ignore Git's Windows CRLF conversion in JSON, never in model binaries."""
    content = path.read_bytes()
    if path.suffix == ".json":
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def save_model_bundle(model: PathDSPFNN, feature_names: list[str], feature_groups: dict[str, list[int]],
                      medians: np.ndarray, means: np.ndarray, scales: np.ndarray, threshold: float,
                      path: Path, training_metrics: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "architecture": "PathDSP FNN (1000, 800, 500, 100; ELU; dropout=0.1)",
        "target": "negative log10(IC50 in micromolar), as represented by supplied PathDSP response file",
        "ic50_conversion": "IC50_uM = 10 ** (-target)",
        "threshold_neglog10_ic50": float(threshold),
        "feature_names": feature_names,
        "feature_groups": feature_groups,
        "imputation_medians": medians.astype(float).tolist(),
        "scaler_means": means.astype(float).tolist(),
        "scaler_scales": scales.astype(float).tolist(),
        "training_metrics": training_metrics or {},
    }
    validate_metadata(metadata)
    # Stage complete files; readers fail integrity validation during a mixed-version window.
    # Serving replicas retain their cached model until explicitly restarted.
    with tempfile.TemporaryDirectory(dir=path) as directory:
        staged = Path(directory)
        torch.save(model.state_dict(), staged / "model.pt")
        (staged / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest = {filename: artifact_digest(staged / filename) for filename in ("model.pt", "metadata.json")}
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for filename in ("model.pt", "metadata.json", "manifest.json"):
            os.replace(staged / filename, path / filename)


def load_model_bundle(path: Path, expected_features: list[str] | None = None) -> tuple[PathDSPFNN, dict]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if expected_features is not None and metadata["feature_names"] != expected_features:
        raise ValueError("The model's feature order does not match the loaded PathDSP matrices.")
    validate_metadata(metadata)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Model integrity manifest is missing. Restore artifacts/manifest.json with its matching model and metadata.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename in ("model.pt", "metadata.json"):
        if artifact_digest(path / filename) != manifest.get(filename):
            raise ValueError(f"Artifact integrity check failed for {filename}. Restore a matching model bundle.")
    model = PathDSPFNN(len(metadata["feature_names"]))
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Model weights contain nonfinite values.")
    model.load_state_dict(state)
    model.eval()
    return model, metadata

