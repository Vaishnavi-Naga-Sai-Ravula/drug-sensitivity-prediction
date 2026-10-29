from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import torch

from .model import PathDSPFNN


def exact_group_shap(
    model: PathDSPFNN,
    sample_scaled: np.ndarray,
    groups: dict[str, list[int]],
    *,
    baseline: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute exact Shapley values over feature groups (five PathDSP modalities).

    The default baseline is the all-zero standardized vector, i.e. each feature
    at its training mean after z-score scaling. Exact enumeration is small here:
    PathDSP has five groups and therefore only 32 coalitions.
    """
    x = np.asarray(sample_scaled, dtype=np.float32).reshape(-1)
    base = np.zeros_like(x) if baseline is None else np.asarray(baseline, dtype=np.float32).reshape(-1)
    names = list(groups)
    m = len(names)
    if m == 0:
        return {}
    coalitions = list(itertools.product((0, 1), repeat=m))
    inputs = np.repeat(base[None, :], len(coalitions), axis=0)
    for row, coalition in enumerate(coalitions):
        for j, active in enumerate(coalition):
            if active:
                idx = groups[names[j]]
                inputs[row, idx] = x[idx]
    model.eval()
    with torch.no_grad():
        outputs = model(torch.from_numpy(inputs)).squeeze(1).cpu().numpy()
    value_by_mask = {coalition: float(value) for coalition, value in zip(coalitions, outputs)}

    result: dict[str, float] = {}
    for j, name in enumerate(names):
        contribution = 0.0
        for coalition in coalitions:
            if coalition[j]:
                continue
            size = sum(coalition)
            weight = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
            with_group = list(coalition)
            with_group[j] = 1
            contribution += weight * (value_by_mask[tuple(with_group)] - value_by_mask[coalition])
        result[name] = contribution
    return result


def shap_feature_values(
    model: PathDSPFNN,
    background_scaled: np.ndarray,
    sample_scaled: np.ndarray,
    *,
    nsamples: int = 64,
) -> np.ndarray:
    """Calculate feature attributions with SHAP GradientExplainer (expected gradients)."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install the optional SHAP dependency with `pip install shap` to calculate feature-level attributions.") from exc
    model.eval()
    background = torch.as_tensor(np.asarray(background_scaled, dtype=np.float32))
    sample_array = np.asarray(sample_scaled, dtype=np.float32)
    if sample_array.ndim == 1:
        sample_array = sample_array.reshape(1, -1)
    sample = torch.as_tensor(sample_array)
    explainer = shap.GradientExplainer(model, background)
    values = explainer.shap_values(sample, nsamples=nsamples, rseed=42)
    if isinstance(values, list):
        values = values[0]
    result = np.asarray(values)
    if result.ndim == 3 and result.shape[-1] == 1:
        result = result[..., 0]
    elif result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if len(sample_array) == 1:
        return result.reshape(-1)
    return result.reshape(len(sample_array), -1)


def summarize_feature_shap(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != len(feature_names):
        raise ValueError("SHAP values do not match the model feature count.")
    groups = [name.split("__", 1)[0] for name in feature_names]
    return pd.DataFrame({
        "feature": feature_names,
        "feature_group": groups,
        "mean_absolute_shap": np.mean(np.abs(matrix), axis=0),
        "mean_signed_shap": np.mean(matrix, axis=0),
    }).sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)

