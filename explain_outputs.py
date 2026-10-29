from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from pathdsp.data import build_dataset, normalize_id
from pathdsp.explain import shap_feature_values, summarize_feature_shap
from pathdsp.model import load_model_bundle, standardize_transform
from pathdsp.visualizations import save_feature_shap_figure


ROOT = Path(__file__).resolve().parent


def main() -> None:
    bundle = build_dataset("GDSC", include_examples=False)
    model, metadata = load_model_bundle(ROOT / "artifacts", bundle.feature_names)
    predictions = pd.read_csv(ROOT / "outputs" / "holdout_predictions.csv")
    rng = np.random.default_rng(49)
    sample_count = min(12, len(predictions))
    selected = predictions.sample(n=sample_count, random_state=49).reset_index(drop=True)
    holdout_pairs = {(normalize_id(row.drug), normalize_id(row.cell)) for row in predictions.itertuples()}
    train_rows = [i for i, row in bundle.meta.iterrows()
                  if (normalize_id(row.drug), normalize_id(row.cell)) not in holdout_pairs]
    background_count = min(16, len(train_rows))
    background_ids = rng.choice(train_rows, size=background_count, replace=False)

    selected_raw = np.stack([bundle.pair_features(row.drug, row.cell) for row in selected.itertuples()])
    background_raw = np.stack([
        bundle.pair_features(bundle.meta.iloc[i]["drug"], bundle.meta.iloc[i]["cell"])
        for i in background_ids
    ])
    medians = np.asarray(metadata["imputation_medians"], dtype=np.float32)
    means = np.asarray(metadata["scaler_means"], dtype=np.float32)
    scales = np.asarray(metadata["scaler_scales"], dtype=np.float32)
    sample_scaled = standardize_transform(selected_raw, medians, means, scales)
    background_scaled = standardize_transform(background_raw, medians, means, scales)
    values = shap_feature_values(model, background_scaled, sample_scaled, nsamples=32)

    out = ROOT / "outputs"
    value_frame = pd.DataFrame(values, columns=metadata["feature_names"])
    value_frame.insert(0, "cell", selected["cell"].astype(str))
    value_frame.insert(0, "drug", selected["drug"].astype(str))
    value_frame.to_csv(out / "sampled_feature_shap_values.csv", index=False)
    summary = summarize_feature_shap(values, metadata["feature_names"])
    summary.to_csv(out / "global_feature_shap.csv", index=False)
    save_feature_shap_figure(out / "figures", summary)

    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["feature_shap_examples"] = sample_count
    metrics["feature_shap_background_examples"] = background_count
    metrics["feature_shap_model"] = "final deployable model (fit on all GDSC pairs)"
    metrics["feature_shap_method"] = "SHAP GradientExplainer (expected gradients), nsamples=32"
    metrics["top_feature_shap"] = summary.head(10).to_dict(orient="records")
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"sample_count": sample_count, "background_count": background_count,
                      "top_features": summary.head(10).to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()

