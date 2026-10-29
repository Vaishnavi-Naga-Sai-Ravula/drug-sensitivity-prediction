"""Read-only, bounded inference shared by the dashboard and optional HTTP API."""
from functools import lru_cache
from pathlib import Path
import hashlib
import threading
import numpy as np
import pandas as pd
from .data import build_dataset, normalize_id
from .model import load_model_bundle, standardize_transform, predict_scaled
from .validation import MAX_BATCH_ROWS, distribution_warning

ROOT = Path(__file__).resolve().parents[1]
_INFERENCE_SLOTS = threading.BoundedSemaphore(2)

@lru_cache(maxsize=1)
def model_resource():
    model, metadata = load_model_bundle(ROOT / 'artifacts')
    return model, metadata, hashlib.sha256((ROOT / 'artifacts/model.pt').read_bytes()).hexdigest()[:16]

@lru_cache(maxsize=2)
def dataset_resource(name):
    return build_dataset(name, include_examples=False)



@lru_cache(maxsize=1)
def _split_roles(modified_ns, size):
    frame=pd.read_csv(ROOT/'outputs/model_split_membership.csv')
    return {(normalize_id(row.drug),normalize_id(row.cell)):row.model_split for row in frame.itertuples()}


def pair_scope(dataset,drug,cell):
    if dataset.upper()!='GDSC':
        return 'external; not used for fitting'
    path=ROOT/'outputs/model_split_membership.csv'
    if not path.is_file():
        return 'development data; split not recorded'
    stat=path.stat()
    return _split_roles(stat.st_mtime_ns,stat.st_size).get((normalize_id(drug),normalize_id(cell)),'unmeasured pair; no fitting label')

def infer_features(raw, model, metadata):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != len(metadata['feature_names']) or not 1 <= len(raw) <= MAX_BATCH_ROWS:
        raise ValueError('Feature batch has invalid dimensions or exceeds the batch limit.')
    if np.isinf(raw).any() or np.any(np.isnan(raw).mean(axis=1) > .2):
        raise ValueError('Feature batch contains infinite values or too many missing features.')
    scaled = standardize_transform(raw, np.asarray(metadata['imputation_medians']), np.asarray(metadata['scaler_means']), np.asarray(metadata['scaler_scales']))
    if not np.isfinite(scaled).all():
        raise ValueError('Feature scaling produced nonfinite values.')
    if not _INFERENCE_SLOTS.acquire(timeout=10):
        raise RuntimeError('Inference is busy. Retry shortly.')
    try:
        predictions = predict_scaled(model, scaled, batch_size=256)
    finally:
        _INFERENCE_SLOTS.release()
    if not np.isfinite(predictions).all():
        raise RuntimeError('Model returned nonfinite predictions.')
    return predictions, scaled, distribution_warning(scaled)


def predict_pairs(dataset: str, pairs: list[tuple[str, str]]) -> dict:
    dataset = dataset.upper()
    if dataset not in ('GDSC', 'CCLE') or not 1 <= len(pairs) <= 256:
        raise ValueError('Select GDSC or CCLE and supply 1–256 drug/cell pairs.')
    bundle = dataset_resource(dataset)
    model, metadata, version = model_resource()
    if bundle.feature_names != metadata['feature_names']:
        raise ValueError('Dataset feature order does not match the model.')
    try:
        canonical = [(bundle.drug_ids[bundle.drug_index[normalize_id(drug)]], bundle.cell_ids[bundle.cell_index[normalize_id(cell)]]) for drug, cell in pairs]
    except KeyError as exc:
        raise ValueError('A drug or cell identifier is not present in the selected dataset.') from exc
    raw = np.stack([bundle.pair_features(drug, cell) for drug, cell in canonical])
    predictions, _, shifted = infer_features(raw, model, metadata)
    results = []
    for (drug, cell), prediction in zip(canonical, predictions):
        actual = bundle.observed_response(drug, cell)
        error = None if actual is None else abs(float(prediction) - actual)
        results.append({'drug': drug, 'cell': cell, 'predicted_neglog10_ic50': float(prediction),
                        'predicted_ic50_uM_assuming_source_unit': float(10 ** -np.clip(prediction, -20, 20)),
                        'observed_neglog10_ic50': actual, 'absolute_log_error': error, 'model_split':pair_scope(dataset,drug,cell),
                        'concentration_fold_error': None if error is None else float(10 ** min(error, 20))})
    warnings = ['Research-use model; predictions are not clinical recommendations.']
    if dataset == 'CCLE':
        warnings.append('Poor cross-study transfer: the deployment model has external CCLE R² < 0.')
    if shifted:
        warnings.append('Some features are far outside training scaling; this is a heuristic distribution warning.')
    return {'dataset': dataset, 'model_version': version, 'results': results, 'warnings': warnings}
