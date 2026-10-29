"""Explicit contracts for model artifacts and uploaded feature tables."""
from io import BytesIO
import csv
import numpy as np
import pandas as pd

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_BATCH_ROWS = 2048


def validate_metadata(metadata: dict) -> None:
    if not isinstance(metadata, dict):
        raise ValueError('Model metadata must be an object.')
    names = metadata.get('feature_names', [])
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names) or len(names) != len(set(names)):
        raise ValueError('Model feature names must be nonempty and unique.')
    for key in ('imputation_medians', 'scaler_means', 'scaler_scales'):
        values = np.asarray(metadata.get(key, []), dtype=np.float32)
        if values.shape != (len(names),) or not np.isfinite(values).all():
            raise ValueError(f'Model {key} must contain one finite value per feature.')
        if key == 'scaler_scales' and np.any(values <= 0):
            raise ValueError('Model feature scales must be positive.')
    groups = metadata.get('feature_groups', {})
    if not isinstance(groups, dict) or not all(isinstance(values, list) for values in groups.values()):
        raise ValueError('Model feature groups must be lists of feature indices.')
    indices = [i for values in groups.values() for i in values]
    if not all(type(i) is int for i in indices) or sorted(indices) != list(range(len(names))):
        raise ValueError('Model feature groups must cover each input feature exactly once.')


def validate_feature_frame(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    if frame.empty or len(frame) > MAX_BATCH_ROWS:
        raise ValueError(f'Upload must contain 1–{MAX_BATCH_ROWS:,} rows.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing {len(missing)} required model feature columns. Download the template first.')
    selected = frame[names]
    numeric = selected.apply(pd.to_numeric, errors='coerce')
    if (selected.notna() & numeric.isna()).any().any():
        raise ValueError('Feature columns contain nonnumeric text. Only numeric values or blank cells are accepted.')
    raw = numeric.to_numpy(dtype=np.float32)
    if np.isinf(raw).any():
        raise ValueError('Infinite or out-of-range feature values are not accepted.')
    if np.any(np.isnan(raw).mean(axis=1) > .2):
        raise ValueError('A row is missing more than 20% of its features. Supply complete feature definitions before prediction.')
    return raw


def read_feature_upload(content: bytes, separator: str, names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError('Upload exceeds the 20 MB limit.')
    try:
        text = content.decode('utf-8-sig')
        header = next(csv.reader(text.splitlines(), delimiter=separator))
        if len(header) != len(set(header)):
            raise ValueError('Duplicate column names are not allowed.')
        frame = pd.read_csv(BytesIO(content), sep=separator, nrows=MAX_BATCH_ROWS + 1)
    except (UnicodeError, StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as exc:
        raise ValueError('Upload must be a valid UTF-8 CSV or TSV with a header.') from exc
    return frame, validate_feature_frame(frame, names)


def distribution_warning(scaled: np.ndarray) -> bool:
    """A heuristic flag, not a calibrated probability of error."""
    return bool(np.any(np.mean(np.abs(scaled) > 6, axis=1) > .1))
