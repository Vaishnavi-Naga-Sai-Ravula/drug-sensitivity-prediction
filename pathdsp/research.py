"""Research-only splits, calibration and ranking metrics; never serves candidates."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


def dataset_fingerprint(bundle):
    digest = hashlib.sha256()
    digest.update(bundle.meta[['drug', 'cell']].to_csv(index=False, lineterminator='\n').encode())
    digest.update(np.asarray(bundle.y, dtype='<f4').tobytes())
    digest.update(np.asarray(bundle.X, dtype='<f4').tobytes())
    digest.update('\n'.join(bundle.feature_names).encode())
    return digest.hexdigest()


def partition(meta, mode, seed=20261030):
    """Fit/tune/calibration/test; group identity never crosses any role."""
    if mode not in ('pair', 'drug', 'cell', 'both'):
        raise ValueError('Unknown split mode')
    roles = ('fit', 'tune', 'calibration', 'test')
    def allocate(values, split_seed):
        fit, remainder = train_test_split(np.asarray(values), test_size=.4, random_state=split_seed)
        tune, remainder = train_test_split(remainder, test_size=.75, random_state=split_seed + 1)
        calibration, test = train_test_split(remainder, test_size=2/3, random_state=split_seed + 2)
        return dict(zip(roles, (fit, tune, calibration, test)))
    if mode == 'pair':
        return allocate(np.arange(len(meta)), seed)
    identifiers = {}
    for column in ('drug', 'cell'):
        identifiers[column] = allocate(sorted(meta[column].astype(str).unique()), seed + (column == 'cell'))
    result = {}
    for role in roles:
        mask = np.ones(len(meta), dtype=bool)
        for column in ('drug', 'cell') if mode == 'both' else (mode,):
            mask &= meta[column].astype(str).isin(identifiers[column][role]).to_numpy()
        result[role] = np.flatnonzero(mask)
    return result


def conformal_radius(actual, predicted, alpha=.1):
    errors = np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
    if not 0 < alpha < 1 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError('Finite calibration data and 0 < alpha < 1 required')
    rank = int(np.ceil((len(errors) + 1) * (1 - alpha)))
    if rank > len(errors):
        raise ValueError('Calibration set too small for requested coverage')
    return float(np.sort(errors)[rank - 1])


def ranking_metrics(meta, actual, predicted, k=5):
    frame = meta[['drug', 'cell']].copy().reset_index(drop=True)
    frame['actual'] = np.asarray(actual)
    frame['predicted'] = np.asarray(predicted)
    rows = []
    for cell, group in frame.groupby('cell', sort=True):
        if len(group) < max(k, 3):
            continue
        observed = set(group.nlargest(k, 'actual').drug)
        selected = set(group.nlargest(k, 'predicted').drug)
        rho = spearmanr(group.actual, group.predicted).statistic if group.actual.nunique() > 1 and group.predicted.nunique() > 1 else None
        rows.append({'cell':cell, 'measured_drugs':len(group), 'top_k':k,
                     'top_k_overlap_fraction':len(observed & selected)/k,
                     'within_cell_spearman':None if rho is None or not np.isfinite(rho) else float(rho)})
    return pd.DataFrame(rows, columns=['cell','measured_drugs','top_k','top_k_overlap_fraction','within_cell_spearman'])
