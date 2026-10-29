"""Exploratory GDSC pair-holdout comparison; never replaces deployment weights."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from pathdsp.data import build_dataset

ROOT = Path(__file__).resolve().parents[1]

def compare():
    bundle = build_dataset('GDSC')
    train, test = train_test_split(np.arange(len(bundle.y)), test_size=.2, random_state=42)
    previous = pd.read_csv(ROOT / 'outputs/holdout_predictions.csv')
    if not np.allclose(previous['actual_neglog10_ic50'], bundle.y[test]):
        raise ValueError('Saved baseline is not from this pair holdout. Re-establish a common baseline first.')
    for column in ('drug', 'cell'):
        if list(previous[column].astype(str)) != list(bundle.meta.iloc[test][column].astype(str)):
            raise ValueError('Saved baseline pair identities do not match this split.')
    imputer = SimpleImputer(strategy='median')
    x, xt = imputer.fit_transform(bundle.X[train]), imputer.transform(bundle.X[test])
    baseline = json.loads((ROOT / 'outputs/metrics.json').read_text())['holdout']
    report = {'scope': 'Exploratory model selection on the existing GDSC seed-42 80/20 pair holdout; not an independent final validation. No CCLE adaptation or deployment weight change.',
              'seed':42, 'split':'pair', 'n_train':len(train), 'n_test':len(test),
              'baseline_fnn': {'mae':baseline['mae_neglog10_ic50'], 'rmse':baseline['rmse_neglog10_ic50'], 'r2':baseline['r2']}, 'candidates':{}}
    models = [
        ('hist_gradient_boosting', HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=31, l2_regularization=5, random_state=42)),
        ('extra_trees', ExtraTreesRegressor(n_estimators=128, min_samples_leaf=2, max_features=.6, n_jobs=4, random_state=42)),
        ('hist_gradient_boosting_tuned', HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=63, l2_regularization=1, learning_rate=.06, random_state=42))]
    with threadpool_limits(limits=4):
        for name, model in models:
            started = time.perf_counter()
            model.fit(x, bundle.y[train])
            prediction = model.predict(xt)
            result = {'mae':float(mean_absolute_error(bundle.y[test], prediction)), 'rmse':float(root_mean_squared_error(bundle.y[test], prediction)), 'r2':float(r2_score(bundle.y[test], prediction)), 'seconds':time.perf_counter()-started}
            report['candidates'][name] = result
            print(name, result, flush=True)
    report['decision'] = 'Retain FNN; no candidate beats its saved MAE and RMSE.' if all(r['rmse'] >= report['baseline_fnn']['rmse'] for r in report['candidates'].values()) else 'A candidate warrants fresh independent validation; no automatic deployment.'
    return report

if __name__ == '__main__':
    report = compare()
    (ROOT / 'outputs/model_comparison.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
