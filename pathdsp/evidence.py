"""Explicit evidence boundaries for the deployed model, not candidate validation."""
import hashlib
from functools import lru_cache
from pathlib import Path
import pandas as pd
from .inference import pair_scope

ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=8)
def _file_digest(path, modified_ns, size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_digest(path):
    stat=path.stat()
    return _file_digest(str(path),stat.st_mtime_ns,stat.st_size)


def prediction_evidence(dataset,drug,cell,shifted=False):
    external=dataset.upper()!='GDSC'
    scope=pair_scope(dataset,drug,cell)
    return {
        'dataset':dataset,'drug':drug,'cell':cell,'model_split':scope,
        'model_sha256':file_digest(ROOT/'artifacts/model.pt'),
        'data_archive_sha256':file_digest(ROOT/'app_data/pathdsp_core.zip'),
        'target_definition':'Supplied −log10(IC50); source unit requires assay verification',
        'concentration_unit_status':'μM conversion conditional; original assay metadata not bundled',
        'validation_scope':'Historical GDSC pair holdout reused during development; CCLE external transfer poor',
        'applicability':'unsupported cross-study transfer' if external else 'feature distribution warning' if shifted else 'repository feature catalogue; individual accuracy not guaranteed',
        'prediction_interval_status':'Unavailable for serving checkpoint: no untouched independent calibration set',
        'patient_health_status':'Unavailable: cell-line data contain no patient records',
        'clinical_use_status':'Research only; not validated for diagnosis, drug selection, dose or duration',
        'pathway_evidence':'Numeric enrichment and SHAP associations; no verified drug-to-pathway prescription map',
        'ranking_scope':'Within-dataset cell-line research comparison; no toxicity or exposure adjustment',
    }


def evidence_table(dataset,drug,cell,shifted=False):
    return pd.DataFrame([{'Evidence':key.replace('_',' ').capitalize(),'Details':value}
                         for key,value in prediction_evidence(dataset,drug,cell,shifted).items()])


def research_tables():
    directory=ROOT/'research/validation-20261030'
    names=('benchmark_summary','within_cell_rankings','fnn_summary','fnn_within_cell_rankings')
    return {'research_'+name:pd.read_csv(directory/(name+'.csv')) for name in names if (directory/(name+'.csv')).is_file()}
