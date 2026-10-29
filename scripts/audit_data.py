"""Audit the bundled processed tables without changing rows or inventing labels."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from pathdsp.data import DATA_FILES, _open_table, build_dataset


def audit():
    result = {'source': 'repository bundled processed tables; external ZIP archives are not required', 'datasets': {}}
    for name, files in DATA_FILES.items():
        tables = {}
        for role, filename in files.items():
            with _open_table(filename) as stream:
                frame = pd.read_csv(stream, sep='\t', index_col=0, low_memory=False)
            numeric = frame.apply(pd.to_numeric, errors='coerce')
            values = numeric.to_numpy(dtype=float)
            tables[role] = {'rows': len(frame), 'columns': len(frame.columns),
                'duplicate_normalized_ids': int(pd.Index(frame.index.astype(str).str.strip().str.casefold()).duplicated().sum()),
                'duplicate_columns': int(frame.columns.duplicated().sum()),
                'nonnumeric_nonblank_cells': int((frame.notna() & numeric.isna()).to_numpy().sum()),
                'missing_cells': int(np.isnan(values).sum()), 'infinite_cells': int(np.isinf(values).sum()),
                'min_finite': float(values[np.isfinite(values)].min()), 'max_finite': float(values[np.isfinite(values)].max())}
        bundle = build_dataset(name, include_examples=False)
        result['datasets'][name] = {'tables': tables, 'usable_response_pairs': len(bundle.y),
            'duplicate_pairs': int(bundle.meta.duplicated(['drug', 'cell']).sum()),
            'usable_drugs': len(bundle.drug_ids), 'usable_cells': len(bundle.cell_ids),
            'features': len(bundle.feature_names),
            'response_pairs_missing_required_features': int(tables['response']['rows'] * tables['response']['columns'] - tables['response']['missing_cells'] - len(bundle.y))}
    return result

if __name__ == '__main__':
    result = audit()
    path = Path(__file__).resolve().parents[1] / 'outputs/data_quality_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for name, dataset in result['datasets'].items():
        print(name, 'usable pairs:', dataset['usable_response_pairs'], 'excluded for missing features:', dataset['response_pairs_missing_required_features'])
        for role, table in dataset['tables'].items():
            print(role, {key: table[key] for key in ('duplicate_normalized_ids', 'nonnumeric_nonblank_cells', 'missing_cells', 'infinite_cells')})
