"""Describe identity overlap and missing assay provenance without guessing units."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pathdsp.data import build_dataset,normalize_id,CORE_ARCHIVE

root=Path(__file__).resolve().parents[1]
a=build_dataset('GDSC');b=build_dataset('CCLE')
result={'archive_sha256':hashlib.sha256(CORE_ARCHIVE.read_bytes()).hexdigest(),
        'target':'supplied −log10(IC50)',
        'unit_verified':False,'assay_protocol_verified':False,'replicate_identity_available':False,
        'unit_evidence':'Processed response filenames name the transformation; original concentration unit and assay protocols are not bundled. No guessed unit correction applied.',
        'pathway_mapping':'Numeric features, not verified drug indications or causal pathway effects',
        'overlap':{}}
for column in ('drug','cell'):
    left=set(a.meta[column].map(normalize_id));right=set(b.meta[column].map(normalize_id))
    result['overlap'][column]={'gdsc':len(left),'ccle':len(right),'shared_normalized_ids':len(left&right),
        'limitation':'Name matches only; cross-study biological identity not independently verified'}
for bundle in (a,b):
    # Repeated numeric rows are investigated, never silently removed.
    result[bundle.name]={'duplicate_feature_rows':int(__import__('pandas').DataFrame(bundle.X).duplicated().sum()),
                        'constant_feature_columns':int(np.sum(np.nanstd(bundle.X,axis=0)==0)),
                        'target_min':float(bundle.y.min()),'target_max':float(bundle.y.max()),
                        'at_minimum_count':int(np.sum(bundle.y==bundle.y.min())),'at_maximum_count':int(np.sum(bundle.y==bundle.y.max())),
                        'boundary_interpretation':'Repeated boundary values may reflect assay limits; original protocols are needed before censoring adjustments',
                        'action':'Retain valid rows; investigate identical features and assay provenance before external claims'}
(root/'research/provenance_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
