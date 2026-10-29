import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from pathdsp.inference import model_resource,dataset_resource,infer_features,pair_scope

class ReleaseConsistencyTests(unittest.TestCase):
    def test_served_checkpoint_matches_saved_predictions_and_training_audit(self):
        root=Path(__file__).resolve().parents[1]
        model,metadata,version=model_resource();bundle=dataset_resource('GDSC')
        audit=json.loads((root/'outputs/mse_training_comparison.json').read_text())
        self.assertEqual(version,audit['release_model_sha256'][:16])
        table=pd.read_csv(root/'outputs/holdout_predictions.csv')
        chosen=table.iloc[np.linspace(0,len(table)-1,16,dtype=int)]
        raw=np.stack([bundle.pair_features(row.drug,row.cell) for row in chosen.itertuples()])
        prediction,_,_=infer_features(raw,model,metadata)
        np.testing.assert_allclose(prediction,chosen['predicted_neglog10_ic50'],atol=1e-5)
        example=audit['example']
        prediction,_,_=infer_features(bundle.pair_features(example['drug'],example['cell']).reshape(1,-1),model,metadata)
        self.assertAlmostEqual(float(prediction[0]),example['predicted'],places=5)
        self.assertEqual(pair_scope('GDSC',example['drug'],example['cell']),example['scope'])
