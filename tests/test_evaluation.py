from io import BytesIO
import json
import unittest
import zipfile
import numpy as np
import pandas as pd
from pathdsp.evaluation import build_evaluation, sample_features, render_report, evidence_zip, add_holdout_view
from pathdsp.inference import dataset_resource, model_resource, infer_features
from pathdsp.validation import read_feature_upload

class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=dataset_resource('GDSC')
        cls.model,cls.metadata,_=model_resource()
        cls.drug,cls.cell=cls.bundle.drug_ids[0],cls.bundle.cell_ids[0]
        cls.payload=build_evaluation(cls.bundle,cls.model,cls.metadata,cls.drug,cls.cell)

    def test_sample_csv_and_tsv_are_populated_and_uploadable(self):
        sample=sample_features(self.bundle)
        for separator in (',','\t'):
            frame,raw=read_feature_upload(sample.to_csv(index=False,sep=separator).encode(),separator,self.bundle.feature_names)
            self.assertEqual(len(frame),5)
            prediction,_,_=infer_features(raw,self.model,self.metadata)
            self.assertTrue(np.isfinite(prediction).all())
            self.assertEqual(raw.shape,(5,len(self.metadata['feature_names'])))

    def test_ranking_is_for_one_cell_and_predictions_are_independent(self):
        table=self.payload['tables']['cell_line_drug_rankings']
        self.assertEqual(len(table),len(self.bundle.drug_ids))
        self.assertEqual(table['cell'].unique().tolist(),[self.cell])
        self.assertTrue(table['predicted_ic50_uM_assuming_source_unit'].is_monotonic_increasing)
        selected=self.payload['tables']['selected_pair'].iloc[0]
        expected,_,_=infer_features(self.bundle.pair_features(self.drug,self.cell).reshape(1,-1),self.model,self.metadata)
        self.assertAlmostEqual(selected['predicted_neglog10_ic50'],float(expected[0]),places=5)
        self.assertTrue(table['patient_health_status'].str.contains('Unavailable').all())
        self.assertLess(abs(self.payload['context']['group_shap_additivity_residual']),1e-4)
        self.assertFalse(self.payload['tables']['local_pathway_shap'].empty)
        self.assertFalse(self.payload['tables']['local_pathway_shap']['feature'].str.startswith('CHEM__').any())

    def test_unmeasured_pairs_keep_missing_observations(self):
        from unittest.mock import patch
        with patch.object(type(self.bundle),'observed_response',return_value=None):
            payload=build_evaluation(self.bundle,self.model,self.metadata,self.drug,self.cell)
        rankings=payload['tables']['cell_line_drug_rankings']
        self.assertTrue(rankings['observed_ic50_uM_assuming_source_unit'].isna().all())
        self.assertTrue(rankings['absolute_log_error'].isna().all())
        self.assertTrue(np.isfinite(rankings['predicted_ic50_uM_assuming_source_unit']).all())
        self.assertIn(b'Unavailable',render_report(payload))

    def test_report_and_zip_include_every_table_row_and_figure(self):
        view=self.payload['tables']['saved_holdout_predictions']['drug'].iloc[0]
        add_holdout_view(self.payload,view)
        self.assertEqual(self.payload['context']['interactive_holdout_filter'],view)
        html=render_report(self.payload)
        self.assertIn(b'Patient health status',html)
        self.assertIn(b'not a clinical recommendation',html)
        self.assertIn(b'data:image/png;base64,',html)
        self.assertIn(b'local_pathway_shap',html)
        with zipfile.ZipFile(BytesIO(evidence_zip(self.payload,html))) as archive:
            self.assertEqual(archive.read('evaluation_report.html'),html)
            context=json.loads(archive.read('context.json'))
            self.assertEqual(context['cell'],self.cell)
            for name,frame in self.payload['tables'].items():
                exported=pd.read_csv(BytesIO(archive.read('tables/'+name+'.csv')))
                self.assertEqual(len(exported),len(frame),name)
                self.assertEqual(list(exported.columns),list(frame.columns),name)
            for name,figure in self.payload['figures'].items():
                self.assertEqual(archive.read('figures/'+name+'.png'),figure['png'])
