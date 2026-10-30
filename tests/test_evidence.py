import unittest
from pathdsp.evidence import prediction_evidence

class EvidenceTests(unittest.TestCase):
    def test_scope_and_unavailable_uncertainty_cannot_be_mistaken_for_clinical_validation(self):
        evidence=prediction_evidence('CCLE','x','y')
        self.assertEqual(evidence['applicability'],'unsupported cross-study transfer')
        self.assertIn('Unavailable',evidence['prediction_interval_status'])
        self.assertIn('not validated',evidence['clinical_use_status'])
        self.assertEqual(len(evidence['model_sha256']),64)
        self.assertIn('requires assay verification',evidence['target_definition'])

    def test_comparison_keeps_observations_and_assumed_units_separate(self):
        import pandas as pd
        from pathdsp.comparison_ui import comparison_chart
        frame=pd.DataFrame({'drug':['a','b'],'predicted_ic50_uM_assuming_source_unit':[10.,20.],
                            'observed_neglog10_ic50':[-2.,None]})
        chart=comparison_chart(frame)
        self.assertEqual(list(chart.data[0].y),[10.,20.])
        self.assertEqual(list(chart.data[1].y),[100.])
        self.assertEqual(list(chart.data[1].x),['a'])
