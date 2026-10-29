import math
import unittest
from pathdsp.ic50 import concentration_from_target, concentration_comparison, clinical_availability, scale_table, interpretation_data
from pathdsp.inference import dataset_resource

class IC50Tests(unittest.TestCase):
    def test_log_and_concentration_scales(self):
        self.assertAlmostEqual(concentration_from_target(-math.log10(16.5)),16.5)
        self.assertAlmostEqual(concentration_from_target(-1),10)
        self.assertAlmostEqual(concentration_from_target(0),1)
        self.assertAlmostEqual(concentration_from_target(1),.1)
        comparison=concentration_comparison(16.5,9)
        self.assertAlmostEqual(comparison['reference_divided_by_comparison'],16.5/9)
        self.assertAlmostEqual(comparison['target_change_comparison_minus_reference'],math.log10(16.5/9))
        self.assertIn('no inference about patient',comparison['interpretation'])

    def test_invalid_concentrations_fail_and_clinical_fields_are_not_invented(self):
        for value in (0,-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):concentration_comparison(value,9)
        self.assertTrue(clinical_availability()['Status'].isin(['Unavailable','Not determined from IC50','Not supplied']).all())
        self.assertEqual(len(scale_table()),8)

    def test_same_drug_context_and_unmeasured_observation(self):
        bundle=dataset_resource('GDSC')
        tables,figures=interpretation_data(bundle,bundle.drug_ids[0],-1.2,None)
        self.assertNotIn('ic50_prediction_observation_comparison',tables)
        self.assertIn('ic50_concentration_scale',figures)
        self.assertIn('clinical_dose_duration_cost_availability',tables)
        self.assertGreater(len(figures['ic50_concentration_scale']['png']),1000)
