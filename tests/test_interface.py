import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from pathdsp.interface import search_options

class SearchTests(unittest.TestCase):
    def test_partial_case_insensitive_and_no_invented_ids(self):
        self.assertEqual(search_options(['Drug-B', 'drug-A', 'Other'], ' DRUG '), ['drug-A', 'Drug-B'])
        self.assertEqual(search_options(['A549'], 'unknown'), [])

    def test_landing_navigation_and_search(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
        app.toggle[0].set_value(True).run()
        self.assertFalse(app.exception)
        app.toggle[0].set_value(False).run()
        app.button[0].click().run()
        self.assertEqual(app.radio[0].value, 'Predict')
        for source in ('GDSC', 'CCLE'):
            app.radio[1].set_value(source).run()
            drug = app.selectbox[0].options[-1]
            cell = app.selectbox[1].options[-1]
            app.text_input[0].set_value(drug.swapcase()).run()
            app.text_input[1].set_value(cell.swapcase()).run()
            self.assertIn(drug, app.selectbox[0].options)
            self.assertIn(cell, app.selectbox[1].options)
            app.selectbox[0].set_value(drug)
            app.selectbox[1].set_value(cell)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Predicted' in metric.label for metric in app.metric))
            app.text_input[0].set_value('no-such-drug-xyz').run()
            self.assertTrue(app.button[0].disabled)
            self.assertFalse(app.exception)
            app.text_input[0].set_value('').run()
            app.text_input[1].set_value('').run()

    def test_serving_mode_does_not_allow_retraining(self):
        from unittest.mock import patch
        with patch.dict('os.environ', {'PATHDSP_ENABLE_TRAINING':'1'}, clear=True):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
            app.radio[0].set_value('Methods & deploy').run()
            training = next(button for button in app.button if 'Train model' in button.label)
            self.assertTrue(training.disabled)
            self.assertFalse(app.exception)

if __name__ == '__main__':
    unittest.main()

class ComparisonTests(unittest.TestCase):
    def test_comparison_persists_and_captures_complete_session_results(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=90).run()
        app.radio[0].set_value('Predict').run()
        choices=app.multiselect[0].options[:2]
        app.multiselect[0].set_value(choices).run()
        next(button for button in app.button if button.label=='Run matched-cell comparison').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['matched_comparison']['response']['results']),2)
        app.run()
        self.assertFalse(app.exception)
        snapshots=app.session_state['workspace_outputs']
        comparisons=[value for value in snapshots.values() if value['section']=='comparison']
        self.assertEqual(len(comparisons),1)
        self.assertEqual(len(comparisons[0]['tables']['drug_comparison']),2)
