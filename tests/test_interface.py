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
        app.sidebar.toggle[0].set_value(True).run()
        self.assertFalse(app.exception)
        app.sidebar.toggle[0].set_value(False).run()
        app.button[0].click().run()
        self.assertEqual(app.sidebar.radio[0].value, 'Predict')
        for source in ('GDSC', 'CCLE'):
            app.radio[0].set_value(source).run()
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

if __name__ == '__main__':
    unittest.main()
