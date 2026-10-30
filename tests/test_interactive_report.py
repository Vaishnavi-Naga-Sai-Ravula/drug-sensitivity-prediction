import base64
from io import BytesIO
import json
import re
import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathdsp.evaluation import build_evaluation,render_report,evidence_zip
from pathdsp.inference import dataset_resource,model_resource
from pathdsp.interactive import build_charts,gallery_html,evaluation_charts,EVALUATION_CHART_NAMES
from pathdsp.workspace_report import record_output,attach_workspace_outputs,full_table_download

class InteractiveReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bundle=dataset_resource('GDSC');model,metadata,_=model_resource()
        cls.payload=build_evaluation(bundle,model,metadata,bundle.drug_ids[0],bundle.cell_ids[0])

    def test_all_existing_figures_have_real_data_interactive_equivalents(self):
        charts=build_charts(self.payload)
        self.assertFalse(set(self.payload['figures'])-set(charts))
        self.assertEqual(len(charts['saved_holdout_predictions_agreement'].data[0].x),7547)
        self.assertEqual(len(charts['saved_ccle_external_predictions_agreement'].data[0].x),10602)
        self.assertEqual(sum(charts['saved_target_distribution'].data[0].y),37732)
        self.assertEqual(np.asarray(charts['saved_confusion_matrix'].data[0].z).sum(),7547)
        matrix=np.asarray(charts['saved_correlation_heatmap'].data[0].z)
        self.assertEqual(matrix.shape,(21,21))
        np.testing.assert_allclose(np.diag(matrix),1,atol=1e-6)
        self.assertEqual(charts['ic50_concentration_scale'].layout.xaxis.type,'log')

    def test_evaluation_has_exactly_three_requested_graphs_but_report_keeps_all(self):
        visible=evaluation_charts(self.payload)
        self.assertEqual(tuple(visible),EVALUATION_CHART_NAMES)
        self.assertEqual(len(visible),3)
        all_charts=build_charts(self.payload)
        self.assertGreater(len(all_charts),len(visible))
        self.assertIn('local_group_shap',all_charts)
        self.assertIn('ic50_concentration_scale',all_charts)
        for name,count in [('gdsc_agreement_density_bins',7547),('ccle_agreement_density_bins',10602)]:
            self.assertEqual(self.payload['tables'][name]['pair_count'].sum(),count)
        for name in ('gdsc_agreement_density','ccle_agreement_density'):
            self.assertTrue(visible[name].layout.xaxis.fixedrange)
            self.assertEqual(visible[name].data[0].marker.symbol,'hexagon')

    def test_each_displayed_evaluation_table_is_exported_and_inventory_is_complete(self):
        tables=self.payload['tables']
        report=render_report(self.payload)
        self.assertIn(b'Key validation statistics',report)
        self.assertIn(b'Evaluation table coverage',report)
        for row in tables['evaluation_table_inventory'].itertuples():
            self.assertIn(row.Table,tables)
            self.assertEqual(row.Rows,len(tables[row.Table]))
            self.assertIn(('download="'+row.Table+'.csv"').encode(),report)
        self.assertIn(b'Complete Overview section text',report)
        self.assertIn(b'complete interpretation guide',report)
        self.assertIn(b'Complete Methods scientific content',report)

    def test_offline_gallery_has_cursor_coordinates_and_safe_script_data(self):
        malicious='</script><script>alert("hover-test")</script>'
        fig=go.Figure(go.Scatter(x=[1],y=[2],customdata=[malicious]))
        html=gallery_html({'safe':fig})
        self.assertIn('Plotly.newPlot',html)
        self.assertIn('pointermove',html)
        self.assertIn('xaxis.p2d(px)',html)
        self.assertNotIn('<script src=',html)
        self.assertNotIn(malicious,html)
        raw=re.search(r'id="chart-specifications">(.*?)</script>',html,re.S).group(1)
        self.assertEqual(json.loads(raw)[0]['figure']['data'][0]['customdata'][0],malicious)

    def test_embedded_csv_is_complete_even_when_html_preview_is_small(self):
        frame=pd.DataFrame({'value':range(100),'note':['quoted, value\nsecond line']*100})
        html=full_table_download('example',frame)
        csv=base64.b64decode(re.search(r'base64,([^" ]+)',html).group(1))
        pd.testing.assert_frame_equal(pd.read_csv(BytesIO(csv)),frame)
        self.assertIn('Preview: 25 rows',html)

    def test_workspace_capture_is_complete_and_does_not_change_with_input_mutation(self):
        state={};batch=pd.DataFrame({'predicted_neglog10_ic50':[1.,2.]})
        record_output(state,'batch',{'batch_predictions':batch},{'model_sha256':'example'})
        batch.loc[0,'predicted_neglog10_ic50']=999
        attach_workspace_outputs(self.payload,state['workspace_outputs'],(16.5,9))
        name=next(name for name in self.payload['tables'] if name.startswith('workspace_batch_'))
        self.assertEqual(self.payload['tables'][name].iloc[0,0],1)
        html=render_report(self.payload)
        for section in ('Overview','Predict','Evaluation','Explainability','Methods'):
            self.assertIn(section.encode(),html)
        self.assertIn(b'chart-specifications',html)
        self.assertNotIn(b'streamlit run app.py',html)
        self.assertNotIn(b'pip install -r requirements',html)
