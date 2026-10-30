import unittest
import numpy as np
import pandas as pd
from pathdsp.research import partition,conformal_radius,ranking_metrics

class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.meta=pd.DataFrame([(f'd{i}',f'c{j}') for i in range(40) for j in range(50)],columns=['drug','cell'])

    def test_split_roles_and_unseen_entities_are_disjoint_and_reproducible(self):
        for mode in ('pair','drug','cell','both'):
            parts=partition(self.meta,mode)
            again=partition(self.meta,mode)
            for role,index in parts.items():
                np.testing.assert_array_equal(index,again[role]);self.assertGreater(len(index),0)
                for other,index2 in parts.items():
                    if role==other:continue
                    self.assertFalse(set(index)&set(index2))
                    for entity in ('drug','cell') if mode=='both' else (mode,) if mode!='pair' else ():
                        self.assertFalse(set(self.meta.iloc[index][entity]) & set(self.meta.iloc[index2][entity]))

    def test_conformal_uses_finite_sample_rank_and_rejects_invalid_data(self):
        self.assertEqual(conformal_radius(np.arange(1,10),np.zeros(9)),9)
        for actual,predicted in (([],[]),([np.nan],[0]),([1],[0])):
            with self.assertRaises(ValueError):conformal_radius(actual,predicted)

    def test_rankings_compare_only_within_cell_and_high_target_is_sensitive(self):
        meta=pd.DataFrame({'drug':list('abcde')*2,'cell':['x']*5+['y']*5})
        actual=np.tile(np.arange(5),2)
        values=ranking_metrics(meta,actual,actual,k=2)
        self.assertEqual(list(values.top_k_overlap_fraction),[1,1])
        inverse=ranking_metrics(meta,actual,-actual,k=2)
        self.assertEqual(list(inverse.top_k_overlap_fraction),[0,0])

class RecordedResearchTests(unittest.TestCase):
    def test_recorded_data_hashes_and_split_identity(self):
        import hashlib,json
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]/'research/validation-20261030'
        for filename in ('protocol.json','fnn_protocol.json'):
            protocol=json.loads((root/filename).read_text())
            for name,digest in protocol['files'].items():
                self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),digest,name)
        summary=pd.read_csv(root/'benchmark_summary.csv')
        self.assertEqual(len(summary),20)
        for mode,rows in summary.groupby('split'):
            self.assertEqual(rows.loc[rows.tune_rmse.idxmin(),'selected_on_tune'],True)
            membership=pd.read_csv(root/f'{mode}_membership.csv')
            if mode!='pair':
                for entity in ('drug','cell') if mode=='both' else (mode,):
                    used=membership[membership.role!='excluded_cross_role']
                    self.assertEqual(used.groupby(entity).role.nunique().max(),1)
        fnn=pd.read_csv(root/'fnn_summary.csv')
        self.assertEqual(set(fnn.split),{'pair','drug','cell','both'})
