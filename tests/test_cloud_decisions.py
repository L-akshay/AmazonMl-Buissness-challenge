import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.cloud_decisions import competition,probabilities,scan_grid,audit
from src.cloud_model import entity_metrics
from src.cloud_store import atomic_json,write_parquet


class DecisionTests(unittest.TestCase):
    def test_secondary_competition_keeps_multiple_targets_and_abstains_on_ties(self):
        b={'ri':np.array([0,1,0,2,3,0]),'tid':np.array([21,21,22,31,31,32]),
           'p':np.array([.9,.8,.7,.9,.9,.6],dtype=np.float32)}
        winner,second=competition(b)
        np.testing.assert_array_equal(winner,[True,False,True,False,False,True])
        self.assertAlmostEqual(second[0],.8)
        b.update(winner=winner,second=second)
        np.testing.assert_allclose(probabilities(b,'exclusive'),[.9,0,.7,0,0,.6])
        np.testing.assert_allclose(probabilities(b,'margin_075'),[.9,0,.7,0,0,.6])
        self.assertLess(probabilities(b,'soft_050')[0],b['p'][0])
        # Labels are never an input to winner selection.
        b['y']=np.array([0,1,0,1,0,0])
        np.testing.assert_array_equal(competition(b)[0],winner)
        b['ri'][1]=0
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            competition(b)

    def test_histograms_match_direct_entity_metrics_with_missing_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'batch.parquet'
            b={'ri':np.array([0,0,1,2,3]),'tid':np.array([1,2,1,3,4]),
               'y':np.array([1,0,0,1,0]),'p':np.array([.9,.5,.8,.6,.7],dtype=np.float32)}
            b['winner'],b['second']=competition(b);write_parquet(path,b)
            ref={'ri':np.arange(5),'fold':np.arange(5),'truth_count':np.array([2,0,2,0,1]),
                 'country_norm':np.array(['us','india','us','india','us'])}
            for mode in ('independent','exclusive','soft_025'):
                grid=scan_grid([path],ref,mode,np.array([.5,.7,.9],dtype=np.float32))
                for row in grid:
                    keep=probabilities(b,mode)>=row['threshold']
                    count=np.bincount(b['ri'][keep],minlength=5)
                    hits=np.bincount(b['ri'][keep & (b['y']>0)],minlength=5)
                    expected=entity_metrics(ref['truth_count'],count,hits,ref['fold']<4)
                    self.assertEqual(row['development']['overall'],expected)
                self.assertEqual(grid[0]['selection']['overall']['macro_f05'],
                                 max(r['selection']['overall']['macro_f05'] for r in grid))

    def test_audit_resume_and_oracle_account_for_unretrieved_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache/cloud'; cache.mkdir(parents=True)
            refs={'ri':np.arange(5,dtype=np.int32),'fold':np.arange(5,dtype=np.int32),
                  'truth_count':np.array([2,1,1,0,1],dtype=np.int32),
                  'country_norm':np.array(['us','india','us','india','us'])}
            write_parquet(cache/'references_train.parquet',refs)
            counts=np.array([1,0,1,0,0]);hits=counts.copy()
            expected=entity_metrics(refs['truth_count'],counts,hits,refs['fold']!=4)
            atomic_json(root/'reports/cloud_model_validation.json',{'comparisons':{'gbdt':{'oof':{'overall':expected}}}})
            for fold in range(4):
                folder=cache/'scores'/f'gbdt_fold_{fold}'
                # Folds 0/1 compete for the same secondary record; fold 1's
                # true secondary was never retrieved, lowering the oracle.
                tid=(2<<32)+(1 if fold<2 else fold)
                batch={'ri':np.array([fold],dtype=np.int32),'tid':np.array([tid],dtype=np.uint64),
                       'y':np.array([int(fold in (0,2))],dtype=np.uint8),
                       'p':np.array([(.9,.5,.8,.1)[fold]],dtype=np.float32)}
                write_parquet(folder/'batch_00000.parquet',batch)
                atomic_json(folder/'complete.json',{'files':['batch_00000.parquet']})
                atomic_json(folder/'config.json',{'fold':fold})
            with patch('src.cloud_decisions.check_headroom'):
                result=audit(root,{})
                self.assertEqual(result['unretrieved_true_links'],2)
                self.assertLess(result['candidate_oracle']['overall']['macro_f05'],1)
                self.assertEqual(result['baseline']['overall'],expected)
                (cache/'decision_audit/report.json').unlink()
                with patch('src.cloud_decisions.competition',side_effect=AssertionError('recomputed competition')):
                    resumed=audit(root,{})
                self.assertEqual(result,resumed)
            # A forged OOF source containing reserved rows cannot enter the audit.
            import shutil
            shutil.rmtree(cache/'decision_audit')
            batch['ri']=np.array([4],dtype=np.int32)
            write_parquet(cache/'scores/gbdt_fold_3/batch_00000.parquet',batch)
            with patch('src.cloud_decisions.check_headroom'),self.assertRaisesRegex(ValueError,'fitted or reserved'):
                audit(root,{})
