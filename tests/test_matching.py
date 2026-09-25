import unittest
import numpy as np
from scipy import sparse
from src.blocking import prune_rows,fused_pairs
from src.features import FEATURE_NAMES,prepare,row_features


class MatchingTests(unittest.TestCase):
    def test_pruning_and_fusion(self):
        matrix=sparse.csr_matrix(np.array([[1,3,2],[0,0,0]],dtype=np.float32))
        pruned=prune_rows(matrix,2)
        self.assertEqual(pruned.nnz,2)
        self.assertAlmostEqual(float(pruned.multiply(pruned).sum()),1,places=5)
        results={"name":sparse.csr_matrix([[.9,.8,0]]),"address":sparse.csr_matrix([[0,.7,.6]]),"token":sparse.csr_matrix([[0,0,.9]])}
        pairs=list(fused_pairs(results,max_candidates=2,min_candidates=1,ratio=.5))
        self.assertEqual(len(pairs),2)
        self.assertEqual({r for q,r,m in pairs},{0,2})

    def test_soft_numeric_conflict_and_multilingual_country(self):
        a=prepare("royal traders","shop 28 main road","france")
        b=prepare("royal traders","shop 82 main rd","france")
        row=row_features(a,b,[.9,.8,.7,1,1,1],{"royal traders":20},{},{})
        features=dict(zip(FEATURE_NAMES,row))
        self.assertEqual(len(row),len(FEATURE_NAMES))
        self.assertEqual(features["name_exact"],1)
        self.assertEqual(features["number_conflict"],1)
        self.assertEqual(features["same_country"],1)
        self.assertGreater(features["address_ratio"],.7)

    def test_empty_evidence_is_not_agreement(self):
        row=row_features(prepare("","",""),prepare("","",""),[0]*6,{},{},{})
        features=dict(zip(FEATURE_NAMES,row))
        for key in ["name_exact","address_exact","number_exact","same_country","name_ratio","address_ratio"]:
            self.assertEqual(features[key],0)



if __name__=="__main__":
    unittest.main()
