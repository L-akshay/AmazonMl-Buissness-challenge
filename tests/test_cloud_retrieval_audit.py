import unittest
import numpy as np
from scipy import sparse
from src.cloud_retrieval_audit import _raw_group,_finish,trim_top_k


class RetrievalDepthAuditTests(unittest.TestCase):
    def test_deeper_top_k_adds_owner_without_truncating_union(self):
        data=np.linspace(.99,.10,20,dtype=np.float32)
        matrices=[]
        for offset in (0,20,40):
            matrices.append(sparse.csr_matrix((data,np.arange(offset,offset+20,dtype=np.int32),
                np.array([0,20],dtype=np.int32)),shape=(1,60)))
        trimmed=trim_top_k(matrices[0],6)
        self.assertEqual(trimmed.nnz,6)
        self.assertEqual(trimmed.indices.tolist(),list(range(6)))
        found={k:{channel:trim_top_k(matrix,k) for channel,matrix in zip(
            ('name','address','token'),matrices)} for k in (6,12,20)}
        raw=_raw_group([('S2-1','name','address','us')],np.array([59],dtype=np.int32),found,(6,12,20))
        result=_finish(raw)['overall']
        self.assertEqual(result['6']['recalled_links'],0)
        self.assertEqual(result['12']['recalled_links'],0)
        self.assertEqual(result['20']['recalled_links'],1)
        self.assertAlmostEqual(result['20']['link_recall_delta_from_top6'],1.0)
        self.assertEqual(result['6']['candidate_pairs'],18)
        self.assertEqual(result['12']['candidate_pairs'],36)
        self.assertEqual(result['20']['candidate_pairs'],60)


if __name__=='__main__':
    unittest.main()
