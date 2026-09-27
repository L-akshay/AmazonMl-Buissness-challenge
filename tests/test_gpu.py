import os
import unittest
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from src.gpu_sparse import GpuSparse,SCALE


@unittest.skipUnless(os.environ.get("RUN_GPU_TESTS")=="1","Optional CUDA verification; set RUN_GPU_TESTS=1")
class GpuTests(unittest.TestCase):
    def test_matches_dense_integer_oracle_including_empty_queries(self):
        rng=np.random.default_rng(42)
        b=normalize(sparse.csr_matrix(rng.integers(0,3,size=(30,20)).astype(np.float32)))
        q=normalize(sparse.csr_matrix(rng.integers(0,3,size=(5,20)).astype(np.float32)))
        q.data[q.indptr[0]:q.indptr[1]]=0
        q.eliminate_zeros()
        backend=GpuSparse({"test":b.T.tocsr()})
        result=backend.search(q,"test",k=6)
        expected=np.rint(q.toarray()*SCALE).astype(np.int64) @ np.rint(b.toarray()*SCALE).astype(np.int64).T
        for row in range(q.shape[0]):
            order=np.argsort(-expected[row],kind="stable")[:6]
            order=order[expected[row,order]>.05*SCALE*SCALE]
            lo,hi=result.indptr[row:row+2]
            np.testing.assert_array_equal(result.indices[lo:hi],order)
            np.testing.assert_allclose(result.data[lo:hi],expected[row,order]/(SCALE*SCALE),atol=1e-7)
