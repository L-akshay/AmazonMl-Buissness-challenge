"""Optional GPU acceleration for deterministic quantized sparse cosine retrieval."""

import os
from pathlib import Path
import numpy as np
from scipy import sparse

SCALE=30000
KERNEL=r'''
extern "C" __global__ void accumulate(
    const int* qp, const int* qj, const int* qv,
    const int* bp, const int* bj, const int* bv,
    const int* offsets,int* candidates,int* scores, const int nref,const int width) {
    int row=blockIdx.x;
    for(int p=qp[row]+blockIdx.y;p<qp[row+1];p+=gridDim.y){
        int feature=qj[p]; int value=qv[p];
        for(int j=bp[feature]+threadIdx.x;j<bp[feature+1];j+=blockDim.x){
            atomicAdd(&scores[(long long)row*nref+bj[j]], value*bv[j]);
            candidates[(long long)row*width+offsets[p]+j-bp[feature]]=bj[j];
        }
    }
}
extern "C" __global__ void gather(
    const int* scores,const int* candidates,const int* counts,
    long long* keys,const int nref,const int width,const int nrows) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)nrows*width) return;
    int row=i/width; int col=i%width;
    if(col>=counts[row]) {keys[i]=-1; return;}
    int ref=candidates[i]; int value=scores[(long long)row*nref+ref];
    keys[i]=(long long)value*(nref+1)+(nref-ref);
}
'''


class GpuSparse:
    def __init__(self,matrices):
        cache=Path(__file__).resolve().parents[1]/"cache"/"cuda_kernels"
        cache.mkdir(exist_ok=True)
        os.environ.setdefault("CUPY_CACHE_DIR",str(cache))
        import cupy as cp
        self.cp=cp
        # WDDM can spill device allocations into host RAM. Keep a hard pool
        # budget instead of allowing differently sized batches to accumulate.
        cp.get_default_memory_pool().set_limit(size=3*1024**3)
        self.kernel=cp.RawKernel(KERNEL,"accumulate")
        self.gather=cp.RawKernel(KERNEL,"gather")
        self.matrices={}
        self.nref={}
        self.frequency={}
        for c,m in matrices.items():
            self.matrices[c]=(cp.asarray(m.indptr,dtype=cp.int32),cp.asarray(m.indices,dtype=cp.int32),cp.asarray(np.rint(m.data*SCALE).astype(np.int32)))
            self.nref[c]=m.shape[1]
            self.frequency[c]=np.diff(m.indptr)

    def search(self,query,channel,k=6,batch_size=128):
        cp=self.cp
        bp,bj,bv=self.matrices[channel]
        nref=self.nref[channel]
        parts=[]
        for start in range(0,query.shape[0],batch_size):
            q=query[start:start+batch_size]
            qp=cp.asarray(q.indptr,dtype=cp.int32)
            qj=cp.asarray(q.indices,dtype=cp.int32)
            qv=cp.asarray(np.rint(q.data*SCALE).astype(np.int32))
            lengths=self.frequency[channel][q.indices]
            offsets=np.empty(len(lengths),dtype=np.int32)
            counts=np.empty(q.shape[0],dtype=np.int32)
            for row in range(q.shape[0]):
                a,b=q.indptr[row:row+2]
                offsets[a:b]=np.r_[0,np.cumsum(lengths[a:b])][:-1]
                counts[row]=lengths[a:b].sum()
            width=max(int(counts.max()),1)
            candidates=cp.empty((q.shape[0],width),dtype=cp.int32)
            keys=cp.empty((q.shape[0],width),dtype=cp.int64)
            gpu_counts=cp.asarray(counts)
            scores=cp.zeros((q.shape[0],nref),dtype=cp.int32)
            self.kernel((q.shape[0],16),(128,),(qp,qj,qv,bp,bj,bv,cp.asarray(offsets),candidates,scores,np.int32(nref),np.int32(width)))
            indices=[]
            values=[]
            rows=cp.arange(q.shape[0])
            for _ in range(k):
                self.gather(((q.shape[0]*width+255)//256,),(256,),(scores,candidates,gpu_counts,keys,np.int32(nref),np.int32(width),np.int32(q.shape[0])))
                best_position=cp.argmax(keys,axis=1)
                best=cp.where(gpu_counts>0,candidates[rows,best_position],0)
                value=cp.where(gpu_counts>0,scores[rows,best],0)
                indices.append(cp.asnumpy(best))
                values.append(cp.asnumpy(value)/(SCALE*SCALE))
                scores[rows,best]=-1
            idx=np.stack(indices,axis=1)
            vals=np.stack(values,axis=1).astype(np.float32)
            valid=vals>.05
            indptr=np.r_[0,np.cumsum(valid.sum(axis=1))].astype(np.int32)
            parts.append(sparse.csr_matrix((vals[valid],idx[valid].astype(np.int32),indptr),shape=(q.shape[0],nref)))
            del scores,candidates,keys,qp,qj,qv,gpu_counts
        return sparse.vstack(parts,format="csr")
