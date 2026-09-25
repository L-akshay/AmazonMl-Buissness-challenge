"""Bounded sparse retrieval against all S1 references, without country filtering."""

from pathlib import Path
import gc
import json
import pickle
from time import perf_counter
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn
from src.data import connect

CHANNELS = {
    "name": {"field": "name_norm", "analyzer": "char_wb", "ngram_range": (3, 4), "keep": 12},
    "address": {"field": "address_norm", "analyzer": "char_wb", "ngram_range": (4, 4), "keep": 16},
    "token": {"field": "combined", "analyzer": "word", "ngram_range": (1, 1), "keep": 12},
}
INDEX_VERSION = 1


def prune_rows(matrix, keep):
    """Retain strongest IDF-weighted terms; deterministic ties use feature index."""
    matrix = matrix.tocsr(copy=True)
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row:row+2]
        if end-start > keep:
            values = matrix.data[start:end]
            # Stable sort keeps lexical vocabulary order when weights tie.
            selected = np.argsort(-values, kind="stable")[:keep]
            mask = np.ones(end-start, dtype=bool)
            mask[selected] = False
            values[mask] = 0
    matrix.eliminate_zeros()
    return normalize(matrix, copy=False)


def texts(records, channel):
    # Records contain (entity_id, normalized name, normalized address, country).
    if channel == "name":
        return [r[1] for r in records]
    if channel == "address":
        return [r[2] for r in records]
    return [r[1]+" "+r[2] for r in records]


def reference_records(root, split):
    db = connect(root)
    result = db.execute(f"SELECT entity_id,name_norm,address_norm,country_norm FROM {split}_source1_norm ORDER BY entity_id").fetchall()
    db.close()
    return result


def build_index(root, split, channels=tuple(CHANNELS)):
    root = Path(root)
    folder = root / "cache" / f"sparse_v{INDEX_VERSION}_{split}"
    folder.mkdir(exist_ok=True)
    if (folder/"reference_ids.json").exists() and all((folder/f"{c}.pkl").exists() and (folder/f"{c}.npz").exists() for c in channels):
        return folder
    records = reference_records(root, split)
    (folder/"reference_ids.json").write_text(json.dumps([r[0] for r in records]),encoding="utf-8")
    for channel in channels:
        vector_path, matrix_path = folder/f"{channel}.pkl", folder/f"{channel}.npz"
        if vector_path.exists() and matrix_path.exists():
            continue
        start = perf_counter()
        config = CHANNELS[channel]
        print(f"Building {split} {channel} index: {len(records):,} references",flush=True)
        vectorizer = TfidfVectorizer(analyzer=config["analyzer"], ngram_range=config["ngram_range"],
            lowercase=False, dtype=np.float32, binary=True, min_df=1, max_df=.005,
            max_features=700_000, norm=None, token_pattern=r"(?u)\b\w+\b")
        matrix = vectorizer.fit_transform(texts(records,channel))
        # Clip fitted IDF, then rebuild weights so indexed and query texts agree.
        vectorizer.idf_ = np.minimum(vectorizer.idf_,8).astype(np.float32)
        del matrix
        gc.collect()
        matrix = vectorizer.transform(texts(records,channel))
        matrix = prune_rows(matrix,config["keep"])
        sparse.save_npz(matrix_path,matrix,compressed=False)
        with vector_path.open("wb") as f:
            pickle.dump(vectorizer,f,protocol=5)
        info={"split":split,"channel":channel,"references":len(records),"features":matrix.shape[1],
              "nnz":matrix.nnz,"seconds":perf_counter()-start,"idf_clip":8,"max_df":.005,"keep":config["keep"]}
        (folder/f"{channel}.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
        print(json.dumps(info),flush=True)
        del matrix,vectorizer
        gc.collect()
    return folder


class Retriever:
    def __init__(self, folder, channels=tuple(CHANNELS), posting_cap=None):
        self.channels=channels
        self.vectorizers={}
        self.matrices={}
        self.reference_matrices={}
        self.allowed={}
        for channel in channels:
            with (Path(folder)/f"{channel}.pkl").open("rb") as f:
                self.vectorizers[channel]=pickle.load(f)
            self.reference_matrices[channel]=sparse.load_npz(Path(folder)/f"{channel}.npz")
            self.matrices[channel]=self.reference_matrices[channel].T.tocsr()
            if posting_cap is not None:
                self.allowed[channel]=np.diff(self.matrices[channel].indptr)<=posting_cap
                matrix=self.reference_matrices[channel]
                matrix.data[~self.allowed[channel][matrix.indices]]=0
                matrix.eliminate_zeros()
                self.reference_matrices[channel]=normalize(matrix,copy=False)
                self.matrices[channel]=self.reference_matrices[channel].T.tocsr()

    def search(self, records, top_k=6):
        result={}
        self.last_timing={}
        for channel in self.channels:
            tick=perf_counter()
            query=self.vectorizers[channel].transform(texts(records,channel))
            query=prune_rows(query,CHANNELS[channel]["keep"])
            if channel in self.allowed:
                query.data[~self.allowed[channel][query.indices]]=0
                query.eliminate_zeros()
                query=normalize(query,copy=False)
            result[channel]=sp_matmul_topn(query,self.matrices[channel],top_n=top_k,
                threshold=.05,sort=True,n_threads=8)
            self.last_timing[channel]=perf_counter()-tick
        return result

    def search_selected(self,records,backend="cpu"):
        """Frozen pilot policy: token top-6 plus name/address rescue when ambiguous."""
        from src.gpu_sparse import SCALE
        if backend=="gpu" and not hasattr(self,"gpu"):
            from src.gpu_sparse import GpuSparse
            self.gpu=GpuSparse(self.matrices)
        def search_channel(items,channel,k):
            q=prune_rows(self.vectorizers[channel].transform(texts(items,channel)),CHANNELS[channel]["keep"])
            if backend=="gpu":
                return self.gpu.search(q,channel,k)
            if not hasattr(self,"integer_matrices"):
                self.integer_matrices={}
            if channel not in self.integer_matrices:
                b=self.matrices[channel].copy()
                b.data=np.rint(b.data*SCALE).astype(np.int32)
                self.integer_matrices[channel]=b
            q.data=np.rint(q.data*SCALE).astype(np.int32)
            m=sp_matmul_topn(q,self.integer_matrices[channel],top_n=k,threshold=int(.05*SCALE*SCALE),sort=True,n_threads=8)
            return m.astype(np.float32)/(SCALE*SCALE)
        token=search_channel(records,"token",6)
        rescue=[]
        for i in range(len(records)):
            lo,hi=token.indptr[i:i+2]
            scores=token.data[lo:hi]
            strong=len(scores)>0 and scores[0]>=.6 and (len(scores)<2 or scores[0]-scores[1]>=.15)
            if not strong:
                rescue.append(i)
        result={"token":token}
        for channel,k in (("name",2),("address",3)):
            counts=np.zeros(len(records),dtype=np.int32)
            if rescue:
                m=search_channel([records[i] for i in rescue],channel,k)
                counts[rescue]=np.diff(m.indptr)
                result[channel]=sparse.csr_matrix((m.data,m.indices,np.r_[0,np.cumsum(counts)].astype(np.int32)),shape=token.shape)
            else:
                result[channel]=sparse.csr_matrix(token.shape,dtype=np.float32)
        return {c:result[c] for c in CHANNELS}
    def search_fast(self,records,top_k=6,probe_terms=3,pool_size=40):
        result={}
        for channel in self.channels:
            query=prune_rows(self.vectorizers[channel].transform(texts(records,channel)),CHANNELS[channel]["keep"])
            probe=query.copy()
            frequency=np.diff(self.matrices[channel].indptr)
            for row in range(probe.shape[0]):
                lo,hi=probe.indptr[row:row+2]
                indices=probe.indices[lo:hi]
                priority=np.where(frequency[indices]>0,frequency[indices],np.iinfo(np.int32).max)
                selected=np.argsort(priority,kind="stable")[:probe_terms]
                mask=np.ones(hi-lo,dtype=bool)
                mask[selected]=False
                probe.data[lo:hi][mask]=0
            probe.eliminate_zeros()
            pool=sp_matmul_topn(probe,self.matrices[channel],top_n=pool_size,threshold=.01,sort=True,n_threads=8)
            qi=np.repeat(np.arange(len(records)),np.diff(pool.indptr))
            full=np.asarray(query[qi].multiply(self.reference_matrices[channel][pool.indices]).sum(axis=1)).ravel()
            out_indices=[]
            out_values=[]
            indptr=[0]
            for row in range(len(records)):
                lo,hi=pool.indptr[row:row+2]
                order=np.argsort(-full[lo:hi],kind="stable")[:top_k]
                order=order[full[lo:hi][order]>.05]
                out_indices.extend(pool.indices[lo:hi][order])
                out_values.extend(full[lo:hi][order])
                indptr.append(len(out_indices))
            result[channel]=sparse.csr_matrix((np.asarray(out_values,dtype=np.float32),np.asarray(out_indices,dtype=np.int32),np.asarray(indptr,dtype=np.int32)),shape=(len(records),pool.shape[1]))
        return result


def fused_pairs(results, max_candidates=8, min_candidates=3, ratio=.65):
    """Union channel hits; retain metadata and adaptively cap per-secondary fanout."""
    channels=list(results)
    size=results[channels[0]].shape[0]
    for qi in range(size):
        hits={}
        for ci,channel in enumerate(channels):
            matrix=results[channel]
            start,end=matrix.indptr[qi:qi+2]
            for rank,p in enumerate(range(start,end),1):
                ri=int(matrix.indices[p])
                item=hits.setdefault(ri,[0.0]*len(channels)+[0]*len(channels))
                item[ci]=float(matrix.data[p])
                item[len(channels)+ci]=rank
        ordered=sorted(hits.items(),key=lambda x:(-max(x[1][:len(channels)]),-sum(v>0 for v in x[1][:len(channels)]),x[0]))
        best=max(ordered[0][1][:len(channels)]) if ordered else 0
        for position,(ri,values) in enumerate(ordered):
            if position>=max_candidates:
                break
            if position>=min_candidates and max(values[:len(channels)])<best*ratio:
                break
            yield qi,ri,values


def selected_pairs(results):
    # The channel budgets and rescue gate perform the only pruning. Every pair
    # in this union is passed to the classifier and included in the output.
    return fused_pairs(results,max_candidates=11,min_candidates=11,ratio=0)
