"""Generic lexical, numeric, frequency and retrieval evidence for a pair."""

from collections import Counter
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import math
import os
import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler,Levenshtein
from src.normalize import SUFFIXES,STREETS,accent_fold

FEATURE_NAMES = [
    "name_retrieval","address_retrieval","token_retrieval","name_rank","address_rank","token_rank",
    "channel_support","rrf","best_retrieval","rank_spread",
    "name_exact","core_exact","name_ratio","core_ratio","name_token_sort","name_token_set",
    "name_jaro","name_levenshtein","name_jaccard","name_containment","name_length_ratio","name_token_count_ratio",
    "acronym_agreement","address_exact","address_ratio","address_token_sort","address_token_set",
    "address_jaccard","address_containment","address_length_ratio","number_jaccard","number_exact",
    "first_number_equal","number_conflict","one_number_missing","postal_equal","postal_conflict",
    "same_country","different_country","country_missing","name_missing","address_missing",
    "log_name_frequency","log_address_frequency","shared_rare_tokens","max_shared_idf","sum_shared_idf","weighted_jaccard",
    "strong_name_number_conflict","weak_name_strong_address","name_address_product",
]


@lru_cache(maxsize=100000)
def prepare(name,address,country):
    nt=name.split()
    core=list(nt)
    while len(core)>1 and core[-1] in SUFFIXES:
        core.pop()
    core=accent_fold(" ".join(core))
    address=" ".join(STREETS.get(t,t) for t in address.split())
    numbers=tuple(re.findall(r"\d+",address))
    postal=frozenset(n for n in numbers if len(n) in (5,6))
    return (name,address,country,core,frozenset(nt),frozenset(address.split()),
            frozenset(numbers),numbers,postal,"".join(t[0] for t in core.split() if t))


def overlap(a,b):
    shared=len(a & b)
    return shared/max(len(a|b),1),shared/max(min(len(a),len(b)),1)


def length_ratio(a,b):
    return min(len(a),len(b))/max(len(a),len(b),1)


def row_features(a,b,meta,name_frequency,address_frequency,idf):
    an,aa,ac,an_core,ant,aat,ann,ano,ap,acr=a
    bn,ba,bc,bn_core,bnt,bat,bnn,bno,bp,bcr=b
    nj,nc=overlap(ant,bnt)
    aj,atc=overlap(aat,bat)
    numj,_=overlap(ann,bnn)
    nr=fuzz.ratio(an,bn)/100 if an and bn else 0
    ar=fuzz.ratio(aa,ba)/100 if aa and ba else 0
    conflict=float(bool(ann and bnn and not ann & bnn))
    left,right=ant|aat,bnt|bat
    shared=left&right
    weights=[float(idf.get(t,1.0)) for t in shared]
    shared_weight=sum(weights)
    union_weight=sum(float(idf.get(t,1.0)) for t in left|right)
    scores=meta[:3]
    ranks=[v for v in meta[3:] if v>0]
    values=list(meta)+[
        sum(v>0 for v in scores),sum(1/(60+r) for r in ranks),max(scores),max(ranks)-min(ranks) if ranks else 0,
        float(bool(an) and an==bn),float(bool(an_core) and an_core==bn_core),nr,
        fuzz.ratio(an_core,bn_core)/100 if an_core and bn_core else 0,
        fuzz.token_sort_ratio(an,bn)/100 if an and bn else 0,
        fuzz.token_set_ratio(an,bn)/100 if an and bn else 0,
        JaroWinkler.normalized_similarity(an,bn) if an and bn else 0,
        Levenshtein.normalized_similarity(an,bn) if an and bn else 0,
        nj,nc,length_ratio(an,bn),length_ratio(ant,bnt),
        float(bool(acr and bcr) and (acr==bn_core.replace(" ","") or bcr==an_core.replace(" ",""))),
        float(bool(aa) and aa==ba),ar,
        fuzz.token_sort_ratio(aa,ba)/100 if aa and ba else 0,
        fuzz.token_set_ratio(aa,ba)/100 if aa and ba else 0,
        aj,atc,length_ratio(aa,ba),numj,float(bool(ann) and ann==bnn),
        float(bool(ano and bno) and ano[0]==bno[0]),conflict,float(bool(ann)!=bool(bnn)),
        float(bool(ap&bp)),float(bool(ap and bp and not ap&bp)),
        float(bool(ac and bc) and ac==bc),float(bool(ac and bc) and ac!=bc),float(not ac or not bc),
        float(not an or not bn),float(not aa or not ba),
        math.log1p(name_frequency.get(an,0)),math.log1p(address_frequency.get(aa,0)),
        sum(w>=5 for w in weights),max(weights,default=0),shared_weight,shared_weight/max(union_weight,1),
        nr*conflict,(1-nr)*ar,nr*ar,
    ]
    return values


class FeatureBuilder:
    def __init__(self, references, token_vectorizer):
        self.references=references
        self.name_frequency=Counter(r[1] for r in references)
        self.address_frequency=Counter(" ".join(STREETS.get(t,t) for t in r[2].split()) for r in references if r[2])
        self.idf={t:float(token_vectorizer.idf_[i]) for t,i in token_vectorizer.vocabulary_.items()}

    def transform(self,records,qidx,ridx,metadata):
        result=np.empty((len(qidx),len(FEATURE_NAMES)),dtype=np.float32)
        for i,(qi,ri,meta) in enumerate(zip(qidx,ridx,metadata)):
            r=self.references[int(ri)]
            q=records[int(qi)]
            result[i]=row_features(prepare(*r[1:4]),prepare(*q[1:4]),meta,
                self.name_frequency,self.address_frequency,self.idf)
        return result


_WORKER_IDF={}


def _initialize_feature_worker(idf):
    global _WORKER_IDF
    _WORKER_IDF=idf


def _feature_chunk(payload):
    references,records,qi,ri,metadata,name_frequency,address_frequency=payload
    builder=FeatureBuilder.__new__(FeatureBuilder)
    builder.references=references
    builder.name_frequency=name_frequency
    builder.address_frequency=address_frequency
    builder.idf=_WORKER_IDF
    return builder.transform(records,qi,ri,metadata)


class ParallelFeatures:
    """Workers receive only the records needed by a small pair batch.

    The complete reference corpus and its frequency counters stay in the parent;
    Windows workers never duplicate the multi-million-record reference list.
    """
    def __init__(self,builder,workers=None,chunk_size=2000):
        self.builder=builder
        self.workers=workers if workers is not None else int(os.environ.get("ENTITY_FEATURE_WORKERS","4"))
        if self.workers<1:
            raise ValueError("Feature worker count must be positive")
        self.chunk_size=chunk_size
        self.pool=ProcessPoolExecutor(max_workers=self.workers,initializer=_initialize_feature_worker,
                                      initargs=(builder.idf,)) if self.workers>1 else None

    def transform(self,records,qi,ri,metadata):
        if self.pool is None or not len(qi):
            return self.builder.transform(records,qi,ri,metadata)
        futures=[]
        for start in range(0,len(qi),self.chunk_size):
            end=start+self.chunk_size
            unique_r,local_r=np.unique(ri[start:end],return_inverse=True)
            unique_q,local_q=np.unique(qi[start:end],return_inverse=True)
            references=[self.builder.references[int(i)] for i in unique_r]
            names={r[1]:self.builder.name_frequency[r[1]] for r in references}
            addresses={" ".join(STREETS.get(t,t) for t in r[2].split()) for r in references}
            frequencies={a:self.builder.address_frequency.get(a,0) for a in addresses}
            payload=(references,[records[int(i)] for i in unique_q],local_q,local_r,metadata[start:end],names,frequencies)
            futures.append(self.pool.submit(_feature_chunk,payload))
        return np.vstack([future.result() for future in futures])

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()
            self.pool=None
