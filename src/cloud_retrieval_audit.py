"""Measure recall gains from expanding cached sparse retrieval depth.

This is an exact, label-aware diagnostic over every training query and the
complete S1 index. It writes aggregate metrics only; it never alters candidate
caches or submissions. It compares deeper retrieval before any model refit.
"""

import json
from pathlib import Path
import numpy as np
from scipy import sparse
from src.blocking import CHANNELS, Retriever, fused_pairs, prune_rows, texts
from src.cloud_store import atomic_json, check_headroom, claim_config, digest
from src.data import connect
from src.gpu_sparse import SCALE
from src.streaming import label_lookup, secondary_batches

TOP_KS=(6,12,20)
BATCHES_PER_CHECKPOINT=10


def trim_top_k(matrix,k):
    """Keep the first k score-sorted results in every sparse query row."""
    counts=np.minimum(np.diff(matrix.indptr),k).astype(np.int64)
    pointers=np.r_[0,np.cumsum(counts)].astype(np.int32)
    indices=np.empty(pointers[-1],dtype=matrix.indices.dtype)
    values=np.empty(pointers[-1],dtype=matrix.data.dtype)
    for row,(start,end) in enumerate(zip(matrix.indptr[:-1],matrix.indptr[1:])):
        count=counts[row]; target=pointers[row]
        indices[target:target+count]=matrix.indices[start:start+count]
        values[target:target+count]=matrix.data[start:start+count]
    return sparse.csr_matrix((values,indices,pointers),shape=matrix.shape)


def _raw_group(records,owners,found_by_k,ks):
    groups={'overall':np.ones(len(records),dtype=bool)}
    groups.update({str(country):np.fromiter((r[3]==country for r in records),bool,len(records))
                   for country in sorted({r[3] for r in records})})
    raw={name:{} for name in groups}
    outcomes={}
    for k in ks:
        channels=found_by_k[k]
        counts=np.zeros(len(records),dtype=np.int32)
        hits=np.zeros(len(records),dtype=np.uint8)
        for qi,ri,_ in fused_pairs(channels,max_candidates=len(channels)*k,
                                   min_candidates=len(channels)*k,ratio=0):
            counts[qi]+=1
            if owners[qi]>=0 and ri==owners[qi]:
                hits[qi]=1
        outcomes[k]=(counts,hits)
    for name,mask in groups.items():
        positive=(owners>=0)&mask
        for k in ks:
            counts,hits=outcomes[k]
            histogram=np.bincount(counts[mask],minlength=len(found_by_k[k])*k+1)
            raw[name][str(k)]={'queries':int(mask.sum()),'labeled_links':int(positive.sum()),
                'recalled_links':int(hits[positive].sum()),'candidate_pairs':int(counts[mask].sum()),
                'fanout_histogram':histogram.tolist()}
    return raw


def _merge(left,right):
    for group,depths in right.items():
        target=left.setdefault(group,{})
        for depth,values in depths.items():
            current=target.setdefault(depth,{'queries':0,'labeled_links':0,'recalled_links':0,
                                              'candidate_pairs':0,'fanout_histogram':[0]*len(values['fanout_histogram'])})
            if len(current['fanout_histogram'])!=len(values['fanout_histogram']):
                raise ValueError('Retrieval fanout histogram size changed')
            for key in ('queries','labeled_links','recalled_links','candidate_pairs'):
                current[key]+=values[key]
            current['fanout_histogram']=[a+b for a,b in zip(current['fanout_histogram'],values['fanout_histogram'])]
    return left


def _finish(raw):
    result={}
    for group,depths in raw.items():
        result[group]={}
        baseline=None
        for depth in TOP_KS:
            item=depths.get(str(depth),{'queries':0,'labeled_links':0,'recalled_links':0,
                                       'candidate_pairs':0,'fanout_histogram':[0]*(3*depth+1)})
            histogram=np.asarray(item['fanout_histogram'],dtype=np.int64)
            p95_index=int(np.searchsorted(np.cumsum(histogram),.95*max(item['queries'],1),side='left'))
            recall=item['recalled_links']/max(item['labeled_links'],1)
            if baseline is None:
                baseline=recall
            result[group][str(depth)]={'queries':item['queries'],
                'labeled_links':item['labeled_links'],'recalled_links':item['recalled_links'],
                'link_recall':recall,'link_recall_delta_from_top6':recall-baseline,
                'candidate_pairs':item['candidate_pairs'],
                'mean_candidates_per_query':item['candidate_pairs']/max(item['queries'],1),
                'p95_candidates_per_query':p95_index}
    return result


def _top_results(retriever,records,k):
    """Mirror production's integer sparse search while retaining deeper ranks."""
    results={}
    if not hasattr(retriever,'integer_matrices'):
        retriever.integer_matrices={}
    for channel in retriever.channels:
        query=prune_rows(retriever.vectorizers[channel].transform(texts(records,channel)),
                         CHANNELS[channel]['keep'])
        if channel not in retriever.integer_matrices:
            matrix=retriever.matrices[channel].copy()
            matrix.data=np.rint(matrix.data*SCALE).astype(np.int32)
            retriever.integer_matrices[channel]=matrix
            if not retriever.retain_forward:
                del retriever.matrices[channel]
        query.data=np.rint(query.data*SCALE).astype(np.int32)
        from sparse_dot_topn import sp_matmul_topn
        result=sp_matmul_topn(query,retriever.integer_matrices[channel],top_n=k,
            threshold=int(.05*SCALE*SCALE),sort=True,n_threads=int(__import__('os').environ.get('ER_THREADS','4')))
        results[channel]=result.astype(np.float32)/(SCALE*SCALE)
    return results


def audit_retrieval(root,config):
    root=Path(root); index=root/'cache/sparse_v1_train'
    required=[index/'reference_ids.json',*(index/f'{channel}.{suffix}'
              for channel in CHANNELS for suffix in ('pkl','npz','json'))]
    if any(not path.is_file() for path in required):
        raise ValueError('Complete cached training sparse index is required; it is never rebuilt by this audit')
    output=root/'cache/cloud/retrieval_depth_audit'
    identity={'code':digest(Path(__file__)),
        'index_reports':{channel:digest(index/f'{channel}.json') for channel in CHANNELS},
        'query_population':'all training S2/S3 records',
        'top_k':list(TOP_KS),'threshold':.05,'integer_scale':SCALE,
        'policy':'complete union; no final candidate cap'}
    claim_config(output,identity)
    final=output/'report.json'
    if final.exists():
        return json.loads(final.read_text())
    database=connect(root)
    labels=label_lookup(database,root)
    query_count=sum(database.execute(f'SELECT count(*) FROM train_source{s}_norm').fetchone()[0]
                    for s in (2,3))
    batch_size=config['batch_size']
    batch_count=(query_count+batch_size-1)//batch_size
    chunk_count=(batch_count+BATCHES_PER_CHECKPOINT-1)//BATCHES_PER_CHECKPOINT
    retriever=Retriever(index,retain_forward=False)
    index_references=retriever.matrices['name'].shape[1]
    batches=iter(secondary_batches(database,'train',batch_size,labels))
    merged={}; processed=0
    for chunk_id in range(chunk_count):
        marker=output/f'chunk_{chunk_id:05d}.json'
        chunk_metrics={}; chunk_queries=0
        for _ in range(BATCHES_PER_CHECKPOINT):
            try:
                batch=next(batches)
            except StopIteration:
                break
            chunk_queries+=len(batch)
            if not marker.exists():
                check_headroom(root)
                records=[tuple(row[:4]) for row in batch]
                owners=np.asarray([row[4] for row in batch],dtype=np.int32)
                found={6:retriever.search_selected(records,backend='cpu',policy='full6')}
                for depth in TOP_KS[1:]:
                    found[depth]=_top_results(retriever,records,depth)
                chunk_metrics=_merge(chunk_metrics,_raw_group(records,owners,found,TOP_KS))
        if marker.exists():
            saved=json.loads(marker.read_text())
            if saved.get('identity')!=digest(output/'config.json'):
                raise ValueError('Retrieval checkpoint configuration differs')
            if saved.get('queries')!=chunk_queries:
                raise ValueError('Retrieval checkpoint input population differs')
            merged=_merge(merged,saved['metrics'])
        else:
            merged=_merge(merged,chunk_metrics)
            atomic_json(marker,{'identity':digest(output/'config.json'),
                                'queries':chunk_queries,'metrics':chunk_metrics})
        processed+=chunk_queries
        if chunk_id%10==0:
            print(f'Retrieval-depth audit: checkpointed {chunk_id+1}/{chunk_count} query groups; '
                  f'{processed:,}/{query_count:,} queries',flush=True)
    database.close()
    if processed!=query_count:
        raise ValueError('Retrieval depth audit did not cover every training query')
    report={'scope':'All training S2/S3 queries and full S1 index; labels used only to measure link recall',
        'population':{'evaluated_queries':processed,'query_population':query_count,
                      'index_references':index_references},
        'retrieval':_finish(merged),
        'controls':['Original IDF/vectorizers/index reused; no features, fitting or labels used to retrieve.',
                    'Training labels are read only to measure whether the true owner is in each candidate set.',
                    'Top-6 reproduces the production integer sparse search; top-12/top-20 deepen all three channels.',
                    'All channel candidates are retained in each diagnostic union; no cap is applied.',
                    'This retrieval diagnostic is not model validation or a submission.']}
    atomic_json(root/'reports/cloud_retrieval_depth_audit.json',report)
    atomic_json(final,report)
    return report
