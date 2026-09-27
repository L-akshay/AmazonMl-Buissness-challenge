"""Full OOF decision diagnostics over saved scores; never changes submissions.

Secondary ownership is unique in the supplied training labels. Evaluate a
competition rule, rather than silently assuming it improves the matcher.
Existing candidate/feature/model caches and reserved labels remain untouched.
"""

import json
from pathlib import Path
import numpy as np
from src.cloud_features import references
from src.cloud_model import entity_metrics
from src.cloud_store import (atomic_json, check_headroom, claim_config, digest,
                             parquet_files, read_parquet, write_parquet)
from src.predict import decode_secondary

MODES=('independent','exclusive','margin_025','margin_075','soft_025','soft_050')
THRESHOLDS=np.array(sorted(set([float(x) for x in np.arange(.1,.951,.025)]
                               +[.625,.975,.99,.995])),dtype=np.float32)


def competition(batch):
    """Return unique winners and runner-up scores; exact top ties abstain."""
    n=len(batch['ri']); winner=np.zeros(n,dtype=bool); second=np.zeros(n,dtype=np.float32)
    if not n:
        return winner,second
    p=batch['p']; ri=batch['ri']; tid=batch['tid']
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid saved probabilities')
    by_pair=np.lexsort((ri,tid))
    if np.any((tid[by_pair][1:]==tid[by_pair][:-1]) & (ri[by_pair][1:]==ri[by_pair][:-1])):
        raise ValueError('Duplicate candidate pair in OOF union')
    order=np.lexsort((ri,-p,tid)); targets=tid[order]
    starts=np.r_[0,1+np.flatnonzero(targets[1:]!=targets[:-1])]
    ends=np.r_[starts[1:],n]; first=order[starts]
    has_other=ends-starts>1
    runner=np.zeros(len(first),dtype=np.float32)
    runner[has_other]=p[order[starts[has_other]+1]]
    winner[first]=(~has_other) | (p[first]>runner)
    second[first]=runner
    return winner,second


def probabilities(batch,mode):
    p=batch['p']
    if mode=='independent':
        return p
    winner=batch['winner'].astype(bool); second=batch['second']
    if mode=='exclusive':
        return np.where(winner,p,0)
    if mode in ('margin_025','margin_075'):
        gap=.025 if mode=='margin_025' else .075
        return np.where(winner & (p-second>=gap),p,0)
    if mode in ('soft_025','soft_050'):
        power=.25 if mode=='soft_025' else .5
        return np.where(winner,p*np.power(1-second,power),0)
    raise ValueError('Unknown decision mode')


def summarize(ref,count,tp,scope):
    return {'overall':entity_metrics(ref['truth_count'],count,tp,scope),
            'countries':{str(c):entity_metrics(ref['truth_count'],count,tp,scope & (ref['country_norm']==c))
                         for c in np.unique(ref['country_norm'][scope])}}


def scan_grid(files,ref,mode,thresholds=THRESHOLDS,root=None):
    """At most two int32 histograms in RAM; no pair or negative sampling."""
    n=len(ref['ri']); size=len(thresholds)
    counts=np.zeros((n,size+1),dtype=np.int32); hits=np.zeros_like(counts)
    for path in files:
        if root is not None:
            check_headroom(root)
        b=read_parquet(path); scores=probabilities(b,mode)
        bins=np.searchsorted(thresholds,scores,side='right')
        np.add.at(counts,(b['ri'],bins),1)
        positive=b['y']>0
        np.add.at(hits,(b['ri'][positive],bins[positive]),1)
    count=np.zeros(n,dtype=np.int64); tp=np.zeros(n,dtype=np.int64)
    selection=np.isin(ref['fold'],[0,1]); confirmation=np.isin(ref['fold'],[2,3])
    ranking=[]
    for index in range(size-1,-1,-1):
        count+=counts[:,index+1]; tp+=hits[:,index+1]
        ranking.append({'mode':mode,'threshold':float(thresholds[index]),
                        'selection':summarize(ref,count,tp,selection),
                        'confirmation':summarize(ref,count,tp,confirmation),
                        'development':summarize(ref,count,tp,selection|confirmation)})
    # Selection order depends only on folds 0/1, never confirmation/leaderboard.
    ranking.sort(key=lambda r:(r['selection']['overall']['macro_f05'],
                              r['selection']['overall']['link_precision'],r['threshold']),reverse=True)
    return ranking


def audit(root,config):
    ref=read_parquet(references(root,'train'),'ri,fold,truth_count,country_norm')
    dev=ref['fold']!=4; n=len(ref['ri'])
    sources=[root/'cache/cloud/scores'/f'gbdt_fold_{fold}' for fold in range(4)]
    inputs=[parquet_files(path) for path in sources]
    names=[p.name for p in inputs[0]]
    if any([p.name for p in files]!=names for files in inputs):
        raise ValueError('OOF shard boundaries differ')
    out=root/'cache/cloud/decision_audit'
    identity={'code':digest(Path(__file__)),'references':digest(references(root,'train')),
              'sources':[{'complete':digest(p/'complete.json'),'config':digest(p/'config.json')} for p in sources],
              'thresholds':THRESHOLDS.tolist(),'modes':MODES}
    # JSON roundtrip preserves equality when tuples have been serialized as lists.
    claim_config(out,json.loads(json.dumps(identity)))
    if (out/'report.json').exists():
        return json.loads((out/'report.json').read_text())
    files=[]; candidate_hits=np.zeros(n,dtype=np.int64)
    baseline_count=np.zeros(n,dtype=np.int64); baseline_tp=np.zeros(n,dtype=np.int64)
    conflicts=0; previous=None
    for index,name in enumerate(names):
        check_headroom(root)
        target=out/'scores'/name
        marker=target.with_suffix('.json')
        if target.exists() and marker.exists():
            info=json.loads(marker.read_text()); b=read_parquet(target)
        else:
            parts=[read_parquet(paths[index]) for paths in inputs]
            for fold,part in enumerate(parts):
                if np.any(ref['fold'][part['ri']]!=fold):
                    raise ValueError('OOF shard contains a fitted or reserved reference')
            b={key:np.concatenate([p[key] for p in parts]) for key in ('ri','tid','y','p')}
            b['winner'],b['second']=competition(b)
            unique=np.unique(b['tid'])
            boundaries=[decode_secondary(int(code)) for code in unique]
            info={'first':min(boundaries) if boundaries else None,'last':max(boundaries) if boundaries else None,
                  'rows':len(b['ri'])}
            write_parquet(target,b); atomic_json(marker,info)
        # Source batches are lexicographically sorted by secondary ID. This
        # proves all competitors for an ID were considered in one joined shard.
        if info['first'] is not None:
            if previous is not None and info['first']<=previous:
                raise ValueError('Secondary IDs cross shard boundaries')
            previous=info['last']
        if len(b['ri'])!=info['rows']:
            raise ValueError('Incomplete joined score shard')
        np.add.at(candidate_hits,b['ri'][b['y']>0],1)
        chosen=b['p']>=.625
        np.add.at(baseline_count,b['ri'][chosen],1)
        np.add.at(baseline_tp,b['ri'][chosen & (b['y']>0)],1)
        _,volumes=np.unique(b['tid'][chosen],return_counts=True)
        conflicts+=int((volumes>1).sum())
        files.append(target)
        if index%50==0:
            print(f'Decision audit: joined {index+1}/{len(names)} score shards',flush=True)
    if np.any(candidate_hits>ref['truth_count']):
        raise ValueError('Retrieved positive counts exceed ground truth')
    baseline=summarize(ref,baseline_count,baseline_tp,dev)
    expected=json.loads((root/'reports/cloud_model_validation.json').read_text())['comparisons']['gbdt']['oof']['overall']
    if abs(baseline['overall']['macro_f05']-expected['macro_f05'])>1e-10:
        raise ValueError('Saved OOF population does not reproduce the baseline')
    oracle=summarize(ref,candidate_hits,candidate_hits,dev)
    rankings={}
    for mode in MODES:
        path=out/f'grid_{mode}.json'
        if path.exists():
            rankings[mode]=json.loads(path.read_text())
        else:
            print(f'Tuning {mode} on all saved OOF pairs',flush=True)
            rankings[mode]=scan_grid(files,ref,mode,root=root)
            atomic_json(path,rankings[mode])
    selected=max((rows[0] for rows in rankings.values()),
                 key=lambda r:r['selection']['overall']['macro_f05'])
    atomic_json(out/'selected_before_confirmation.json',{'mode':selected['mode'],'threshold':selected['threshold'],
                                                       'selection':selected['selection']})
    confirmation=np.isin(ref['fold'],[2,3])
    confirmation_baseline=summarize(ref,baseline_count,baseline_tp,confirmation)
    difference=selected['confirmation']['overall']['macro_f05']-confirmation_baseline['overall']['macro_f05']
    country_changes={c:row['macro_f05']-confirmation_baseline['countries'][c]['macro_f05']
                     for c,row in selected['confirmation']['countries'].items()}
    result={'scope':'All existing OOF candidates for development folds 0-3; no reserved labels or test labels',
            'baseline':baseline,'candidate_oracle':oracle,'baseline_conflicting_secondary_ids':conflicts,
            'unretrieved_true_links':int((ref['truth_count']-candidate_hits)[dev].sum()),
            'retrieved_but_missed_true_links':int((candidate_hits-baseline_tp)[dev].sum()),
            'false_positive_links':int((baseline_count-baseline_tp)[dev].sum()),
            'rankings':rankings,'selected':selected,'confirmation_baseline':confirmation_baseline,
            'confirmation_macro_delta':difference,'confirmation_country_deltas':country_changes,
            'supports_new_submission':bool(difference>.0005 and min(country_changes.values())>=-.002),
            'limits':['Candidate oracle assumes perfect labels and is not achievable model performance.',
                      'Competition uses the 80% development reference pool; full test competition can differ.',
                      'Confirmation is cross-fitted OOF evidence, not a new untouched or nested-model holdout.',
                      'No prediction file is changed automatically by this audit.']}
    atomic_json(out/'report.json',result)
    atomic_json(root/'reports/cloud_decision_audit.json',result)
    return result
