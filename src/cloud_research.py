"""Full-cache diagnostics and leakage-safe country/feature experiments.

These are opt-in experiments. They never touch reserved fold 4 or change a
selected submission automatically. Keep components only after comparing reports.
"""

import json
from pathlib import Path
import numpy as np
from src.cloud_features import references
from src.cloud_model import train_one,score_model,tune_scores,evaluate_scores
from src.cloud_store import atomic_json,read_parquet,parquet_files,claim_config,digest
from src.features import FEATURE_NAMES


ABLATIONS={
    'address': {'address_retrieval','address_rank','address_exact','address_ratio','address_token_sort',
                'address_token_set','address_jaccard','address_containment','address_length_ratio',
                'number_jaccard','number_exact','first_number_equal','number_conflict','one_number_missing',
                'postal_equal','postal_conflict','address_missing','log_address_frequency',
                'strong_name_number_conflict','weak_name_strong_address','name_address_product'},
    'numeric': {'number_jaccard','number_exact','first_number_equal','number_conflict','one_number_missing',
                'postal_equal','postal_conflict','strong_name_number_conflict'},
    'frequency': {'log_name_frequency','log_address_frequency','shared_rare_tokens',
                  'max_shared_idf','sum_shared_idf','weighted_jaccard'},
    'retrieval': set(FEATURE_NAMES[:10]),
}


def candidate_diagnostics(root):
    refs=read_parquet(references(root,'train'),'ri,truth_count,country_norm')
    n=len(refs['ri']); total=0
    volumes=np.zeros((4,n),dtype=np.int32); hits=np.zeros_like(volumes)
    unique_hits=np.zeros(3,dtype=np.int64); unique_volume=np.zeros(3,dtype=np.int64)
    for path in parquet_files(root/'cache/cloud/candidates_train'):
        b=read_parquet(path,'ri,y,m0,m1,m2')
        present=np.column_stack([b[f'm{i}']>0 for i in range(3)])
        total+=len(b['ri'])
        for channel in range(4):
            keep=present[:,channel] if channel<3 else np.ones(len(b['ri']),dtype=bool)
            np.add.at(volumes[channel],b['ri'][keep],1)
            np.add.at(hits[channel],b['ri'][keep & (b['y']>0)],1)
        sole=present.sum(axis=1)==1
        unique_volume+=(present & sole[:,None]).sum(axis=0)
        unique_hits+=(present & (sole & (b['y']>0))[:,None]).sum(axis=0)
    truth=refs['truth_count']; metrics={}
    for i,name in enumerate(('name','address','token','union')):
        metrics[name]={'candidate_pairs':int(volumes[i].sum()),
                       'link_recall':float(hits[i].sum()/max(truth.sum(),1)),
                       'all_true_matches_covered':float((hits[i]==truth).mean()),
                       'mean_per_s1':float(volumes[i].mean()),'p95_per_s1':float(np.quantile(volumes[i],.95))}
        if i<3:
            metrics[name].update(unique_true_links=int(unique_hits[i]),unique_pairs=int(unique_volume[i]))
    shape=json.loads((root/'cache/cloud/candidates_train/shape.json').read_text())
    report={'scope':'All training references and full candidate union; descriptive blocking diagnostics',
            'channels':metrics,'reduction_ratio':1-total/(n*shape['query_count']),
            'note':'Channel ablation uses cached full top-6 lists; it does not simulate a new retriever.'}
    atomic_json(root/'reports/cloud_retrieval_contributions.json',report)
    return report


def masks(ref,experiment):
    dev=ref['fold']!=4
    if experiment.startswith('country-'):
        country=experiment.removeprefix('country-')
        available=set(ref['country_norm'][dev])
        if country not in available or len(available)<2:
            raise ValueError('Country experiment requires both a fitting and held-out country')
        fit=dev & (ref['country_norm']==country)
        test=dev & (ref['country_norm']!=country)
        return fit,test,tuple(range(len(FEATURE_NAMES)))
    if experiment.startswith('ablation-'):
        removed=ABLATIONS[experiment.removeprefix('ablation-')]
        return dev,dev,tuple(i for i,n in enumerate(FEATURE_NAMES) if n not in removed)
    raise ValueError('Unknown research experiment')


def run_experiment(root,experiment,config,fingerprint):
    if experiment=='contributions':
        return candidate_diagnostics(root)
    ref=read_parquet(references(root,'train'),'ri,fold,truth_count,country_norm')
    fit,test,columns=masks(ref,experiment)
    folder=root/'cache/cloud/research'/experiment
    research_identity={'base':fingerprint,'research_code':digest(Path(__file__)),
                       'experiment':experiment,'columns':list(columns),
                       'trees':config['trees'],'seed':config['seed']}
    claim_config(folder,research_identity)
    if (folder/'report.json').exists():
        return json.loads((folder/'report.json').read_text())
    scores=[]
    for fold in range(4):
        name=f'research/{experiment}/fold_{fold}'
        model=train_one(root,name,fit & (ref['fold']!=fold),'gbdt',config,fingerprint,columns)
        scores.extend(score_model(root,model,'train',fit & (ref['fold']==fold),name,config))
    ranking=tune_scores(scores,ref['truth_count'],fit)
    policy={k:ranking[0][k] for k in ('threshold','relative')}
    result={'experiment':experiment,'feature_columns':[FEATURE_NAMES[i] for i in columns],
            'policy':policy,'development_oof':evaluate_scores(scores,ref,fit,policy),
            'folds':{},'reserved_fold_used':False,
            'scope':'Full candidate cache; development folds 0-3 only; no pair or negative subsampling'}
    for fold in range(4):
        scope=fit & (ref['fold']==fold)
        fold_ranking=tune_scores(scores,ref['truth_count'],scope)
        result['folds'][str(fold)]={'fixed_policy':evaluate_scores(scores,ref,scope,policy)['overall'],
                                   'descriptive_best_threshold':fold_ranking[0]['threshold']}
    if experiment.startswith('country-'):
        # Freeze the threshold using inner OOF from the fitting country only.
        # No label from the evaluated country enters model or threshold fitting.
        claim_config(folder/'locked',{'policy':policy,'fitting_country':experiment[8:]})
        name=f'research/{experiment}/country_fit'
        model=train_one(root,name,fit,'gbdt',config,fingerprint,columns)
        held=score_model(root,model,'train',test,name,config)
        result['country_held_out']=evaluate_scores(held,ref,test,policy)
        result['interpretation']='A country-shift proxy; does not measure France accuracy.'
    else:
        result['interpretation']='OOF-selected policy on development data; not an untouched performance estimate.'
    atomic_json(folder/'report.json',result)
    atomic_json(root/'reports'/f'cloud_{experiment}.json',result)
    return result
