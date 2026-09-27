"""Checkpointed model comparison, frozen decision and one reserved-fold evaluation."""

import json
import numpy as np
from src.cloud_features import references
from src.cloud_store import atomic_json, read_parquet, parquet_files, claim_config
from src.cloud_model import train_one, score_model, tune_scores, evaluate_scores
from src.evaluation_scope import exclusion_indices


def oof_fold(root,config,fingerprint,kind,fold):
    """One independent full-data OOF fit for a separately scheduled CPU job."""
    if kind not in ('gbdt','logistic') or fold not in range(4):
        raise ValueError('Invalid OOF task')
    ref=read_parquet(references(root,'train'),'ri,fold')
    folds=ref['fold']; dev=folds!=4
    name=f'{kind}_fold_{fold}'
    model=train_one(root,name,dev & (folds!=fold),kind,config,fingerprint)
    return score_model(root,model,'train',folds==fold,name,config)


def candidate_fit(root,config,fingerprint,kind,scope):
    """Fit frozen model configurations early; never inspect reserved outcomes."""
    if kind not in ('gbdt','logistic') or scope not in ('full','reserved'):
        raise ValueError('Invalid candidate fit')
    ref=read_parquet(references(root,'train'),'ri,fold')
    allowed=np.ones(len(ref['ri']),dtype=bool) if scope=='full' else ref['fold']!=4
    return train_one(root,f'{scope}_{kind}',allowed,kind,config,fingerprint)


def candidate_score(root,config,kind,split):
    """Cache scores independently of threshold/model selection; no evaluation."""
    if kind not in ('gbdt','logistic') or split not in ('train','test'):
        raise ValueError('Invalid candidate scoring task')
    scope='full' if split=='test' else 'reserved'
    suffix='model.txt' if kind=='gbdt' else 'model.joblib'
    model=root/'cache/cloud/models'/f'{scope}_{kind}'/suffix
    ref=read_parquet(references(root,split),'ri,fold')
    allowed=np.ones(len(ref['ri']),dtype=bool) if split=='test' else ref['fold']==4
    if split=='train':
        excluded,_=exclusion_indices(root)
        allowed[excluded]=False
    return score_model(root,model,split,allowed,f'{split}_{kind}',config)


def validation(root,config,fingerprint):
    out=root/"cache"/"cloud"/"validation"
    claim_config(out,{"fingerprint":fingerprint,"models":config["models"],"seed":config["seed"],
                      "trees":config["trees"],"logistic_epochs":config["logistic_epochs"]})
    if (out/"report.json").exists():
        return json.loads((out/"report.json").read_text())
    ref=read_parquet(references(root,"train"),"ri,fold,truth_count,country_norm")
    folds=ref["fold"]; dev=folds!=4
    comparisons={}
    for kind in config["models"]:
        report_path=out/f"{kind}.json"
        if report_path.exists():
            comparisons[kind]=json.loads(report_path.read_text())
            continue
        all_scores=[]
        for fold in range(4):
            all_scores.extend(oof_fold(root,config,fingerprint,kind,fold))
        ranking=tune_scores(all_scores,ref["truth_count"],dev)
        selected=ranking[0]
        policy={k:selected[k] for k in ("threshold","relative")}
        report={"policy":policy,"oof":evaluate_scores(all_scores,ref,dev,policy),
                "folds":{str(f):evaluate_scores(all_scores,ref,folds==f,policy)["overall"] for f in range(4)},
                "threshold_grid":ranking}
        atomic_json(report_path,report)
        comparisons[kind]=report
    kind=max(comparisons,key=lambda k:comparisons[k]["oof"]["overall"]["macro_f05"])
    policy=comparisons[kind]["policy"]
    # Freeze all upload variants before looking at the reserved fold. They reuse
    # the same full test scores; no leaderboard score is used for this selection.
    candidates=comparisons[kind]["threshold_grid"]
    near=[r for r in candidates if r["macro_f05"]>=candidates[0]["macro_f05"]-.005]
    choices=[("oof_best",candidates[0]),("precision",max(near,key=lambda r:(r["link_precision"],r["macro_f05"]))),
             ("recall",max(near,key=lambda r:(r["link_recall"],r["macro_f05"])))]
    variants=[]; seen=set()
    for name,r in choices:
        key=(r["threshold"],r["relative"])
        if key not in seen:
            variants.append({"name":name,"threshold":key[0],"relative":key[1],"development_macro_f05":r["macro_f05"]})
            seen.add(key)
    frozen={"kind":kind,"policy":policy,"variants":variants,"fingerprint":fingerprint}
    claim_config(out/"locked",frozen)
    holdout=folds==4
    excluded,exclusion_report=exclusion_indices(root)
    holdout[excluded]=False
    candidate_fit(root,config,fingerprint,kind,'reserved')
    scores=candidate_score(root,config,kind,'train')
    result={"scope":"All training S1 entities and every full-union candidate; entity folds 0-3 OOF, fold 4 reserved",
            "comparisons":comparisons,"selected_model":kind,"selected_policy":policy,
            "variants":variants,"reserved":evaluate_scores(scores,ref,holdout,policy),
            "reserved_exclusions":exclusion_report,"fingerprint":fingerprint,
            "logistic_solver":"SGD log-loss with train-only streaming standardization; no negative subsampling",
            "calibration":"none; thresholds selected from OOF probabilities",
            "pending_experiments":["nested country-held-out refits","OOF hard-negative reweighting",
                                   "full feature and normalization ablations","multilingual dense retrieval (conditional)"]}
    atomic_json(out/"report.json",result)
    atomic_json(root/"reports"/"cloud_model_validation.json",result)
    return result


def final_fit(root,config,fingerprint):
    report=json.loads((root/"cache"/"cloud"/"validation"/"report.json").read_text())
    if report["fingerprint"]!=fingerprint:
        raise ValueError("Validation does not match current features")
    model=candidate_fit(root,config,fingerprint,report['selected_model'],'full')
    atomic_json(root/"cache"/"cloud"/"final.json",{"model":str(model.relative_to(root)),
                "kind":report["selected_model"],"policy":report["selected_policy"],"variants":report["variants"],
                "score_name":'test_'+report['selected_model']})
    return model
