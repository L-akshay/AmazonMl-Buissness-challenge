"""Entity-grouped development OOF, locked holdout, and final LightGBM fitting."""

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import pickle
from pathlib import Path
from time import perf_counter
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from src.features import FEATURE_NAMES


def make_model(kind="gbdt",seed=42):
    if kind=="logistic":
        return make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=250,random_state=seed))
    return lgb.LGBMClassifier(n_estimators=350,num_leaves=31,learning_rate=.065,
        min_child_samples=100,colsample_bytree=.9,reg_lambda=3,n_jobs=8,
        random_state=seed,deterministic=True,force_col_wise=True,histogram_pool_size=256,verbosity=-1)


def aggregate(y,prob,groups,truth,scope,threshold,relative=0):
    accept=prob>=threshold
    if relative:
        best=np.zeros(len(truth),dtype=np.float32)
        np.maximum.at(best,groups,prob)
        accept &= prob>=best[groups]*relative
    count=np.bincount(groups[accept],minlength=len(truth))
    tp=np.bincount(groups[accept & (y>0)],minlength=len(truth))
    t=truth[scope]
    p=count[scope]
    hits=tp[scope]
    den=5*hits+t-hits+4*(p-hits)
    scores=np.divide(5*hits,den,out=np.zeros(len(t),dtype=float),where=den>0)
    scores[t==0]=(p[t==0]==0).astype(float)
    return {"entities":int(len(t)),"macro_f05":float(scores.mean()),
        "link_precision":float(hits.sum()/p.sum()) if p.sum() else 0,
        "link_recall":float(hits.sum()/t.sum()) if t.sum() else 0,
        "singleton_fp_rate":float((p[t==0]>0).mean()) if (t==0).any() else None,
        "predicted_singletons":int((p==0).sum()),"true_links":int(t.sum()),"predicted_links":int(p.sum())}


def tune(y,prob,groups,truth,scope):
    best=None
    for relative in (0,.5,.8):
        for threshold in np.r_[np.arange(.10,.96,.025),.97,.98,.99,.995]:
            score=aggregate(y,prob,groups,truth,scope,float(threshold),relative)
            record={"threshold":float(threshold),"relative":relative,**score}
            if best is None or (score["macro_f05"],score["link_precision"],-relative)>(best["macro_f05"],best["link_precision"],-best["relative"]):
                best=record
    return best


def decision_mask(prob,groups,n,policy):
    result=prob>=policy["threshold"]
    if policy["relative"]:
        best=np.zeros(n,dtype=np.float32)
        np.maximum.at(best,groups,prob)
        result &= prob>=best[groups]*policy["relative"]
    return result


def diagnostics(y,prob,groups,truth,scope,folds,policy):
    result={"by_match_count":{},"by_fold":{}}
    for name,mask in (("singleton",truth==0),("one",truth==1),("multiple",truth>1)):
        result["by_match_count"][name]=aggregate(y,prob,groups,truth,scope&mask,**policy)
    for fold in range(4):
        result["by_fold"][str(fold)]=aggregate(y,prob,groups,truth,scope&(folds==fold),**policy)
    values=[r["macro_f05"] for r in result["by_fold"].values()]
    result["fold_score_std"]=float(np.std(values))
    result["fold_score_range"]=[min(values),max(values)]
    return result


def feature_ablations(x,y,g,qids,fit,pair_folds,truth,sample,folds,kind,policy,baseline):
    # Diagnostic only: the same held-out development fold and frozen policy.
    # The locked fold is never involved in these fits or comparisons.
    groups={
        "address_text":["address_exact","address_ratio","address_token_sort","address_token_set","address_jaccard","address_containment","address_length_ratio","address_missing","log_address_frequency","weak_name_strong_address","name_address_product"],
        "numbers":["number_jaccard","number_exact","first_number_equal","number_conflict","one_number_missing","postal_equal","postal_conflict","strong_name_number_conflict"],
        "rare_token_features":["shared_rare_tokens","max_shared_idf","sum_shared_idf","weighted_jaccard"],
        "retrieval_metadata":FEATURE_NAMES[:10],
    }
    train=(pair_folds>0)&(pair_folds<4)&fit
    valid=pair_folds==0
    scope=sample&(folds==0)
    base=aggregate(y[valid],baseline[valid],g[valid],truth,scope,**policy)
    report={"scope":"Development fold 0 only; identical frozen OOF policy. Descriptive diagnostics, not independent model selection.","baseline":base,"without":{}}
    for name,removed in groups.items():
        columns=[i for i,f in enumerate(FEATURE_NAMES) if f not in removed]
        model=make_model(kind)
        model.fit(x[train][:,columns],y[train])
        prob=model.predict_proba(x[valid][:,columns])[:,1]
        result=aggregate(y[valid],prob,g[valid],truth,scope,**policy)
        result["macro_f05_delta"]=result["macro_f05"]-base["macro_f05"]
        report["without"][name]=result
        print(f"Ablation without {name}: {result['macro_f05']:.5f}",flush=True)
    random_neg=((g.astype(np.uint64)*2654435761+qids.astype(np.uint64)*2246822519)%10)==0
    simple=(pair_folds>0)&(pair_folds<4)&((y>0)|random_neg)
    model=make_model(kind)
    model.fit(x[simple],y[simple])
    prob=model.predict_proba(x[valid])[:,1]
    report["without"]["hard_negative_retention"]=aggregate(y[valid],prob,g[valid],truth,scope,**policy)
    xv=x[valid]
    negative=y[valid]==0
    before=decision_mask(prob,g[valid],len(truth),policy)
    after=decision_mask(baseline[valid],g[valid],len(truth),policy)
    name=xv[:,FEATURE_NAMES.index("name_ratio")]
    address=xv[:,FEATURE_NAMES.index("address_ratio")]
    typed={"same_name_different_address":(name>.9)&(address<.5),
        "different_name_similar_address":(name<.5)&(address>.9),
        "numeric_conflict":xv[:,FEATURE_NAMES.index("number_conflict")]>0,
        "frequent_name":xv[:,FEATURE_NAMES.index("log_name_frequency")]>=np.log(11)}
    report["hard_negative_buckets"]={}
    for name,mask in typed.items():
        mask &= negative
        report["hard_negative_buckets"][name]={"candidate_count":int(mask.sum()),
            "mean_score_without_retention":float(prob[mask].mean()) if mask.any() else None,
            "mean_score_with_retention":float(baseline[valid][mask].mean()) if mask.any() else None,
            "false_positives_without_retention":int((before&mask).sum()),
            "false_positives_with_retention":int((after&mask).sum())}
    model=make_model(kind,seed=17)
    model.fit(x[train],y[train])
    prob=model.predict_proba(x[valid])[:,1]
    report["seed_17"]=aggregate(y[valid],prob,g[valid],truth,scope,**policy)
    return report


def validate(root):
    start=perf_counter()
    folder=root/"cache"/"development"
    arrays={n:np.load(folder/f"{n}.npy",mmap_mode="r") for n in ("x","y","groups","qids","fit")}
    x,y,g,fit=arrays["x"],arrays["y"],arrays["groups"],arrays["fit"]
    with (root/"cache"/"training_metadata.pkl").open("rb") as f:
        meta=pickle.load(f)
    folds=np.asarray(meta["fold"])
    truth=np.asarray(meta["truth_count"])
    sample=np.asarray(meta["sample"])
    country=np.asarray(meta["country"])
    pair_folds=folds[g]
    development=sample & (folds!=4)
    dev_pairs=pair_folds!=4
    report={"scope":"4% deterministic S1 sample; ALL training secondary records searched against ALL S1 references. Fold 4 locked until selection is frozen.","models":{},"feature_names":FEATURE_NAMES}
    report["reserved_scope"]=meta.get("holdout_exclusions",{})
    oofs={}
    for kind in ("logistic","gbdt"):
        oof=np.full(len(y),np.nan,dtype=np.float32)
        for fold in range(4):
            train=(pair_folds!=fold)&dev_pairs&fit
            validation=pair_folds==fold
            model=make_model(kind)
            print(f"{kind}: fold {fold}, fit {int(train.sum()):,}, validate {int(validation.sum()):,}",flush=True)
            model.fit(x[train],y[train])
            oof[validation]=model.predict_proba(x[validation])[:,1]
        oofs[kind]=oof
        policy=tune(y[dev_pairs],oof[dev_pairs],g[dev_pairs],truth,development)
        report["models"][kind]={"oof_policy":policy}
        print(json.dumps({kind:policy}),flush=True)
    selected=max(report["models"],key=lambda k:report["models"][k]["oof_policy"]["macro_f05"])
    policy=report["models"][selected]["oof_policy"]
    report["selected_model"]=selected
    report["selected_policy"]={"threshold":policy["threshold"],"relative":policy["relative"]}
    prob=oofs[selected][dev_pairs]
    report["oof_diagnostics"]=diagnostics(y[dev_pairs],prob,g[dev_pairs],truth,development,folds,report["selected_policy"])
    report["fold_tuned_policies"]={str(f):tune(y[pair_folds==f],oofs[selected][pair_folds==f],g[pair_folds==f],truth,development&(folds==f)) for f in range(4)}
    report["feature_ablations"]=feature_ablations(x,y,g,arrays["qids"],fit,pair_folds,truth,sample,folds,selected,report["selected_policy"],oofs[selected])
    # Monotone calibration cannot improve a freely tuned scalar threshold's
    # ranking; report Brier diagnostics without using locked-fold outcomes.
    calibration=[]
    for fold in range(4):
        idx=pair_folds[dev_pairs]==fold
        train=~idx
        target=y[dev_pairs]
        iso=IsotonicRegression(out_of_bounds="clip").fit(prob[train],target[train])
        logits=np.log(np.clip(prob,1e-6,1-1e-6)/(1-np.clip(prob,1e-6,1-1e-6))).reshape(-1,1)
        platt=LogisticRegression().fit(logits[train],target[train])
        calibration.append({"fold":fold,"raw_brier":float(np.mean((prob[idx]-target[idx])**2)),
            "platt_brier":float(np.mean((platt.predict_proba(logits[idx])[:,1]-target[idx])**2)),
            "isotonic_brier":float(np.mean((iso.predict(prob[idx])-target[idx])**2))})
    report["calibration"]={"diagnostics":calibration,"selected":"none","reason":"Keep raw scores and OOF-tuned decision; calibration diagnostics use cross-fitted OOF scores and are descriptive, not the locked holdout."}
    report["oof_by_country"]={}
    for c in sorted(set(country[development])):
        report["oof_by_country"][c]=aggregate(y[dev_pairs],prob,g[dev_pairs],truth,development&(country==c),**report["selected_policy"])
    report["country_held_out"]={}
    for c in sorted(set(country[development])):
        source_pairs=dev_pairs & (country[g]!=c)
        train=source_pairs & fit
        valid=dev_pairs & (country[g]==c)
        source_oof=np.full(len(y),np.nan,dtype=np.float32)
        for fold in range(4):
            inner_train=train&(pair_folds!=fold)
            inner_valid=source_pairs&(pair_folds==fold)
            inner_model=make_model(selected)
            inner_model.fit(x[inner_train],y[inner_train])
            source_oof[inner_valid]=inner_model.predict_proba(x[inner_valid])[:,1]
        training_country=development&(country!=c)
        train_policy=tune(y[source_pairs],source_oof[source_pairs],g[source_pairs],truth,training_country)
        model=make_model(selected)
        model.fit(x[train],y[train])
        p=model.predict_proba(x[valid])[:,1]
        report["country_held_out"][c]={**aggregate(y[valid],p,g[valid],truth,development&(country==c),threshold=train_policy["threshold"],relative=train_policy["relative"]),
            "threshold":train_policy["threshold"],"relative":train_policy["relative"],
            "scope":"Frozen model family; fitting and nested OOF threshold tuning use only the other country. A transfer proxy, not evidence of France accuracy."}
        print(f"Country-held-out {c}: {report['country_held_out'][c]['macro_f05']:.5f}",flush=True)
    # One untouched final check after the model family and policy are frozen.
    signature=hashlib.sha256(b"".join((root/"src"/name).read_bytes() for name in ("model.py","features.py","normalize.py","training_data.py","evaluation_scope.py"))).hexdigest()
    frozen={"model":selected,"policy":report["selected_policy"],"code_sha256":signature}
    report["frozen_configuration"]=frozen
    locked_path=root/"cache"/"locked_holdout.json"
    if locked_path.exists():
        locked=json.loads(locked_path.read_text())
        if locked["frozen"]!=frozen:
            raise ValueError("The reserved fold has already been evaluated for a different configuration; do not retune using it")
    else:
        (root/"cache"/"selection_frozen.json").write_text(json.dumps(frozen,indent=2),encoding="utf-8")
        model=make_model(selected)
        model.fit(x[dev_pairs & fit],y[dev_pairs & fit])
        held=pair_folds==4
        p=model.predict_proba(x[held])[:,1]
        locked={"frozen":frozen,
            "overall":aggregate(y[held],p,g[held],truth,sample&(folds==4),**report["selected_policy"]),
            "by_country":{c:aggregate(y[held],p,g[held],truth,sample&(folds==4)&(country==c),**report["selected_policy"]) for c in sorted(set(country[sample&(folds==4)]))}}
        locked_path.write_text(json.dumps(locked,indent=2),encoding="utf-8")
    report["locked_holdout"]=locked["overall"]
    report["locked_holdout_by_country"]=locked["by_country"]
    report["candidate_recall"]=float(y.sum()/truth[sample].sum())
    covered=np.bincount(g[y>0],minlength=len(truth))
    report["blocking_misses_on_development"]=int((truth[development]-covered[development]).sum())
    accepted=decision_mask(prob,g[dev_pairs],len(truth),report["selected_policy"])
    false_positive=accepted&(y[dev_pairs]==0)
    missed=(~accepted)&(y[dev_pairs]>0)
    xdev=x[dev_pairs]
    report["error_buckets"]={"scope":"Overlapping heuristic categories on development OOF; no manual identity lookup or prediction edits.",
        "false_positive_pairs":int(false_positive.sum()),"rejected_true_candidate_pairs":int(missed.sum()),"buckets":{}}
    buckets={"numeric_conflict":xdev[:,FEATURE_NAMES.index("number_conflict")]>0,
        "missing_address":xdev[:,FEATURE_NAMES.index("address_missing")]>0,
        "generic_name_frequency_at_least_10":xdev[:,FEATURE_NAMES.index("log_name_frequency")]>=np.log(11),
        "weak_name_similarity_below_0_5":xdev[:,FEATURE_NAMES.index("name_ratio")]<.5,
        "strong_name_weak_address":(xdev[:,FEATURE_NAMES.index("name_ratio")]>.85)&(xdev[:,FEATURE_NAMES.index("address_ratio")]<.5)}
    for name,mask in buckets.items():
        report["error_buckets"]["buckets"][name]={"false_positive_pairs":int((false_positive&mask).sum()),"rejected_true_pairs":int((missed&mask).sum())}
    # Keep row-level diagnostic examples local; they never change predictions.
    error_rows=np.flatnonzero(false_positive|missed)
    dev_groups=g[dev_pairs]
    dev_queries=arrays["qids"][dev_pairs]
    key=(dev_groups[error_rows].astype(np.uint64)*2654435761+dev_queries[error_rows].astype(np.uint64)*2246822519)%2**32
    chosen=error_rows[np.argsort(key,kind="stable")[:50]]
    blocking=np.flatnonzero(development&(covered<truth))
    order=(blocking.astype(np.uint64)*2654435761)%2**32
    blocking=blocking[np.argsort(order,kind="stable")[:50]]
    with (root/"reports"/"trained_errors.tsv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f,delimiter="\t")
        writer.writerow(["source1_entity_id","secondary_row_index","fold","first_failure","label","score","accepted","heuristic_causes"])
        for i in chosen:
            ri=int(dev_groups[i])
            writer.writerow([meta["ids"][ri],int(dev_queries[i]),int(folds[ri]),"matcher_or_decision",int(y[dev_pairs][i]),float(prob[i]),bool(accepted[i]),",".join(name for name,mask in buckets.items() if mask[i])])
        for ri in blocking:
            writer.writerow([meta["ids"][int(ri)],"",int(folds[ri]),"blocking","","","",f"{truth[ri]-covered[ri]} unretrieved true links"])
    del xdev
    report["runtime_seconds"]=perf_counter()-start
    report["split_counts"]={str(f):int((sample&(folds==f)).sum()) for f in range(5)}
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for path in (root/"reports"/"model_validation.json",root/"experiments"/f"E09_E19_{stamp}.json"):
        path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    np.savez_compressed(root/"cache"/"development_oof.npz",**oofs)
    print(json.dumps(report,indent=2),flush=True)


def train_final(root):
    report=json.loads((root/"reports"/"model_validation.json").read_text())
    folder=root/"cache"/"full"
    x=np.load(folder/"x.npy",mmap_mode="r")
    y=np.load(folder/"y.npy",mmap_mode="r")
    model=make_model(report["selected_model"])
    print(f"Final fit: {len(y):,} sampled candidate pairs from full training population",flush=True)
    model.fit(x,y)
    out=root/"cache"/"model"
    out.mkdir(exist_ok=True)
    with (out/"matcher.pkl").open("wb") as f:
        pickle.dump(model,f,protocol=5)
    config={"kind":report["selected_model"],"policy":report["selected_policy"],"features":FEATURE_NAMES,
        "frozen_configuration":report["frozen_configuration"],
        "feature_code_sha256":hashlib.sha256(b"".join((root/"src"/n).read_bytes() for n in ("features.py","normalize.py"))).hexdigest(),
        "training_pairs":len(y),"positive_pairs":int(y.sum()),"seed":42,
        "pretrained_models":[],"library_license":"LightGBM MIT; scikit-learn BSD-3-Clause",
        "negative_sampling":"All retrieved positives, score>=0.85 rank-one hard negatives, plus 10% deterministic other negatives."}
    if hasattr(model,"booster_"):
        model.booster_.save_model(str(out/"matcher.txt"))
        trees=model.booster_.dump_model()["tree_info"]
        config["scalar_tree_parameters"]=int(sum(3*t["num_leaves"]-2 for t in trees))
    else:
        linear=model.steps[-1][1]
        config["linear_coefficients_and_intercept"]=int(linear.coef_.size+linear.intercept_.size)
    (out/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    (root/"reports"/"final_model.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print(json.dumps(config,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["validate","final"],required=True)
    args=parser.parse_args()
    (validate if args.stage=="validate" else train_final)(Path(__file__).resolve().parents[1])
