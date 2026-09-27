"""All-pair, entity-grouped remote training without a dense full feature matrix."""

import gc
import hashlib
import json
import os
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from src.cloud_store import atomic_json, claim_config, check_headroom, read_parquet, write_parquet, parquet_files, digest
from src.cloud_features import references
from src.features import FEATURE_NAMES

_SHARD_CACHE = [None, None]


def feature_matrix(batch, columns=None):
    columns=range(len(FEATURE_NAMES)) if columns is None else columns
    return np.column_stack([batch[f"f{i}"] for i in columns]).astype(np.float32,copy=False)


class ParquetSequence(lgb.Sequence):
    """Only one decompressed shard is retained across all sequence objects."""
    def __init__(self,path,allowed,columns=None,batch_size=8192):
        self.path=Path(path); self.allowed=allowed; self.batch_size=batch_size
        self.columns=tuple(range(len(FEATURE_NAMES)) if columns is None else columns)
        meta=read_parquet(path,"ri,y")
        selected=allowed[meta["ri"]]
        self.labels=meta["y"][selected].astype(np.float32)
        self.size=len(self.labels)

    def __len__(self):
        return self.size

    def __getitem__(self,index):
        key=(str(self.path),id(self.allowed),self.columns)
        if _SHARD_CACHE[0]!=key:
            b=read_parquet(self.path,"ri,"+",".join(f"f{i}" for i in self.columns))
            x=feature_matrix(b,self.columns)[self.allowed[b["ri"]]]
            # Retain the mask object as well, preventing id() reuse between folds.
            _SHARD_CACHE[:]=[key,(x,self.allowed)]
        # LightGBM's Sequence sampling path requires doubles. Only the requested
        # row/batch is converted; the reusable Parquet cache stays float32.
        return _SHARD_CACHE[1][0][index].astype(np.float64)


class StreamingDataset(lgb.Dataset):
    """LightGBM 4.7 adapter: resume a model using batched Sequence initial scores.

LightGBM's default resume path sends a list of Sequences to predict(), which
does not accept that input type. Keep this version-pinned override covered by
the resumed-versus-uninterrupted training test.
"""
    def _set_init_score_by_predictor(self,predictor,data,used_indices):
        if predictor is not None and isinstance(data,list) and all(isinstance(s,lgb.Sequence) for s in data):
            if used_indices is not None:
                raise ValueError("Use explicit group masks instead of Dataset.subset")
            values=np.empty(sum(len(s) for s in data),dtype=np.float64)
            offset=0
            for sequence in data:
                for start in range(0,len(sequence),sequence.batch_size):
                    x=sequence[start:start+sequence.batch_size]
                    values[offset:offset+len(x)]=predictor.predict(x,raw_score=True)
                    offset+=len(x)
            return self.set_init_score(values)
        return super()._set_init_score_by_predictor(predictor,data,used_indices)


def train_one(root,name,allowed,kind,config,fingerprint,columns=None):
    files=parquet_files(root/"cache"/"cloud"/"features_train")
    out=root/"cache"/"cloud"/"models"/name
    columns=tuple(range(len(FEATURE_NAMES)) if columns is None else columns)
    claim_config(out,{"fingerprint":fingerprint,"kind":kind,"columns":list(columns),
                      "allowed_sha256":hashlib.sha256(allowed.tobytes()).hexdigest(),
                      "trees":config["trees"],"seed":config["seed"],
                      "logistic_epochs":config["logistic_epochs"],"min_leaf":config.get("min_leaf",100)})
    path=out/("model.txt" if kind=="gbdt" else "model.joblib")
    if (out/"complete.json").exists():
        if digest(path)!=json.loads((out/"complete.json").read_text())["sha256"]:
            raise ValueError("Completed model changed")
        return path
    if kind=="gbdt":
        sequences=[ParquetSequence(p,allowed,columns,config["feature_chunk"]) for p in files]
        sequences=[s for s in sequences if len(s)]
        n=sum(len(s) for s in sequences)
        if not n:
            raise ValueError("No pairs in fitting scope")
        import psutil
        estimate=n*(len(columns)+40)*1.4+1024**3
        if estimate>psutil.virtual_memory().available-2*1024**3:
            raise MemoryError(f"Estimated fitting memory {estimate/1024**3:.1f} GiB exceeds current headroom; use a larger RAM session, never drop pairs")
        labels=np.concatenate([s.labels for s in sequences])
        for s in sequences:
            s.labels=None
        dataset=StreamingDataset(sequences,label=labels,feature_name=[FEATURE_NAMES[i] for i in columns],
                                 params={"max_bin":255,"feature_pre_filter":False},free_raw_data=True)
        params={"objective":"binary","verbosity":-1,"learning_rate":.065,"num_leaves":31,
                "min_data_in_leaf":config.get("min_leaf",100),"feature_fraction":.9,"lambda_l2":3,
                "num_threads":config["threads"],"seed":config["seed"],"deterministic":True,
                "force_col_wise":True,"histogram_pool_size":256,"feature_pre_filter":False}
        checkpoint=out/"checkpoint.txt"
        previous=lgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else None
        remaining=config["trees"]-(previous.current_iteration() if previous else 0)
        def save_checkpoint(env):
            if (env.iteration+1)%25==0:
                temporary=out/"checkpoint.tmp.txt"
                env.model.save_model(str(temporary))
                temporary.replace(checkpoint)
                check_headroom(root)
        if remaining>0:
            model=lgb.train(params,dataset,num_boost_round=remaining,init_model=previous,
                            callbacks=[save_checkpoint])
        else:
            model=previous
        temp=path.with_suffix(".tmp.txt")
        model.save_model(str(temp)); temp.replace(path)
        parameter_bound=8*sum(2*t["num_leaves"]-1 for t in model.dump_model()["tree_info"])
        del model,dataset,sequences,labels
        _SHARD_CACHE[:]=[None,None]
        gc.collect()
    elif kind=="logistic":
        # Logistic regression with a streaming optimizer: all allowed candidates,
        # not a capacity-driven negative sample. Scaler sees fitting entities only.
        checkpoint=out/"checkpoint.joblib"
        if checkpoint.exists():
            state=joblib.load(checkpoint)
        else:
            state={"scaler":StandardScaler(),"model":SGDClassifier(loss="log_loss",alpha=.0001,
                   random_state=config["seed"],shuffle=False),"epoch":-1,"next_file":0,"rows":0}
        for epoch in range(state["epoch"],config["logistic_epochs"]):
            start=state["next_file"] if epoch==state["epoch"] else 0
            for index in range(start,len(files)):
                check_headroom(root)
                b=read_parquet(files[index])
                mask=allowed[b["ri"]]
                x=feature_matrix(b,columns)[mask]; y=b["y"][mask]
                if len(x):
                    if epoch==-1:
                        state["scaler"].partial_fit(x)
                        state["rows"]+=len(x)
                    else:
                        state["model"].partial_fit(state["scaler"].transform(x),y,classes=np.array([0,1]))
                state.update(epoch=epoch,next_file=index+1)
                temp=out/"checkpoint.tmp.joblib"; joblib.dump(state,temp); temp.replace(checkpoint)
            state.update(epoch=epoch+1,next_file=0)
        n=state["rows"]
        if not n:
            raise ValueError("No pairs in logistic fitting scope")
        temp=path.with_suffix(".tmp.joblib"); joblib.dump(state,temp); temp.replace(path)
        parameter_bound=4*len(columns)+16
    else:
        raise ValueError("Unknown model family")
    if parameter_bound>8_000_000_000:
        raise ValueError("Model exceeds the parameter budget")
    atomic_json(out/"complete.json",{"rows":n,"sha256":digest(path),"kind":kind,"columns":list(columns),
                                    "conservative_scalar_parameter_upper_bound":parameter_bound,"pretrained_models":[]})
    return path


def score_model(root,model_path,split,allowed,name,config):
    out=root/"cache"/"cloud"/"scores"/name
    config_path=model_path.parent/"complete.json"
    model_info=json.loads(config_path.read_text())
    claim_config(out,{"model_sha256":digest(model_path),"split":split,
                      "scope":hashlib.sha256(allowed.tobytes()).hexdigest(),"columns":model_info["columns"]})
    if (out/"complete.json").exists():
        return parquet_files(out)
    model=lgb.Booster(model_file=str(model_path)) if model_info["kind"]=="gbdt" else joblib.load(model_path)
    outputs=[]
    for path in parquet_files(root/"cache"/"cloud"/f"features_{split}"):
        check_headroom(root)
        target=out/path.name
        if not target.exists():
            b=read_parquet(path); selected=allowed[b["ri"]]
            x=feature_matrix(b,model_info["columns"])[selected]
            probabilities=[]
            for start in range(0,len(x),config["prediction_batch"]):
                chunk=x[start:start+config["prediction_batch"]]
                p=(model.predict(chunk,num_threads=config["threads"]) if model_info["kind"]=="gbdt"
                   else model["model"].predict_proba(model["scaler"].transform(chunk))[:,1])
                probabilities.append(p)
            p=np.concatenate(probabilities).astype(np.float32) if probabilities else np.empty(0,dtype=np.float32)
            if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
                raise ValueError("Invalid probabilities")
            write_parquet(target,{"ri":b["ri"][selected],"tid":b["tid"][selected],"y":b["y"][selected],"p":p})
        outputs.append(target.name)
    atomic_json(out/"complete.json",{"files":outputs,"model_sha256":digest(model_path)})
    return parquet_files(out)


def entity_metrics(truth,count,tp,scope):
    t,c,h=truth[scope],count[scope],tp[scope]
    den=t+4*c
    scores=np.divide(5*h,den,out=np.zeros(len(t)),where=den>0)
    scores[t==0]=(c[t==0]==0)
    return {"entities":len(t),"macro_f05":float(scores.mean()) if len(t) else None,
            "link_precision":float(h.sum()/max(c.sum(),1)),"link_recall":float(h.sum()/max(t.sum(),1)),
            "singleton_fp_rate":float((c[t==0]>0).mean()) if (t==0).any() else None,
            "true_links":int(t.sum()),"predicted_links":int(c.sum()),"true_positive_links":int(h.sum())}


def tune_scores(files,truth,scope,thresholds=None,relatives=(0,.5,.8)):
    """Exact counts at a fixed threshold grid, including unretrieved truth links."""
    thresholds=np.array(thresholds if thresholds is not None else list(np.arange(.1,.96,.025))+[.97,.98,.99,.995],dtype=np.float32)
    best=np.zeros(len(truth),dtype=np.float32)
    for path in files:
        b=read_parquet(path,"ri,p"); np.maximum.at(best,b["ri"],b["p"])
    results=[]
    for relative in relatives:
        counts=np.zeros((len(truth),len(thresholds)+1),dtype=np.int32)
        hits=np.zeros_like(counts)
        for path in files:
            b=read_parquet(path,"ri,y,p")
            keep=scope[b["ri"]] & (b["p"]>=relative*best[b["ri"]])
            ri=b["ri"][keep]; bins=np.searchsorted(thresholds,b["p"][keep],side="right")
            np.add.at(counts,(ri,bins),1)
            positive=b["y"][keep]>0
            np.add.at(hits,(ri[positive],bins[positive]),1)
        c=np.zeros(len(truth),dtype=np.int64); h=np.zeros_like(c)
        for index in range(len(thresholds)-1,-1,-1):
            c+=counts[:,index+1]; h+=hits[:,index+1]
            results.append({"threshold":float(thresholds[index]),"relative":relative,
                            **entity_metrics(truth,c,h,scope)})
        del counts,hits
    results.sort(key=lambda r:(r["macro_f05"],r["link_precision"],-r["relative"],r["threshold"]),reverse=True)
    return results


def evaluate_scores(files,ref,scope,policy):
    n=len(ref["ri"]); count=np.zeros(n,dtype=np.int64); tp=np.zeros(n,dtype=np.int64)
    best=np.zeros(n,dtype=np.float32)
    for path in files:
        b=read_parquet(path,"ri,p"); np.maximum.at(best,b["ri"],b["p"])
    for path in files:
        b=read_parquet(path,"ri,y,p")
        keep=(b["p"]>=policy["threshold"]) & (b["p"]>=policy["relative"]*best[b["ri"]])
        np.add.at(count,b["ri"][keep],1); np.add.at(tp,b["ri"][keep & (b["y"]>0)],1)
    truth=ref["truth_count"]
    return {"overall":entity_metrics(truth,count,tp,scope),
            "countries":{str(c):entity_metrics(truth,count,tp,scope & (ref["country_norm"]==c))
                         for c in np.unique(ref["country_norm"][scope])},
            "match_counts":{name:entity_metrics(truth,count,tp,scope & mask) for name,mask in
                            (("singleton",truth==0),("one",truth==1),("multiple",truth>1))}}
