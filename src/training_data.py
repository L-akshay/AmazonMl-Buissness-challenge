"""Build development features or a bounded-negative full training matrix."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from src.features import FeatureBuilder,ParallelFeatures,FEATURE_NAMES
from src.evaluation_scope import exclusion_indices


def load_builder(root,split="train"):
    folder=root/"cache"/f"candidates_v1_{split}"
    with (folder/"references.pkl").open("rb") as f:
        refs=pickle.load(f)
    with (root/"cache"/f"sparse_v1_{split}"/"token.pkl").open("rb") as f:
        vectorizer=pickle.load(f)
    builder=FeatureBuilder([r[1:5] for r in refs],vectorizer)
    return refs,builder


def retained_for_fitting(y,ri,qid,metadata):
    # Keep every retrieved positive and real confusing negatives, plus a
    # reproducible 10% of other negatives. Validation scores ALL candidates.
    hashed=(ri.astype(np.uint64)*2654435761+qid.astype(np.uint64)*2246822519)%10
    hard=(metadata[:,:3].max(axis=1)>=.85)&((metadata[:,3:]==1).any(axis=1))
    return (y>0)|hard|(hashed==0)


def build(root,mode):
    folder=root/"cache"/"candidates_v1_train"
    summary=json.loads((folder/"summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Full candidate generation must complete before validation")
    signature=hashlib.sha256(b"".join((root/"src"/name).read_bytes() for name in ("training_data.py","features.py","normalize.py","evaluation_scope.py"))).hexdigest()
    target=root/"cache"/mode
    target.mkdir(exist_ok=True)
    if (target/"complete.json").exists():
        saved=json.loads((target/"complete.json").read_text())
        if saved.get("code_sha256")!=signature:
            raise ValueError("Completed features use a different implementation; use a fresh feature cache")
        print(f"Reusing completed {mode} features",flush=True)
        return
    refs,builder=load_builder(root)
    engine=ParallelFeatures(builder)
    sample=np.array([r[7] for r in refs],dtype=bool)
    folds=np.array([r[5] for r in refs],dtype=np.int8)
    excluded,exclusion_report=exclusion_indices(root)
    withheld=excluded[(folds[excluded]==4)&sample[excluded]]
    sample[withheld]=False
    exclusion_report={**exclusion_report,"removed_from_reserved_sample":len(withheld),
        "reserved_sample_entities":int((sample&(folds==4)).sum())}
    meta={"ids":[r[1] for r in refs],"country":[r[4] for r in refs],
          "fold":[int(r[5]) for r in refs],"truth_count":[int(r[6]) for r in refs],
          "sample":sample,"feature_names":FEATURE_NAMES,"holdout_exclusions":exclusion_report}
    with (root/"cache"/"training_metadata.pkl").open("wb") as f:
        pickle.dump(meta,f,protocol=5)
    files=sorted(folder.glob("batch_*.npz"))
    total=0
    for file in files:
        batch=np.load(file)
        qi,ri,y=batch["qi"],batch["ri"],batch["y"]
        qid=qi.astype(np.uint32)+int(file.stem.split("_")[1])*10000
        mask=sample[ri] if mode=="development" else retained_for_fitting(y,ri,qid,batch["meta"])
        total+=int(mask.sum())
    progress=target/"progress.json"
    state=json.loads(progress.read_text()) if progress.exists() else {"offset":0,"next_file":0,"rows":total,"code_sha256":signature}
    if state["rows"]!=total or state["code_sha256"]!=signature:
        raise ValueError("Partial feature cache does not match this run")
    access="r+" if progress.exists() else "w+"
    x=np.lib.format.open_memmap(target/"x.npy",mode=access,dtype=np.float32,shape=(total,len(FEATURE_NAMES)))
    ys=np.lib.format.open_memmap(target/"y.npy",mode=access,dtype=np.uint8,shape=(total,))
    groups=np.lib.format.open_memmap(target/"groups.npy",mode=access,dtype=np.int32,shape=(total,))
    qids=np.lib.format.open_memmap(target/"qids.npy",mode=access,dtype=np.uint32,shape=(total,))
    fit=np.lib.format.open_memmap(target/"fit.npy",mode=access,dtype=bool,shape=(total,))
    offset=state["offset"]
    for index,file in enumerate(files):
        if index<state["next_file"]:
            continue
        batch=np.load(file)
        qi,ri,y,metadata=batch["qi"],batch["ri"],batch["y"],batch["meta"]
        qid=qi.astype(np.uint32)+int(file.stem.split("_")[1])*10000
        selected=retained_for_fitting(y,ri,qid,metadata)
        mask=sample[ri] if mode=="development" else selected
        with file.with_suffix(".pkl").open("rb") as f:
            records=pickle.load(f)
        n=int(mask.sum())
        x[offset:offset+n]=engine.transform(records,qi[mask],ri[mask],metadata[mask])
        ys[offset:offset+n]=y[mask]
        groups[offset:offset+n]=ri[mask]
        qids[offset:offset+n]=qid[mask]
        fit[offset:offset+n]=selected[mask]
        offset+=n
        if int(file.stem.split("_")[1])%25==0:
            print(f"{mode} features: {offset:,}/{total:,}",flush=True)
            for a in (x,ys,groups,qids,fit):
                a.flush()
            checkpoint={"offset":offset,"next_file":index+1,"rows":total,"code_sha256":signature}
            temporary=progress.with_suffix(".tmp")
            temporary.write_text(json.dumps(checkpoint),encoding="utf-8")
            temporary.replace(progress)
    for a in (x,ys,groups,qids,fit):
        a.flush()
    engine.close()
    (target/"complete.json").write_text(json.dumps({"rows":total,"features":len(FEATURE_NAMES),"mode":mode,"sample_entities":int(sample.sum()),"code_sha256":signature}),encoding="utf-8")
    print(f"Saved {mode}: {total:,} pairs, {len(FEATURE_NAMES)} features",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["development","full"],required=True)
    build(Path(__file__).resolve().parents[1],parser.parse_args().mode)
