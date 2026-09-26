"""Full-population retrieval and reusable compressed feature shards for remote runs."""

import json
import pickle
import time
import numpy as np
from scipy import sparse
from src.blocking import build_index, Retriever, fused_pairs
from src.cloud_store import atomic_json, claim_config, check_headroom, read_parquet, write_parquet, parquet_files
from src.data import connect
from src.features import FeatureBuilder, ParallelFeatures, FEATURE_NAMES
from src.generate_candidates import reference_metadata
from src.streaming import label_lookup, secondary_batches, encode_secondary


def complete_candidates(folder):
    """Publish a global manifest only after every disjoint batch is present."""
    shape=json.loads((folder/"shape.json").read_text())
    files=[]; totals={"queries":0,"pairs":0,"covered":0,"true_links":0,"legacy_token_batches":0}
    for index in range(shape["batch_count"]):
        stem=folder/f"batch_{index:05d}"
        if not all(p.exists() for p in (stem.with_suffix(".json"),stem.with_suffix(".parquet"),folder/(stem.name+"_records.parquet"))):
            return False
        info=json.loads(stem.with_suffix(".json").read_text())
        files.append(stem.with_suffix(".parquet").name)
        for name in totals:
            totals[name]+=info[name]
    if totals["queries"]!=shape["query_count"]:
        raise ValueError("Partition union does not cover all secondary records")
    atomic_json(folder/"complete.json",{"files":files,**totals,"policy":"all three top-6 lists, full union"})
    return True


def complete_features(source,out):
    if not (source/"complete.json").exists():
        return False
    files=parquet_files(source); total=0
    for path in files:
        target=out/path.name
        if not target.exists() or not target.with_suffix(".json").exists():
            return False
        total+=json.loads(target.with_suffix(".json").read_text())["rows"]
    expected=json.loads((source/"complete.json").read_text())["pairs"]
    if total!=expected:
        raise ValueError("Feature cache does not cover all candidates")
    atomic_json(out/"complete.json",{"files":[p.name for p in files],"rows":total,"feature_count":len(FEATURE_NAMES)})
    return True


def assemble(root,split):
    source=root/"cache/cloud"/f"candidates_{split}"
    if not complete_candidates(source):
        raise ValueError("Some retrieval partitions are missing")
    if not complete_features(source,root/"cache/cloud"/f"features_{split}"):
        raise ValueError("Some feature partitions are missing")
    if split=="train":
        coverage(root)


def references(root, split):
    path=root/"cache"/"cloud"/f"references_{split}.parquet"
    if not path.exists():
        rows=reference_metadata(root,split)
        names=("ri","entity_id","name_norm","address_norm","country_norm","fold","truth_count")
        arrays={name:np.array([r[i] for r in rows],dtype=np.int32 if i in (0,5,6) else object)
                for i,name in enumerate(names)}
        write_parquet(path,arrays)
    return path


def legacy_token(root,split,index,records,nref,batch_size):
    """Reuse complete token-6 retrieval from the earlier cache when boundaries match."""
    if batch_size!=10000:
        return None
    stem=root/"cache"/f"candidates_v1_{split}"/f"batch_{index:05d}"
    if not stem.with_suffix(".json").exists():
        return None
    info=json.loads(stem.with_suffix(".json").read_text())
    if (info["first_id"],info["last_id"])!=(records[0][0],records[-1][0]):
        raise ValueError("Legacy cached source ordering differs")
    with np.load(stem.with_suffix(".npz")) as b:
        mask=b["meta"][:,2]>0
        qi,ri,meta=b["qi"][mask],b["ri"][mask],b["meta"][mask]
    order=np.lexsort((meta[:,5],qi))
    qi,ri,meta=qi[order],ri[order],meta[order]
    counts=np.bincount(qi,minlength=len(records))
    return sparse.csr_matrix((meta[:,2],ri,np.r_[0,np.cumsum(counts)]),shape=(len(records),nref))


def retrieve(root,split,config,fingerprint):
    out=root/"cache"/"cloud"/f"candidates_{split}"
    claim_config(out,{"fingerprint":fingerprint,"batch_size":config["batch_size"],
                      "policy":"full6","backend":config["backend"]})
    if (out/"complete.json").exists():
        parquet_files(out)
        return
    references(root,split)
    folder=build_index(root,split)
    db=connect(root)
    query_count=sum(db.execute(f"SELECT count(*) FROM {split}_source{s}_norm").fetchone()[0] for s in (2,3))
    shape={"query_count":query_count,"batch_count":(query_count+config["batch_size"]-1)//config["batch_size"]}
    atomic_json(out/"shape.json",shape)
    labels=label_lookup(db,root) if split=="train" else None
    retriever=Retriever(folder,retain_forward=False)
    nref=len(json.loads((folder/"reference_ids.json").read_text()))
    files=[]; totals={"queries":0,"pairs":0,"covered":0,"true_links":0,"legacy_token_batches":0}
    for index,records in enumerate(secondary_batches(db,split,config["batch_size"],labels)):
        if index<config.get("batch_start",0):
            continue
        if config.get("batch_stop") is not None and index>=config["batch_stop"]:
            break
        check_headroom(root)
        stem=out/f"batch_{index:05d}"
        marker=stem.with_suffix(".json")
        if marker.exists():
            info=json.loads(marker.read_text())
            if (info["first_id"],info["last_id"])!=(records[0][0],records[-1][0]):
                raise ValueError("Source ordering changed while resuming")
            if not stem.with_suffix(".parquet").exists() or not (out/(stem.name+"_records.parquet")).exists():
                raise ValueError("Completed retrieval shard is missing")
        else:
            tick=time.perf_counter()
            token=legacy_token(root,split,index,records,nref,config["batch_size"])
            found=retriever.search_selected(records,backend=config["backend"],policy="full6",token_override=token)
            pairs=list(fused_pairs(found,max_candidates=18,min_candidates=18,ratio=0))
            qi=np.array([p[0] for p in pairs],dtype=np.int32)
            ri=np.array([p[1] for p in pairs],dtype=np.int32)
            meta=np.array([p[2] for p in pairs],dtype=np.float32).reshape(-1,6)
            owners=np.array([r[4] for r in records],dtype=np.int32)
            y=(ri==owners[qi]).astype(np.uint8)
            write_parquet(stem.with_suffix(".parquet"),{"qi":qi,"ri":ri,"y":y,
                           **{f"m{i}":meta[:,i] for i in range(6)}})
            write_parquet(out/(stem.name+"_records.parquet"),{
                name:np.array([r[i] for r in records],dtype=np.int32 if i==4 else object)
                for i,name in enumerate(("entity_id","name","address","country","owner"))})
            info={"first_id":records[0][0],"last_id":records[-1][0],"queries":len(records),
                  "pairs":len(ri),"covered":int(y.sum()),"true_links":int((owners>=0).sum()),
                  "legacy_token_batches":int(token is not None),"seconds":time.perf_counter()-tick}
            atomic_json(marker,info)
        files.append(stem.with_suffix(".parquet").name)
        for name in totals:
            totals[name]+=info[name]
        print(f"{split} retrieval {index+1}: {totals['queries']:,} records, {totals['pairs']:,} pairs",flush=True)
    db.close()
    complete_candidates(out)


def features(root,split,config,fingerprint):
    source=root/"cache"/"cloud"/f"candidates_{split}"
    if config.get("batch_stop") is not None:
        shape=json.loads((source/"shape.json").read_text())
        files=[source/f"batch_{i:05d}.parquet" for i in range(config.get("batch_start",0),min(config["batch_stop"],shape["batch_count"]))]
        if not files or any(not p.exists() or not p.with_suffix(".json").exists() for p in files):
            raise ValueError("Retrieve the complete requested partition before computing features")
    else:
        files=parquet_files(source)
    out=root/"cache"/"cloud"/f"features_{split}"
    claim_config(out,{"fingerprint":fingerprint,"features":FEATURE_NAMES,"selection":"every candidate"})
    if (out/"complete.json").exists():
        parquet_files(out)
        return
    ref=read_parquet(references(root,split))
    if not np.array_equal(ref["ri"],np.arange(len(ref["ri"]))):
        raise ValueError("Reference order changed")
    records=list(zip(ref["entity_id"],ref["name_norm"],ref["address_norm"],ref["country_norm"]))
    with (root/"cache"/f"sparse_v1_{split}"/"token.pkl").open("rb") as f:
        vectorizer=pickle.load(f)
    builder=FeatureBuilder(records,vectorizer)
    engine=ParallelFeatures(builder,workers=config["feature_workers"],chunk_size=config["feature_chunk"])
    total=0
    try:
        for index,path in enumerate(files):
            check_headroom(root)
            target=out/path.name
            marker=target.with_suffix(".json")
            if marker.exists():
                if not target.exists():
                    raise ValueError("Completed feature shard is missing")
                info=json.loads(marker.read_text())
            else:
                batch=read_parquet(path)
                text=read_parquet(path.parent/(path.stem+"_records.parquet"))
                queries=list(zip(text["entity_id"],text["name"],text["address"],text["country"]))
                meta=np.column_stack([batch[f"m{i}"] for i in range(6)])
                x=engine.transform(queries,batch["qi"],batch["ri"],meta)
                if not np.isfinite(x).all():
                    raise ValueError("Nonfinite features")
                codes=np.array([encode_secondary(r[0]) for r in queries],dtype=np.uint64)
                write_parquet(target,{"ri":batch["ri"],"y":batch["y"],"tid":codes[batch["qi"]],
                    "fold":ref["fold"][batch["ri"]].astype(np.int8),
                    **{f"f{i}":x[:,i] for i in range(len(FEATURE_NAMES))}})
                info={"rows":len(x),"bytes":target.stat().st_size}
                atomic_json(marker,info)
            total+=info["rows"]
            print(f"{split} features {index+1}/{len(files)}: {total:,} pairs",flush=True)
    finally:
        engine.close()
    complete_features(source,out)


def coverage(root):
    ref=read_parquet(references(root,"train"),"ri,country_norm,truth_count")
    n=len(ref["ri"]); volume=np.zeros(n,dtype=np.int32); hits=np.zeros(n,dtype=np.int32)
    for path in parquet_files(root/"cache"/"cloud"/"candidates_train"):
        b=read_parquet(path,"ri,y")
        np.add.at(volume,b["ri"],1); np.add.at(hits,b["ri"][b["y"]>0],1)
    truth=ref["truth_count"]
    if np.any(hits>truth):
        raise ValueError("Retrieved positives exceed truth")
    def stats(mask):
        t,h,v=truth[mask],hits[mask],volume[mask]
        return {"entities":int(mask.sum()),"candidate_recall":float(h.sum()/max(t.sum(),1)),
                "all_true_matches_covered":float((t==h).mean()),"mean_candidates":float(v.mean()),
                "median_candidates":float(np.median(v)),"p95_candidates":float(np.quantile(v,.95)),
                "oracle_macro_f05":float(np.divide(5*h,t+4*h,out=np.ones(len(t)),where=(t+4*h)>0).mean())}
    report={"scope":"Full train population, all three top-6 lists, full union",
            "overall":stats(np.ones(n,dtype=bool)),
            "countries":{str(c):stats(ref["country_norm"]==c) for c in np.unique(ref["country_norm"])}}
    atomic_json(root/"reports"/"cloud_candidate_coverage.json",report)
    return report
