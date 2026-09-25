"""Score every retained test candidate and export deterministic submission TSVs."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from src.data import connect
from src.features import FEATURE_NAMES,ParallelFeatures
from src.streaming import encode_secondary
from src.training_data import load_builder


def score(root):
    model_path=root/"cache"/"model"/"matcher.pkl"
    signature=hashlib.sha256(model_path.read_bytes()).hexdigest()
    config=json.loads((model_path.parent/"config.json").read_text())
    if config["features"]!=FEATURE_NAMES:
        raise ValueError("Saved model feature schema differs from this code")
    if config["feature_code_sha256"]!=hashlib.sha256((root/"src"/"features.py").read_bytes()).hexdigest():
        raise ValueError("Saved model feature implementation differs from this code")
    inputs=root/"cache"/"candidates_v1_test"
    summary=json.loads((inputs/"summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Test retrieval must finish before scoring")
    out=root/"cache"/"test_scores"
    out.mkdir(exist_ok=True)
    marker=out/"model.json"
    if marker.exists() and json.loads(marker.read_text())["sha256"]!=signature:
        raise ValueError("Scores belong to a different model; use a fresh test_scores directory")
    marker.write_text(json.dumps({"sha256":signature}),encoding="utf-8")
    with model_path.open("rb") as f:
        model=pickle.load(f)
    refs,builder=load_builder(root,"test")
    engine=ParallelFeatures(builder)
    count=0
    for path in sorted(inputs.glob("batch_*.npz")):
        target=out/path.name
        with np.load(path) as batch:
            qi,ri,meta=batch["qi"],batch["ri"],batch["meta"]
        if target.exists():
            with np.load(target) as saved:
                if not np.array_equal(saved["ri"],ri):
                    raise ValueError("Score checkpoint does not match candidates")
        else:
            with path.with_suffix(".pkl").open("rb") as f:
                records=pickle.load(f)
            x=engine.transform(records,qi,ri,meta)
            p=model.predict_proba(x)[:,1].astype(np.float32)
            if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
                raise ValueError("Invalid matcher probability")
            codes=np.array([encode_secondary(r[0]) for r in records],dtype=np.uint64)
            temp=target.with_suffix(".tmp.npz")
            np.savez_compressed(temp,ri=ri,tid=codes[qi],p=p)
            temp.replace(target)
        count+=len(ri)
        if int(path.stem.split("_")[1])%25==0:
            print(f"Scored {count:,}/{summary['pairs']:,} test pairs",flush=True)
    if count!=summary["pairs"]:
        raise ValueError("Not every candidate was scored")
    engine.close()
    (out/"complete.json").write_text(json.dumps({"pairs":count,"sha256":signature,"references":len(refs)}),encoding="utf-8")


def decode_secondary(value):
    value=int(value)
    source=value>>32
    if source not in (2,3):
        raise ValueError("Invalid secondary source")
    return f"S{source}-{value&0xffffffff}"


def write_grouped_outputs(ids,edges,best,policy,out):
    """Edges must be unique and sorted by (reference index, numeric target ID)."""
    out.mkdir(exist_ok=True,parents=True)
    iterator=iter(edges)
    edge=next(iterator,None)
    counts={"entities":len(ids),"candidate_pairs":0,"matched_pairs":0,"predicted_singletons":0}
    with (out/"matching_results.tsv").open("w",encoding="utf-8",newline="") as matching, \
         (out/"candidate_pairs.tsv").open("w",encoding="utf-8",newline="") as candidates:
        matching.write("source1_entity_id\tmatched_entity_ids\n")
        candidates.write("source1_entity_id\tcandidate_entity_ids\n")
        for ri,sid in enumerate(ids):
            proposed,accepted=[],[]
            previous=-1
            while edge is not None and edge[0]==ri:
                _,tid,p=edge
                if tid<=previous:
                    raise ValueError("Duplicate or unsorted candidate edge")
                if not np.isfinite(p) or not 0<=p<=1:
                    raise ValueError("Invalid score in export")
                previous=tid
                name=decode_secondary(tid)
                proposed.append(name)
                if p>=policy["threshold"] and p>=best[ri]*policy["relative"]:
                    accepted.append(name)
                edge=next(iterator,None)
            # The accepted list is constructed only from the scored candidates.
            candidates.write(sid+"\t"+",".join(proposed)+"\n")
            matching.write(sid+"\t"+",".join(accepted)+"\n")
            counts["candidate_pairs"]+=len(proposed)
            counts["matched_pairs"]+=len(accepted)
            counts["predicted_singletons"]+=int(not accepted)
            if edge is not None and edge[0]<ri:
                raise ValueError("Candidate reference ordering is invalid")
        if edge is not None:
            raise ValueError("Candidate refers to an unknown S1 index")
    return counts


def export(root):
    inputs=root/"cache"/"test_scores"
    complete=json.loads((inputs/"complete.json").read_text())
    config=json.loads((root/"cache"/"model"/"config.json").read_text())
    ids=json.loads((root/"cache"/"sparse_v1_test"/"reference_ids.json").read_text(encoding="utf-8"))
    if len(ids)!=complete["references"] or ids!=sorted(set(ids)):
        raise ValueError("Reference ID coverage/order mismatch")
    db=connect(root)
    db.execute("SET memory_limit='2GB'")
    db.execute("SET threads=1")
    db.execute("CREATE OR REPLACE TABLE test_scored_edges(ri INTEGER,tid UBIGINT,p FLOAT)")
    best=np.zeros(len(ids),dtype=np.float32)
    for path in sorted(inputs.glob("batch_*.npz")):
        with np.load(path) as batch:
            values={n:batch[n] for n in ("ri","tid","p")}
        if (values["ri"]<0).any() or (values["ri"]>=len(ids)).any():
            raise ValueError("Unknown reference index in scores")
        np.maximum.at(best,values["ri"],values["p"])
        db.register("score_batch",values)
        db.execute("INSERT INTO test_scored_edges SELECT * FROM score_batch")
        db.unregister("score_batch")
    cursor=db.execute("SELECT ri,tid,p FROM test_scored_edges ORDER BY ri,tid")
    def edges():
        while True:
            batch=cursor.fetchmany(50000)
            if not batch:
                return
            yield from batch
    result=write_grouped_outputs(ids,edges(),best,config["policy"],root/"output"/"trained")
    db.close()
    if result["candidate_pairs"]!=complete["pairs"]:
        raise ValueError("Export differs from the exact scored candidate set")
    result.update({"policy":config["policy"],"model_sha256":complete["sha256"]})
    (root/"reports"/"trained_output.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["score","export"],required=True)
    args=parser.parse_args()
    (score if args.stage=="score" else export)(Path(__file__).resolve().parents[1])
