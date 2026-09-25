"""Checkpointed candidate batches; all secondary records see the full S1 index."""

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter
import numpy as np
from src.blocking import build_index,Retriever,selected_pairs
from src.data import connect


def reference_metadata(root,split):
    db=connect(root)
    table=f"ml_{split}_references"
    db.execute(f"""CREATE TABLE IF NOT EXISTS {table} AS SELECT
        (row_number() OVER (ORDER BY entity_id)-1)::INTEGER AS ri,* FROM {split}_source1_norm""")
    if split=="train":
        result=db.execute(f"""SELECT r.*,f.fold,coalesce(t.n,0) AS truth_count,
            hash(r.entity_id)%25=0 AS development_sample
            FROM {table} r JOIN folds f ON r.entity_id=f.sid
            LEFT JOIN (SELECT sid,count(*) n FROM truth_links GROUP BY sid) t ON r.entity_id=t.sid ORDER BY ri""").fetchall()
    else:
        result=db.execute(f"SELECT *, -1 AS fold,0 AS truth_count,false AS development_sample FROM {table} ORDER BY ri").fetchall()
    db.close()
    return result


def run(root,split,batch_size=10000,limit_batches=None,backend="cpu"):
    if batch_size!=10000:
        raise ValueError("Checkpoint format v1 uses fixed batches of 10,000 records")
    start=perf_counter()
    folder=build_index(root,split)
    out=root/"cache"/f"candidates_v1_{split}"
    out.mkdir(exist_ok=True)
    if not (out/"references.pkl").exists():
        refmeta=reference_metadata(root,split)
        with (out/"references.pkl").open("wb") as f:
            pickle.dump(refmeta,f,protocol=5)
        del refmeta
    retriever=Retriever(folder)
    db=connect(root)
    db.execute("SET memory_limit='2GB'")
    db.execute("SET threads=2")
    if split=="train":
        sql="""SELECT s.*,coalesce(r.ri,-1) AS owner FROM
            (SELECT * FROM train_source2_norm UNION ALL SELECT * FROM train_source3_norm) s
            LEFT JOIN truth_links t ON s.entity_id=t.tid
            LEFT JOIN ml_train_references r ON t.sid=r.entity_id ORDER BY s.entity_id"""
    else:
        sql="""SELECT *, -1 AS owner FROM
            (SELECT * FROM test_source2_norm UNION ALL SELECT * FROM test_source3_norm) ORDER BY entity_id"""
    cursor=db.execute(sql)
    batch_index=total_pairs=total_queries=covered=actual=0
    while True:
        batch=cursor.fetchmany(batch_size)
        if not batch or (limit_batches is not None and batch_index>=limit_batches):
            break
        stem=out/f"batch_{batch_index:05d}"
        marker=stem.with_suffix(".json")
        if marker.exists():
            info=json.loads(marker.read_text())
            if (info["first_id"],info["last_id"])!=(batch[0][0],batch[-1][0]):
                raise ValueError("Cached candidate batch does not match source ordering")
        else:
            tick=perf_counter()
            found=retriever.search_selected(batch,backend=backend)
            pairs=list(selected_pairs(found))
            qi=np.array([p[0] for p in pairs],dtype=np.int32)
            ri=np.array([p[1] for p in pairs],dtype=np.int32)
            meta=np.array([p[2] for p in pairs],dtype=np.float32).reshape(-1,6)
            owner=np.array([r[4] for r in batch],dtype=np.int32)
            y=(ri==owner[qi]).astype(np.uint8)
            np.savez_compressed(stem.with_suffix(".npz"),qi=qi,ri=ri,meta=meta,y=y)
            with stem.with_suffix(".pkl").open("wb") as f:
                pickle.dump(batch,f,protocol=5)
            info={"queries":len(batch),"pairs":len(pairs),"true_links":int((owner>=0).sum()),
                  "covered":int(y.sum()),"seconds":perf_counter()-tick,"first_id":batch[0][0],"last_id":batch[-1][0]}
            marker.write_text(json.dumps(info),encoding="utf-8")
        total_queries+=info["queries"]
        total_pairs+=info["pairs"]
        covered+=info["covered"]
        actual+=info["true_links"]
        batch_index+=1
        print(f"{split} batch {batch_index}: {total_queries:,} records, {total_pairs:,} pairs, recall {covered/max(actual,1):.5f}, last batch {info['seconds']:.1f}s",flush=True)
    db.close()
    report={"split":split,"queries":total_queries,"pairs":total_pairs,"covered":covered,
            "true_links":actual,"candidate_recall":covered/actual if actual else None,
            "mean_candidates_per_secondary":total_pairs/max(total_queries,1),"seconds":perf_counter()-start,
            "complete":limit_batches is None}
    (out/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--split",choices=["train","test"],required=True)
    parser.add_argument("--batch-size",type=int,default=10000)
    parser.add_argument("--limit-batches",type=int)
    parser.add_argument("--backend",choices=["cpu","gpu"],default="cpu")
    args=parser.parse_args()
    run(Path(__file__).resolve().parents[1],args.split,args.batch_size,args.limit_batches,args.backend)
