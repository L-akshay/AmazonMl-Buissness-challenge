"""Verify GPU sparse retrieval recall and speed on the established pilot sample."""

import json
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
from src.data import connect
from src.blocking import Retriever,prune_rows,texts,CHANNELS
from src.gpu_sparse import GpuSparse


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/"cache"/"sparse_v1_train"
    refs=json.loads((folder/"reference_ids.json").read_text())
    indices={s:i for i,s in enumerate(refs)}
    db=connect(root)
    records=db.execute("SELECT * FROM (SELECT * FROM train_source2_norm UNION ALL SELECT * FROM train_source3_norm) WHERE hash(entity_id)%2000=0 ORDER BY entity_id").fetchall()
    truth=dict(db.execute("SELECT tid,sid FROM truth_links WHERE hash(tid)%2000=0").fetchall())
    db.close()
    r=Retriever(folder)
    gpu=GpuSparse(r.matrices)
    start=perf_counter()
    result={}
    for c in CHANNELS:
        tick=perf_counter()
        q=prune_rows(r.vectorizers[c].transform(texts(records,c)),CHANNELS[c]["keep"])
        result[c]=gpu.search(q,c)
        print(f"GPU {c}: {perf_counter()-tick:.3f}s",flush=True)
    hits=volume=0
    for i,row in enumerate(records):
        ids=set()
        for c,k in (("name",2),("address",3),("token",6)):
            m=result[c]
            lo,hi=m.indptr[i:i+2]
            ids.update(m.indices[lo:min(hi,lo+k)])
        volume+=len(ids)
        if row[0] in truth:
            hits+=int(indices[truth[row[0]]] in ids)
    report={"queries":len(records),"true_links":len(truth),"recall":hits/len(truth),"mean_pairs":volume/len(records),"seconds":perf_counter()-start,
        "quantization_scale":30000,"note":"Integer accumulation is deterministic; exact-score ties choose smallest reference index."}
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (root/"reports"/"gpu_metrics.json").write_text(json.dumps(report,indent=2))
    (root/"experiments"/f"E08_gpu_{stamp}.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
