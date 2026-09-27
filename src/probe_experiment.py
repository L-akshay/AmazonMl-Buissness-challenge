"""Compare rare-term candidate probing to the full sparse pilot, on identical rows."""

import json
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
from src.data import connect
from src.blocking import Retriever


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
    experiments=[]
    for terms in (2,3,5):
        start=perf_counter()
        hits=volume=0
        channel_hits={c:0 for c in r.channels}
        for offset in range(0,len(records),2000):
            batch=records[offset:offset+2000]
            found=r.search_fast(batch,probe_terms=terms)
            for i,row in enumerate(batch):
                candidates=set()
                for c,k in (("name",2),("address",3),("token",6)):
                    m=found[c]
                    lo,hi=m.indptr[i:i+2]
                    ids=set(m.indices[lo:min(hi,lo+k)])
                    candidates.update(ids)
                    if row[0] in truth:
                        channel_hits[c]+=int(indices[truth[row[0]]] in ids)
                volume+=len(candidates)
                if row[0] in truth:
                    hits+=int(indices[truth[row[0]]] in candidates)
        record={"probe_terms":terms,"pool_size":40,"recall":hits/len(truth),"mean_pairs":volume/len(records),
                "seconds":perf_counter()-start,"channel_recall":{c:n/len(truth) for c,n in channel_hits.items()},
                "queries":len(records),"true_links":len(truth)}
        experiments.append(record)
        print(json.dumps(record),flush=True)
    report={"scope":"Same uniform secondary sample and complete S1 index as retrieval_metrics.json", "experiments":experiments}
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (root/"reports"/"probe_metrics.json").write_text(json.dumps(report,indent=2))
    (root/"experiments"/f"E08_probe_{stamp}.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
