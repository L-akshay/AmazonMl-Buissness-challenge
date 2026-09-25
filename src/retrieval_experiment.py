"""Measure reverse candidate retrieval on a uniform sample, with full S1 competition."""

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from src.data import connect
from src.blocking import CHANNELS,build_index,Retriever,fused_pairs


def run(root, modulus):
    start=perf_counter()
    folder=build_index(root,"train")
    refs=json.loads((folder/"reference_ids.json").read_text(encoding="utf-8"))
    ref_index={s:i for i,s in enumerate(refs)}
    db=connect(root)
    records=db.execute(f"""SELECT * FROM (
        SELECT * FROM train_source2_norm UNION ALL SELECT * FROM train_source3_norm)
        WHERE hash(entity_id)%{int(modulus)}=0 ORDER BY entity_id""").fetchall()
    truth=dict(db.execute(f"SELECT tid,sid FROM truth_links WHERE hash(tid)%{int(modulus)}=0").fetchall())
    db.close()
    retriever=Retriever(folder)
    counts={c:{str(k):0 for k in (1,2,4,6)} for c in CHANNELS}
    union_hits=adaptive_hits=volume=0
    policy_counts={"name2_address3_token6":{"hits":0,"pairs":0},"name4_address4_token6":{"hits":0,"pairs":0},"union6":{"hits":0,"pairs":0}}
    search_seconds=0
    channel_seconds={c:0.0 for c in CHANNELS}
    gates={str(t):{"hits":0,"pairs":0,"rescue_queries":0} for t in (.6,.7,.8,.9)}
    country={}
    for offset in range(0,len(records),2000):
        batch=records[offset:offset+2000]
        tick=perf_counter()
        results=retriever.search(batch)
        search_seconds+=perf_counter()-tick
        for c in CHANNELS:
            channel_seconds[c]+=retriever.last_timing[c]
        full={i:set() for i in range(len(batch))}
        adaptive={i:set() for i in range(len(batch))}
        for i,j,_ in fused_pairs(results):
            adaptive[i].add(j)
            volume+=1
        for i,row in enumerate(batch):
            token=results["token"]
            lo,hi=token.indptr[i:i+2]
            ts=token.data[lo:hi]
            for threshold,d in gates.items():
                strong=len(ts)>0 and ts[0]>=float(threshold) and (len(ts)<2 or ts[0]-ts[1]>=.15)
                ids=set(token.indices[lo:hi])
                if not strong:
                    d["rescue_queries"]+=1
                    for c,k in (("name",2),("address",3)):
                        m=results[c]
                        a,b=m.indptr[i:i+2]
                        ids.update(m.indices[a:min(b,a+k)])
                d["pairs"]+=len(ids)
                if row[0] in truth:
                    d["hits"]+=int(ref_index[truth[row[0]]] in ids)
            for pname,budget in [("name2_address3_token6",(2,3,6)),("name4_address4_token6",(4,4,6)),("union6",(6,6,6))]:
                ids=set()
                for (c,m),k in zip(results.items(),budget):
                    lo,hi=m.indptr[i:i+2]
                    ids.update(m.indices[lo:min(hi,lo+k)])
                policy_counts[pname]["pairs"]+=len(ids)
                if row[0] in truth:
                    policy_counts[pname]["hits"]+=int(ref_index[truth[row[0]]] in ids)
            owner=truth.get(row[0])
            if owner is None:
                continue
            target=ref_index[owner]
            for c,matrix in results.items():
                lo,hi=matrix.indptr[i:i+2]
                hits=list(matrix.indices[lo:hi])
                full[i].update(hits)
                for k in counts[c]:
                    counts[c][k]+=int(target in hits[:int(k)])
            union_hits+=int(target in full[i])
            adaptive_hits+=int(target in adaptive[i])
            bucket=country.setdefault(row[3],{"true_links":0,"covered":0})
            bucket["true_links"]+=1
            bucket["covered"]+=int(target in adaptive[i])
        print(f"Retrieval pilot: {min(offset+2000,len(records)):,}/{len(records):,}",flush=True)
    report={"experiment":"E02-E08","sample":"uniform secondary hash sample against ALL training S1 records",
        "modulus":modulus,"query_records":len(records),"sample_true_links":len(truth),"references":len(refs),
        "channel_recall_at_k":{c:{k:n/len(truth) for k,n in d.items()} for c,d in counts.items()},
        "union_recall":union_hits/len(truth),"adaptive_recall":adaptive_hits/len(truth),
        "mean_candidates_per_secondary":volume/len(records),"country":country,
        "search_seconds":search_seconds,"queries_per_second":len(records)/search_seconds,
        "channel_seconds":channel_seconds,
        "gated_rescue":{k:{"recall":v["hits"]/len(truth),"mean_pairs":v["pairs"]/len(records),"rescue_fraction":v["rescue_queries"]/len(records)} for k,v in gates.items()},
        "policies":{k:{"recall":v["hits"]/len(truth),"mean_pairs":v["pairs"]/len(records)} for k,v in policy_counts.items()},
        "runtime_seconds":perf_counter()-start,
        "scope":"Retrieval-only sample estimate. Not full S1 entity recall or matcher validation.",
        "unsupervised_fit":"S1 text only within each split; no labels in TF-IDF or index fitting."}
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for path in [root/"experiments"/f"E02_{stamp}.json",root/"reports"/"retrieval_metrics.json"]:
        path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--modulus",type=int,default=1000)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    run(args.root,args.modulus)
