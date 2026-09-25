"""Full-population candidate coverage and volume from checkpointed retrieval."""

from datetime import datetime,timezone
import json
import pickle
from pathlib import Path
import numpy as np


def run(root):
    folder=root/"cache"/"candidates_v1_train"
    summary=json.loads((folder/"summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Retrieval is incomplete")
    with (folder/"references.pkl").open("rb") as f:
        refs=pickle.load(f)
    truth=np.array([r[6] for r in refs],dtype=np.int32)
    country=np.array([r[4] for r in refs])
    del refs
    covered=np.zeros(len(truth),dtype=np.int32)
    volume=np.zeros(len(truth),dtype=np.int32)
    incremental={c:{"pairs":0,"true_links":0} for c in ("token","name","address")}
    for path in sorted(folder.glob("batch_*.npz")):
        with np.load(path) as batch:
            ri,y,meta=batch["ri"],batch["y"],batch["meta"]
        np.add.at(volume,ri,1)
        np.add.at(covered,ri[y>0],1)
        prior=np.zeros(len(ri),dtype=bool)
        for channel,index in (("token",2),("name",0),("address",1)):
            present=meta[:,index]>0
            added=present&~prior
            incremental[channel]["pairs"]+=int(added.sum())
            incremental[channel]["true_links"]+=int((added&(y>0)).sum())
            prior|=present
    def metrics(mask):
        t,c,v=truth[mask],covered[mask],volume[mask]
        oracle=np.divide(5.0*c,t+4*c,out=np.ones(len(t)),where=(t+4*c)>0)
        return {"entities":int(mask.sum()),"link_recall":float(c.sum()/t.sum()),
                "oracle_macro_f05_ceiling":float(oracle.mean()),
                "all_true_matches_covered":float((t==c).mean()),
                "nonsingleton_all_matches_covered":float((t[t>0]==c[t>0]).mean()),
                "mean_candidates":float(v.mean()),"median_candidates":float(np.median(v)),
                "p95_candidates":float(np.quantile(v,.95)),"max_candidates":int(v.max())}
    report={"scope":"All training S1 and all training S2/S3; frozen token-6 with name-2/address-3 rescue policy.",
            "overall":metrics(np.ones(len(truth),dtype=bool)),
            "by_country":{c:metrics(country==c) for c in sorted(set(country))},
            "incremental_in_order":incremental,"queries":summary["queries"],"pairs":summary["pairs"],
            "reduction_ratio":1-summary["pairs"]/(len(truth)*summary["queries"])}
    if int(covered.sum())!=summary["covered"] or int(volume.sum())!=summary["pairs"]:
        raise ValueError("Checkpoint totals do not match retrieval summary")
    if int(truth.sum())!=summary["true_links"] or (covered>truth).any():
        raise ValueError("Candidate labels do not agree with the complete ground truth")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for path in (root/"reports"/"full_candidate_metrics.json",root/"experiments"/f"E02_full_{stamp}.json"):
        path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    run(Path(__file__).resolve().parents[1])
