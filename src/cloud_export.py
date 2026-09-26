"""Stream one shared candidate file and validated OOF-selected submission variants."""

from contextlib import ExitStack
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from src.cloud_store import atomic_json, read_parquet, parquet_files, digest
from src.cloud_features import references
from src.data import connect
from src.predict import decode_secondary
from src.validate_output import validate_rows


def write_variants(ids,edges,best,policies,out):
    out.mkdir(parents=True,exist_ok=True)
    counts={p["name"]:{"entities":0,"candidate_pairs":0,"matched_pairs":0,"predicted_singletons":0} for p in policies}
    with ExitStack() as stack:
        candidate=stack.enter_context((out/"candidate_pairs.tsv").open("w",encoding="utf-8",newline=""))
        candidate.write("source1_entity_id\tcandidate_entity_ids\n")
        matching={}
        for policy in policies:
            folder=out/policy["name"]; folder.mkdir(exist_ok=True)
            f=stack.enter_context((folder/"matching_results.tsv").open("w",encoding="utf-8",newline=""))
            f.write("source1_entity_id\tmatched_entity_ids\n"); matching[policy["name"]]=f
        iterator=iter(edges); edge=next(iterator,None)
        for ri,sid in enumerate(ids):
            if edge is not None and edge[0]<ri:
                raise ValueError("Unknown or unsorted reference index")
            proposed=[]; accepted={p["name"]:[] for p in policies}; previous=-1
            while edge is not None and edge[0]==ri:
                _,tid,prob=edge
                if tid<=previous or not np.isfinite(prob) or not 0<=prob<=1:
                    raise ValueError("Duplicate/invalid scored edge")
                previous=tid; name=decode_secondary(tid); proposed.append(name)
                for p in policies:
                    if prob>=p["threshold"] and prob>=p["relative"]*best[ri]:
                        accepted[p["name"]].append(name)
                edge=next(iterator,None)
            candidate.write(sid+"\t"+",".join(proposed)+"\n")
            for p in policies:
                name=p["name"]; found=accepted[name]
                matching[name].write(sid+"\t"+",".join(found)+"\n")
                count=counts[name]; count["entities"]+=1; count["candidate_pairs"]+=len(proposed)
                count["matched_pairs"]+=len(found); count["predicted_singletons"]+=int(not found)
        if edge is not None:
            raise ValueError("Unknown reference index after final S1")
    return counts


def export(root,config):
    final=json.loads((root/"cache"/"cloud"/"final.json").read_text())
    policies=final["variants"]
    files=parquet_files(root/"cache"/"cloud"/"scores"/"test")
    ids=read_parquet(references(root,"test"),"entity_id")["entity_id"]
    best=np.zeros(len(ids),dtype=np.float32)
    total=0
    for path in files:
        b=read_parquet(path,"ri,p"); np.maximum.at(best,b["ri"],b["p"]); total+=len(b["ri"])
    expected=json.loads((root/"cache"/"cloud"/"features_test"/"complete.json").read_text())["rows"]
    if total!=expected:
        raise ValueError("Not all test candidates were scored")
    db=connect(root)
    cursor=db.execute("SELECT ri,tid,p FROM read_parquet(?) ORDER BY ri,tid",[[str(p) for p in files]])
    def edges():
        while True:
            rows=cursor.fetchmany(config["prediction_batch"])
            if not rows:
                return
            yield from rows
    out=root/"output"/"cloud"
    counts=write_variants(ids,edges(),best,policies,out)
    valid=db.execute("""SELECT ((substr(entity_id,2,1)::UBIGINT << 32)+substr(entity_id,4)::UBIGINT) code
        FROM (SELECT entity_id FROM test_source2 UNION ALL SELECT entity_id FROM test_source3) ORDER BY code""").fetchnumpy()["code"]
    db.close()
    reports={}
    for policy in policies:
        name=policy["name"]; matching=out/name/"matching_results.tsv"
        with matching.open(encoding="utf-8",newline="") as m, (out/"candidate_pairs.tsv").open(encoding="utf-8",newline="") as c:
            result=validate_rows(csv.reader(m,delimiter="\t"),csv.reader(c,delimiter="\t"),iter(ids),valid)
        if result!=counts[name]:
            raise ValueError("Export totals differ from validation")
        work=root/"cache"/"cloud"/"organizer_validation"; work.mkdir(exist_ok=True)
        run=subprocess.run([sys.executable,"-X","utf8",str(root/"utils"/"validate_submission.py"),
            "--matching",str(matching),"--test-dir",str(root/"dataset"/"test"),"--check-ids"],
            cwd=work,capture_output=True,text=True,encoding="utf-8")
        log=run.stdout+run.stderr
        (out/name/"organizer_validator.txt").write_text(log,encoding="utf-8")
        if run.returncode or "PASS" not in log:
            raise ValueError(f"Organizer validation failed for {name}")
        reports[name]={**result,"sha256":digest(matching),"organizer":"PASS",
                       "candidate_checks":"Full shared file checked by streaming validator","policy":policy}
    atomic_json(out/"validation.json",reports)
    atomic_json(root/"reports"/"cloud_output_validation.json",reports)
    return reports
