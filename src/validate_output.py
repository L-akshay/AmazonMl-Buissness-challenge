"""Bounded-memory strict output checks, followed by the organizer validator."""

import argparse
import csv
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from src.data import connect
from src.streaming import encode_secondary


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def validate_rows(matching,candidates,required,valid_codes):
    """Check complete coverage, list uniqueness, provenance and exact subset."""
    if next(matching,None)!=["source1_entity_id","matched_entity_ids"]:
        raise ValueError("Invalid matching header")
    if next(candidates,None)!=["source1_entity_id","candidate_entity_ids"]:
        raise ValueError("Invalid candidate header")
    counts={"entities":0,"candidate_pairs":0,"matched_pairs":0,"predicted_singletons":0}
    pending=[]
    def check_pending():
        queries=np.array(pending,dtype=np.uint64)
        positions=np.searchsorted(valid_codes,queries)
        if (positions>=len(valid_codes)).any() or not np.array_equal(valid_codes[positions],queries):
            raise ValueError("Output contains a target ID absent from test sources")
        pending.clear()
    previous=None
    for row,proposal,sid in zip_longest(matching,candidates,required):
        if row is None or proposal is None or sid is None:
            raise ValueError("Missing or extra S1 rows")
        if len(row)!=2 or len(proposal)!=2 or row[0]!=sid or proposal[0]!=sid:
            raise ValueError("Output schema, ordering or S1 coverage mismatch")
        if previous is not None and sid<=previous:
            raise ValueError("Duplicate or unsorted S1 row")
        previous=sid
        matches=row[1].split(",") if row[1] else []
        candidate_ids=proposal[1].split(",") if proposal[1] else []
        if len(matches)!=len(set(matches)) or len(candidate_ids)!=len(set(candidate_ids)):
            raise ValueError("Repeated target ID within a row")
        if not set(matches)<=set(candidate_ids):
            raise ValueError("Final match is absent from the scored candidate set")
        pending.extend(encode_secondary(tid) for tid in candidate_ids)
        if len(pending)>=250000:
            check_pending()
        counts["entities"]+=1
        counts["candidate_pairs"]+=len(candidate_ids)
        counts["matched_pairs"]+=len(matches)
        counts["predicted_singletons"]+=int(not matches)
    if pending:
        check_pending()
    return counts


def validate(root,publish=False):
    folder=root/"output"/"trained"
    db=connect(root)
    db.execute("SET memory_limit='2GB'")
    db.execute("SET threads=1")
    valid=db.execute("""SELECT ((substr(entity_id,2,1)::UBIGINT << 32)
        + substr(entity_id,4)::UBIGINT) AS code FROM
        (SELECT entity_id FROM test_source2 UNION ALL SELECT entity_id FROM test_source3)
        ORDER BY code""").fetchnumpy()["code"]
    cursor=db.execute("SELECT entity_id FROM test_source1 ORDER BY entity_id")
    def required():
        while True:
            batch=cursor.fetchmany(20000)
            if not batch:
                return
            yield from (r[0] for r in batch)
    with (folder/"matching_results.tsv").open(encoding="utf-8",newline="") as m, \
         (folder/"candidate_pairs.tsv").open(encoding="utf-8",newline="") as c:
        result=validate_rows(csv.reader(m,delimiter="\t"),csv.reader(c,delimiter="\t"),required(),valid)
    db.close()
    del valid
    exported=json.loads((root/"reports"/"trained_output.json").read_text())
    if any(result[k]!=exported[k] for k in result):
        raise ValueError("Output file counts differ from scored export")
    print("Strict full-file validation PASS, including all candidate ID checks",flush=True)
    # The supplied validator keeps all candidate strings/sets in RAM. Its own
    # documentation recommends omitting candidates when --check-ids is large.
    # Validate the complete scored submission there; both complete files and
    # their subset relationship have already passed the streaming checks above.
    validator_workdir=root/"cache"/"organizer_matching_only"
    validator_workdir.mkdir(exist_ok=True)
    process=subprocess.run([sys.executable,"-X","utf8",str(root/"utils"/"validate_submission.py"),
        "--matching",str(folder/"matching_results.tsv"),"--test-dir",str(root/"dataset"/"test"),"--check-ids"],
        cwd=validator_workdir,capture_output=True,text=True,encoding="utf-8")
    log=process.stdout+process.stderr
    (root/"reports"/"trained_organizer_validator.txt").write_text(log,encoding="utf-8")
    print(log,flush=True)
    if process.returncode or "PASS" not in log:
        raise ValueError("Organizer validator did not pass")
    result.update({"strict_full_files":"PASS","organizer_matching_with_id_checks":"PASS",
        "organizer_candidate_check":"Omitted for memory; complete candidate file checked by strict streaming validator",
        "sha256":{name:digest(folder/name) for name in ("matching_results.tsv","candidate_pairs.tsv")}})
    (root/"reports"/"trained_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if publish:
        for name in ("matching_results.tsv","candidate_pairs.tsv"):
            temporary=root/"output"/(name+".tmp")
            shutil.copyfile(folder/name,temporary)
            temporary.replace(root/"output"/name)
        print(f"Validated submission ready: {root/'output'/'matching_results.tsv'}",flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--publish",action="store_true",help="Replace local baseline outputs after all checks pass")
    args=parser.parse_args()
    validate(Path(__file__).resolve().parents[1],args.publish)
