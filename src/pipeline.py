"""Reproducible training-to-submission workflow, with isolated process stages."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def complete(path):
    return path.exists() and json.loads(path.read_text()).get("complete",False)


def stage(root,module,*args):
    command=[sys.executable,"-u","-m",module,*args]
    print("Running: "+" ".join(command),flush=True)
    environment=os.environ.copy()
    environment.setdefault("OPENBLAS_NUM_THREADS","1")
    environment["PYTHONUTF8"]="1"
    with (root/"cache"/"pipeline.log").open("a",encoding="utf-8") as log:
        process=subprocess.Popen(command,cwd=root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8")
        for line in process.stdout:
            print(line,end="",flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f"Stage {module} failed; inspect cache/pipeline.log. Completed checkpoints are retained.")


def prepare(root):
    from src.data import connect,load
    from src.run_pipeline import normalize_tables
    db=connect(root)
    load(db,root)
    normalize_tables(db)
    db.execute("""CREATE TABLE IF NOT EXISTS folds AS SELECT entity_id AS sid,
        (row_number() OVER (ORDER BY md5('42|' || entity_id))-1)%5 AS fold FROM train_source1""")
    db.close()


def run(root,backend):
    (root/"cache").mkdir(exist_ok=True)
    (root/"reports").mkdir(exist_ok=True)
    (root/"experiments").mkdir(exist_ok=True)
    if not (root/"reports"/"data_audit.json").exists():
        stage(root,"src.audit")
    integrity=json.loads((root/"reports"/"data_audit.json").read_text())
    if any(integrity["integrity"].values()) or any(s["duplicate_ids"] or s["invalid_ids"] for s in integrity["sources"].values()):
        raise ValueError("Resolve data integrity issues before training")
    stage(root,"src.pipeline","--prepare-only")
    if not complete(root/"cache"/"candidates_v1_train"/"summary.json"):
        stage(root,"src.generate_candidates","--split","train","--backend",backend)
    if not (root/"reports"/"full_candidate_metrics.json").exists():
        stage(root,"src.candidate_report")
    stage(root,"src.training_data","--mode","development")
    if not (root/"reports"/"model_validation.json").exists():
        stage(root,"src.model","--stage","validate")
    stage(root,"src.training_data","--mode","full")
    if not (root/"cache"/"model"/"matcher.pkl").exists():
        stage(root,"src.model","--stage","final")
    if not complete(root/"cache"/"candidates_v1_test"/"summary.json"):
        stage(root,"src.generate_candidates","--split","test","--backend",backend)
    if not (root/"cache"/"test_scores"/"complete.json").exists():
        stage(root,"src.predict","--stage","score")
    stage(root,"src.predict","--stage","export")
    stage(root,"src.validate_output","--publish")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--backend",choices=["cpu","gpu"],default="cpu")
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    prepare(root) if args.prepare_only else run(root,args.backend)
