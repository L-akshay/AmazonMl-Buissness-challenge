"""Stage-based Kaggle/AWS entry point; full populations and cached all-pair features."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
from src.cloud_store import atomic_json, claim_config, digest, signature

STAGES=("prepare","retrieve-train","features-train","validate","final",
        "retrieve-test","features-test","score","export","package")


def validate_config(config):
    for key in ("batch_size","threads","feature_workers","feature_chunk","prediction_batch","gpu_batch","trees","logistic_epochs"):
        if not isinstance(config[key],int) or config[key]<1:
            raise ValueError(f"{key} must be a positive integer")
    if config["backend"] not in ("cpu","gpu") or not config["models"] or any(m not in ("gbdt","logistic") for m in config["models"]):
        raise ValueError("Unknown backend or model family")
    if not 0<config["session_hours"]<=11:
        raise ValueError("Use a session budget above zero and at most 11 hours")
    if not 0<config["gpu_memory_gib"] or not 0<config["saved_output_limit_gib"]<=20:
        raise ValueError("Invalid GPU or saved-output budget")
    return config


def environment(config):
    os.environ.update(ER_THREADS=str(config["threads"]),ER_DB_MEMORY=config["db_memory"],
        ER_GPU_BATCH=str(config["gpu_batch"]),ER_GPU_MEMORY_GIB=str(config["gpu_memory_gib"]),
        ENTITY_FEATURE_WORKERS=str(config["feature_workers"]),OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",OMP_NUM_THREADS=str(config["threads"]),PYTHONUTF8="1")
    os.environ["ER_SAVED_OUTPUT_LIMIT_GIB"]=str(config["saved_output_limit_gib"])


def preflight(root,config,stage=None):
    memory=psutil.virtual_memory(); disk=shutil.disk_usage(root)
    report={"python":sys.version.split()[0],"platform":sys.platform,"logical_cpus":os.cpu_count(),
            "ram_total_gib":memory.total/1024**3,"ram_available_gib":memory.available/1024**3,
            "disk_free_gib":disk.free/1024**3,"saved_output_budget_gib":config["saved_output_limit_gib"],
            "backend":config["backend"],"candidate_policy":"full union of three top-6 lists",
            "training_selection":"all candidates; all S1; no negative subsampling",
            "warning":"GPU VRAM does not add system RAM. Full fitting performs an additional row-count-based RAM check."}
    try:
        gpu=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.free","--format=csv,noheader"],
                           capture_output=True,text=True,timeout=10)
        report["gpus"]=gpu.stdout.strip() if gpu.returncode==0 else "unavailable"
    except (OSError,subprocess.TimeoutExpired):
        report["gpus"]="unavailable"
    if config["backend"]=="gpu" and stage in ("retrieve-train","retrieve-test"):
        import cupy as cp
        report["cupy"]=cp.__version__
        available,total=cp.cuda.runtime.memGetInfo()
        if available<(config["gpu_memory_gib"]+.5)*1024**3:
            raise MemoryError("GPU pool plus overhead does not fit free VRAM")
    atomic_json(root/"reports"/"remote_preflight.json",report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def input_manifest(root):
    files={}
    for split in ("train","test"):
        names=[f"{split}_source{i}.tsv" for i in (1,2,3)]
        if split=="train":
            names.append("train_ground_truth.tsv")
        for name in names:
            path=root/"dataset"/split/name
            if not path.is_file():
                raise FileNotFoundError(path)
            files[f"{split}/{name}"]={"bytes":path.stat().st_size,"sha256":digest(path)}
    return files


def prepare(root,config):
    from src.audit import audit
    from src.pipeline import prepare as legacy_prepare
    from src.cloud_features import references
    # Hash once at ingestion. A copied cache must match these exact seven inputs.
    inputs=input_manifest(root)
    claim_config(root/"cache"/"cloud"/"input_identity",inputs)
    audit(root)
    report=json.loads((root/"reports"/"data_audit.json").read_text())
    if any(report["integrity"].values()):
        raise ValueError("Resolve truth ownership/split integrity issues before this pipeline")
    legacy_prepare(root)
    for split in ("train","test"):
        references(root,split)
    atomic_json(root/"cache"/"cloud"/"prepared.json",{"inputs":inputs})


def execute_stage(root,stage,config):
    from src.cloud_features import retrieve,features,coverage,references
    from src.cloud_model import score_model
    from src.cloud_validation import validation,final_fit
    from src.cloud_export import export
    from src.cloud_store import read_parquet
    import numpy as np
    prepared=root/"cache"/"cloud"/"prepared.json"
    if stage=="prepare":
        prepare(root,config)
        return
    if not prepared.exists():
        raise ValueError("Run prepare first")
    key={"inputs":json.loads(prepared.read_text())["inputs"],"backend":config["backend"],
         "batch_size":config["batch_size"],"policy":"full6"}
    fingerprint=signature(Path(__file__).resolve().parents[1],key)
    claim_config(root/"cache"/"cloud"/"pipeline_identity",{"fingerprint":fingerprint})
    if stage.startswith("retrieve-"):
        retrieve(root,stage.split("-")[1],config,fingerprint)
    elif stage.startswith("features-"):
        split=stage.split("-")[1]
        features(root,split,config,fingerprint)
        if split=="train":
            coverage(root)
    elif stage=="validate":
        validation(root,config,fingerprint)
    elif stage=="final":
        final_fit(root,config,fingerprint)
    elif stage=="score":
        final=json.loads((root/"cache"/"cloud"/"final.json").read_text())
        n=len(read_parquet(references(root,"test"),"ri")["ri"])
        score_model(root,root/final["model"],"test",np.ones(n,dtype=bool),"test",config)
    elif stage=="export":
        export(root,config)
    elif stage=="package":
        from src.handoff import package_submission
        package_submission(root,config)
    else:
        raise ValueError("Unknown stage")


def run_stage(root,stage,config_path,config):
    """Monitor the isolated remote worker; completed atomic shards survive a stop."""
    report=preflight(root,config,stage)
    if report["ram_available_gib"]<8 or report["disk_free_gib"]<5:
        raise RuntimeError("Insufficient RAM/disk headroom for a full remote stage")
    environment(config)
    deadline=float(os.environ.get("ER_DEADLINE_EPOCH","0"))
    if not deadline:
        deadline=time.time()+config["session_hours"]*3600
        os.environ["ER_DEADLINE_EPOCH"]=str(deadline)
    limit=min(psutil.virtual_memory().total*.8,psutil.virtual_memory().available-2*1024**3)
    code_root=Path(__file__).resolve().parents[1]
    log_path=root/"reports"/f"remote_{stage}.log"
    with log_path.open("a",encoding="utf-8") as log:
        process=subprocess.Popen([sys.executable,"-u","-m","src.cloud_pipeline","--worker",
            "--stage",stage,"--root",str(root),"--config",str(config_path)],cwd=code_root,
            env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        monitored=psutil.Process(process.pid)
        start=time.monotonic(); peak=0; reason=None; last_log=0; offset=0
        try:
            while process.poll() is None:
                try:
                    family=[monitored]+monitored.children(recursive=True)
                    rss=sum(p.memory_info().rss for p in family if p.is_running()); peak=max(peak,rss)
                except psutil.NoSuchProcess:
                    if process.poll() is not None:
                        break
                    continue
                if rss>limit or psutil.virtual_memory().available<2*1024**3:
                    reason="RAM budget reached"
                elif shutil.disk_usage(root).free<2*1024**3:
                    reason="Disk reserve reached"
                elif time.time()>deadline:
                    reason="Session time budget reached"
                if reason:
                    for child in reversed(family):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            continue
                    break
                if time.monotonic()-last_log>30:
                    with log_path.open(encoding="utf-8",errors="replace") as reader:
                        reader.seek(offset); lines=reader.read(); offset=reader.tell()
                    if lines:
                        print(lines[-12000:],end="",flush=True)
                    last_log=time.monotonic()
                time.sleep(1)
        finally:
            if process.poll() is None:
                for child in monitored.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        continue
                process.kill()
            process.wait()
    atomic_json(root/"reports"/f"remote_{stage}_resources.json",{"peak_rss_gib":peak/1024**3,
        "wall_seconds":time.monotonic()-start,"stop_reason":reason,"exit_code":process.returncode})
    if process.returncode or reason:
        raise RuntimeError(f"Stage stopped: {reason or 'worker failure'}. Read {log_path}; save outputs and resume")
    print(f"Completed {stage}. Log: {log_path}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"configs"/"kaggle.json")
    parser.add_argument("--stage",choices=("preflight",)+STAGES+("all",),default="preflight")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args(); root=args.root.resolve(); config_path=args.config.resolve()
    config=validate_config(json.loads(config_path.read_text())); environment(config)
    for directory in ("reports","cache","experiments","output"):
        (root/directory).mkdir(parents=True,exist_ok=True)
    if args.stage=="preflight":
        preflight(root,config)
    elif args.worker:
        execute_stage(root,args.stage,config)
    else:
        if not (os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or str(root).startswith("/kaggle/") or os.environ.get("ER_REMOTE_COMPUTE")=="1"):
            raise RuntimeError("Full stages require remote compute. The laptop is restricted to bounded tests and samples")
        for stage in STAGES if args.stage=="all" else (args.stage,):
            run_stage(root,stage,config_path,config)
