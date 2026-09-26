"""Atomic Parquet shards and fingerprints shared by remote pipeline stages."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import duckdb
import numpy as np

_LAST_OUTPUT_CHECK={}


def output_size(root):
    # Inputs linked from prior saved versions are not independent output files.
    return sum(p.stat().st_size for p in Path(root).rglob("*")
               if p.is_file() and not p.is_symlink() and ".git" not in p.parts)


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2),encoding="utf-8")
    tmp.replace(path)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def sql_path(path):
    return "'"+str(Path(path).resolve()).replace("'","''")+"'"


def memory_db():
    db=duckdb.connect(config={"memory_limit":os.environ.get("ER_SHARD_MEMORY","512MB"),
                              "threads":1})
    return db


def write_parquet(path, arrays):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp.parquet")
    with memory_db() as db:
        db.register("payload",arrays)
        db.execute(f"COPY payload TO {sql_path(temp)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    temp.replace(path)


def read_parquet(path, columns="*"):
    with memory_db() as db:
        return db.execute(f"SELECT {columns} FROM read_parquet({sql_path(path)})").fetchnumpy()


def claim_config(folder, config):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"config.json"
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError(f"Cache configuration changed at {folder}; choose a new run directory")
    if not path.exists():
        atomic_json(path,config)


def signature(root, config):
    names=("blocking.py","gpu_sparse.py","normalize.py","features.py","cloud_store.py",
           "cloud_features.py","cloud_model.py","cloud_pipeline.py","cloud_validation.py","cloud_export.py")
    payload={"settings":config,"code":{n:digest(root/"src"/n) for n in names}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def check_headroom(root, reserve_gib=2):
    import psutil
    if psutil.virtual_memory().available<reserve_gib*1024**3:
        raise MemoryError("Less than reserved system RAM remains; completed shards are safe")
    if shutil.disk_usage(root).free<reserve_gib*1024**3:
        raise OSError("Less than reserved disk space remains; completed shards are safe")
    deadline=float(os.environ.get("ER_DEADLINE_EPOCH","0"))
    if deadline and time.time()>=deadline:
        raise TimeoutError("Stage session budget reached; save outputs and resume")
    budget=float(os.environ.get("ER_SAVED_OUTPUT_LIMIT_GIB","0"))
    key=str(root)
    if budget and time.monotonic()-_LAST_OUTPUT_CHECK.get(key,0)>10:
        _LAST_OUTPUT_CHECK[key]=time.monotonic()
        if output_size(root)>budget*1024**3:
            raise OSError("Saved-output budget reached. Save this version and attach its checkpoints to a fresh session; do not discard candidates/features")


def parquet_files(folder):
    manifest=Path(folder)/"complete.json"
    if not manifest.exists():
        raise ValueError(f"Stage incomplete: {folder}")
    meta=json.loads(manifest.read_text())
    files=[Path(folder)/name for name in meta["files"]]
    if any(not p.is_file() for p in files):
        raise ValueError("Completed shard is missing")
    return files
