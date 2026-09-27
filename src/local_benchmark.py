"""Small real-cache timing probe inside a Windows memory/CPU bounded process.

This is a resource benchmark, not model selection or a submission pipeline.
Sample reference frequencies intentionally differ from production frequencies.
"""

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GIB = 1024 ** 3


class WindowsJob:
    """Hard aggregate committed-memory limit; close kills contained processes."""

    def __init__(self, memory_bytes):
        if os.name != "nt":
            raise RuntimeError("This benchmark guard requires Windows")

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in
                        ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                     ctypes.c_void_p, wintypes.DWORD]
        self.api.SetInformationJobObject.restype = wintypes.BOOL
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x100 | 0x200
        limits.ProcessMemoryLimit = memory_bytes
        limits.JobMemoryLimit = memory_bytes
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def stop_reason(elapsed, available, free_disk, timeout):
    if elapsed > timeout:
        return "time_limit"
    if available < 4 * GIB:
        return "system_available_memory_below_4_GiB"
    if free_disk < 5 * GIB:
        return "disk_free_below_5_GiB"
    return None


def run_bounded(root, output, timeout=300, tests=False):
    import psutil
    import shutil
    output.mkdir(parents=True, exist_ok=False)
    reason = stop_reason(0, psutil.virtual_memory().available, shutil.disk_usage(root).free, timeout)
    if reason:
        raise RuntimeError(f"Benchmark refused: {reason}")
    gate = output / "ready"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", ENTITY_FEATURE_WORKERS="1",
               PYTHONHASHSEED="42", PYTHONIOENCODING="utf-8")
    job = WindowsJob(3 * GIB)
    process = None
    start = time.perf_counter()
    peak_rss = 0
    min_available = psutil.virtual_memory().available
    stop = None
    try:
        with (output / "worker.log").open("w", encoding="utf-8") as log:
            command = [sys.executable, "-m", "src.local_benchmark", "--worker", "--output", str(output)]
            if tests:
                command.append("--tests")
            process = subprocess.Popen(command, cwd=root, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            # Worker imports no numerical packages and does no data work until released.
            job.assign(process)
            monitored = psutil.Process(process.pid)
            allowed = monitored.cpu_affinity()
            monitored.cpu_affinity(allowed[:2])
            monitored.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            gate.touch()
            while process.poll() is None:
                available = psutil.virtual_memory().available
                min_available = min(min_available, available)
                try:
                    processes = [monitored] + monitored.children(recursive=True)
                    peak_rss = max(peak_rss, sum(p.memory_info().rss for p in processes))
                except psutil.NoSuchProcess:
                    pass
                stop = stop_reason(time.perf_counter() - start, available,
                                   shutil.disk_usage(root).free, timeout)
                if stop:
                    job.close()
                    break
                time.sleep(.25)
            process.wait(timeout=10)
    finally:
        job.close()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    result = {"exit_code": process.returncode, "stop_reason": stop,
              "wall_seconds": round(time.perf_counter() - start, 3),
              "peak_process_tree_rss_GiB": round(peak_rss / GIB, 3),
              "minimum_system_available_GiB": round(min_available / GIB, 3),
              "hard_job_committed_memory_GiB": 3, "logical_CPU_limit": 2,
              "priority": "below_normal", "timeout_seconds": timeout,
              "GPU_used": False, "monitor_interval_seconds": .25}
    (output / "resources.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if process.returncode or stop:
        raise RuntimeError(f"Benchmark stopped; inspect {output / 'worker.log'}")


def worker(root, output, tests=False):
    gate = output / "ready"
    deadline = time.monotonic() + 30
    while not gate.exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Parent did not establish resource limits")
        time.sleep(.05)
    if tests:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(root / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise RuntimeError("Unit tests failed")
        return
    import pickle
    import numpy as np
    import duckdb
    from src.features import FeatureBuilder, FEATURE_NAMES
    from src.model import make_model

    start = time.perf_counter()
    folder = root / "cache" / "candidates_v1_train"
    markers = sorted(folder.glob("batch_*.json"))
    if len(markers) < 5:
        raise RuntimeError("Need five completed cached batches")
    chosen = [markers[i] for i in np.linspace(0, len(markers)-1, 5, dtype=int)]
    rng = np.random.default_rng(42)
    samples = []
    needed = set()
    for marker in chosen:
        with np.load(marker.with_suffix(".npz")) as batch:
            take = np.sort(rng.choice(len(batch["y"]), min(20000, len(batch["y"])), replace=False))
            item = {k: batch[k][take] for k in ("qi", "ri", "meta", "y")}
        with marker.with_suffix(".pkl").open("rb") as f:
            item["records"] = pickle.load(f)
        needed.update(int(i) for i in item["ri"])
        samples.append(item)
    # Read just the referenced rows. The production database is opened read-only.
    db = duckdb.connect(str(root / "cache" / "entities.duckdb"), read_only=True,
                        config={"memory_limit": "384MB", "threads": 1,
                                "temp_directory": str(output / "duckdb_temp"),
                                "max_temp_directory_size": "512MB"})
    wanted = np.array(sorted(needed), dtype=np.int32)
    db.register("wanted", {"ri": wanted})
    refs = db.execute("SELECT r.ri,r.entity_id,r.name_norm,r.address_norm,r.country_norm "
                      "FROM ml_train_references r JOIN wanted w ON r.ri=w.ri").fetchall()
    db.close()
    mapping = {r[0]: i for i, r in enumerate(refs)}
    with (root / "cache" / "sparse_v1_train" / "token.pkl").open("rb") as f:
        vectorizer = pickle.load(f)
    builder = FeatureBuilder([r[1:] for r in refs], vectorizer)
    setup_seconds = time.perf_counter() - start
    feature_timings = []
    matrices = []
    for item in samples:
        local_ri = np.array([mapping[int(i)] for i in item["ri"]], dtype=np.int32)
        tick = time.perf_counter()
        x = builder.transform(item["records"], item["qi"], local_ri, item["meta"])
        assert np.isfinite(x).all()
        matrices.append(x)
        feature_timings.append({"pairs": len(x), "seconds": time.perf_counter() - tick})
        print("Features", feature_timings[-1], flush=True)
    x = np.concatenate(matrices)
    y = np.concatenate([s["y"] for s in samples])
    # Timings only: no reported accuracy or held-out score from this sampling scheme.
    fits = []
    for size in (25000, 50000, len(y)):
        index = rng.choice(len(y), size, replace=False)
        model = make_model("gbdt")
        model.set_params(n_jobs=2)
        tick = time.perf_counter()
        model.fit(x[index], y[index])
        seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        model.predict_proba(x)
        fits.append({"training_pairs": size, "trees": 350, "fit_seconds": seconds,
                     "prediction_pairs": len(y), "prediction_seconds": time.perf_counter() - tick})
        print("GBDT", fits[-1], flush=True)
    report = {"purpose": "resource timing only, not model evaluation",
              "sampling": "20,000 random cached pairs from each of five evenly spaced completed batches; seed 42",
              "limitations": ["Existing narrower retrieval policy, not broad top-6 union",
                              "Name/address frequencies use sampled references; full production setup memory not measured",
                              "Mostly S2 cached records; no full test or France timing",
                              "Model fitting timings do not scale reliably linearly to full data",
                              "No full-scale OOF, logistic model, export or GPU benchmark"],
              "sample_pairs": len(y), "sample_positive_pairs": int(y.sum()),
              "sample_references": len(refs), "features": len(FEATURE_NAMES),
              "feature_matrix_MiB": x.nbytes / 2**20, "setup_seconds": setup_seconds,
              "features_batches": feature_timings, "gbdt_fits": fits}
    (output / "timings.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tests", action="store_true", help="Run unit tests under the same resource limits")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "cache" / "benchmarks" / time.strftime("%Y%m%d_%H%M%S")
    if args.worker:
        worker(root, output, args.tests)
    else:
        run_bounded(root, output, tests=args.tests)
