# Managed private Kaggle execution

The user authorized remote job management after CLI browser authentication on
2026-09-27. Authentication stays outside the repository. All generated jobs are
private and use only `lakshaytechai/amazon-er-private-inputs` plus explicitly
attached prior checkpoint notebooks.

`scripts/build_kaggle_job.py` builds a reviewable job from a clean committed
revision. Its embedded ZIP contains tracked source only. It does not start a job.
For example, from the repository root:

```powershell
python scripts/build_kaggle_job.py --slug amazon-er-train-000 --stages retrieve-train features-train --checkpoints lakshaytechai/amazon-er-prepare --overrides cache/job-config.json
kaggle kernels push -p cache/kaggle_jobs/amazon-er-train-000 -t 14400
```

The override file sets `batch_start`, exclusive `batch_stop`, worker settings and
`session_hours`. The default batch is 10,000 secondary records against the full
reference index. Partitioning divides computation, not the dataset or candidates.
Attach every required predecessor explicitly, in oldest-to-newest order.

Each job writes `REMOTE_CHECKPOINT.json`. Read its `status` and error field:
Kaggle kernel completion alone is not proof that all requested stages completed.
The wrapper catches stage interruptions so atomic checkpoints can still be saved.
Only newly produced files are persisted. Links to input files are detached without
deleting their targets. The database is mounted read-only after preparation.
Prior inputs must remain attached when resuming or aggregating partitions.

Run `assemble-train` / `assemble-test` after attaching all corresponding candidate
and feature partitions. Global completion markers are published only when every
batch exists and candidate/feature totals agree. Training and inference require
these global manifests. No partial dataset is silently treated as complete.

Retrieval/features have a separate fingerprint from model and validation code.
Changing matcher experiments therefore preserves the expensive upstream caches.
Changing the actual normalization/blocking/features invalidates them as intended.

## First measured remote environment

Private resource probe: `lakshaytechai/amazon-er-resource-check`, version 1.
Python 3.12.13; four CPU cores; cgroup RAM limit 30 GiB; host-reported RAM 31.35 GiB;
available RAM 30.46 GiB; working filesystem free space 19.50 GiB; 1,713 attached
files. GPU quota reported 30 hours unused; availability is checked separately.

The initial preparation attempt failed before ingestion because Kaggle's Python
omits `ensurepip`. The notebook and managed runner now create a venv without pip,
then use host `pip --python` to bootstrap pip 26.2.1 inside that isolated venv.
No project dependencies are installed into the preinstalled Kaggle environment.

Local checks for partitioning and checkpoint lifecycle: 34 synthetic tests,
33 passed and one optional GPU test skipped; Windows job capped at 3 GiB and two
logical CPUs. Measured peak resident memory 0.214 GiB, 130.23 seconds wall time.
This does not establish full-data model quality or full-scale runtime.
