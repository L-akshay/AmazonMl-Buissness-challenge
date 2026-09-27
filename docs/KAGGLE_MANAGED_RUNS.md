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

## Planned diagnostic jobs

`research-contributions` reports name/address/token unique true links and unique
candidate volume from the complete existing union. No blocking recomputation.
`research-country-us` and `research-country-india` fit only one country's
development entities, freeze a threshold from inner grouped OOF in that country,
then evaluate the other country. Reserved fold 4 is excluded throughout.

`research-ablation-address`, `research-ablation-numeric`,
`research-ablation-frequency`, and `research-ablation-retrieval` remove declared
feature groups, retain every candidate and reuse the original entity folds.
The address-feature group is not a removal of the address retrieval channel;
indirect address information can remain in combined token/frequency evidence.
Reports are experiments, not automatic changes to the submission policy.

OOF model work can also be scheduled independently with `oof-gbdt-0` through
`oof-gbdt-3`, and `oof-logistic-0` through `oof-logistic-3`. Each task uses the
same full candidate cache and saved S1 folds. Attach these outputs to `validate`;
it reuses finished models and score shards before selecting the policy and
performing the reserved evaluation. This parallelization changes neither the
training population nor the validation protocol.

`fit-full-{gbdt,logistic}`, `fit-reserved-{gbdt,logistic}` and
`score-{train,test}-{gbdt,logistic}` allow fixed model fits and scoring to overlap
with the OOF work. Full fits retain every training entity; reserved models fit
development folds only. These jobs do not compute reserved metrics or select a
policy. Validation freezes the choice from OOF before evaluating the selected
reserved score. Final export reuses the chosen model's cached test scores.
The full plan allows up to four private CPU jobs, with dependency-safe priorities
for assembly, long independent fits and deliverables. Account-side queueing can
still limit actual concurrency.

## Local control process

`scripts/manage_kaggle.py` can supervise already-built private CPU jobs from a
reviewed JSON dependency plan. It performs CLI/network polling only; it never
loads or trains on the challenge data locally. It records a durable journal and
stops dependent work when a remote job reports an interruption. A remote kernel
marked COMPLETE is insufficient: its project completion marker must also agree.
An ambiguous launch is not automatically retried. Completed artifacts remain on
Kaggle for inspection and resumption. Routine polling downloads only small
reports/logs. Completed export jobs can additionally stream validated TSVs to
disk using `scripts/download_kaggle_results.py`, with one-megabyte chunks and
resumable partial files. A downloaded TSV must match its remote validation SHA256
before it receives its final filename. Supporting ZIP delivery is also streamed.

`python scripts/build_kaggle_plan.py` builds the initial resource-check plan and
the complete dependency plan from a clean committed source revision. The initial
plan checks the first full-index retrieval/feature batches and test-index resources
before automatically continuing to the full plan. It requires at most 23 GiB
measured peak RSS, at most 16 GiB projected output per 200-batch partition, and
at most eight hours projected compute after doubling the measured time estimate.
Missing, interrupted, or oversized evidence stops the continuation for review;
it never reduces candidates or data. These are initial-train-batch projections,
not guaranteed timings for every population. Remote runtime guards remain active.
Both plans use the same durable journal, so completed initial jobs are reused.
The package job explicitly attaches the train assembly
report: checkpoint dependencies are not transitive because each saved notebook
contains only the files it newly produced.

The controller can continue between chat turns while the computer is awake.
Closing or sleeping the computer does not cancel already launched Kaggle jobs,
but new dependent jobs require the controller to run again. It does not submit
to the separate challenge portal or publish any dataset/notebook publicly.
