# Kaggle handoff for the entity-resolution challenge

The scoring portal expects **matching_results.tsv**. It does not run this notebook
on your behalf. Run the code in Kaggle, download the validated TSV, then upload it
to the challenge portal. A supporting methodology/code ZIP is also generated.

## Send your teammate these items

1. `output/handoff/kaggle_handoff.zip`, or this repository's
   `experiment/e02-sparse-retrieval` branch at the revision in `HANDOFF_REVISION.txt`.
2. The original organizer resource ZIP containing the seven dataset TSVs.
   The dataset is deliberately absent from Git and the code bundle.
3. Optionally, `output/handoff/retrieval_checkpoints.zip` if supplied. It contains
   existing sparse indexes and completed token-retrieval checkpoints. It does not
   contain a trained model. Completed token-6 results can be reused; the broader
   name/address top-6 lists still need to be computed.
4. The submission deadline and access to the team's challenge submission portal.
   Do not share passwords or API keys in the notebook or repository.

Use a **private Kaggle dataset** and **private notebook**. Grant the teammate access
to both separately: notebook access does not automatically grant dataset access.
Use only challenge-permitted team members and data access.

## Kaggle setup

1. Create a private Dataset from the organizer ZIP. Kaggle may unpack it; either
   the extracted directory or the original ZIP is supported by the setup cell.
2. Create a Notebook and import `notebooks/kaggle_handoff.ipynb`.
3. Use **Add Input** to attach the private dataset. Attach the code ZIP as another
   private input if you are not cloning GitHub.
4. Enable Internet for dependency installation/GitHub only. Matching never queries
   internet identity services. Start with a CPU session for preparation/features/
   training; use the automatic hardware report to inspect actual RAM and disk.
5. Run setup, configure the input path and run the preflight cell.
6. Run one stage at a time. Read the stage log if it stops; do not skip the checks.

The notebook creates an isolated Python environment in scratch storage. Python
3.11 uses SciPy 1.15.3; Python >=3.12 uses SciPy 1.18.1. All other dependencies are
pinned in requirements.txt. Runtime versions are recorded in the setup log.

## Resource expectations

Do not assume a GPU session has more system RAM. This pipeline's model fitting is
CPU/RAM work. It retains every candidate and uses a batched LightGBM Sequence over
Parquet, so the full float32 feature matrix is never loaded at once. LightGBM still
needs its binned dataset, labels and training buffers in RAM. The row-count-based
fit estimate must pass before fitting. If it refuses, use a larger RAM session;
do not remove rows or reduce candidate quality.

Kaggle's current documentation lists 12-hour CPU/GPU sessions and up to 20 GB saved
notebook output. Account quotas and available accelerators vary. The default stage
budget is 10 hours to leave time to save outputs. `/kaggle/temp` is scratch and
must not be treated as persistent. See [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks).

The earlier full-union pilot implies roughly 152 million training pairs. That is
an estimate, not the actual full-cache count. The fit estimate is approximately
`rows * (feature_count + 40) * 1.4 + 1 GiB`, plus a 2 GiB system reserve. It is a
conservative planning heuristic, not a measured peak. A separate worker monitor
stops the process tree if its resident memory exceeds its budget or RAM/disk
reserve is exhausted. It is not a kernel-enforced container memory limit.

Feature files use lossless ZSTD Parquet. Actual compression must be measured from
the produced shards. Split work across saved notebook versions if the accumulated
artifacts approach the output quota; do not assume the whole project fits in one
saved version. Runtime free disk and saved-output quota are different limits.

## Stages and files

From the project root, using the notebook's isolated Python executable:

```bash
python -m src.cloud_pipeline --stage preflight
python -m src.cloud_pipeline --stage prepare
python -m src.cloud_pipeline --stage retrieve-train
python -m src.cloud_pipeline --stage features-train
python -m src.cloud_pipeline --stage validate
python -m src.cloud_pipeline --stage final
python -m src.cloud_pipeline --stage retrieve-test
python -m src.cloud_pipeline --stage features-test
python -m src.cloud_pipeline --stage score
python -m src.cloud_pipeline --stage export
python -m src.cloud_pipeline --stage package
```

| Stage | What it produces |
|---|---|
| prepare | Full audit, normalized DuckDB tables, source hashes, saved entity folds |
| retrieve-* | All three top-6 lists, complete union, Parquet record/pair shards |
| features-* | All 51 features for every candidate, cached once as Parquet |
| validate | Four entity-grouped OOF folds for GBDT and streaming logistic regression; frozen decision; reserved fold 4 evaluation |
| final | Selected model fitted on all training entities and all candidate pairs |
| score | Cached probabilities for every test candidate |
| export | One shared candidate TSV and up to three distinct OOF-selected matching variants, all validated |
| package | Supporting code/methodology/output ZIP based on actual run results |

Each completed model, fold, scoring shard and feature shard is reused. LightGBM
saves a tree checkpoint every 25 iterations and resumes from it using batched
initial-score calculation. Logistic regression checkpoints after each shard.
Interrupted writes use temporary names; completion markers are published last.
An interrupted preparation table is rebuilt transactionally; completed tables
remain. Export is streamed but restarts if interrupted before final validation.

## Saving and resuming

1. Before the session ends, stop at a completed stage or shard boundary and use
   **Save Version** with outputs included. Confirm the saved version actually has
   the checkpoint files. An interactive filesystem alone is not a backup.
2. Attach the saved notebook output to the next notebook using **Add Input**.
3. Set `CHECKPOINT_SOURCE` in the setup cell to that saved project folder, then
   run the restore cell. The writable DuckDB database is copied; immutable files
   are linked to the attached input. Keep that input attached throughout the run.
4. Re-run the same stage with the same config and code revision. Completed work
   is reused. A changed cache fingerprint fails instead of mixing experiments.
5. Save the new version's newly generated files. Keep earlier input versions as
   well: links to old checkpoints are not independent backups. Check the saved
   output size before saving and verify that the intended files are present.

Optional legacy cache ZIP: extract its trusted `cache/` tree into the project
before retrieval. Do not overwrite a newer run. Its token results are reused only
when the fixed 10,000-record batch boundaries match.

## Multiple submissions

Start with the validated baseline already available on the original machine if
the deadline is close. After this run, upload:

`output/cloud/oof_best/matching_results.tsv`

Additional `precision/` or `recall/` folders are generated only when development
OOF scores support distinct policies within 0.005 macro F0.5 of the best. All
variants use the same cached model probabilities; retrieval/features are not
recomputed. There may be fewer than three variants if the policies coincide.
The portal score is not guaranteed to improve, and submitting identical files
does not create additional accuracy.

`output/cloud/validation.json` contains counts, hashes and validator results.
The shared `candidate_pairs.tsv` is supporting material, not the leaderboard
upload when the portal asks only for matching results.

## Verification status

Local validation uses small synthetic fixtures inside a Windows process with a
3 GiB hard memory cap, two logical CPUs and a five-minute timeout. Full-data
training and Kaggle execution are not performed on the local laptop. The notebook
must still be executed in the teammate's actual Kaggle account to establish
hardware compatibility, resource fit, total runtime and model quality.

The code bundle is a runnable handoff, not evidence that every experimental branch
in the original brief has been run. Read [BRIEF_COVERAGE.md](BRIEF_COVERAGE.md).
