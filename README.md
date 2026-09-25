# Business entity resolution

[![Tests](https://github.com/L-akshay/AmazonMl-Buissness-challenge/actions/workflows/tests.yml/badge.svg)](https://github.com/L-akshay/AmazonMl-Buissness-challenge/actions/workflows/tests.yml)

Implementation of the supplied V3 brief for the Amazon ML Challenge 2026.

The first completed stage is a full-data audit, exact entity-level F0.5 scorer,
multi-view normalization, and an untuned exact-match baseline. This is an initial
development baseline, not the final V3 trained system or a leaderboard score.

The sparse retrieval milestone is also implemented and measured: the selected
pilot policy recovers 95.21% of sampled true links against all training S1
references. See [retrieval results](reports/retrieval_summary.md). Full-data
candidate generation and supervised model validation are in progress; the current
`output/` TSVs still belong to E01 until the trained pipeline replaces them.

## Environment and reproduction

Tested with Python 3.13.5, DuckDB 1.5.5, and NumPy 2.2.6 on Windows. NumPy is
required by DuckDB's Python function registration. From this directory:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.audit
python -m src.sample_pairs
python -m src.run_pipeline
```

Place the seven organizer TSVs under `dataset/train` and `dataset/test` first.
The supplied validator must be at `utils/validate_submission.py`; it is preserved
unchanged. The pipeline runs it with `--check-ids` and saves its complete result.
All paths default to this project directory. The dataset and both original ZIPs
remain unchanged. `reports/input_manifest.json` records archive provenance.

Raw input is loaded into `cache/entities.duckdb`, with a 6 GB DuckDB memory limit
and four worker threads. Allow additional memory for Python and the organizer
validator. Cached tables make subsequent baseline runs faster. If the dataset or
normalization implementation changes, use a fresh `cache` directory before
rerunning; cache invalidation is not automatic in this initial version.

## What the baseline does

1. Read every file as strict UTF-8 TSV and verify its column names.
2. Preserve original fields; normalize Unicode NFKC, case, punctuation and spaces.
3. Generate candidates with an exact, nonempty normalized name **and** address.
4. Score every candidate by country consistency; accept when countries agree or
   one country is missing. No country labels are hardcoded.
5. Write one row per test S1, including empty results, and validate both outputs.

Missing names/addresses are never treated as an exact match. Candidate IDs come
only from the corresponding split's S2/S3 files. Multiple matches are allowed.
The final scored candidate table is exactly what gets exported. Accepted matches
are selected from that same table, so the subset constraint holds by construction.

`normalize.py` additionally implements Latin accent folding, legal-suffix/core
views, small generic street expansions, and ordered address-number extraction.
Those auxiliary views are tested but are **not used by E01**. They are intended for
subsequent retrieval/features experiments. Indic combining marks are preserved.
Suffix and street vocabularies are general normalization aids, not identity data.

## Evaluation and artifacts

- [Data audit](reports/data_audit.md), with detailed counts in `data_audit.json`.
- [Measured baseline result](reports/baseline_summary.md): macro F0.5 0.08282;
  full test output passes the organizer validator with ID checks enabled.
- [Audit notebook](notebooks/data_audit.ipynb) and exact SQL in `src/audit.py`.
- `reports/positive_pair_sample.tsv`: 2,000 reproducibly sampled true pairs, with
  overlapping **heuristic** noise tags, not verified linguistic annotations.
- `reports/baseline_metrics.json`: macro F0.5, link precision/recall, candidate
  recall, singleton false-positive rate, country and match-count breakdowns.
- `reports/error_analysis.md` and `baseline_errors.tsv`: initial retrieval errors.
- `reports/baseline_validator.txt`: organizer validator with ID checks enabled.
- `experiments/folds.tsv`: five saved, deterministic S1-grouped folds (seed 42,
  ordering by MD5, then balanced round-robin assignment).
- `experiments/results.csv`: append-only experiment summaries.
- `experiments/E01_*.json`: immutable detailed results for subsequent baseline runs.
- `output/candidate_pairs.tsv` and `output/matching_results.tsv`: full test output.

E01 uses a fixed rule selected before examining scores. Its reported performance
is measured over the complete labeled training population; no model is fitted and
no threshold is tuned. Fold breakdowns are descriptive and are **not OOF model
predictions**. Actual test F0.5 is unknown. The SQL scorer is checked against the
Python scorer on a synthetic end-to-end case, including singletons and multi-match
entities. The organizer example evaluates to 5/7.

## Next experiment

Complete the checkpointed training candidate run, then compare logistic and GBDT
models using saved entity folds. Tune decisions on development OOF predictions,
check country-held-out performance, and evaluate the frozen choice on the reserved
fold. The final test TSV will be regenerated only after this validation.

### Sparse retrieval commands

```powershell
python -m src.retrieval_experiment --modulus 2000
python -m src.generate_candidates --split train --backend cpu
```

### Trained submission workflow

The complete workflow is implemented and covered by synthetic integration tests.
Its first full-data run is in progress; real supervised validation results and
the trained submission are not yet available.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m src.pipeline --backend gpu
```

Use `--backend cpu` when CUDA is unavailable. The workflow audits/prepares the
data, generates all training candidates, measures full candidate coverage, builds
features, compares logistic regression and LightGBM using entity-grouped OOF
scores, freezes a decision policy, checks a reserved fold, fits the final model,
then scores and validates the complete test population. Stages run in separate
processes so training, GPU retrieval, and output validation do not compete for
memory. Checkpoints and `cache/pipeline.log` support resuming interrupted runs.

Development uses a deterministic 4% sample of S1 entities, with all secondary
records searched against the full S1 index. Four saved entity folds supply OOF
model/threshold selection; fold 4 is reserved. The final model uses the complete
training S1 population, every retrieved positive, rank-one high-score negatives,
and a deterministic 10% sample of remaining negatives. Validation scores every
retrieved candidate, and its recall denominator includes unretrieved truth links.
Unsupervised TF-IDF and reference frequencies use all S1 texts in each split;
validation labels never fit the supervised matcher.

The reserved matcher sample excludes entities used in earlier labeled retrieval
pilots and row-level diagnostic samples. The exclusion rule is deterministic and
its counts are reported. Earlier full-training baseline/retrieval aggregates were
viewed, so this is a reserved **supervised matcher** evaluation, not a completely
untouched evaluation of the entire development process.

Country transfer checks refit on one training country and tune thresholds using
only that country's nested entity OOF scores before evaluating the other country.
These are proxies for geographic shift, not measured France test performance.
The workflow also records feature ablations, fold spread, singleton errors,
calibration diagnostics, and heuristic error categories. No pretrained weights
or external identity data are needed.

Features use four CPU workers by default. Each receives only a small batch of
records; the full reference corpus remains in the parent process. Set
`ENTITY_FEATURE_WORKERS=1` for serial execution or a lower memory budget.
Parallel and serial feature values/order are tested for equivalence.

The main upload artifact is `output/matching_results.tsv`. The exact scored
candidate set is also saved as `output/candidate_pairs.tsv`. Both files must pass
strict streaming checks for complete S1 coverage, existing target IDs, duplicate
IDs, and the match-subset invariant. The unmodified organizer validator checks
the complete matching TSV with `--check-ids`; its large in-memory candidate map
is omitted as recommended in its documentation. Full candidate validation is
performed by the project's bounded-memory validator. Trained outputs replace the
baseline files only after both validation stages pass.

The optional GPU backend is tested on an RTX 4060 with a CUDA 13.1-compatible
driver. Install `requirements-gpu.txt`, then select `--backend gpu`. GPU caches
live under the ignored `cache/` directory, and device cache growth is capped.
Set `RUN_GPU_TESTS=1` to include the numerical GPU-versus-integer-oracle test.
The portable test workflow does not install GPU dependencies.

Candidate generation uses 10,000-record checkpoints and resumes automatically.
Secondary texts are sorted one source at a time with a 2 GB DuckDB budget;
training ownership is attached through compact sorted NumPy arrays. This avoids
materializing a ten-million-row join of labels and text. Batches preserve the
original global ID ordering, including the S2/S3 boundary. Numeric ID encoding
is used only for label bookkeeping and never enters model features.

The remaining V3 experiments, trained model selection, final methodology and
submission ZIP have not been completed. No pretrained models, hosted matchers,
external identity lookup, or manual prediction edits are used in this baseline.
`reference/` holds the original organizer README and documentation template;
`brief/` holds the supplied project brief.

## Repository maintenance

[GitHub repository](https://github.com/L-akshay/AmazonMl-Buissness-challenge).
Working milestones are committed with their tests and aggregate evidence.
See [CONTRIBUTING.md](CONTRIBUTING.md) for branch, review, and commit practices.
GitHub Actions runs the test suite on Linux and Windows without challenge data.

Datasets, prediction files, row-level report TSVs, local input manifests and the
generated `experiments/folds.tsv` stay local and are regenerated by the documented
commands. Links to those local artifacts describe pipeline outputs; those files
are intentionally absent from a fresh clone.
