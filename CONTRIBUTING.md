# Development workflow

## Milestone commits

A milestone is one working, verified result: a completed audit, a retriever with
measured recall, a model with grouped validation, or a validated output pipeline.
Commit when that result and its evidence are reviewable. Use short present-tense
messages such as `add name character retrieval diagnostics` or
`tune entity decisions using out-of-fold scores`.

Before pushing:

```powershell
git fetch origin
git status --short
python -m unittest discover -s tests -v
git diff --cached --check
git diff --cached --stat
```

Stage explicit paths after reviewing changes. Check that data, samples, outputs,
caches, and secrets remain ignored. Preserve remote work and use normal pushes.

## Experiments

Use branches such as `experiment/e02-name-char-retrieval` for new experiments.
Include the hypothesis, measured changes, validation scope, candidate volume,
runtime, and decision in the pull request. Promote a component only when evidence
supports it. Update the README/status when an accepted result changes the pipeline.

Use the saved entity folds consistently and distinguish descriptive fixed-rule
scores from trained OOF predictions. Do not select thresholds using in-sample
predictions. Preserve previous experiment records; save new detailed snapshots.

## Checks

GitHub Actions runs the synthetic unit/integration suite on Python 3.13 on Linux
and Windows. It does not require or download challenge data. Full-data runs and
the organizer's output validator remain local milestone checks. A green test
workflow does not establish model quality or competition compliance by itself.

When dependencies change, also run the suite in a fresh virtual environment
installed only from `requirements.txt`. DuckDB's Python function registration
requires NumPy even though basic SQL queries work without it.

For changed prediction generation, regenerate the appropriate outputs and run:

```powershell
python -X utf8 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## Versioned artifacts

Commit source, tests, documentation, aggregate metrics and experiment summaries.
Keep `dataset/`, `cache/`, `output/`, row-level report TSVs, the local input
manifest, generated fold assignments, and submission ZIPs outside Git. They can
be regenerated with the documented commands. Avoid embedding dataset rows in
notebook outputs or issue/PR descriptions.

## Current milestones

- E00: full audit, exact scorer, normalization, and positive-pair diagnostics.
- E01: fixed-rule baseline, complete test outputs, organizer validator PASS.
- Next: E02 character-name retrieval with recall/volume measurements.

The trained V3 matcher and final submission packaging remain pending.
