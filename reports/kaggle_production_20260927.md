# Production Kaggle evidence, 2026-09-27

Full preparation completed successfully in private job `amazon-er-v3-prepare`.
The stage took 2,470.39 seconds with 3.13 GiB peak RSS; all organizer source files
were retained and all reported ownership/split integrity checks were zero.
The persisted project contains 3,171,611,420 newly produced bytes.

The first two full-index batches (20,000 secondary queries) completed in private
job `amazon-er-v3-first-batches`. Retrieval took 48.06 seconds and peaked at
1.95 GiB RSS. Computing all 51 features took 32.04 seconds and peaked at 6.34 GiB
aggregate process RSS. The job retained 17,559,451 new output bytes. These timings
include stage setup. They do not establish accuracy or throughput for every
country/source partition. The configured 200-batch projection doubles measured
time and checks output/RAM headroom before continuing.

To shorten the critical path without changing the modeling protocol, fixed full
and development-only model fits can run independently while OOF jobs proceed.
Their reserved/test scores are cached without evaluating the reserved outcomes.
The selected model and policy still come from all four development OOF folds;
only then is the selected reserved score evaluated. Final selection reuses the
corresponding full-data model and its test scores. Both model families, all rows,
all candidates, 350 GBDT trees, and the existing decision grid are retained.

Private test job `amazon-er-delivery-check` version 4 ran 47 synthetic tests in
37.58 seconds: 45 passed and two platform-specific tests were skipped. Snapshot
SHA256: `60b89acbb462538fb93ab9a6aa1e314d2fbf0dc4c8c638a4a10479bab0320e70`.
Tests check no premature reserved evaluation, all-row fitting, completed-model
reuse, and dependency-safe job priority. Full production model scores remain
unmeasured at this milestone; no trained submission or leaderboard gain is claimed.

## Full candidate and feature caches verified, 05:18 UTC

All six training partitions and all five test partitions have completed. The
assembly jobs verified complete batch coverage and equal candidate/feature row
counts. Test assembly finished with exit code zero in 22.02 seconds, peaking at
0.151 GiB RSS. Its saved manifests contain 997 batches, 9,969,589 secondary
queries, and 147,697,378 candidate pairs with all 51 features. The candidate
policy remains the full union of three top-6 retrieval lists. Zero test truth
counts in this manifest mean labels are unavailable, not zero recall.

The full training candidate coverage report spans 2,206,821 S1 entities:

| Population | Candidate link recall | All true matches covered | Mean candidates |
| --- | ---: | ---: | ---: |
| Overall | 0.955892 | 0.883585 | 68.907 |
| India | 0.921669 | 0.809182 | 71.623 |
| US | 0.978764 | 0.933229 | 67.095 |

These are candidate-retrieval diagnostics on known training truth, not classifier
validation or leaderboard scores. Full/reserved GBDT fits and the first two
development OOF fits are running. No trained model score or improved submission
has been produced yet. Sources are the private train/test assembly job outputs
under `cache/managed_kaggle/results`; row-level artifacts remain uncommitted.
