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
