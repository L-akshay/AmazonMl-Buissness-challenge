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

## Full-data GBDT fit verified, 05:32 UTC

`amazon-er-v3-full-gbdt` completed successfully. The saved model metadata records
152,065,947 training pairs and all 51 features. The downloaded model has exactly
350 trees, matching its configuration, and its SHA-256 matches `complete.json`.
Its conservative scalar parameter bound is 170,800. Fitting took 5,047.09 seconds
(84.12 minutes), peaking at 15.052 GiB aggregate process RSS on the private CPU
session. The remote resource guard recorded no stop reason and exit code zero.

The controller started `amazon-er-v3-test-score-gbdt` using this saved model and
the assembled full test feature cache. Reserved and OOF fitting/scoring remain
in progress. No threshold has been selected from test data, and no validation
score or trained submission is claimed at this milestone. Model artifacts stay
in private Kaggle outputs and ignored local cache, not Git.

## Model selection and reserved evaluation verified, 10:29 UTC

All eight development OOF jobs and both reserved scoring jobs completed. The
validation job finished with exit code zero in 2,866.72 seconds, peaking at
1.086 GiB RSS. On 1,765,457 development entities, macro F0.5 was 0.936898 for
GBDT versus 0.821198 for streaming logistic regression. GBDT was selected with
threshold 0.625 and relative-to-best cutoff zero using development OOF only.

After freezing the model and policy, reserved matcher evaluation on 439,240
entities yielded macro F0.5 0.937302, link precision 0.976287 and link recall
0.884442. Reserved macro F0.5 was 0.913934 for India and 0.952849 for the US.
These are internal measurements, not leaderboard results. Earlier baseline and
retrieval aggregates had been viewed; the reserved result is a supervised
matcher holdout, not a completely untouched end-to-end pipeline holdout.
Previously inspected examples were excluded under the recorded exclusion rule.

Three distinct upload policies were frozen from development OOF before reserved
evaluation: best (threshold 0.625, relative 0), precision (approximately 0.775,
relative 0.8), and recall (approximately 0.475, relative 0). The latter two have
development macro F0.5 0.932554 and 0.932378. Their reserved performance has not
been used for selection. Aggregate evidence is in
`reports/kaggle_validation_20260927.json`; the private job retains the complete
threshold grid. Final model reuse and export remain pending at this milestone.

## Export recovery: writable sort spill

The first export attempt stopped before producing submissions. DuckDB's external
sort tried to create its default temporary folder beside the checkpoint database,
whose symlink resolves into Kaggle's read-only input mount. The worker exited
with an I/O error after 60.06 seconds, peaking at 3.918 GiB RSS. The saved model,
test scores, selected policies, and completed validation remain unchanged.

Database connections now explicitly use writable `cache/duckdb_spill`, capped
at 8 GB. Operators can override the directory with `ER_DB_TEMP_DIRECTORY` and
the cap with `ER_DB_MAX_TEMP_DIRECTORY_SIZE`. This changes storage placement,
not candidate selection, features, fitting, or output decisions. Recovery reruns
export against the same cached scores and retains the failed attempt's evidence.

The bounded local suite passed 53 tests (one skipped) in 105.98 seconds, with
0.218 GiB peak process-tree RSS. A new regression forces a 300,000-row sort to
spill under a 32 MB DuckDB memory budget, verifies complete sorted output, and
checks that the read-only database's SHA-256 is unchanged. The test spill cap is
128 MB; production uses the 8 GB cap above.

## Resume validation of saved TSVs after CSV field-limit failure

The corrected export produced all TSV variants and the shared candidate file,
then stopped in strict validation because a candidate field exceeded Python's
default 131,072-character CSV limit. No candidate list is truncated. The parser
limit now accommodates the complete target population (up to 14 characters per
encoded target including separators), and restores the caller's limit afterward.

The explicit `validate-export` recovery stage validates existing TSVs against
all required S1 rows, valid target IDs, candidate-subset invariants, and the
expected full candidate count. It runs the unmodified organizer validator and
persists the validated files from immutable checkpoint inputs. It avoids
repeating retrieval, features, training, scoring, and the completed export sort.
The failed export remains available as a private checkpoint; a distinct recovery
job supplies validated outputs to packaging and downloads.

The bounded suite passed 54 tests (one skipped) in 118.53 seconds, peaking at
0.206 GiB process-tree RSS. Regression coverage includes a real CSV field over
128 KiB, restoring parser limits on success/failure, revalidating saved exports
without regeneration, and rejecting an incorrect expected candidate count.
