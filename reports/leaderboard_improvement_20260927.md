# E03: decision diagnostics after first leaderboard feedback

The user reported a portal score of **0.918** for `matching_results.tsv` on
2026-09-27 and requested improvement toward 0.99. The primary file had been
recommended immediately before this feedback; the portal's uploaded-file hash
has not been independently verified. Record this as user-reported leaderboard
feedback, separate from local validation. The user confirmed the filename but
did not independently confirm the displayed metric label; the organizer README
defines entity-macro F0.5.

The existing primary TSV and two alternatives remain unchanged. The public
leaderboard is a subset of test entities and final ranking uses a private subset.
A 0.99 outcome is an aspiration, not a measured or guaranteed result.

## First improvement experiment

`src.cloud_decisions` reuses all four existing GBDT OOF score checkpoints. It
joins score shards by their original secondary-record batch, validates complete
competitor grouping and OOF reference membership, and saves enriched Parquet
shards. It does not rerun retrieval, features, fitting, or full test scoring.

The experiment measures the candidate-set oracle, unretrieved true links,
retrieved-but-rejected links, false-positive links, and secondary IDs assigned to
multiple references by the baseline. The oracle is a mathematical ceiling using
training labels, not an achievable model or a test-score forecast.

Six decision families are compared: independent scores, unique secondary winner,
two winner-margin gates, and two competitor-score penalties. Exact top-score
ties abstain instead of using an arbitrary ID tie-break. Multiple secondary
matches to one reference remain allowed. No candidate is dropped from the cache.

Threshold/rule selection uses only OOF folds 0/1. Folds 2/3 supply confirmation
metrics and per-country regression checks. Fold 4 and test labels are not read.
Because the underlying models are cross-fitted, this confirmation is not a new
untouched or fully nested model holdout. Competition occurs within the 80%
development reference pool, so full-test competition can differ. A new submission
is supported only after confirmation improves macro F0.5 by more than 0.0005 and
neither country's macro F0.5 falls by more than 0.002. The audit itself never
overwrites or generates a submission.

## Resource and continuation plan

Use an existing private Kaggle CPU session class (approximately 31.3 GiB RAM,
four CPUs, no GPU). One threshold scan needs approximately 0.75 GiB for its two
int32 histograms; including references, joined shards, and processing buffers,
budget below 4 GiB working RAM. Estimate enriched score storage at up to 5 GiB;
keep the existing 18 GiB output cap and stage headroom checks. Worker threads and
session duration remain configurable. Cache each joined batch and completed rule
grid so interruptions resume without recomputing earlier work.

Existing country-transfer and feature-ablation jobs continue. If the oracle
shows retrieval prevents the target, prioritize additional retrieval channels
and cache only new candidate features. If decision errors dominate, use the
OOF evidence to select a decision or model improvement. Do not infer French
accuracy from the public score alone or tune row-level predictions manually.

## Retrieval ceiling and next experiment

The full-training top-6 candidate audit reports an oracle macro F0.5 of
0.98336072 overall, 0.96848434 for India, and 0.99328692 for the US. This is
the score ceiling for those training candidates under perfect decisions; it is
not a forecast of France-heavy test performance. The same audit reports
0.955892 candidate-link recall and 0.883585 coverage of all true matches.
Threshold changes cannot recover true links absent from the candidate union.

The next Kaggle CPU diagnostic evaluates every training S2/S3 query against the
complete S1 index at per-channel top-6, top-12, and top-20. It reuses the frozen
vectorizers and sparse index, keeps the complete union without a candidate cap,
and reads labels only to measure candidate recall and fanout. It checkpoints
aggregate metrics every ten input batches and writes no row-level labels or
scores. The full-data run is queued behind the four currently running private
country and ablation jobs; no duplicate jobs are launched.

If deeper retrieval recovers enough missed links, the follow-up is to generate
features for the expanded union, fit and compare models on complete cached
candidates, and validate per-country OOF/confirmation metrics before preparing
another TSV. The report does not claim that top-20 alone will produce 0.99.
The user-reported portal score of 0.918 is not independently tied to an upload
hash, so it remains separate from the reproducible training diagnostics.

## Implementation checks

The bounded local suite completed 59 tests: 58 passed and one optional GPU test
was skipped in 110.38 seconds. Peak process-tree RSS was 0.212 GiB and minimum
available system RAM was 7.257 GiB, under the 3 GiB/two-core/below-normal
limits. The retrieval-depth unit test verifies that the uncapped union can
contain 60 distinct candidates across three channels. Heavy retrieval and
feature experiments remain on Kaggle; no full-data run is performed on the
laptop.
New tests check exact competition ties, many targets for one reference, rejection
of duplicate pairs and reserved rows, exact histogram metrics against direct
counts, unretrieved truth in the oracle, and resumed cache reuse without repeating
competition processing. No full-data diagnostic has run locally.
