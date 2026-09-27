# Validated country-routed submission

The new submission is ready locally at
`output/final_submission_v4/matching_results.tsv`. Upload this TSV directly to
the Amazon ML leaderboard portal. The previous submission remains unchanged in
`output/final_submission/`. Do not upload the validation receipt or substitute
the supporting package for a portal field requesting matching results.

The private Kaggle job `lakshaytechai/amazon-er-v4-routed-export` completed at
source revision `a357abfec1c56a576fb70d43c591720e24616b88`. The controller downloaded
the validated artifact automatically. A local streaming check independently
verified its header, row count and SHA-256; a separate delivery copy has the same
hash. Full matching, candidate, entity-ID and organizer checks ran on Kaggle.

| Check | Result |
|---|---:|
| Test reference rows, excluding header | 1,732,544 |
| Complete test candidate pairs | 147,697,378 |
| Accepted matches | 5,534,847 |
| Predicted singletons | 110,480 |
| File size, bytes | 93,779,797 |
| Unchanged organizer validator with ID checking | PASS |
| Streaming coverage and candidate-subset checks | PASS |
| Downloaded and delivery-copy hashes match remote | PASS |

SHA-256: `2fc86d9fc7943b2b2247fd41b52bf9a1c3e5f78536cf1779e1e7ae790ce0d2e5`.

India uses its full-country GBDT at threshold 0.625; US uses its full-country
GBDT at float32 threshold 0.6499999761581421. Relative thresholds are zero.
France and all other unseen countries keep the original mixed-model scores
and threshold 0.625. Exact per-shard candidate alignment, scope and model
identities were checked before assembly; every candidate was retained.

| Completed remote stage | Worker minutes | Peak RSS GiB |
|---|---:|---:|
| US test scoring | 15.99 | 0.393 |
| India test scoring | 13.83 | 0.399 |
| Score assembly | 4.97 | 0.411 |
| TSV export and validation | 15.43 | 4.315 |

Every stage exited successfully with no resource guard stop. Full training
resources and population counts are documented in
`country_final_fits_20260928.json`. The complete score/export manifests,
resource measurements, thresholds and validation receipt are preserved in
`country_routed_submission_20260928.json`. No row-level predictions or model
binaries are committed.

Internal exploratory confirmation macro F0.5 improved from 0.937069 to 0.944247
on grouped OOF folds 2/3 after policy selection on folds 0/1. As detailed in
`country_routing_20260927.md`, this is cross-fitted exploratory evidence, not an
untouched holdout or a leaderboard result. France has no training-country labels.
The last user-reported portal score remains 0.918 for the previous submission;
the new portal score and the target of 0.99 are unverified. Submit the new TSV and
compare the portal score before choosing between submissions.

At delivery, the original single controller continues the full retrieval-depth/
decision audit and the frequency/retrieval-feature ablations. These remaining
research jobs are separate from this completed submission. Full experiments
remain on Kaggle; local work in this delivery step was limited to file copying,
streaming integrity checks and aggregate evidence maintenance.
