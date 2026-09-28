# Validated V5 country-decision submission

Upload `output/final_submission_v5/matching_results.tsv` to the Amazon ML
matching-results field. This new TSV has been downloaded and independently
checked locally; V4 remains available in `output/final_submission_v4/`.

The private export job `lakshaytechai/amazon-er-v5-decision-export` completed
successfully at source revision `84a1f97a5486711602395b1bdbb2b45784998257`.
The strict streaming validator and unchanged organizer validator with ID checking
passed remotely. Local streaming checks confirmed the header, row count and
SHA-256, including a separate delivery copy. No full matching workload ran locally.

| Check | Result |
|---|---:|
| Test reference rows, excluding header | 1,732,544 |
| Complete retained candidate pairs | 147,697,378 |
| Accepted matches | 5,537,442 |
| Predicted singletons | 110,381 |
| File size, bytes | 93,813,159 |
| Organizer and strict validators | PASS |
| Download and delivery-copy hashes match remote | PASS |

SHA-256: `3573f848e36f9b65811ade60ad9f485954d3316afc25d6dfc25412b46d43f16d`.

The new decision rule uses the same cached country specialist models as V4.
India applies exclusive secondary ownership; the US additionally discounts
the winning score using the runner-up probability to power 0.25. Both use the
frozen float32 threshold 0.6000000238418579. Exact best-score ties abstain for
these countries. Each reference can retain multiple secondary matches. All
candidate pairs remain present, including rejected competitors with zero score.
Unknown countries retain their original probabilities and threshold 0.625.

| Remote stage | Worker minutes | Peak RSS GiB |
|---|---:|---:|
| Apply confirmed decisions to cached test scores | 5.14 | 0.269 |
| TSV export and validation | 15.62 | 4.293 |

Both stages exited successfully without resource-guard stops. Cached blocking,
features and models were reused. Source shard hashes, exact candidate totals and
secondary batch boundaries were checked before export. Aggregate receipts and
resources are in `country_decision_submission_20260928.json`; row-level data and
TSVs are excluded from Git.

Exploratory cross-fitted confirmation F0.5 improved from V4's 0.9442466793 to
0.9468676332, with gains in both labeled countries. See
`routed_decision_confirmation_20260928.md` for selection, confirmation and limits.
This is not an untouched holdout or leaderboard result; France has no training
labels, and full-test secondary competition can differ from development.
The last user-reported portal score remains 0.918. Submit V5 and compare its
portal result before choosing the final submission. Neither a 0.99 score nor
an improvement on the hidden leaderboard has been established.

At delivery, the single controller remains active for the full retrieval-depth
audit. No additional training, duplicate jobs or candidate reduction were launched
as part of delivery. The completed code milestone passed 77 synthetic tests
(one optional GPU skip) under the bounded local runner before remote execution.
