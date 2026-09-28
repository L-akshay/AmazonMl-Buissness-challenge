# Country decision confirmation and V5 export plan

The private `amazon-er-v5-routed-decision-audit` completed successfully using all
122,061,620 unchanged development OOF pairs and cached country specialist scores.
It reproduced V4's baseline before comparing decision rules. The complete stage
took 54.24 minutes and peaked at 1.142 GiB process RSS on Kaggle CPU.

Selection on folds 0/1 chose exclusive secondary ownership at threshold
0.6000000238418579 for India, and exclusive ownership with a soft runner-up
penalty (power 0.25), at the same threshold, for the US. Exact best-score ties
abstain for these countries. Each reference may still receive many secondary
matches. This is not a one-to-one matching constraint on references.

On cross-fitted confirmation folds 2/3, macro F0.5 rose from 0.9442466793 to
0.9468676332 (+0.0026209538). India improved by 0.0037786998 and US by 0.0018467566.
Link precision increased from 0.980733 to 0.983130 and recall from 0.892466 to
0.895618. These exploratory confirmation folds are not a fresh untouched holdout;
no leaderboard score has been measured for this change. Full-test secondary
competition includes all references, unlike development-only OOF competition.

The confirmed policy supports preparing another submission. The new stage
`apply-country-decisions` transforms cached V4 test probabilities and retains
every candidate pair, including rejected competitors with score zero. All
unknown-country probabilities and the 0.625 fallback threshold remain unchanged.
The stage binds the decision evidence to V4's original routing audit, checks
source shard hashes and secondary boundaries, saves each transformed Parquet
shard with a checksum, and resumes completed shards without rescoring models.
Its separate final manifest feeds the existing strict and organizer validators.
The previous V4 delivery stays available.

Resource plan: existing Kaggle CPU, four threads, no GPU. Allow 4 GiB for the
per-shard transform, and 8 GiB for export (previous measured export peak 4.315
GiB). Both fit the approximately 31 GiB worker with reserved headroom. Existing
eight-hour worker, nine-hour external and 18 GiB saved-output limits apply.
Separate transform and export jobs preserve resumability. No blocking, features,
training or candidate expansion are repeated for this submission variant.

During monitoring, the cached OAuth access token expired at 04:30:12 UTC.
Installed CLI credentials logic delayed automatic expiry recognition; a single
refresh through the existing OAuth credentials restored access. No credentials
were exposed, permissions changed, jobs duplicated or remote workers restarted.
The controller resumed normal reads and downloaded the completed audit evidence.

Validation: all 77 synthetic tests passed (one optional GPU skip) in 115.51 seconds. The bounded Windows supervisor completed in 136.51 seconds with 0.214 GiB peak process-tree RSS and at least 7.933 GiB available system RAM. New checks verify tie abstention, all-pair and unknown-country preservation, full transform resume without recomputation, and rejection of changed source shards. The 3 GiB / two-CPU / five-minute limits remained enabled.
