# Completed decision audit and routed follow-up

The completed private `amazon-er-v5-decision-audit` evaluated all existing
development OOF candidate scores, without refitting, sampling or changing the
submission. Selection used folds 0/1; confirmation used folds 2/3. These are
cross-fitted exploratory results, not untouched validation or leaderboard scores.

The rule selected on folds 0/1 was exclusive secondary ownership with a soft
runner-up penalty (power 0.25), threshold 0.5749999881. Confirmation macro F0.5
increased from 0.9370690779 to 0.9395049882 (+0.0024359103). Both labeled countries
improved. This remains below the delivered V4 country model result of
0.9442466793 on the same confirmation population. The V4 TSV is therefore kept.
We do not choose another rule by its confirmation score.

Across all four development folds, 270,099 true links were absent from the
candidate pool, and the baseline missed another 439,358 retrieved true links.
Perfect classification of the existing candidates has an oracle macro F0.5 of
0.9833418271 (India 0.9684664747; US 0.9932736045). This is a ceiling for this
development candidate pool, not a bound on the hidden test score or achievable
model performance. Reaching 0.99 on this development population requires better
retrieval as well as better decisions. The separate full-population top-6/12/20
retrieval audit continues from its preserved aggregate checkpoints.

The completed audit took 32.66 minutes with 1.129 GiB peak process RSS. Its
aggregate results and provenance are recorded in `decision_audit_20260928.json`.

The next `routed-decision-audit` evaluates the same six decision families with
the cached country scores used by V4. It rechecks original grouped fitting and
scoring identities, exact candidate coverage, secondary shard boundaries, and
reproduces V4's internal baseline before accepting a result. It selects a rule
and threshold per country using folds 0/1 only, then compares confirmation
against V4. Unknown-country behavior is not changed; France has no labels.
No export is authorized by a weaker result against the old mixed-model baseline.

All candidate/model/feature artifacts are reused. Joined score shards are cached
as Parquet and each completed threshold grid as JSON. Interrupted work resumes
from completed shards/grids. The previous audit's measured 1.129 GiB plus the
additional per-shard specialist reads supports a conservative 4 GiB process
estimate on the existing four-CPU, roughly 31 GiB Kaggle CPU environment; no GPU
is needed. Allow eight worker hours, nine external hours, and 18 GiB saved output
with existing runtime guards. Expected runtime is approximately 35–60 minutes,
based on the previous audit, and can vary with Kaggle I/O.

Validation: all 75 synthetic tests passed (one optional GPU skip) under the bounded Windows runner. Supervision took 112.23 seconds, peak process-tree RSS was 0.200 GiB, and available system RAM stayed above 8.023 GiB. Tests cover complete routed-pair coverage, V4 baseline reproduction, checkpoint reuse without recomputing competition, and rejection of missing prior audit evidence.
