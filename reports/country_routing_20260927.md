# Country transfer findings and routing confirmation

Both private country experiments completed on the full cached candidates with
350-tree GBDT models and seed 42. Each selects a threshold from four grouped OOF
folds in its fitting country, then freezes that policy and its country-trained
model before evaluating the other country's development entities. Reserved fold
4 is excluded. The candidate index and unsupervised representation are shared;
the foreign country's labels are excluded from model and threshold fitting.

| Fitting country | Same-country OOF macro F0.5 | Other country | Transfer macro F0.5 | Transfer singleton FP rate |
|---|---:|---|---:|---:|
| US | 0.958001 | India | 0.784538 | 0.367033 |
| India | 0.923010 | US | 0.890308 | 0.331376 |

The US experiment covered 1,058,639 US development entities and 706,818 India
transfer entities; the India experiment reverses those populations. The mixed
model's same-country OOF scores were 0.952701 for US and 0.913228 for India. The
country-only models therefore support investigating specialists for known
countries; their transfer results do not support substituting either specialist
for all test countries. The 259,452 France test references have no corresponding
training-country labels, so France performance remains unmeasured.

The US worker completed in 20,727.56 seconds with peak process-tree RSS 7.517 GiB;
India completed in 9,886.29 seconds with peak RSS 5.565 GiB. Both resource guards
report exit code zero and no stop reason. Source hashes, complete aggregate
reports, configurations and resource measurements are preserved in
`country_transfer_20260927.json`. No row-level records are committed.

## Follow-up protocol

`src.cloud_country_audit` reuses all original mixed-model and country-specialist
OOF score shards. It verifies model-fitting masks, scoring masks, model hashes,
complete manifests, reference folds, country membership and exact candidate/label
alignment before comparing scores. It does not fit models, recompute features,
rerun blocking or truncate candidates. Reserved scores never enter the audit.

For each known country, choose between mixed and specialist scores and select
an absolute threshold using folds 0/1 only. Relative threshold remains zero,
as selected by all three preceding model experiments. Check those frozen choices
on folds 2/3 against the deployed mixed-model policy (threshold 0.625).
The gate requires overall macro F0.5 improvement above 0.0005 and no country's
macro F0.5 falling by more than 0.002. Unseen countries retain the mixed model and
its existing threshold in any subsequent production experiment.

This is exploratory cross-fitted confirmation: the routing hypothesis followed
inspection of all-development aggregate scores, and OOF models share other
folds during fitting. It is not an untouched, nested or independently selected
holdout. Neither the audit nor the existing country models automatically changes
the submission. Passing the gate supports a follow-up full-country fit, test
scoring and complete TSV validation; it does not establish a leaderboard gain.

## Resources, checkpoints and launch

Use the existing private Kaggle CPU class, approximately 31.3 GiB RAM and four
CPUs, with no GPU. Budget below 4 GiB working RAM for two source histograms,
references and shard buffers. The NumPy checkpoint and its atomic replacement
need approximately 3 GiB of temporary disk headroom; retain the 18 GiB output
budget and runtime resource guard. These are preflight estimates; the completed
remote job must record actual usage.

After every 128 fold/shard groups, atomically persist the two complete count/hit
histograms and the next input offset in one NumPy archive. A resumed run reuses
that checkpoint, repeating only work since its last commit. The interval,
threads and session budget are configurable in `configs/country_route_audit.json`.
Final policies and aggregate grids are JSON reports. All source Parquet scores
remain immutable checkpoint inputs.

Build the private `amazon-er-v4-country-routing` job with the
`country-route-audit` stage and the above overrides. Its checkpoint dependencies
are `amazon-er-v3-prepare`, `amazon-er-v3-gbdt-0` through `amazon-er-v3-gbdt-3`,
`amazon-er-v3-country-us`, `amazon-er-v3-country-india`, and
`amazon-er-v3-validation`, all under `lakshaytechai`. Add it to the existing
controller plan only after synthetic checks pass and the source is committed.
Preserve the other remote jobs and the single controller when loading that plan.

## Implementation verification

The bounded local full suite ran 64 tests in 199.97 seconds: 63 passed and one
optional GPU test was skipped. The resource guard recorded 224.48 seconds total,
0.210 GiB peak process-tree RSS, and 5.841 GiB minimum available system RAM, with
the hard 3 GiB memory limit, two logical CPUs, below-normal priority and five-minute
timeout retained. All input data for these tests were synthetic.

New checks compare histogram/routed metrics with direct entity counts, ensure
confirmation metrics cannot change policy selection, reject reserved/fitted
references and mismatched candidates, validate model-fitting scopes, and simulate
an interruption after an atomic histogram checkpoint. Resume skips the completed
shard; completed results avoid rescanning scores. No full-data local run occurred.

## Completed confirmation and production follow-up

The private `amazon-er-v4-country-routing` job completed at source revision
`d02e93062f4e753da0dfb8ce750e333b8b0d3adb`. It checked 122,061,620 OOF score rows
per model family. Its worker took 1,270.31 seconds and peaked at 1.730 GiB RSS,
with exit code zero and no guard stop. Complete aggregate evidence is in
`country_routing_confirmation_20260927.json`.

Selection folds 0/1 chose the India specialist at threshold 0.625 and the US
specialist at the float32 threshold 0.6499999761581421. Both relative thresholds
are zero. The frozen rules passed the predefined confirmation gate on folds 2/3:

| Confirmation population | Mixed baseline F0.5 | Routed F0.5 | Difference |
|---|---:|---:|---:|
| All 882,728 references | 0.937069 | 0.944247 | +0.007178 |
| India | 0.913493 | 0.923241 | +0.009748 |
| US | 0.952834 | 0.958293 | +0.005459 |

Routed confirmation link precision was 0.980733 and recall was 0.892466. The
unchanged exploratory/cross-fitting limitations above still apply. This is not a
new leaderboard score and does not establish that the user's target of 0.99 is
reachable. The most recent user-reported leaderboard score remains 0.918.

`src.cloud_country_final` fits one final 350-tree model on **all training folds**
of each selected country, using every saved candidate pair and all 51 features.
These production fits are not evaluated on their own training labels. Scoring
uses all test candidates of the corresponding country. A separate assembly job
checks exact candidate alignment against the existing full mixed-model scores,
retains their original probabilities for France and any other unseen country,
and preserves the complete 147,697,378-pair test union. Export applies frozen
country thresholds directly to raw model probabilities. It retains the mixed
threshold of 0.625 for unseen countries, includes all test references, and runs
the strict streaming and unchanged organizer validators.

Model checkpoints retain the existing 25-tree resume interval. Test scores and
routed merges checkpoint each Parquet shard; merge markers include SHA-256 and
row counts. Cache identities include the audit, references, model/score manifests,
and production source. Mismatched scopes, identities, candidates or altered
completed shards stop the run. Original submission artifacts remain unchanged.

Prepare six private CPU jobs with `scripts/build_country_final_plan.py` and
`configs/country_final.json`: two fits, two test scorers, score assembly, then
validated export. The builder creates a proposed plan only; the existing single
controller must be reloaded safely with its current journal. New deliverables
download to `output/kaggle_delivery/country_routed`, separate from the original
submission. All original running audit/ablation jobs retain their state.

Resource planning uses the previously measured full mixed fit (15.052 GiB peak
RSS) and specialist experiments (5.565 GiB India, 7.517 GiB US, across smaller
cross-validation fits). Budget up to 16 GiB working RAM per final country fit,
below 6 GiB per scorer/merge, and below 12 GiB for export/sorting/validation. Use
the existing approximately 31.3 GiB Kaggle CPU class, four threads, with no GPU.
These are estimates, not new measurements; row-count RAM checks, output/disk
limits and remote process guards remain enabled. Candidate generation and
features are reused unchanged. No AWS jobs or full-data local runs are needed.

Production implementation checks: the complete bounded synthetic suite ran
69 tests in 110.46 seconds (68 passed, one optional GPU test skipped). The guard
recorded 114.30 seconds total, 0.205 GiB peak RSS and at least 6.143 GiB available
system RAM, with two logical CPUs, a 3 GiB hard memory limit and five-minute
timeout. New tests verify full-country fitting masks including fold 4, test-only
country scoring, exact candidate preservation, unchanged unseen-country scores,
frozen float32 threshold boundaries, interruption/resume, completed shard hashes,
gate rejection, and rejection of incorrect scopes or missing/duplicate candidates.
