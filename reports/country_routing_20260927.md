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
