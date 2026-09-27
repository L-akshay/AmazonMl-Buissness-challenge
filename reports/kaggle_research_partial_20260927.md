# Completed research findings, 2026-09-27

The full-data address and numeric feature ablations both reduce development
macro F0.5. Retain both feature groups. These completed experiments do not
justify replacing the currently delivered TSV.

## Comparable feature ablations

All three runs use the same full candidate cache, 1,765,457 development entities,
6,109,380 true links, grouped folds 0-3, 350-tree GBDT and seed 42. Each experiment
selects its own threshold on development OOF scores: 0.625 for all features and
approximately 0.65 for both ablations; relative cutoff zero throughout. There is
no pair/negative subsampling and no new use of reserved fold 4. These are
OOF-selected development measurements, not untouched holdout or portal scores.

| Feature experiment | Overall macro F0.5 | India | US | Delta vs all features |
|---|---:|---:|---:|---:|
| All 51 features | 0.936898 | 0.913228 | 0.952701 | 0 |
| Remove address group | 0.890090 | 0.866358 | 0.905934 | -0.046808 |
| Remove numeric/postal group | 0.916149 | 0.894319 | 0.930724 | -0.020749 |

Both ablations regress in every development fold and in both countries.
The address group includes address retrieval scores/ranks, address similarities,
numeric/postal evidence and address interactions; it is broader than just the
address string. Numeric/postal removal includes its name-conflict interaction.
Neither experiment removes candidate pairs or reruns blocking.

## India-only fitting and country transfer

The India-only experiment scores 0.923010 on its 706,818 India development
entities using inner grouped OOF. Its model and threshold then freeze before
scoring 1,058,639 US development entities, yielding macro F0.5 0.890308 and
singleton false-positive rate 0.331376. This is evidence of transfer difficulty,
not a France accuracy estimate. The mixed-country model has a different fitting
population; its US OOF metric is not a controlled country-transfer comparison.
Country-specific fitting remains a research hypothesis, not a promoted change.

## Remote resources and evidence checks

| Completed experiment | Worker hours | Peak process-tree RSS (GiB) |
|---|---:|---:|
| ablation-address | 4.811 | 7.963 |
| ablation-numeric | 5.668 | 9.138 |
| country-india | 2.746 | 5.565 |

All three saved project manifests report complete; resource reports have exit
code zero and no stop reason. The local check read only aggregate JSON reports,
verified comparable entity/truth counts and every-fold regression, and saved
source SHA-256 values, configurations, policies and metrics in
`kaggle_research_partial_20260927.json`. No full-data local computation ran.
No executable code changed; the numerical evidence checks passed without
rerunning the synthetic implementation suite.

The US transfer experiment, frequency and retrieval-feature ablations, and
retrieval-depth/decision audit were running at 17:28 UTC. The separate retrieval
feature ablation removes score/rank inputs to the classifier; it does not test
deeper retrieval. The pending depth audit is the experiment comparing top-6,
top-12 and top-20 against the complete training index. Further model changes
require measured improvements; no new submission or leaderboard gain is claimed.

## Frequency ablation completed (2026-09-28 IST)

The private `amazon-er-v3-ablation-frequency` experiment completed at revision
`5ae44ad5889dcdabe40eeea2619ca9ddc601c71c`. It removed six frequency/rarity
features: log name frequency, log address frequency, shared rare tokens, maximum
shared IDF, summed shared IDF and weighted Jaccard. All original candidate pairs
were retained; the classifier used the remaining 45 features, 350 trees and
seed 42. Reserved fold 4 was excluded. Both compared OOF policies selected the
same absolute threshold 0.625 and relative threshold zero.

| Grouped development OOF population | All 51 features | Without frequency features | Difference |
|---|---:|---:|---:|
| All 1,765,457 references | 0.936898 | 0.930356 | -0.006541 |
| India, 706,818 references | 0.913228 | 0.904852 | -0.008376 |
| US, 1,058,639 references | 0.952701 | 0.947385 | -0.005317 |

All four fold-level macro F0.5 values also decreased. Link precision fell from
0.976242 to 0.972379 and recall from 0.883874 to 0.873217. Retain all six
frequency-related features; this experiment provides no reason to refit or
change the already validated country-routed submission. This is an exploratory
OOF-selected development comparison, not an untouched holdout or a leaderboard
score, and it supplies no labelled France evaluation.

The worker took 19,343.71 seconds (5.373 hours), peaking at 9.668 GiB RSS, with
exit code zero and no resource stop. The local evidence check verified identical
entity/truth populations and policies, plus regression in both countries and
every fold. Complete aggregate reports, resource measurements, configuration,
checkpoint provenance and source hashes are in `frequency_ablation_20260928.json`.
Only aggregate metadata was processed locally; no executable code changed.
The retrieval-feature ablation and retrieval-depth/decision audit remain active.
