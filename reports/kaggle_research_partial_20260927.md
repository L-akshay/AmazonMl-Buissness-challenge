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
