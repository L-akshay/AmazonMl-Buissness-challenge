# Validation Rules

## 1. Split by S1 entity

Never randomly split candidate pairs.

Use GroupKFold or reproducible grouped splits on `source1_entity_id`.

All pairs for one S1 remain in one fold.

Save fold assignments and reuse them across experiments.

## 2. Leakage rules

Supervised components must respect fold boundaries.

This includes:

- classifier;
- calibration;
- threshold;
- model-based hard-negative mining.

Document how unsupervised TF-IDF/IDF fitting is handled.

## 3. OOF predictions

Use OOF scores for:

- threshold selection;
- calibration;
- iterative hard-negative mining where practical.

Never choose final thresholds from training predictions.

## 4. Country-held-out validation

Run:

- train US -> validate India;
- train India -> validate US.

This is only a proxy for France shift, not a faithful simulation.

Report per-country scores.

## 5. Blocking diagnostics

After each blocking change report:

- link-level candidate recall;
- fraction of S1 entities with all true matches covered;
- reduction ratio;
- mean/median/p95 candidates per S1;
- incremental recall contributed by each retriever;
- incremental candidate volume per retriever;
- recall@K for fixed-K tests.

## 6. Adaptive-candidate diagnostics

Compare adaptive vs fixed K on:

- candidate recall;
- mean candidate count;
- p95 candidate count;
- true matches pruned by adaptive logic;
- runtime.

Do not accept meaningful recall loss just for elegance.

## 7. Matcher diagnostics

Primary: entity-level macro F0.5.

Break down by:

- country;
- singleton;
- exactly one match;
- multiple matches.

Also report singleton false-positive rate and pair precision/recall as diagnostics.

## 8. Decision diagnostics

For threshold/margin policies report:

- chosen threshold;
- chosen margin rule;
- predicted singleton count;
- true singleton accuracy;
- false matches removed/added;
- non-singleton recall impact;
- fold-to-fold threshold variability.

## 9. Typed hard-negative diagnostics

For each bucket report:

- count;
- average pre-mining score;
- false-positive contribution;
- post-retraining behavior;
- F0.5 change.

Avoid dominance by one negative type.

## 10. Rare-token ablation

Compare:

- no rare-token features;
- raw IDF;
- clipped IDF.

Check whether extreme weights create false positives.

## 11. France-normalization ablation

Compare:

- general normalization only;
- general + small French auxiliary vocabulary.

The system should not depend critically on French-specific rules.

## 12. Embedding branch validation

If embeddings are added, report:

- true matches recovered only by dense retrieval;
- extra candidate volume;
- dense-specific false positives;
- F0.5 before/after dense cosine feature;
- country-held-out effect.

Remove if value is not measurable.

## 13. Error analysis

Sample errors reproducibly and categorize first as:

- blocking;
- matcher;
- decision;
- conflict handling.

Then assign cause where possible:

- generic name;
- typo;
- suffix;
- transliteration;
- partial address;
- numeric conflict;
- acronym;
- rare-token overconfidence;
- dense semantic collision;
- missing data;
- country shift.

Experiments should target observed failure modes.

## 14. Final ablations

Remove one component at a time:

- address features;
- numeric features;
- rare-token features;
- clipped IDF;
- BM25/token retriever;
- dense retriever;
- retrieval metadata;
- hard-negative mining;
- calibration;
- margin features;
- singleton model;
- conflict handling.

Keep complexity only if it produces reproducible value.

## 15. Stability

For final candidates, repeat across multiple folds/seeds and report spread.

Do not select a fragile one-off score.
