# Experiment Queue and Decision Rules

Do not run advanced experiments automatically. Each experiment should answer a concrete question.

## E00 — Dataset audit

Establish sizes, missingness, singleton rate, match-count distribution, collisions and secondary-ID ownership.

## E01 — Exact normalized baseline

Build the end-to-end floor and verify scoring/output.

## E02 — Name char retrieval

Compare char/char_wb and n-gram ranges. Sweep K. Measure recall and candidate volume.

## E03 — Address char retrieval

Measure incremental true links recovered beyond name retrieval.

## E04 — Sparse union

Union name + address retrieval and quantify unique contribution.

## E05 — Token / BM25 retrieval

Add only if it recovers useful additional true links at acceptable volume.

## E06 — Rare-token retrieval

Compare raw IDF and clipped IDF behavior.

## E07 — Numeric retrieval

Measure whether generic numeric keys recover missed links or mainly add noise.

## E08 — Adaptive candidate budget

Compare fixed K against min-K / max-K / relative-score retention. Protect recall first.

## E09 — Logistic regression baseline

Check whether engineered evidence is already linearly separable.

## E10 — GBDT baseline

Compare LightGBM and/or XGBoost on identical folds and candidate sets.

## E11 — Typed hard negatives

Buckets:

- same-name/different-address;
- same-address/different-name;
- rare-token collision;
- numeric conflict;
- acronym collision;
- retrieval false positive.

Compare with untyped negative sampling.

## E12 — Iterative hard-negative mining

Use OOF false positives. Retrain once or twice. Stop when improvement saturates.

## E13 — Address / numeric feature expansion

Add number overlap, contradiction, postal-like agreement and containment.

## E14 — Frequency-aware features

Add shared IDF, weighted Jaccard and clipped variants. Run ablation.

## E15 — Retrieval metadata

Add rank, score, retriever count, reciprocal rank and rank disagreement.

## E16 — Calibration

Compare none vs Platt vs isotonic on OOF scores.

## E17 — Margin / ambiguity decision

Compare threshold-only against threshold + margin/ambiguity features and optional singleton model.

## E18 — Secondary-ID conflict handling

Only if training labels support uniqueness. Compare none vs greedy dominant claim vs margin-aware rejection.

## E19 — Leave-one-country-out

Train US -> India and India -> US. Identify geographically brittle components.

## E20 — French-normalization ablation

Compare general-only normalization against general + small French auxiliary vocabulary.

## E21 — Dense multilingual retrieval

Only after sparse system is strong. Measure incremental true links, candidate growth and dense false positives.

## E22 — Dense cosine feature

If E21 is useful, add dense similarity to the matcher and measure incremental value.

## E23 — Sparse+dense fusion

Compare union, weighted score fusion and RRF.

## E24 — Final ablation

Remove each major component individually.

## E25 — Seed/fold stability

Repeat strongest variants. Reject fragile one-off wins.

## E26 — Final train + test inference

Freeze normalization, blocking, candidate policy, features, model, calibration, threshold and conflict policy. Retrain on full train and generate final TSVs.

# Decision rules

Keep a component only if it improves at least one of:

1. candidate recall at acceptable volume;
2. macro F0.5;
3. precision at similar recall;
4. country-held-out robustness;
5. runtime/memory without meaningful quality loss.

# Highest-ROI priorities

1. exact metric;
2. leakage-safe folds;
3. blocking recall;
4. hard negatives;
5. numeric/address contradictions;
6. threshold tuning;
7. singleton protection;
8. retrieval metadata;
9. clipped rare-token weighting;
10. adaptive candidate budget;
11. embeddings only after the above are strong.

# Failure-mode-driven development

Good experiment:

```text
Observation: false positives are dominated by same-name/different-address records.
Hypothesis: numeric contradiction features reduce these errors.
Experiment: add numeric conflict features.
Decision: compare macro F0.5 and singleton FP rate.
```

Bad experiment:

```text
Add a transformer because transformers are powerful.
```
