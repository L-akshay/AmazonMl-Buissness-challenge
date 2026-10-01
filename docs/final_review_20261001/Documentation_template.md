# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** L-akshay

**Team Members:** Lakshay Dawar

**Submission Date:** 2026-10-01

## 1. Executive Summary

We resolve business entities using three sparse retrieval channels followed by
a supervised LightGBM classifier. The submitted system retains the complete
union of the three top-six candidate lists, computes 51 comparison features,
and applies an entity-grouped-validation-selected threshold of 0.625.
This write-up describes the original and only leaderboard submission, reported
by the user at approximately 0.91, not later unsubmitted experiments.

## 2. Methodology

### 2.1 Problem Analysis

Business names and addresses exhibit spelling, punctuation, legal-suffix and
formatting variation. Missing or generic fields create ambiguity, while numeric
address/postal disagreements can distinguish otherwise similar businesses.
Training labels cover India and the US; no France accuracy is inferred from
training-country metrics. Singletons and one-to-many matching are evaluated
at the reference-entity level rather than treating random pairs as independent.

### 2.2 Solution Strategy

**Approach Type:** Sparse blocking plus supervised pair classifier.

**Core Innovation:** Complementary name, address and token retrieval with an
uncapped channel union, followed by reusable all-pair features and precision-aware
entity-level threshold selection. All records and retained negatives are used.

Normalization includes Unicode NFKC, case/punctuation/space normalization,
accent-folded and legal-suffix name views, generic street expansions, and ordered
address-number extraction. IDs are used only for bookkeeping, joins and deterministic
fold assignment; their numeric values are not model features.

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Sparse TF-IDF name, address and token channels against
  the complete S1 reference index for each split. Each S2/S3 query retrieves six
  candidates per channel above the configured similarity threshold (0.05).
- **Candidate pairs generated:** 152,065,947 frozen training pairs and
  **147,697,378 test pairs**. The submitted candidate file covers all
  **1,732,544 test reference entities**, including empty candidate lists.
- **How true-match loss was measured:** All training references were evaluated
  against labeled links. Frozen training candidate recall was 0.9558922623;
  perfect-classifier entity-macro F0.5 on that full training candidate pool was
  0.9833607229. These are retrieval diagnostics, not achievable model scores.
  Candidate recall is not perfect; missing matches are explicitly counted.
- **Retention:** Every pair in the three-channel union is scored and included
  in the blocking output. There is no additional union cap or negative sampling.

## 4. Matching Model

**Features used:** 51 cached numeric features covering name/address similarity,
edit and token overlaps, normalized/core name variants, ordered numeric/postal
agreement or conflicts, country consistency, frequency/IDF evidence, retrieval
scores/ranks and interactions. Exact feature names/order are in `src/features.py`.

**Model type:** LightGBM binary GBDT, 350 trees, seed 42, minimum leaf size 100.
The final model was fitted on all 152,065,947 retained training pairs. Its
conservative scalar-parameter upper bound is 170,800, well below eight billion.
Streaming logistic regression was evaluated as a comparator, not an ensemble
component in the submitted predictions.

**Threshold selection method:** Four deterministic S1-grouped OOF development
folds selected model and decision by macro F0.5, including unretrieved truths in
the denominator. The frozen rule accepts probability >= **0.625**, with relative
cutoff zero. Calibration is not applied. Multiple secondary matches per reference
are allowed; the original submission uses no later country-specialist or
secondary-competition rule.

All preprocessing and features use organizer inputs only. No external business
identity lookup, hosted matcher, pretrained neural weights, manual pair edits,
or test labels are used. LightGBM is MIT licensed; the scikit-learn comparator
uses BSD-3-Clause. Exact library versions are pinned in `requirements.txt`.

## 5. Results & Error Analysis

| Measurement | Macro F0.5 |
|---|---:|
| Full development OOF, 1,765,457 S1 entities | 0.93689774199 |
| Reserved matcher evaluation, 439,240 S1 entities | 0.93730237901 |
| Leaderboard, sole submitted file | Approximately 0.91, user reported |

The reserved matcher result excludes 10,703 references touched by earlier
row-level diagnostics. Earlier aggregate retrieval/baseline results had been
viewed, so this is not a wholly untouched end-to-end pipeline holdout. Detailed
exclusion rules and country/match-count metrics are in the bundled original
`cloud_model_validation.json`. Reserved macro F0.5 is 0.9139338770 for India and
0.9528490426 for the US. These are not estimates of France performance.

- **False positives / wrong merges:** Repeated or generic names/addresses and
  incomplete distinguishing fields can trigger incorrect matches. Reserved
  link precision is 0.9762871482 and singleton false-positive rate 0.0864781156.
  These mechanisms are qualitative interpretations; no manual test fixes were made.
- **False negatives / missed matches:** Some true owners are absent from the
  candidate pool; noisy/short records and score thresholding can also miss
  retrieved positives. Reserved link recall is 0.8844424357.
- **Output checks:** 5,520,920 accepted test links and 110,901 predicted singletons.
  The unchanged organizer validator with ID checks passed. Strict streaming checks
  passed for reference coverage, duplicate/valid IDs and matches being subsets
  of the candidate lists. Predictions are not proof of test correctness.

## 6. Conclusion

Complementary sparse retrieval and all-pair supervised matching provide a
reproducible, resource-controlled solution. Grouped validation makes candidate
misses and singleton errors visible while selecting a precision-aware threshold.
The package preserves the exact submitted output and its original model; later
unsubmitted improvements are deliberately excluded from this solution's results.

## Appendix

### A. Code Artefacts

The ZIP contains `output/matching_results.tsv`, `output/candidate_pairs.tsv`, and
the self-contained code under `code/business_entity_resolution/`. The code folder
includes all original pipeline source, the unchanged organizer validator, pinned
dependencies, original run evidence, and the fitted original model trained only
on organizer data. All additional reproduction logic is in `src/reproduce_original.py`.

From the code folder, follow `README.md` to install Python 3.12 dependencies and
place the organizer TSVs. On a remote CPU worker, run:

```bash
export ER_REMOTE_COMPUTE=1
python -m src.reproduce_original --mode inference --stage all
```

This regenerates both outputs from the organizer data and bundled model, then
checks their hashes against the supplied files. A separate fresh workspace can
run `--mode retrain --stage all` to reproduce the training/validation procedure.
Original training reused partial token-retrieval caches; fresh retraining can
differ slightly. The fitted original model is included to avoid substituting
new weights when reproducing the actual submitted predictions.

Full jobs use remote compute; local finalization uses streaming file checks and
bounded synthetic tests. A fresh full-data clean-room run was not performed for
this packaging step. See README resource and reproduction limitations.

### B. Additional Results

Original aggregate validation/candidate reports are in
`code/business_entity_resolution/resources/original_submission/` and `run_reports/`.
The original matching TSV SHA-256 is
`4095f52419dad13ee9b24650d340de54330e52971bacfecbd68250371181d42f`.
Candidate and model checksums are in `resources/original_submission/manifest.json`.
The root `PACKAGE_VERIFICATION.json` records final packaging integrity checks.

No later validation score is presented as the score of the original submission.
