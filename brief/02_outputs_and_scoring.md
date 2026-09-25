# Outputs and Scoring

## Required files

Both files belong in `output/` and are tab-separated.

### `matching_results.tsv`

```text
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S3-00812
S1-00002
```

Rules:

- exactly one row per test S1;
- S2/S3 IDs only;
- IDs must exist in test data;
- no duplicate IDs;
- empty means no match.

### `candidate_pairs.tsv`

```text
source1_entity_id    candidate_entity_ids
S1-00001             S2-00047,S3-00812,S3-00999
```

This must contain the exact final set scored by the matcher after all blocking/fusion/pruning. Every final match must appear in the candidates for that S1.

Assert this before writing output.

## Validator

Run before every submission:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Do not submit unless it prints PASS.

## Metric

Per S1 entity:

```text
F0.5 = (1.25 * P * R) / (0.25 * P + R)
```

Equivalent:

```text
F0.5 = 5TP / (5TP + FN + 4FP)
```

So one FP contributes four times as much as one FN in the denominator.

Singleton behavior:

- true singleton + predicted empty -> 1.0;
- true singleton + any predicted match -> 0.0;
- true non-singleton + no correct prediction -> 0.0.

Implement this exactly in `evaluate.py`.

Unit-test the organizers' example:

Prediction: `[S2-00047, S2-00193, S3-00812]`

Truth: `[S2-00047, S3-00812]`

Expected F0.5 ≈ `0.714`.

## Model-selection metric

Primary:

- entity-level macro F0.5.

Diagnostics:

- precision;
- recall;
- singleton false-positive rate;
- per-country F0.5;
- per-match-count F0.5;
- candidate recall;
- calibration.

Pair accuracy and ROC-AUC are not final selection objectives.

## Thresholding

Threshold 0.5 has no special status. Tune on out-of-fold predictions.

Also evaluate:

- best score;
- second-best score;
- top1-top2 margin;
- ambiguity/entropy;
- retrieval-channel support.
