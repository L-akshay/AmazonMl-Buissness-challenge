# Style, Structure and Build Order

## Code style

- no placeholder files;
- no `pass` or TODOs left behind;
- concise docstrings;
- comment only non-obvious logic;
- prefer plain functions;
- avoid unnecessary classes and abstraction;
- no invented benchmark numbers;
- never claim code ran unless it actually ran.

## Documentation style

Docs should say:

- what was done;
- why;
- what was observed;
- what it means.

Avoid marketing language, emojis, vague SOTA claims and undocumented assumptions.

## Repository layout

```text
business_entity_resolution/
├── .gitignore
├── README.md
├── requirements.txt
├── Documentation_template.md
├── dataset/
│   ├── train/
│   └── test/
├── src/
│   ├── normalize.py
│   ├── blocking.py
│   ├── features.py
│   ├── model.py
│   ├── evaluate.py
│   └── run_pipeline.py
├── notebooks/
├── experiments/
│   ├── folds.tsv
│   └── results.csv
├── reports/
│   ├── data_audit.md
│   └── error_analysis.md
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
└── utils/
    └── validate_submission.py
```

Datasets and outputs should normally be git-ignored.

## File creation policy

Create files only when the current step needs them. Do not generate empty future modules.

## Build order

1. Setup — repo, `.gitignore`, pinned requirements.
2. Data inspection — verify files/schema, audit data, inspect truth structure.
3. `evaluate.py` — exact scorer and unit test.
4. Multi-view normalization.
5. Rule-based end-to-end baseline and validator pass.
6. Blocking retrievers and blocking diagnostics.
7. Adaptive candidate-budget experiments.
8. Pairwise features.
9. Logistic baseline + LightGBM/XGBoost with GroupKFold.
10. Typed hard-negative mining and OOF iterative mining.
11. Threshold, calibration and relative-margin decision tests.
12. Secondary-ID conflict check and optional resolution.
13. Error analysis by blocking/matcher/decision failures.
14. Optional embedding branch.
15. Final ablations and fold/seed stability.
16. Retrain selected architecture on full training data.
17. Run test inference.
18. Write both outputs and run validator.
19. Complete methodology write-up and README.
20. Package final zip.

## Reproducibility

Use:

- fixed random seeds;
- saved fold assignments;
- relative paths;
- pinned dependencies;
- deterministic preprocessing;
- one documented final run command.

## Final package

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

## Commit messages

Use short present-tense messages, e.g.:

- `add exact entity scorer`
- `add multi-view normalization`
- `add name char blocker`
- `add adaptive candidate pruning`
- `add typed hard negatives`
- `tune precision-first decision rule`
