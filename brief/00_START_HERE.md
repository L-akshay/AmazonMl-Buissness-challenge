# Business Entity Resolution (Amazon ML Challenge 2026) — Start Here

This is the master brief for the project. Read the numbered files in order.

## Files

| File | Purpose |
|---|---|
| `01_problem_and_data.md` | Task, schema, data audit |
| `02_outputs_and_scoring.md` | Output files, validator, exact F0.5 |
| `03_constraints.md` | Competition restrictions and licensing |
| `04_approach.md` | Decided V3 architecture |
| `05_hard_cases.md` | Difficult match patterns and expected signals |
| `06_validation.md` | Splits, leakage, diagnostics, ablations |
| `07_style_and_structure.md` | Repo layout, code style, build order |
| `08_experiments.md` | Required experiment queue and decision rules |
| `09_AGENT_MASTER_PROMPT.md` | Master operating prompt for the coding agent |

## Read-first order

Before writing code, read 01, 02, 03 and 07. Then summarize:

- the task;
- the data sources;
- scoring edge cases;
- hard constraints;
- validation strategy;
- build order.

Do not begin modelling until the dataset files are located and their schemas are verified.

## Core philosophy

This is not just pair classification. The full problem is:

1. retrieval / candidate generation;
2. evidence extraction;
3. pair scoring;
4. conservative entity-level decision making.

Candidate generation sets the recall ceiling. The metric is precision-heavy, so no-match must be treated as a first-class outcome.

The V3 refinements are:

- multi-view normalization;
- multi-retriever blocking;
- adaptive candidate budgets;
- retrieval-channel agreement features;
- clipped rare-token weighting;
- typed hard-negative mining;
- OOF thresholding and calibration;
- top-1/top-2 margin and ambiguity features;
- explicit singleton analysis;
- secondary-ID conflict handling only if supported by training labels;
- optional multilingual embeddings only if validation shows value.

## Source of truth

If files conflict, use this priority:

1. `03_constraints.md`
2. `02_outputs_and_scoring.md`
3. `01_problem_and_data.md`
4. `04_approach.md`
5. `06_validation.md`
6. `07_style_and_structure.md`
7. `08_experiments.md`
8. `05_hard_cases.md`

Never invent dataset properties. Never claim a score or run unless it actually happened.
