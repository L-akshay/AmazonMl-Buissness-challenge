# Brief coverage and execution status

The attached brief describes both a submission pipeline and a research agenda.
The code does not claim unexecuted experiments or fabricate validation scores.

| Requirement | Implementation / evidence |
|---|---|
| Input audit, schemas, ownership and split integrity | `src/audit.py`; full original-data audit already completed |
| Exact entity macro F0.5, including singletons | `src/evaluate.py`, `src/model.py`, `src/cloud_model.py`; synthetic checks |
| Full test coverage and valid IDs | Strict streaming checks plus unchanged organizer validator |
| Name/address/token sparse retrieval | Full top-6 union in remote path; prior narrower and full-union pilots measured |
| Rare-token clipping and numeric evidence | Existing index clipping and 51 pair features |
| Avoid capacity-driven data/candidate reduction | Remote path uses all entities and every candidate; no negative sampling |
| Train-only supervised fitting | Four S1-grouped OOF folds; reserved fold 4 after configuration freeze |
| Logistic and GBDT comparison | Remote streaming logistic and batched LightGBM on the same feature cache |
| Conservative decision and singleton analysis | OOF threshold/relative-best comparison and per-match-count reports |
| Model/feature checkpoints and tunable resource settings | Atomic Parquet shards, fold/model checkpoints, config and preflight |
| Multiple scored submissions | Distinct OOF-supported policies over the same cached full-test scores |
| Methodology, code and supporting package | Generated from actual completed run reports by `src.handoff` |
| Nested country-held-out diagnostics, calibration and several ablations | Implemented in original `src/model.py`; not yet ported/executed at full scope in the new remote path |
| OOF iterative hard-negative reweighting | Not implemented in remote path; all existing candidate negatives are retained |
| Secondary conflict policy | Ownership audited; enforcement remains off until a leakage-safe comparison supports it |
| Explicit top-two ambiguity/singleton model | Optional further experiment; current relative-best gate does not implement a separate singleton classifier |
| BM25, numeric retrieval, raw-vs-clipped IDF and French-normalization sweeps | Research agenda not fully executed; no benefit claimed |
| Multilingual embeddings / neural branch | Conditional on sparse-system evidence; not selected, downloaded or run |

The source tree and submission path are operationally complete only after the
remote notebook finishes all stages and its actual outputs pass validation.
Before that, the existing E01 TSV remains the only full-data validated submission.
Green synthetic tests demonstrate code paths, not leaderboard quality or full-scale
runtime/memory. No completion claim is made for the entire research agenda.
