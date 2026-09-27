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
| Multiple submission variants | Three OOF-supported policies over the same cached full-test scores; all passed full-data organizer validation and download checksum checks |
| Methodology, code and supporting package | Generated from actual completed run reports by `src.handoff` |
| Nested country-held-out diagnostics and feature-group ablations | Full-cache implementations in `src/cloud_research.py`; inner country-only OOF freezes the threshold before evaluating the other country. Awaiting full-data execution |
| OOF calibration comparison | Original implementation in `src/model.py`; remote comparison remains pending |
| OOF iterative hard-negative reweighting | Not implemented in remote path; all existing candidate negatives are retained |
| Secondary conflict policy | Ownership audited; enforcement remains off until a leakage-safe comparison supports it |
| Explicit top-two ambiguity/singleton model | Optional further experiment; current relative-best gate does not implement a separate singleton classifier |
| BM25, numeric retrieval, raw-vs-clipped IDF and French-normalization sweeps | Research agenda not fully executed; no benefit claimed |
| Multilingual embeddings / neural branch | Conditional on sparse-system evidence; not selected, downloaded or run |

The trained submission path completed full-data validation and local delivery on
2026-09-27. The primary TSV is `output/final_submission/matching_results.tsv`,
with precision and recall alternatives in subdirectories. All variants cover
1,732,544 test entities; the unchanged candidate union has 147,697,378 pairs.
See `reports/kaggle_production_20260927.md` for integrity evidence and
`reports/kaggle_validation_20260927.json` for model-selection results and caveats.
The supporting code/methodology ZIP is downloaded and all 121 members passed
streaming integrity checks; its embedded primary TSV matches the validated hash.
Country/ablation research remains in progress. No leaderboard score or
completion of the entire research agenda is claimed.
