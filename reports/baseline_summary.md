# E01 baseline result

Full labeled training population; fixed rule, no fitting or threshold selection.
These are development metrics, not OOF model validation or test/leaderboard scores.

| Measure | Result |
|---|---:|
| Macro entity F0.5 | 0.0828189461 |
| Link precision | 1.000000 |
| Link recall / candidate recall | 0.012997939 |
| Singleton false-positive rate | 0.000000 |
| Training candidate links | 99,283 |
| Training true links | 7,638,365 |
| US macro F0.5 | 0.0900156101 |
| India macro F0.5 | 0.0720333127 |
| Test S1 output rows per file | 1,732,544 |
| Test S1s with nonempty matches | 77,282 |

The organizer validator completed with `--check-ids`: PASS, no warnings.
Both TSVs have the required columns, complete S1 coverage, valid target IDs, and
no duplicate IDs. Matches are a subset of the scored candidates.

The blocker missed 7,539,082 labeled links. Increasing retrieval recall is the
next priority. Test quality is unknown; format validation does not measure it.
E02 should compare name character n-gram retrieval, followed by independent
address retrieval and their union. No advanced trained model or final submission
package has been produced yet.

The successful run took 378.69 seconds with raw tables and training S1
normalization already cached. The earlier full ingestion/audit took 188.91
seconds. These are observed stage runtimes, not a cold end-to-end benchmark.

Commands executed successfully:

```powershell
python -m src.audit
python -m src.sample_pairs
python -m unittest discover -s tests -v
python -m src.run_pipeline
python -X utf8 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

The last command was invoked by the pipeline. Nine unit/integration tests pass,
including the exact scorer, Unicode/number handling, and synthetic end-to-end
output consistency. See baseline_metrics.json for full country/fold/match-count
breakdowns and baseline_validator.txt for the validator's output.
