# Full-data audit (E00)

One source row is one business record; truth has one row per reference S1 entity.

| Source | Rows | Duplicate IDs | Missing name | Missing address |
|---|---:|---:|---:|---:|
| train_source1 | 2,206,821 | 0 | 0 | 0 |
| train_source2 | 5,034,616 | 0 | 0 | 168,967 |
| train_source3 | 5,285,603 | 0 | 0 | 175,916 |
| test_source1 | 1,732,544 | 0 | 0 | 0 |
| test_source2 | 4,887,273 | 0 | 0 | 129,408 |
| test_source3 | 5,082,316 | 0 | 0 | 136,098 |

Training singletons: 123,247/2,206,821 (5.58%).

## Integrity checks

- truth_duplicate_s1: 0
- missing_truth_s1: 0
- unknown_truth_s1: 0
- duplicate_truth_links: 0
- unknown_truth_targets: 0
- secondary_ids_with_multiple_s1: 0
- cross_split_id_overlap: 0

## Interpretation and scope

Name/address collisions are expected in entity resolution: exact name or address alone is not sufficient identity evidence. Their full counts and country distributions are in data_audit.json.

Country is an open set. France performance cannot be measured from training labels. No event timestamps are supplied; freshness and temporal drift cannot be assessed.

Raw UTF-8 fields are retained. MacOS resource forks were excluded at extraction. No external identities or lookup services were used.

Inspect the exact queries in src/audit.py and notebooks/data_audit.ipynb. Input ZIP paths, members, byte sizes and CRCs are in input_manifest.json.

Runtime: 188.91 seconds.
