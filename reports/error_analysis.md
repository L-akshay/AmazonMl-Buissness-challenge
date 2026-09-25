# Baseline error analysis

- missed_by_blocking: 7,539,082
- rejected_true_candidates: 0

The exact joint-name/address blocker deliberately establishes a precision-oriented floor. It cannot retrieve most typo, abbreviation, partial-address, transliteration or reordered-text pairs. The next experiment is name character retrieval followed by independent address retrieval, measuring added recall and volume. The 100-row deterministic error sample is baseline_errors.tsv; it is not used to manually edit predictions.
