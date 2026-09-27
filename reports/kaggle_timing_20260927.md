# Kaggle timing evidence, 2026-09-27

Private CPU jobs used the full supplied reference index. These probes measure
throughput and equivalence, not model accuracy or leaderboard performance.

| Probe | Measured result |
|---|---:|
| Python normalization, 300,000 source rows / three fields, one thread | 124.55 s |
| Python normalization, same rows / fields, four threads | 229.64 s |
| SQL ASCII path plus unchanged Unicode fallback, same rows / fields | 1.60 s |
| Full reference sparse index load | 6.62 s |
| Three top-6 lists for 1,000 secondary queries, CPU / four threads | 1.82 s |
| Retrieval worker resident memory after probe | 0.822 GiB |

The normalized output hash aggregate was identical across all three runs:
`2765191090974905195592302`. Native SQL applies only when UTF-8 byte length equals
character length (ASCII). In that case NFKC is the identity and ASCII casefold
equals lowercase. Unicode retains the original Python NFKC/casefold/mark-preserving
normalizer. A synthetic equivalence test covers randomized ASCII, missing values,
Latin accents, Indic combining marks, compatibility characters and Korean text.

Raw columns, full row populations, candidate policies and model features are
unchanged. Source jobs: `amazon-er-timing-check` and `amazon-er-native-check`
under the private `lakshaytechai` Kaggle account. Raw reports/logs remain in
ignored local caches and private Kaggle outputs.

The managed runner itself completed 36 synthetic tests on Kaggle's actual Python
3.12 environment (34 pass, optional GPU and Windows-specific tests skipped).
Local checks after native normalization and scheduler changes: 40 tests,
39 pass and one optional GPU skip, 99.54 seconds wall time, 0.212 GiB peak resident
memory inside the 3 GiB / two-logical-CPU Windows job.
