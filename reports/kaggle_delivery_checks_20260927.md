# Remote delivery and checkpoint checks, 2026-09-27

Private Kaggle job `lakshaytechai/amazon-er-delivery-check`, version 1, tested
the reviewed working-tree snapshot before its milestone commit. Snapshot archive
SHA256: `e91d273b8394ac0c850d849c72826267db7b4858b8c44b326edd02e86d5c2e28`.

Python 3.12 completed 43 synthetic tests in 18.33 seconds: 41 passed; the optional
GPU equivalence and Windows-specific checks were skipped. Exit status was zero.
This includes resumed versus uninterrupted model fitting, partition assembly,
full synthetic export/package validation, normalization equivalence, scheduler
dependency checks, streamed download resumption, and rejection of an incorrect
TSV hash. No full challenge data was used by this test job.

Earlier local delivery checks completed 41 tests (40 passed, one optional GPU
skip) within the Windows 3 GiB/two-core job, with 0.211 GiB peak resident memory.
Two subsequent local attempts stopped when system available RAM fell below
4 GiB. Those interrupted attempts are not passes. Testing moved to Kaggle without
relaxing the laptop guard or changing the full-data pipeline.

The production preparation job remains separate. Synthetic test success does
not establish full-data accuracy, runtime, or leaderboard performance.

## Index recovery follow-up

Version 2 tested snapshot
`5aa99529ba42154adbe6b305fd0076ec8e25f1a19b2ca5bc503cab4b30249268`.
All 44 tests completed in 26.34 seconds: 42 passed, two platform-specific checks
skipped. A simulated interruption during vectorizer serialization leaves no
completed-channel marker; restarting rebuilds that channel, and a subsequent run
reuses the finished index. Sparse matrices, vectorizers, reference IDs and label
arrays now publish through temporary files. The original private train index has
all three completion reports and remains reusable. Retrieval quality/settings
are unchanged. The production partition jobs had not started before this fix.

Authenticated streaming download was also checked against a private Kaggle JSON
output and matched the independently downloaded report's SHA256. Large final
TSVs await the production run.

## Automatic capacity gate

Version 3 tested snapshot
`45ecee7607fcec03c062b581bb692426da7fa87c2feb603514cba763d4f781f9`.
All 45 tests completed in 26.04 seconds: 43 passed and two platform-specific
checks skipped. The initial controller plan can now continue to the full plan
only after verified resource reports pass the configured capacity thresholds.
Tests cover missing evidence, incomplete jobs, safe estimates and excessive
projected runtime. This gate does not replace each remote stage's RAM, disk,
output-size and time guards, or establish full-population performance.
