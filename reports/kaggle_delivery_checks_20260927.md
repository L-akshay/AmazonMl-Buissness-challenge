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
