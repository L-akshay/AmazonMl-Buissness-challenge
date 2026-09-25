# Business entity resolution

Amazon ML Challenge 2026 implementation based on the supplied V3 brief.

## Milestone E00: audit and scoring foundation

Implemented strict TSV ingestion, a full-data integrity audit, reproducible
positive-pair sampling, the exact macro F0.5 scorer, and multi-view Unicode
normalization. Raw data is preserved; external identity lookup is not used.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p test_core.py -v
python -m src.audit
python -m src.sample_pairs
```

Place the seven organizer files under `dataset/train` and `dataset/test`.
The supplied validator is preserved in `utils/validate_submission.py`.
The data audit and aggregate findings are in `reports/`; inspect exact queries
with `notebooks/data_audit.ipynb`. Original task specifications are in `brief/`
and `reference/`. Python 3.13.5 and DuckDB 1.5.5 were used locally.

The full audit covers 24,229,173 source records and 7,638,365 true training links.
Training singletons account for 5.58%. All checked ID/reference invariants pass.
No trained model or test-quality estimate is available at this milestone.

Datasets, caches, predictions, row-level samples, and generated fold assignments
are intentionally excluded from Git. The next milestone is the exact-match
end-to-end baseline, with test output validation.
