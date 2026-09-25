# Repository working agreement

The user requested maintaining this GitHub repository and committing at fair
milestones. Apply this agreement to future work in this project.

- Complete a coherent, reviewable milestone, run the relevant checks, update the
  evidence/documentation, then commit with a short present-tense message.
- Push completed milestone commits to `origin`; do not leave an entire experiment
  sequence uncommitted. Do not commit broken intermediate work merely for cadence.
- Use `experiment/eNN-description` branches for speculative experiments. Keep
  `main` reproducible. Fetch and inspect remote changes before publishing; never
  force-push or rewrite published history without explicit authorization.
- Never commit datasets, row-level samples, fold-ID files, prediction TSVs,
  database caches, credentials, or submission archives. Commit code, aggregate
  metrics, reproducible configurations, and concise interpretation.
- Run `python -m unittest discover -s tests -v` for code changes and
  `git diff --cached --check` before commits. Do not rerun expensive full-data
  experiments unless the change warrants it.
- Keep experiment evidence append-only. Report the population/split, actual
  results, resource assumptions, and limitations. Never label training diagnostics
  as out-of-fold validation or test performance.
- Follow the project's supplied matching constraints. No external identity
  lookup, hosted LLM matcher, manual prediction edits, or pair-randomized splits.
- Preserve organizer files in `reference/` and `utils/validate_submission.py`.
  Enforce stricter output invariants in project code as needed.
- Full experiments belong on AWS: do not reduce dataset size or candidate quality
  just to fit the local laptop. Local work is development, tests and samples only.
  The user specifically authorized a bounded local benchmark, not a full restart.
  Use `python -m src.local_benchmark` for that probe (Windows job memory limit,
  two logical CPUs, low priority, five-minute timeout and system-headroom checks).
- Heavy cloud stages must resume from checkpoints, cache Parquet/NumPy artifacts,
  expose worker/thread/batch controls and reuse blocking/features across models.
  Size CPU/RAM and GPU stages separately before launching billable compute.

See `CONTRIBUTING.md` for the milestone and review workflow.
