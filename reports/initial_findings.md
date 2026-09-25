# Initial findings and development decisions

Evidence: full-data `data_audit.json`, derived by `src/audit.py` from the supplied
seven TSVs. All counts below describe the supplied files, not a live external source.

- **Integrity: passed, high confidence.** Zero duplicate IDs, missing/extra truth
  S1 IDs, duplicate truth links, unknown truth targets, secondary IDs assigned to
  multiple S1s, or train/test ID overlaps. Stable automated checks enforce the
  source-ID and label-reference contract.
- **Name collisions: high modeling risk, high confidence.** 845,385 of 2,206,821
  training S1 records share an identical raw name with another S1. Exact names
  cannot establish identity on their own; address evidence and ambiguity checks
  are required. Collisions are expected data properties, not rows to delete.
- **Missing addresses: medium modeling risk, high confidence.** Training S2/S3
  have 168,967 and 175,916 missing addresses, respectively. An exact joint
  name/address baseline cannot retrieve those records; independent name retrieval
  and explicit missingness features are needed.
- **Country shift: high generalization risk, high confidence.** Test S1 contains
  259,452 France records; training contains no France records. Country must remain
  open-set. US/India held-out checks will be a proxy, not proof of France quality.
- **Singletons: material decision requirement, high confidence.** 123,247 training
  S1s (5.58%) have no match. Output generation preserves empty rows, and the exact
  scorer gives a correctly empty singleton 1.0.
- **Ownership: eligible for a later experiment.** No secondary ID maps to multiple
  S1s in the supplied truth. This permits testing a uniqueness policy later; it
  does not establish that a test-time uniqueness assumption will improve quality.

The 2,000-pair positive sample includes heuristic signals for suffix variants,
typos, token reorder, address abbreviations, script changes and weak-field cases.
These overlapping tags are diagnostic approximations; they are not human-reviewed
noise labels. No event timestamps exist, so temporal anomalies and freshness
cannot be assessed. The original archive metadata forks are excluded; the actual
TSVs loaded successfully as UTF-8 without byte replacement or repair.

E01 is a fixed-rule development floor. E02 should test character name retrieval
against its missed-link population and measure candidate recall before training
a more complex matcher. Advanced V3 experiments and final packaging remain ahead.
