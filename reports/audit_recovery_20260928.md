# Recovery of the remaining audit

The single controller exited after a DNS failure during a Kaggle status read
and a Windows sharing/permission error while replacing its journal. Direct
Kaggle reads subsequently succeeded. The audit notebook reported
`CANCEL_ACKNOWLEDGED`; its saved output lacks `REMOTE_CHECKPOINT.json` and a final
resource report. Its 36,000-second external limit equalled the worker's ten-hour
budget, leaving no allowance for setup and teardown. The saved progress log is
consistent with an external timeout, but the remote cancellation cause is not
independently reported.

There are 85 contiguous, complete aggregate checkpoint files covering 8,500,000
of 10,320,219 training queries. Each references the same unchanged retrieval
configuration. Their file hashes are recorded in the private recovery plan and
checked on Kaggle before mounting the partial output. The remaining 1,820,219
queries retain the full index, all three retrieval channels and top-6/12/20
comparisons. Completed groups skip retrieval on resume. No candidate or dataset
reduction is introduced, and no full computation runs locally.

The recovery runtime accepts a missing lifecycle marker only with an explicit
reviewed partial-checkpoint proof. It validates safe relative cache paths, every
supplied file hash and a unique matching input. This does not mark the cancelled
job complete; the stage's existing identity/population checks still run. The
scheduler now treats a cancelled/error job without a lifecycle marker as needing
investigation, rather than waiting indefinitely for a marker that cannot appear.
Windows journal replacement retries transient sharing failures with a bounded
backoff while preserving the last complete journal.

Resume retrieval and run the cached-score decision analysis in separate private
CPU jobs, preserving one controller and the original journal history. The
decision analysis no longer needs to wait for retrieval-depth diagnostics. Use
the existing approximately 31.3 GiB, four-CPU Kaggle class, with an estimated
working budget below 16 GiB for sparse retrieval and below 6 GiB for decision
histograms. These are planning estimates; the cancelled run has no final peak
measurement. Both recovery jobs retain RAM/disk/output guards, an eight-hour
worker budget and a nine-hour external limit, leaving setup/teardown margin.

The delivered V4 TSV and its validation receipt remain unchanged. Aggregate
checkpoint inventory and incident provenance are in `audit_recovery_20260928.json`.

The full synthetic suite passed under the unchanged Windows guard: 72 tests
(one optional GPU skip), 148.75 seconds including supervision, 0.208 GiB peak
process-tree RSS, and at least 7.166 GiB available system RAM. New checks cover
transient and persistent journal replacement failures, cancellation without a
lifecycle marker, and rejection of unsafe, changed or ambiguous partial inputs.
The three-GiB hard memory limit, two logical CPUs and five-minute timeout remained
enabled. No test used the full organizer data.

## Mounted-output recovery check and fallback

The first resumed retrieval job stopped before computation because the mounted
cancelled-notebook output did not provide a unique hash-verified partial input.
Its failure manifest is retained; it did not repeat retrieval or modify the V4
submission. The independent decision-analysis job started successfully.

The fallback bundles only the already downloaded and verified aggregate
configuration and 85 chunk JSON files into a new private job's source archive.
These are small aggregate counters, not organizer rows or candidate predictions.
The builder restricts restored paths to the cache, verifies each source hash,
and the remote runtime verifies all 86 hashes again before starting. Source
manifests still exclude caches from Git and submission source packaging. Normal
prepared inputs are mounted separately; the unchanged audit reuses its original
configuration and completed chunk markers. No cancelled-notebook input or its
read-only log files need to be mounted for this fallback.
The fallback passed all 73 synthetic tests (one optional GPU skip) in 103.69 seconds. Bounded supervision took 106.57 seconds with 0.218 GiB peak process-tree RSS and at least 7.679 GiB available RAM; the same 3 GiB / two-CPU / five-minute limits remained enabled.
