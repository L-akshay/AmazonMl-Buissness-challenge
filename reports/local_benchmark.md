# Bounded local resource benchmark

Timing probe only; no accuracy claim, final model or submission was produced.
The existing output TSV and production caches were not changed.

## Execution controls

- Windows Job Object: hard 3 GiB aggregate committed-memory limit, kill on close.
- CPU affinity: two logical processors; below-normal process priority.
- Five-minute timeout; monitor checks every 250 ms.
- Stop when system available RAM falls below 4 GiB or project-disk free space below 5 GiB.
- No GPU use. DuckDB opens the production database read-only, with one thread,
  384 MB memory and at most 512 MB temporary spill in the benchmark directory.
- One feature worker, two LightGBM threads and one BLAS thread.

## Measured results

100,000 random cached pairs from five evenly spaced completed training batches
(seed 42), including 10,497 positives and 83,875 distinct references. Reference
name/address frequencies are sampled, while token IDF uses the cached index.
This measures operations on real text but is not a production-quality feature
population, entity-grouped accuracy experiment or test-set evaluation.

| Measurement | Observed |
|---|---:|
| Entire guarded worker, including setup and three fits | 36.09 s |
| Peak sampled process-tree resident memory | 0.666 GiB |
| Minimum available system memory | 13.110 GiB |
| Features, 100,000 pairs / 51 columns | 9.63 s |
| Feature matrix | 19.45 MiB |
| LightGBM 350 trees, 25,000 pairs | 2.20 s |
| LightGBM 350 trees, 50,000 pairs | 3.45 s |
| LightGBM 350 trees, 100,000 pairs | 5.90 s |
| Predict 100,000 pairs, last fitted model | 2.46 s |

The worker exited successfully without triggering any stop condition. The test
suite also ran under the same guard: 24 tests, one optional GPU test skipped.
An additional Windows test confirms that the Job Object rejects an allocation
above its hard memory cap, using a separately restricted child process.

## Implications and limits

Using the earlier broader-union pilot average of 14.71 candidates per secondary
would imply about 151,810,421 training pairs and 146,652,654
test pairs. Actual full-data counts may differ. Their 51-column float32 matrices
alone would require 28.8 GiB and 27.9
GiB respectively. This exceeds the current project drive's 23.4 GiB free space
if both are cached uncompressed; model copies, metadata and scratch add more.

A straight feature-only extrapolation is 8.0
hours at this deliberately restricted CPU setting. This is not an end-to-end
ETA: production context setup, disk I/O, broader retrieval, full model training,
OOF/ablations, prediction and export were not measured. No reliable full-fit RAM
or elapsed-time estimate follows from the 100,000-pair model timings.

The observed peak is the small sample's footprint, not a claim that the full
pipeline fits in 0.67 GiB. Existing retrieval checkpoints use the narrower E02
policy. They do not measure the broader candidate policy requested for full
experiments. Mostly S2 records were sampled; France/test throughput is unmeasured.

Full local execution remains unauthorized. No candidates or training examples
were removed from production to meet benchmark limits. Future threshold variants
should reuse fixed cached model scores; new models need their own evaluation.

Reproduce on Windows from the project root:

```powershell
python -m src.local_benchmark
python -m src.local_benchmark --tests
```

Aggregate measurements and code fingerprints: [local_benchmark.json](local_benchmark.json).
