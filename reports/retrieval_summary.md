# Sparse retrieval milestone

The pilot uniformly sampled 5,143 S2/S3 records using `hash(entity_id) % 2000 = 0`
and searched all 2,206,821 S1 references. There are 3,839 labeled true links in the
sample. This measures sampled link retrieval, not entity-level model F0.5.

| Retriever/policy | Sample link recall | Mean candidates per secondary |
|---|---:|---:|
| Name character top 6 | 48.55% | Up to 6 |
| Address character top 6 | 80.88% | Up to 6 |
| Combined-token top 6 | 94.22% | Up to 6 |
| All three top-6 lists | 95.65% | 14.71 |
| Name 2 + address 3 + token 6 | 95.26% | 8.82 |
| Token 6 + ambiguous-query name/address rescue | 95.21% | 6.79 |
| Original score-ratio adaptive pruning | 91.74% | 6.65 |

Selected policy: keep token top 6 for every secondary. Query name top 2 and address
top 3 unless the best token score is at least 0.6 and exceeds the second score by
at least 0.15. Preserve the complete union for classification. No country filter.

The score-ratio pruning was rejected for losing roughly four percentage points
of recall. Rare-term probing (2/3/5 terms, pool 40) and posting-list caps (100/500/
2000) were also rejected: neither preserved sufficient recall for its speed gain.
All observed runs remain in `experiments/` and the corresponding JSON reports.

The optional GPU implementation quantizes TF-IDF values with scale 30,000 and
uses integer accumulation. Exact-score ties choose the smallest reference index.
On the pilot, the GPU name-2/address-3/token-6 union also recovered 95.26% of links.
First-use GPU timings include CUDA kernel compilation; they are not steady-state
throughput benchmarks. CPU inference remains available. GPU acceleration uses
only sparse lexical evidence; it is not an embedding or neural matching model.

Unsupervised fitting uses S1 text from the same split without labels. The index
is therefore transductive with respect to validation S1 text; all supervised
training, decision tuning and final holdout scoring remain entity-grouped.

Implementation references: [sparse-dot-topn](https://github.com/ing-bank/sparse_dot_topn)
and [CuPy installation documentation](https://docs.cupy.dev/en/stable/install.html).
These were software-documentation checks only; no challenge identity lookup was used.
