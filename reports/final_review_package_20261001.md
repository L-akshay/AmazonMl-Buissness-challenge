# Original leaderboard submission: final review package

On 2026-10-01 the user confirmed that only the original approximately 0.91
submission had been uploaded. V4/V5 remained unsubmitted experiments. The final
review archive therefore preserves the original matching file and its candidate
set rather than substituting a later result with no leaderboard score.

Local delivery: `output/final_review_submission/L-akshay_submission.zip`.
Team metadata uses the established default `L-akshay` and `Lakshay Dawar`; the
user was asked to confirm the final team roster during packaging.

## Required contents

- `output/matching_results.tsv`: identical to the original local upload,
  SHA-256 `4095f52419dad13ee9b24650d340de54330e52971bacfecbd68250371181d42f`.
- `output/candidate_pairs.tsv`: the original full blocking output,
  SHA-256 `f1b1a345737583551c7eaa9c9b98a2e7e8b925db786759f71b8d87d78c29a535`.
- `code/business_entity_resolution/`: archived original source revision
  `5ae44ad5889dcdabe40eeea2619ca9ddc601c71c`, pinned requirements, updated README,
  original trained LightGBM model, and a checked reproduction entry point.
- `Documentation_template.md`: all sections of the organizer methodology filled
  with the actual submitted method, metrics, blocking and evaluation limitations.

Original production settings are 51 features, 350 trees, seed 42, threshold
0.625 and no relative cutoff. The bundled model's SHA-256 is
`da1839c71e5301404414f54ac564916fed6989fa50872d93a2754a453b4ef1aa`.
It was downloaded from the original completed full-training job and verified
against that job's completion manifest. No new Kaggle jobs were launched.

## Reproduction and verification limits

The entry point recomputes blocking/features from organizer inputs, scores with
the original model, validates outputs, and requires both final hashes to match.
It does not copy the top-level supplied output files. Fresh retraining is also
available in a separate workspace. Original training reused partial token caches;
fresh training retrieval can differ slightly, so bitwise retraining equivalence
is not claimed. A new full-data clean-room run was not performed for packaging.

The original algorithm is preserved. The archived cloud configuration guard's
maximum cache allowance was raised from 20 to 512 GiB; the selected standalone
remote configuration uses 128 GiB on a worker with at least 32 GiB RAM and
200 GiB free scratch. This changes resource allowance, not candidate/model logic.

Finalization streams the original files, checks aligned reference rows, pair
counts, list uniqueness, match-subset relationships, source compilation, model
identity, ZIP member CRCs, and rehashes the final compressed outputs. Existing
remote organizer/strict validation remains the evidence for full input-ID checks.
The current repository suite ran 80 tests: 79 passed and one GPU test was skipped.
It used a bounded local process: 2 logical CPUs, 3 GiB committed-memory cap,
below-normal priority and a five-minute timeout; measured peak RSS was 0.215 GiB.

Archive checksums, counts, resource observations and packaged-source test results
are in [the verification receipt](final_review_package_20261001.json). No private
data, prediction TSV, model binary or ZIP is committed to Git.
