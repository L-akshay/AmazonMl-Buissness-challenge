# Reproduce the original leaderboard submission

This package corresponds to the team's **only uploaded leaderboard submission**
(user-reported score approximately 0.91). It uses the original mixed-country
LightGBM matcher, all 51 features, threshold **0.625**, and no relative cutoff.
It does not contain the later V4/V5 predictions or their country decision rules.

The ZIP's `output/matching_results.tsv` is byte-identical to the original local
leaderboard upload. Its SHA-256 is
`4095f52419dad13ee9b24650d340de54330e52971bacfecbd68250371181d42f`.
The complete blocking candidate file is provided alongside it. Both outputs
passed the unchanged organizer validator and strict coverage/subset validation.

## Contents

- `src/`: the archived original pipeline and `reproduce_original.py` entry point.
- `resources/original_submission/model/`: the original fitted LightGBM model,
  its feature columns, training configuration and checksum. It was trained only
  on organizer training data; these are not external pretrained weights.
- `resources/original_submission/manifest.json`: exact model/output checksums.
- `resources/original_submission/cloud_*.json`: original measured run evidence.
- `configs/submission.json`: CPU settings for a standalone remote machine.
- `requirements.txt`: exact dependency pins, including Python-version-specific
  SciPy pins; use Python 3.12 on Linux for the reproduction environment.
- `utils/validate_submission.py`: the unchanged organizer validator.
- `tests/`: synthetic pipeline checks. Other archived reports/docs are historical;
  this README and the root `Documentation_template.md` describe this submission.
- `SOURCE_PROVENANCE.json`: original source revision and packaging-only changes.

No private Kaggle notebook, GitHub checkout, external identity service, or remote
model download is needed. Standard package installation needs a package mirror
or Internet access. The only additional data required are the organizer's seven TSVs.

## Remote environment and input placement

Use a Linux CPU worker with at least **32 GiB RAM** and **200 GiB free scratch disk**.
No GPU is required. The original full GBDT fit peaked at 15.052 GiB process RSS;
allow headroom for the operating system, data preparation and export. Do not run
the complete pipeline on a memory-constrained laptop. Run commands from this
`code/business_entity_resolution/` directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export ER_REMOTE_COMPUTE=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=42
```

Place the original files as follows (do not edit or subsample them):

```text
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

Alternatively, the supplied helper extracts exactly these filenames from the
organizer's ZIP into the folders above:

```bash
python -m src.handoff attach-data --source /absolute/path/to/organizer_data.zip
```

Do not run `attach-data` twice over an already populated destination. A fresh
run must not contain unrelated `cache/` artifacts. The code records input hashes.

## Reproduce both outputs with the submitted fitted model

```bash
python -m src.reproduce_original --mode inference --stage all
```

This executes preparation, full test blocking, all-pair features, inference with
the bundled original model, export and validation, and finally publishes:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

The outputs above are relative to this code directory. They are regenerated
from organizer data; the entry point does not read/copy the ZIP's top-level
prediction files. At publication, it checks both regenerated hashes against the
submitted files and fails explicitly if either differs. Do not ignore that
failure: inspect the recorded environment and cache provenance.

For sessions that cannot finish in one run, execute the same stages separately:

```bash
python -m src.reproduce_original --stage prepare
python -m src.reproduce_original --stage retrieve-test
python -m src.reproduce_original --stage features-test
python -m src.reproduce_original --stage score
python -m src.reproduce_original --stage export
python -m src.reproduce_original --stage publish
```

Each heavy stage runs in an isolated process, keeps its completed checkpoints,
and checks RAM/disk/session headroom. Rerun the interrupted stage with the same
configuration, then continue. Logs and resource reports are under `reports/`.
Keep the complete project workspace on persistent storage between sessions.
Do not clear candidate or feature caches between model experiments.

## Retrain and evaluate from organizer data

Use a **separate fresh copy** of this code folder so that inference and retraining
manifests cannot be confused. Install dependencies and place the same data, then:

```bash
python -m src.reproduce_original --mode retrain --stage all
```

This runs full training blocking/features, four entity-grouped OOF folds, model
and threshold selection, the reserved matcher evaluation, full fitting, test
blocking/features, scoring, export, and publication. Both GBDT and streaming
logistic candidates use every retained pair. Detailed original selection evidence
is bundled for comparison. Individual stages can be selected with `--stage`.

Fresh retraining is an audit of the training procedure; it is not asserted to
produce bitwise-identical model weights across platforms. The original production
run reused earlier token-channel retrieval checkpoints for part of training.
Fresh integer-search top-6 retrieval has a small measured candidate difference
from that frozen training cache. The bundled original model therefore provides
the prediction-reproduction route for the actual submitted model. Retrain mode
prints output hashes and whether they equal the submitted files without relabeling
fresh model results as the historical leaderboard submission.

## Resource settings and validation status

`configs/submission.json` exposes threads (4), feature workers (2), query batches
(10,000), feature chunks (8,192), prediction batches (20,000), DuckDB budget (4 GB),
seed (42), trees (350), and the session budget (11 hours per stage). The original
Kaggle saved-output configuration limit was adapted to allow a 128 GiB cache on
the standalone 200 GiB scratch worker; model/retrieval/feature logic is unchanged.
Kaggle's saved-output quotas require partitioned sessions; do not apply this
single-machine disk configuration to a small Kaggle output volume unchanged.

The original full-data outputs were validated remotely. This review package was
checked for source completeness, pinned requirements, output integrity, model
identity and ZIP CRCs. The reproduction wrapper was tested on synthetic fixtures.
A new full clean-room inference/training run was **not** performed during final
packaging, so no additional real-data reproduction run is claimed.

Additional checks after reproduction:

```bash
python -m unittest discover -s tests -v
python utils/validate_submission.py --matching output/matching_results.tsv --test-dir dataset/test --check-ids
```

The pipeline already validates the entire candidate/match subset relationship
with its streaming validator. The organizer validator stays unchanged.
