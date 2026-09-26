"""Build the reviewable Kaggle handoff notebook (authoring dependency: nbformat)."""
from pathlib import Path
import nbformat as nbf

cells=[]
def md(value):
    cells.append(nbf.v4.new_markdown_cell(value))
def code(value):
    cells.append(nbf.v4.new_code_cell(value))

md("""# Entity resolution: Kaggle handoff

Run the complete dataset on remote compute, cache intermediate work and export
validated `matching_results.tsv` files for the challenge portal.

**Status:** the code paths are tested on synthetic data locally. This notebook has
not been executed in your Kaggle account. Full-data scores and resource fit remain
to be established. Keep the notebook and organizer dataset **private**.

## Goal

Run one stage at a time, save the completed outputs, then continue. Start with a
CPU session; GPU retrieval is optional. GPU memory does not increase system RAM.
Do not shrink the dataset or candidate union to get past a memory check.
""")
md("""## Setup

Add the organizer dataset and supplied code ZIP using **Add Input**. Internet is
needed to install pinned dependencies, and to clone the repo if no code ZIP was
attached. The code never performs external identity lookups.

Set `STAGES_TO_RUN` to the next stage(s). The initial default is preparation only.
For a resumed session, also set `CHECKPOINT_SOURCE` to the saved project folder
under `/kaggle/input`. Keep all earlier checkpoint inputs attached.
""")
code('''from pathlib import Path
import json, os, shutil, subprocess, sys, time, zipfile, stat

assert Path("/kaggle/input").exists(), "Run this notebook on Kaggle, not the laptop"
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/business_entity_resolution")
DATA_INPUT = INPUT  # Or an explicit extracted dataset directory / original ZIP.
CHECKPOINT_SOURCE = None  # Example: Path('/kaggle/input/previous-run/business_entity_resolution')
LEGACY_INPUT = None  # Optional retrieval_checkpoints.zip or its extracted folder; auto-detected when unique.
REVISION = None  # Optional exact Git commit from HANDOFF_REVISION.txt.
STAGES_TO_RUN = ["prepare"]
RETRIEVAL_BACKEND = "cpu"  # Keep this fixed for the run; 'gpu' needs a GPU retrieval session.

os.environ.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="2")
os.environ["ER_REMOTE_COMPUTE"] = "1"
''')
md("### 1. Load the supplied code")
code('''if not (WORK / "src/cloud_pipeline.py").exists():
    folders = [p.parent for p in INPUT.rglob("HANDOFF_FILES.json")
               if (p.parent / "src/cloud_pipeline.py").exists()]
    archives = list(INPUT.rglob("kaggle_handoff.zip"))
    if len(folders) == 1:
        shutil.copytree(folders[0], WORK)
    elif len(archives) == 1:
        parent = WORK.parent
        with zipfile.ZipFile(archives[0]) as archive:
            for member in archive.infolist():
                destination = (parent / member.filename).resolve()
                if not destination.is_relative_to(parent.resolve()) or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('Unsafe code archive member')
            archive.extractall(parent)
    elif folders or archives:
        raise ValueError('Attach exactly one code bundle')
    else:
        subprocess.run(['git', 'clone', '--branch', 'experiment/e02-sparse-retrieval',
                        'https://github.com/L-akshay/AmazonMl-Buissness-challenge.git', str(WORK)], check=True)
        if REVISION:
            subprocess.run(['git', 'checkout', REVISION], cwd=WORK, check=True)
assert (WORK / 'src/cloud_pipeline.py').exists()
revision_file = WORK / 'HANDOFF_REVISION.txt'
revision = revision_file.read_text().strip() if revision_file.exists() else subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=WORK, text=True).strip()
print('Code revision:', revision)
if REVISION and revision != REVISION:
    raise ValueError('Code revision differs from requested handoff')
''')
md("### 2. Install an isolated environment\nThe environment is scratch data and is rebuilt in each fresh session.")
code('''import venv
ENV = Path('/kaggle/temp/entity-resolution-venv')
PYTHON = ENV / 'bin/python'
if not PYTHON.exists():
    ENV.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True).create(ENV)
subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--disable-pip-version-check',
                '-r', str(WORK / 'requirements.txt')], check=True)
gpu_stage = any(s in ('retrieve-train', 'retrieve-test') for s in STAGES_TO_RUN)
if RETRIEVAL_BACKEND == 'gpu' and gpu_stage:
    subprocess.run([str(PYTHON), '-m', 'pip', 'install', 'cupy-cuda12x==13.6.0'], check=True)
freeze = subprocess.check_output([str(PYTHON), '-m', 'pip', 'freeze'], text=True)
(WORK / 'reports').mkdir(exist_ok=True)
(WORK / 'reports/runtime_dependencies.txt').write_text(freeze)
print(freeze)
''')
md("### 3. Attach private data and optional checkpoints\nIf Kaggle kept the original dataset as a ZIP, set `DATA_INPUT` above to that ZIP path.")
code('''def run_module(module, *args):
    subprocess.run([str(PYTHON), '-u', '-m', module, *map(str, args)], cwd=WORK, check=True)

expected = [f'{split}_source{i}.tsv' for split in ('train', 'test') for i in (1, 2, 3)] + ['train_ground_truth.tsv']
attached = all((WORK / 'dataset' / name.split('_')[0] / name).exists() for name in expected)
if not attached:
    if DATA_INPUT == INPUT and not list(INPUT.rglob('train_source1.tsv')):
        matches = []
        for archive_path in INPUT.rglob('*.zip'):
            with zipfile.ZipFile(archive_path) as archive:
                names = {Path(n).name for n in archive.namelist() if '__MACOSX' not in n}
                if set(expected) <= names:
                    matches.append(archive_path)
        if len(matches) != 1:
            raise ValueError('Set DATA_INPUT to the organizer ZIP or extracted dataset directory')
        DATA_INPUT = matches[0]
    run_module('src.handoff', 'attach-data', '--source', DATA_INPUT)
else:
    print('Using already attached dataset. A different dataset requires a fresh project directory.')
if CHECKPOINT_SOURCE:
    run_module('src.handoff', 'restore', '--source', CHECKPOINT_SOURCE)
if LEGACY_INPUT is None and CHECKPOINT_SOURCE is None:
    legacy_folders = [p.parent for p in INPUT.rglob('sparse_v1_train')
                      if p.is_dir() and (p.parent / 'candidates_v1_train').is_dir()]
    legacy_archives = list(INPUT.rglob('retrieval_checkpoints.zip'))
    choices = legacy_folders or legacy_archives
    if len(choices) == 1:
        LEGACY_INPUT = choices[0]
if LEGACY_INPUT:
    run_module('src.handoff', 'import-legacy', '--source', LEGACY_INPUT)

config_path = WORK / 'configs/kaggle.json'
config = json.loads(config_path.read_text())
config['backend'] = RETRIEVAL_BACKEND
config['threads'] = min(4, os.cpu_count() or 1)
config['feature_workers'] = 1
config_path.write_text(json.dumps(config, indent=2))
print(json.dumps(config, indent=2))
run_module('src.cloud_pipeline', '--stage', 'preflight')
''')
md("""## Steps

Run in order: `prepare`, `retrieve-train`, `features-train`, `validate`, `final`,
`retrieve-test`, `features-test`, `score`, `export`, `package`.

The configuration preserves all entities and all three top-six candidate lists.
Changing thread/batch performance settings is different from reducing candidate
quality. Keep code and retrieval configuration fixed when resuming checkpoints.

The worker prints progress approximately every 30 seconds. If a resource check
stops a stage, completed shards remain reusable. Save them and use a larger RAM
session or a fresh output workspace attached to those saved inputs.
""")
code('''# Small synthetic checks, not a full experiment.
subprocess.run([str(PYTHON), '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=WORK, check=True)

# One deadline shared across all selected stages in this notebook execution.
os.environ['ER_DEADLINE_EPOCH'] = str(time.time() + config['session_hours'] * 3600)
for stage in STAGES_TO_RUN:
    print('Starting', stage, flush=True)
    run_module('src.cloud_pipeline', '--stage', stage)
''')
md("""## Checks

Only upload a matching file listed with `organizer: PASS` in `validation.json`.
The shared candidate file has a separate full streaming check. The primary file
is `output/cloud/oof_best/matching_results.tsv`. Precision/recall variants may
also exist if development OOF supports distinct policies. No score is guaranteed.
""")
code('''validation_path = WORK / 'output/cloud/validation.json'
if validation_path.exists():
    print(validation_path.read_text())
    from IPython.display import display, FileLink
    for path in sorted((WORK / 'output/cloud').glob('*/matching_results.tsv')):
        display(FileLink(str(path)))
else:
    print('No trained submission has passed validation yet. Continue the remaining stages.')

local_bytes = sum(p.stat().st_size for p in WORK.rglob('*')
                  if p.is_file() and not p.is_symlink() and '.git' not in p.parts)
print(f'New local project files: {local_bytes / 1024**3:.2f} GiB')
print('Check Kaggle saved-output quota before Save Version. Linked old inputs are not independent backups.')
''')
md("""## Next steps

Use **Save Version** with outputs included; confirm the saved files are present.
Attach that saved notebook output as an input to the next session. Repeat setup,
restore and the next stage. Keep earlier saved inputs attached, because linked
checkpoint files are not duplicated into new independent backups.

Download each validated `matching_results.tsv` and submit it to the challenge
portal. Keep the best accepted submission according to the portal's rules.
`submission_package.zip` is supporting material when requested; it is not the
TSV leaderboard upload.

See `docs/KAGGLE_HANDOFF.md` for resource limits and recovery, and
`docs/BRIEF_COVERAGE.md` for experimental branches that have not been completed.
Sources: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks),
[LightGBM batched datasets](https://lightgbm.readthedocs.io/en/stable/Python-Intro.html).
""")
for index,cell in enumerate(cells):
    cell["id"]=f"handoff-{index:02d}"
notebook=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                                "language_info":{"name":"python","version":"3.11"}})
nbf.validate(notebook)
for cell in cells:
    if cell.cell_type=="code":
        compile(cell.source,"kaggle_handoff.ipynb","exec")
target=Path(__file__).resolve().parents[1]/"notebooks/kaggle_handoff.ipynb"
nbf.write(notebook,target)
print(f"Validated notebook structure and all code-cell syntax: {target}")
