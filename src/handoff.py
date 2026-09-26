"""Private dataset mounting, checkpoint restore, and reviewable delivery archives."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import stat
import subprocess
import zipfile
from src.cloud_store import atomic_json, digest

EXPECTED=[f"{split}_source{i}.tsv" for split in ("train","test") for i in (1,2,3)]+["train_ground_truth.tsv"]


def source_files(root):
    manifest=root/"HANDOFF_FILES.json"
    if manifest.exists():
        files=json.loads(manifest.read_text())
        for name in files:
            if not (root/name).resolve().is_relative_to(root.resolve()):
                raise ValueError("Invalid code manifest path")
        return files
    return subprocess.check_output(["git","ls-files"],cwd=root,text=True).splitlines()


def attach_data(root,source):
    """Extract only the seven exact data filenames, or mount an extracted dataset."""
    source=Path(source)
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            selected={}
            for member in archive.infolist():
                name=Path(member.filename.replace("\\","/")).name
                if name not in EXPECTED or "__MACOSX" in member.filename:
                    continue
                if name in selected or stat.S_ISLNK(member.external_attr>>16):
                    raise ValueError("Duplicate or linked dataset member")
                selected[name]=member
            if set(selected)!=set(EXPECTED):
                raise ValueError("Input ZIP must contain exactly one copy of each required TSV")
            for name,member in selected.items():
                target=root/"dataset"/name.split("_")[0]/name
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    raise FileExistsError(f"Refusing to overwrite existing data: {target}")
                temp=target.with_suffix(".tmp")
                with archive.open(member) as src,temp.open("wb") as dst:
                    shutil.copyfileobj(src,dst,length=1024**2)
                temp.replace(target)
    else:
        for name in EXPECTED:
            matches=[p for p in source.rglob(name) if "__MACOSX" not in p.parts]
            if len(matches)!=1:
                raise ValueError(f"Expected exactly one {name} under {source}")
            target=root/"dataset"/name.split("_")[0]/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if target.resolve()!=matches[0].resolve():
                    raise FileExistsError(target)
            else:
                target.symlink_to(matches[0].resolve())


def restore(root,source):
    """Mount immutable checkpoint files; copy the writable database only.

Keep the source attached to every resumed notebook. A symlink is not a backup.
Only copy trusted outputs from this project's own prior notebook/version.
"""
    source=Path(source).resolve()
    if not (source/"cache"/"cloud"/"prepared.json").exists():
        raise ValueError("Checkpoint must be the project folder from a prior successful prepare stage")
    for base in ("cache","reports","experiments"):
        if not (source/base).exists():
            continue
        for path in (source/base).rglob("*"):
            if not path.is_file() or ".tmp" in path.name or path.suffix==".wal":
                continue
            target=root/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                continue
            if base in ("reports","experiments") or path.name=="entities.duckdb":
                shutil.copyfile(path,target)
            else:
                target.symlink_to(path)


def code_bundle(root):
    out=root/"output"/"handoff"; out.mkdir(parents=True,exist_ok=True)
    if subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip():
        raise ValueError("Commit the reviewed source before building a revision-labelled handoff")
    files=source_files(root)
    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    target=out/"kaggle_handoff.zip"
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name in files:
            path=root/name
            if path.is_file() and not name.startswith(("dataset/","cache/","output/")):
                archive.write(path,"business_entity_resolution/"+name)
        archive.writestr("HANDOFF_REVISION.txt",revision+"\n")
        archive.writestr("business_entity_resolution/HANDOFF_REVISION.txt",revision+"\n")
        archive.writestr("business_entity_resolution/HANDOFF_FILES.json",json.dumps(files))
    atomic_json(out/"manifest.json",{"git_revision":revision,"archive":target.name,"sha256":digest(target),
                                     "contains":"Code, notebook, configuration and docs; no data or credentials"})
    return target


def package_submission(root,config):
    out=root/"output"/"cloud"
    checks=json.loads((out/"validation.json").read_text())
    model_report=json.loads((root/"reports"/"cloud_model_validation.json").read_text())
    selected=out/"oof_best"/"matching_results.tsv"
    if checks["oof_best"]["organizer"]!="PASS" or digest(selected)!=checks["oof_best"]["sha256"]:
        raise ValueError("Final TSV no longer matches validated output")
    method=root/"output"/"Methodology.md"
    oof=model_report["comparisons"][model_report["selected_model"]]["oof"]["overall"]
    reserved=model_report["reserved"]["overall"]
    text=f'''# Business Entity Resolution methodology

Team: {config['team_name']}
Generated: {datetime.now(timezone.utc).isoformat()}

## Data and evidence

Only organizer data was used. Full source integrity results are in reports/data_audit.json.
Every training and test record is retained. All three sparse top-six retrieval lists
are fused without an additional candidate cap. Name, address, numeric, country,
frequency, IDF and retrieval evidence supply {51} features. No external identity
lookups, hosted model calls or pretrained model weights are used.

## Training and validation

Selected model: {model_report['selected_model']}. LightGBM uses an MIT-licensed
library; the streaming logistic implementation uses scikit-learn (BSD-3-Clause).
Exact installed versions are in requirements.txt and the notebook setup log.
Four S1-grouped OOF folds select the model and decision; fold 4 is reserved for
one frozen evaluation after excluding previously inspected examples. Earlier
aggregate retrieval/baseline diagnostics were viewed, so this is a reserved
matcher evaluation rather than a wholly untouched pipeline holdout.
All candidate negatives are retained. Features are cached once as Parquet.
LightGBM loads batches into a binned Dataset; logistic standardization and fitting
stream through the same cache. Final fitting uses all training entities.

Development OOF macro F0.5: {oof['macro_f05']:.8f}.
Reserved matcher macro F0.5: {reserved['macro_f05']:.8f}.
These are validation scores, not leaderboard scores.
Decision: {json.dumps(model_report['selected_policy'])}.
Calibration: none. Conditional embedding and additional experimental branches
are not claimed as completed experiments. See docs/BRIEF_COVERAGE.md.

## Output and reproduction

All matching variants and the complete shared candidate file passed strict
coverage/ID/subset checks. Each matching TSV also passed the unmodified organizer
validator with ID checks. The scoring portal receives matching_results.tsv only.
The accompanying code is run stage-by-stage using src.cloud_pipeline and
configs/kaggle.json, as described in docs/KAGGLE_HANDOFF.md.
Per-country and match-count metrics, candidate recall and exact output hashes
are preserved in reports/cloud_model_validation.json,
reports/cloud_candidate_coverage.json and reports/cloud_output_validation.json.
France accuracy and leaderboard performance remain unknown until external evaluation.
'''
    method.write_text(text,encoding="utf-8")
    target=root/"output"/"submission_package.zip"
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_STORED) as archive:
        archive.write(selected,"output/matching_results.tsv")
        archive.write(out/"candidate_pairs.tsv","output/candidate_pairs.tsv")
        archive.write(method,"Documentation_template.md")
        code_root=Path(__file__).resolve().parents[1]
        names=source_files(code_root)
        for name in names:
            path=code_root/name
            if path.is_file() and not name.startswith(("dataset/","cache/","output/")):
                archive.write(path,"code/business_entity_resolution/"+name)
        archive.writestr("code/business_entity_resolution/HANDOFF_FILES.json",json.dumps(names))
        # Current run reports supersede the source repository's historical reports.
        for name in ("cloud_model_validation.json","cloud_candidate_coverage.json","cloud_output_validation.json"):
            archive.write(root/"reports"/name,"run_reports/"+name)
    print(f"Primary upload: {selected}\nSupporting package: {target}",flush=True)
    return target


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("attach-data","restore","code-bundle"))
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source",type=Path)
    args=parser.parse_args()
    if args.action=="code-bundle":
        print(code_bundle(args.root))
    else:
        if args.source is None:
            parser.error("--source is required")
        (attach_data if args.action=="attach-data" else restore)(args.root,args.source)
