"""Build a private Kaggle job from a committed code revision; does not launch it."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile


def restore_files(archive,source,proof):
    source=Path(source).resolve(); hashes={}
    for name,expected in proof['files'].items():
        relative=Path(name)
        path=source/relative
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name
                or not name.startswith('cache/') or path.is_symlink()
                or not path.resolve().is_relative_to(source)):
            raise ValueError('Unsafe restored checkpoint path')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError('Restored checkpoint hash differs')
        archive.writestr(name,data);hashes[name]=expected
    return hashes


def build(root,output,slug,stages,checkpoints=(),overrides=None,tests=False,recovery=None,restore=None):
    if not set(recovery or {}) <= set(checkpoints):
        raise ValueError('Recovery proofs require the corresponding checkpoint input')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise ValueError('Commit and test the source before publishing a remote job')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    names=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
    config=json.loads((root/'configs/kaggle.json').read_text())
    config.update(overrides or {})
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            if name.startswith(('dataset/','cache/','output/')):
                raise ValueError('Dataset or cache unexpectedly tracked by Git')
            path=root/name
            if path.is_file():
                z.write(path,name)
        z.writestr('HANDOFF_FILES.json',json.dumps(names))
        z.writestr('HANDOFF_REVISION.txt',revision+'\n')
        restored=restore_files(z,*restore) if restore else {}
    job={'id':f'lakshaytechai/{slug}','stages':list(stages),'revision':revision,
         'checkpoint_sources':list(checkpoints),'dataset':'lakshaytechai/amazon-er-private-inputs',
         'config':config,'tests':tests,'recovery_checkpoints':recovery or {},'restored_checkpoint_files':restored}
    encoded=base64.b64encode(payload.getvalue()).decode()
    script=f'''import base64, io, json, sys, zipfile
from pathlib import Path
root=Path('/kaggle/working/business_entity_resolution')
root.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({encoded!r}))) as archive:
    archive.extractall(root)
sys.path.insert(0,str(root))
from src.kaggle_runtime import run
run(root,json.loads({json.dumps(job)!r}))
'''
    compile(script,'run.py','exec')
    output.mkdir(parents=True,exist_ok=True)
    (output/'run.py').write_text(script,encoding='utf-8')
    (output/'job.json').write_text(json.dumps(job,indent=2))
    gpu=config['backend']=='gpu' and any(s.startswith('retrieve-') for s in stages)
    meta={'id':job['id'],'title':slug.replace('-',' ').title(),'code_file':'run.py',
          'language':'python','kernel_type':'script','is_private':True,'enable_gpu':gpu,
          'enable_internet':True,'dataset_sources':[job['dataset']],
          'kernel_sources':list(checkpoints),'competition_sources':[]}
    if gpu:
        meta['machine_shape']='NvidiaTeslaT4'
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({'id':job['id'],'revision':revision,'stages':stages,'config':config},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slug',required=True)
    parser.add_argument('--stages',nargs='+',required=True)
    parser.add_argument('--checkpoints',nargs='*',default=[])
    parser.add_argument('--overrides',type=Path)
    parser.add_argument('--tests',action='store_true')
    parser.add_argument('--recovery',type=Path,help='Reviewed hashes for partial inputs lacking a lifecycle manifest')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    build(root,root/'cache/kaggle_jobs'/args.slug,args.slug,args.stages,args.checkpoints,
          json.loads(args.overrides.read_text()) if args.overrides else {},args.tests,
          json.loads(args.recovery.read_text()) if args.recovery else {})
