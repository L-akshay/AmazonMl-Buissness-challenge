"""Private remote job lifecycle: mount immutable checkpoints and save only new files."""

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import venv


def mount_checkpoint(root, source):
    """Overlay complete files in input order; never follow or copy outside symlinks."""
    source=Path(source).resolve()
    for base in ("cache","reports","experiments","output"):
        if not (source/base).exists():
            continue
        for path in (source/base).rglob("*"):
            if not path.is_file() or path.is_symlink() or ".tmp" in path.name or path.suffix==".wal":
                continue
            target=root/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.is_symlink():
                mutable=path.name in ('checkpoint.txt','checkpoint.joblib')
                if base=='cache' and not mutable and target.resolve()!=path.resolve():
                    def sha(file):
                        with file.open('rb') as handle:
                            return hashlib.file_digest(handle,'sha256').hexdigest()
                    if target.stat().st_size!=path.stat().st_size or sha(target)!=sha(path):
                        raise ValueError(f'Conflicting immutable checkpoint artifact: {target.relative_to(root)}')
                target.unlink()
            elif target.exists():
                # Source-tree historical reports are superseded by measured run reports.
                if base not in ("reports","experiments"):
                    raise ValueError(f"Refusing to replace a new local artifact: {target}")
                target.unlink()
            target.symlink_to(path)


def detach_inputs(root):
    """Remove links, never their targets, so saved outputs contain only new work."""
    count=0
    for path in root.rglob("*"):
        if path.is_symlink():
            path.unlink()
            count+=1
    return count


def run(root, job):
    if not Path('/kaggle/input').exists():
        raise RuntimeError("This runner is restricted to Kaggle")
    result={"job_id":job["id"],"stages":job["stages"],"started":time.time(),
            "revision":job["revision"],"checkpoint_sources":job["checkpoint_sources"]}
    root=Path(root)
    try:
        env=Path('/kaggle/temp/entity-resolution-venv')
        python=env/'bin/python'
        if not python.exists():
            env.parent.mkdir(parents=True,exist_ok=True)
            venv.EnvBuilder(with_pip=False).create(env)
        # Kaggle's distro Python may omit ensurepip. Host pip can install into
        # this isolated interpreter without changing the preinstalled runtime.
        subprocess.run([sys.executable,'-m','pip','--python',str(python),'install',
                        '--disable-pip-version-check','pip==26.2.1'],check=True)
        subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check',
                        '-r',str(root/'requirements.txt')],check=True)
        os.environ.update(ER_REMOTE_COMPUTE='1',ER_THREADS=str(job['config']['threads']),
                          OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                          OMP_NUM_THREADS=str(job['config']['threads']))
        if job['config']['backend']=='gpu' and any(s.startswith('retrieve-') for s in job['stages']):
            subprocess.run([str(python),'-m','pip','install','cupy-cuda12x==13.6.0'],check=True)
        def module(name,*args):
            subprocess.run([str(python),'-u','-m',name,*map(str,args)],cwd=root,check=True)
        input_root=Path('/kaggle/input')
        markers={}
        for marker in input_root.rglob('REMOTE_CHECKPOINT.json'):
            info=json.loads(marker.read_text())
            markers[info['job_id']]=marker.parent
        # The initial handoff preparation predates this runner's marker format.
        if 'lakshaytechai/amazon-er-prepare' in job['checkpoint_sources']:
            prepared=[p.parents[2] for p in input_root.rglob('prepared.json') if p.parent.name=='cloud']
            if len(prepared)!=1:
                raise ValueError('Attach exactly one initial prepared project')
            markers['lakshaytechai/amazon-er-prepare']=prepared[0]
        for name in job['checkpoint_sources']:
            if name not in markers:
                raise ValueError(f'Missing checkpoint input: {name}')
            mount_checkpoint(root,markers[name])
        data_root=input_root/'datasets'/job['dataset']
        if not data_root.exists():
            data_root=input_root/job['dataset'].split('/')[-1]
        if not data_root.exists():
            raise ValueError('Private organizer dataset input is not mounted')
        module('src.handoff','attach-data','--source',data_root)
        legacy=[p.parent for p in data_root.rglob('sparse_v1_train') if p.is_dir()]
        if len(legacy)==1:
            module('src.handoff','import-legacy','--source',legacy[0])
        if 'prepare' not in job['stages']:
            os.environ['ER_DB_READ_ONLY']='1'
        config=root/'configs/kaggle.json'
        config.write_text(json.dumps(job['config'],indent=2))
        (root/'reports').mkdir(exist_ok=True)
        dependency_report=root/'reports/runtime_dependencies.txt'
        if dependency_report.is_symlink():
            dependency_report.unlink()
        dependency_report.write_text(subprocess.check_output([str(python),'-m','pip','freeze'],text=True))
        module('src.cloud_pipeline','--stage','preflight')
        if job.get('tests'):
            subprocess.run([str(python),'-m','unittest','discover','-s','tests','-v'],cwd=root,check=True,
                           env={**os.environ,'ER_DB_READ_ONLY':'0'})
        os.environ['ER_DEADLINE_EPOCH']=str(time.time()+job['config']['session_hours']*3600)
        for stage in job['stages']:
            result['active_stage']=stage
            module('src.cloud_pipeline','--stage',stage)
        result['status']='complete'
    except Exception:
        result['status']='interrupted'
        result['error']=traceback.format_exc()
        print(result['error'],flush=True)
    finally:
        result['finished']=time.time()
        result['detached_input_links']=detach_inputs(root)
        result['new_output_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        (root/'REMOTE_CHECKPOINT.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
    # An interrupted stage is explicitly recorded; the wrapper exits normally to
    # preserve valid checkpoint files. Callers must check status, not kernel exit alone.
    return result
