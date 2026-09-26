"""Lightweight local scheduler for reviewed private Kaggle jobs; never trains locally.

Jobs are built before scheduling. Dependencies are explicit notebook IDs. The
scheduler checks the saved project status, not just Kaggle's COMPLETE flag.
Failed/interrupted work is preserved and halts dependent jobs for investigation.
"""

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time


def atomic(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(path)


def ready_jobs(plan,state):
    return [job for job in plan['jobs'] if state.get(job['id'],{}).get('status','pending')=='pending'
            and all(state.get(dep,{}).get('status')=='complete' for dep in job.get('depends',[]))]


def validate_plan(plan):
    jobs=plan['jobs']; ids=[j['id'] for j in jobs]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate job ID')
    known=set(ids)|set(plan.get('completed_inputs',[]))
    for job in jobs:
        if not job['id'].startswith('lakshaytechai/') or not set(job.get('depends',[]))<=known:
            raise ValueError('Unknown owner or dependency')
        path=Path(job['folder']); meta=json.loads((path/'kernel-metadata.json').read_text())
        if meta['id']!=job['id'] or meta.get('is_private') is not True:
            raise ValueError('Only matching, explicitly private jobs may run')
        if meta.get('enable_gpu'):
            raise ValueError('This CPU scheduler does not allocate GPU quota')
    # Kahn traversal rejects cycles before any remote mutation.
    visited=set(plan.get('completed_inputs',[])); pending=list(jobs)
    while pending:
        ready=[j for j in pending if set(j.get('depends',[]))<=visited]
        if not ready:
            raise ValueError('Cyclic dependency graph')
        visited.update(j['id'] for j in ready)
        pending=[j for j in pending if j not in ready]


def run(plan_path,cli,once=False):
    plan_path=Path(plan_path).resolve(); folder=plan_path.parent
    plan=json.loads(plan_path.read_text()); validate_plan(plan)
    state_path=folder/'scheduler_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    for key in plan.get('completed_inputs',[]):
        state.setdefault(key,{'status':'complete','external_input':True})
    environment={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'}
    def call(*args):
        result=subprocess.run([str(cli),*args],capture_output=True,text=True,encoding='utf-8',
                              errors='replace',env=environment,timeout=120)
        if result.returncode:
            raise RuntimeError((result.stdout+result.stderr)[-3000:])
        return result.stdout+result.stderr
    def log(message):
        line=datetime.now(timezone.utc).isoformat()+' '+message
        print(line,flush=True)
        with (folder/'scheduler.log').open('a',encoding='utf-8') as f:
            f.write(line+'\n')
    while True:
        try:
            for job in plan['jobs']:
                current=state.get(job['id'],{})
                if current.get('status')!='running':
                    continue
                output=call('kernels','status',job['id'])
                match=re.search(r'KernelWorkerStatus\.([A-Z_]+)',output)
                if match is None:
                    raise RuntimeError('Unrecognized remote status: '+output)
                remote=match.group(1)
                current['remote_status']=remote
                if remote in ('QUEUED','RUNNING'):
                    continue
                destination=folder/'results'/job['id'].split('/')[-1]
                destination.mkdir(parents=True,exist_ok=True)
                call('kernels','output',job['id'],'-p',str(destination),'--page-size','200','--file-pattern',
                     r'(REMOTE_CHECKPOINT\.json|remote_.*_resources\.json|cloud_.*\.json|validation\.json)$')
                manifests=list(destination.rglob('REMOTE_CHECKPOINT.json'))
                report=json.loads(manifests[0].read_text()) if len(manifests)==1 else {}
                complete=remote=='COMPLETE' and report.get('status')=='complete' and report.get('job_id')==job['id']
                current.update(status='complete' if complete else 'needs_attention',result=report)
                log(f"{job['id']}: {current['status']}")
                atomic(state_path,state)
            attention=[k for k,v in state.items() if v.get('status')=='needs_attention']
            if attention:
                log('Stopped for investigation; checkpoints retained: '+', '.join(attention))
                return 2
            active=sum(v.get('status')=='running' for v in state.values())
            for job in ready_jobs(plan,state)[:max(0,plan.get('max_parallel',2)-active)]:
                # Save launch intent first. A crash must not trigger an unnoticed duplicate run.
                state[job['id']]={'status':'launching','started':time.time()}
                atomic(state_path,state)
                output=call('kernels','push','-p',job['folder'],'-t',str(job.get('timeout',39600)))
                if 'successfully pushed' not in output:
                    state[job['id']].update(status='needs_attention',error=output[-3000:])
                    atomic(state_path,state)
                    raise RuntimeError('Kaggle did not confirm launch')
                state[job['id']]['status']='running'
                atomic(state_path,state)
                log('Launched '+job['id'])
            if any(v.get('status')=='launching' for v in state.values()):
                log('Ambiguous interrupted launch; verify the remote notebook before resuming.')
                return 2
            if all(state.get(j['id'],{}).get('status')=='complete' for j in plan['jobs']):
                log('All scheduled jobs completed successfully.')
                return 0
            atomic(state_path,state)
            log('Running: '+', '.join(k for k,v in state.items() if v.get('status')=='running'))
        except (OSError,subprocess.TimeoutExpired,RuntimeError) as error:
            log('Scheduler error: '+str(error))
            atomic(state_path,state)
            return 2
        if once:
            return 0
        time.sleep(60)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path)
    parser.add_argument('--cli',type=Path,required=True)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    raise SystemExit(run(args.plan,args.cli,args.once))
