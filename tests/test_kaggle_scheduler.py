import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('manager',Path(__file__).resolve().parents[1]/'scripts/manage_kaggle.py')
manager=importlib.util.module_from_spec(spec); spec.loader.exec_module(manager)


class SchedulerTests(unittest.TestCase):
    def test_atomic_retries_transient_lock_and_preserves_old_state_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.json';path.write_text('{"old":true}')
            replace=Path.replace
            calls=[]
            def locked(source,target):
                calls.append(1)
                if len(calls)<3:
                    raise PermissionError('temporary Windows sharing violation')
                return replace(source,target)
            with patch.object(Path,'replace',locked),patch.object(manager.time,'sleep'):
                manager.atomic(path,{'new':True})
            self.assertEqual(json.loads(path.read_text()),{'new':True})
            with patch.object(Path,'replace',side_effect=PermissionError('persistent lock')) as blocked,patch.object(manager.time,'sleep'):
                with self.assertRaises(PermissionError):
                    manager.atomic(path,{'unsafe':True})
                self.assertEqual(blocked.call_count,8)
            self.assertEqual(json.loads(path.read_text()),{'new':True})

    def test_cancelled_without_manifest_stops_for_recovery_not_relaunch(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); job='lakshaytechai/cancelled'; target=folder/'job';target.mkdir()
            (target/'kernel-metadata.json').write_text(json.dumps({'id':job,'is_private':True}))
            plan=folder/'plan.json';plan.write_text(json.dumps({'jobs':[{'id':job,'folder':str(target)}]}))
            journal=folder/'scheduler_state.json';journal.write_text(json.dumps({job:{'status':'running'}}))
            def api(command,**kwargs):
                self.assertNotIn('push',command)
                return SimpleNamespace(returncode=0,stdout='KernelWorkerStatus.CANCEL_ACKNOWLEDGED' if command[2]=='status' else '',stderr='')
            with patch.object(manager.subprocess,'run',side_effect=api):
                self.assertEqual(manager.run(plan,'kaggle',once=True),2)
            self.assertEqual(json.loads(journal.read_text())[job]['status'],'needs_attention')

    def test_successful_running_status_clears_stale_read_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); job_id='lakshaytechai/a'; target=folder/'a'; target.mkdir()
            (target/'kernel-metadata.json').write_text(json.dumps({'id':job_id,'is_private':True}))
            plan=folder/'plan.json'
            plan.write_text(json.dumps({'jobs':[{'id':job_id,'folder':str(target)}],'max_parallel':1}))
            journal=folder/'scheduler_state.json'
            journal.write_text(json.dumps({job_id:{'status':'running','read_failures':2,
                'last_read_error':'expired token','last_read_error_at':'earlier'}}))
            def api(command,**kwargs):
                self.assertEqual(command[2:4],['status',job_id])
                return SimpleNamespace(returncode=0,stdout='KernelWorkerStatus.RUNNING',stderr='')
            with patch.object(manager.subprocess,'run',side_effect=api):
                self.assertEqual(manager.run(plan,'kaggle',once=True),0)
            saved=json.loads(journal.read_text())[job_id]
            self.assertEqual(saved['status'],'running')
            self.assertEqual(saved['remote_status'],'RUNNING')
            self.assertEqual(saved['read_failures'],2)
            self.assertNotIn('last_read_error',saved)
            self.assertNotIn('last_read_error_at',saved)

    def test_read_failure_preserves_job_and_allows_other_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            jobs=[]
            for name in ('a','b'):
                job_id='lakshaytechai/'+name
                target=folder/name; target.mkdir()
                (target/'kernel-metadata.json').write_text(json.dumps({'id':job_id,'is_private':True}))
                jobs.append({'id':job_id,'folder':str(target)})
            plan=folder/'plan.json'; plan.write_text(json.dumps({'jobs':jobs,'max_parallel':2}))
            journal=folder/'scheduler_state.json'
            journal.write_text(json.dumps({j['id']:{'status':'running'} for j in jobs}))
            result=folder/'results/b'; result.mkdir(parents=True)
            (result/'REMOTE_CHECKPOINT.json').write_text(json.dumps({'job_id':'lakshaytechai/b','status':'complete'}))
            def api(command,**kwargs):
                self.assertNotIn('push',command)
                if command[2]=='status' and command[3]=='lakshaytechai/a':
                    return SimpleNamespace(returncode=1,stdout='',stderr='Temporary permission denial')
                return SimpleNamespace(returncode=0,stdout='KernelWorkerStatus.COMPLETE',stderr='')
            with patch.object(manager.subprocess,'run',side_effect=api):
                self.assertEqual(manager.run(plan,'kaggle',once=True),0)
            saved=json.loads(journal.read_text())
            self.assertEqual(saved['lakshaytechai/a']['status'],'running')
            self.assertEqual(saved['lakshaytechai/a']['read_failures'],1)
            self.assertEqual(saved['lakshaytechai/b']['status'],'complete')

    def test_capacity_gate_requires_complete_evidence_and_headroom(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); job='lakshaytechai/probe'
            gate={'sample_job':job,'sample_batches':2,'partition_batches':200,
                  'time_safety_factor':2,'max_partition_hours':8,
                  'max_partition_output_gib':16,'max_peak_rss_gib':23,
                  'stages':{job:['features-train']}}
            state={job:{'status':'complete','result':{'status':'complete','new_output_bytes':1024**2}}}
            with self.assertRaisesRegex(ValueError,'Missing resource'):
                manager.review_capacity(folder,state,gate)
            target=folder/'results/probe/remote_features-train_resources.json';target.parent.mkdir(parents=True)
            target.write_text(json.dumps({'exit_code':0,'stop_reason':None,'wall_seconds':30,'peak_rss_gib':5}))
            self.assertTrue(manager.review_capacity(folder,state,gate)['passed'])
            target.write_text(json.dumps({'exit_code':0,'stop_reason':None,'wall_seconds':200,'peak_rss_gib':5}))
            with self.assertRaisesRegex(ValueError,'capacity needs review'):
                manager.review_capacity(folder,state,gate)
            state[job]['status']='running'
            with self.assertRaisesRegex(ValueError,'verified completed'):
                manager.review_capacity(folder,state,gate)

    def test_dependencies_require_real_completion(self):
        plan={'jobs':[{'id':'a','depends':[]},{'id':'b','depends':['a']}]}
        self.assertEqual([j['id'] for j in manager.ready_jobs(plan,{'a':{'status':'running'}})],[])
        self.assertEqual([j['id'] for j in manager.ready_jobs(plan,{'a':{'status':'needs_attention'}})],[])
        self.assertEqual([j['id'] for j in manager.ready_jobs(plan,{'a':{'status':'complete'}})],['b'])

    def test_priority_never_bypasses_dependencies(self):
        plan={'jobs':[{'id':'a','priority':20},{'id':'b','priority':0,'depends':['a']},
                      {'id':'c','priority':10}]}
        self.assertEqual([j['id'] for j in manager.ready_jobs(plan,{})],['c','a'])

    def test_public_jobs_and_cycles_are_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            meta={'id':'lakshaytechai/a','is_private':False}
            (folder/'kernel-metadata.json').write_text(json.dumps(meta))
            plan={'jobs':[{'id':meta['id'],'folder':str(folder),'depends':[]}]}
            with self.assertRaisesRegex(ValueError,'private'):
                manager.validate_plan(plan)
            meta['is_private']=True
            (folder/'kernel-metadata.json').write_text(json.dumps(meta))
            plan['jobs'][0]['depends']=[meta['id']]
            with self.assertRaisesRegex(ValueError,'Cyclic'):
                manager.validate_plan(plan)
