import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('manager',Path(__file__).resolve().parents[1]/'scripts/manage_kaggle.py')
manager=importlib.util.module_from_spec(spec); spec.loader.exec_module(manager)


class SchedulerTests(unittest.TestCase):
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
