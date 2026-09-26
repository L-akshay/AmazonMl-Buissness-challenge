import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('manager',Path(__file__).resolve().parents[1]/'scripts/manage_kaggle.py')
manager=importlib.util.module_from_spec(spec); spec.loader.exec_module(manager)


class SchedulerTests(unittest.TestCase):
    def test_dependencies_require_real_completion(self):
        plan={'jobs':[{'id':'a','depends':[]},{'id':'b','depends':['a']}]}
        self.assertEqual([j['id'] for j in manager.ready_jobs(plan,{'a':{'status':'running'}})],[])
        self.assertEqual([j['id'] for j in manager.ready_jobs(plan,{'a':{'status':'needs_attention'}})],[])
        self.assertEqual([j['id'] for j in manager.ready_jobs(plan,{'a':{'status':'complete'}})],['b'])

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
