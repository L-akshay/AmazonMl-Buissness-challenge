import unittest
import json
import pickle
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.model import aggregate,validate,train_final
from src.features import FEATURE_NAMES
from src.evaluate import entity_f05


class ModelMetricTests(unittest.TestCase):
    def test_training_validation_and_frozen_holdout_on_synthetic_entities(self):
        rng=np.random.default_rng(42)
        n=250
        groups=np.repeat(np.arange(n,dtype=np.int32),4)
        truth=np.where(np.arange(n)%7==0,0,2)
        y=((np.tile(np.arange(4),n)<2)&(truth[groups]>0)).astype(np.uint8)
        x=rng.random((len(y),len(FEATURE_NAMES)),dtype=np.float32)
        x[:,10]=y*.8+rng.random(len(y))*.2
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            for directory in ("cache/development","cache/full","reports","experiments","src"):
                (root/directory).mkdir(parents=True,exist_ok=True)
            for name in ("model.py","features.py","training_data.py","evaluation_scope.py"):
                shutil.copyfile(Path(__file__).resolve().parents[1]/"src"/name,root/"src"/name)
            for stage in ("development","full"):
                for name,value in {"x":x,"y":y,"groups":groups,"qids":np.arange(len(y),dtype=np.uint32),"fit":np.ones(len(y),dtype=bool)}.items():
                    np.save(root/"cache"/stage/(name+".npy"),value)
            with (root/"cache/training_metadata.pkl").open("wb") as f:
                pickle.dump({"fold":np.arange(n)%5,"truth_count":truth,"sample":np.ones(n,dtype=bool),
                             "country":np.where(np.arange(n)%2==0,"us","india"),"ids":[f"S1-{i}" for i in range(n)]},f)
            validate(root)
            report=json.loads((root/"reports/model_validation.json").read_text())
            self.assertEqual(report["locked_holdout"]["entities"],50)
            self.assertEqual(report["models"]["gbdt"]["oof_policy"]["entities"],200)
            self.assertEqual(set(report["country_held_out"]),{"us","india"})
            self.assertTrue((root/"cache/locked_holdout.json").exists())
            train_final(root)
            self.assertTrue((root/"cache/model/matcher.pkl").exists())

    def test_grouped_metric_includes_unretrieved_matches(self):
        y=np.array([1,0,0,1],dtype=np.uint8)
        p=np.array([.9,.8,.95,.2])
        groups=np.array([0,0,1,2])
        truth=np.array([2,0,1,0])
        result=aggregate(y,p,groups,truth,np.ones(4,dtype=bool),.5)
        expected=(entity_f05({'a','b'},{'a','wrong'})+0+0+1)/4
        self.assertAlmostEqual(result['macro_f05'],expected)
        self.assertEqual(result['singleton_fp_rate'],.5)
