import json
import csv
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
import unittest
from unittest.mock import patch
import numpy as np
from src.cloud_store import write_parquet,read_parquet,atomic_json,claim_config
from src.cloud_model import ParquetSequence,StreamingDataset,train_one,score_model,tune_scores,evaluate_scores
from src.cloud_export import write_variants,export
from src.cloud_validation import validation,final_fit
from src.cloud_pipeline import validate_config
from src.cloud_pipeline import execute_stage,STAGES
from src.features import FEATURE_NAMES
from src.handoff import attach_data,import_legacy
from src.model import aggregate
from src.blocking import fused_pairs


def fixture(root):
    config=json.loads((Path(__file__).resolve().parents[1]/"configs/kaggle.json").read_text())
    config.update(threads=1,trees=8,logistic_epochs=1,min_leaf=2,prediction_batch=64,feature_chunk=64)
    (root/"reports").mkdir(); (root/"cache"/"cloud").mkdir(parents=True)
    rng=np.random.default_rng(42); n=100
    refs={"ri":np.arange(n,dtype=np.int32),"entity_id":np.array([f"S1-{i:03d}" for i in range(n)],dtype=object),
          "fold":np.arange(n,dtype=np.int8)%5,"truth_count":np.ones(n,dtype=np.int32),
          "country_norm":np.where(np.arange(n)%2==0,"us","india")}
    for split in ("train","test"):
        write_parquet(root/"cache"/"cloud"/f"references_{split}.parquet",refs)
        folder=root/"cache"/"cloud"/f"features_{split}"; folder.mkdir()
        files=[]
        for shard in range(3):
            ri=np.tile(np.arange(n,dtype=np.int32),2)
            y=((np.arange(2*n)<n)&(shard==0)).astype(np.uint8)
            x=rng.random((len(y),len(FEATURE_NAMES)),dtype=np.float32)
            x[:,0]=y*.8+rng.random(len(y))*.2
            payload={"ri":ri,"tid":(2<<32)+np.arange(len(y),dtype=np.uint64)+shard*len(y),
                     "y":y,"fold":refs["fold"][ri],**{f"f{i}":x[:,i] for i in range(len(FEATURE_NAMES))}}
            name=f"batch_{shard:05d}.parquet"; write_parquet(folder/name,payload); files.append(name)
        atomic_json(folder/"complete.json",{"files":files,"rows":600})
    return config,refs


class CloudTests(unittest.TestCase):
    def test_legacy_import_rejects_unexpected_paths_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/"legacy.zip"
            with zipfile.ZipFile(source,"w") as z:
                z.writestr("cache/sparse_v1_train/name.json","{}")
            import_legacy(root,source)
            self.assertEqual((root/"cache/sparse_v1_train/name.json").read_text(),"{}")
            import_legacy(root,source)
            with zipfile.ZipFile(source,"w") as z:
                z.writestr("../outside.txt","bad")
            with self.assertRaises(ValueError):
                import_legacy(root,source)

    def test_full_union_retains_all_eighteen_channel_candidates(self):
        from scipy import sparse
        results={}
        for offset,name in enumerate(("name","address","token")):
            results[name]=sparse.csr_matrix((np.linspace(.9,.6,6),np.arange(offset*6,(offset+1)*6),[0,6]),shape=(1,18))
        pairs=list(fused_pairs(results,max_candidates=18,min_candidates=18,ratio=0))
        self.assertEqual({p[1] for p in pairs},set(range(18)))

    def test_end_to_end_remote_stages_on_synthetic_tsvs(self):
        code_root=Path(__file__).resolve().parents[1]
        config=json.loads((code_root/"configs/kaggle.json").read_text())
        config.update(threads=1,trees=4,models=["gbdt"],batch_size=200,min_leaf=2,
                      feature_chunk=200,prediction_batch=200)
        rng=np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for directory in ("cache","reports","experiments","output","utils"):
                (root/directory).mkdir()
            shutil.copyfile(code_root/"utils/validate_submission.py",root/"utils/validate_submission.py")
            for split,base in (("train",0),("test",100000)):
                folder=root/"dataset"/split; folder.mkdir(parents=True)
                sources={1:[],2:[],3:[]}; truth=[]
                for i in range(250):
                    name="".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"),18))
                    address=f"{i+1} "+"".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"),20))
                    sid=f"S1-{base+i}"; tid=f"S2-{base+i}"; country="France" if split=="test" else ("US" if i%2 else "India")
                    sources[1].append([sid,name,address,country])
                    sources[2].append([tid,name if i%7 else name[::-1],address if i%7 else address[::-1],country])
                    sources[3].append([f"S3-{base+i}",name[::-1],f"{i+9999} remote",country])
                    truth.append([sid,tid if i%7 else ""])
                for source,rows in sources.items():
                    with (folder/f"{split}_source{source}.tsv").open("w",encoding="utf-8",newline="") as f:
                        w=csv.writer(f,delimiter="\t"); w.writerow(["entity_id","business_name","business_address","country"]); w.writerows(rows)
                if split=="train":
                    with (folder/"train_ground_truth.tsv").open("w",encoding="utf-8",newline="") as f:
                        w=csv.writer(f,delimiter="\t"); w.writerow(["source1_entity_id","matched_entity_ids"]); w.writerows(truth)
            with patch("src.cloud_validation.exclusion_indices",return_value=(np.array([],dtype=np.int32),{"scope":"synthetic"})), patch.dict(os.environ,{"ER_THREADS":"1","ER_DB_MEMORY":"256MB"}):
                for stage in STAGES:
                    execute_stage(root,stage,config)
            report=json.loads((root/"output/cloud/validation.json").read_text())
            self.assertEqual(report["oof_best"]["entities"],250)
            self.assertEqual(report["oof_best"]["organizer"],"PASS")
            self.assertTrue((root/"output/submission_package.zip").exists())

    def test_parquet_sequence_and_model_resume(self):
        import lightgbm as lgb
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config,ref=fixture(root)
            allowed=ref["fold"]!=4
            files=list((root/"cache/cloud/features_train").glob("*.parquet"))
            sequences=[ParquetSequence(p,allowed) for p in sorted(files)]
            labels=np.concatenate([s.labels for s in sequences])
            params={"objective":"binary","verbosity":-1,"num_threads":1,"min_data_in_leaf":2,
                    "deterministic":True,"force_col_wise":True,"seed":42}
            whole=lgb.train(params,StreamingDataset(sequences,label=labels),num_boost_round=8)
            first=lgb.train(params,StreamingDataset(sequences,label=labels),num_boost_round=4)
            resumed=lgb.train(params,StreamingDataset(sequences,label=labels),num_boost_round=4,init_model=first)
            x=np.concatenate([s[:] for s in sequences])
            np.testing.assert_allclose(whole.predict(x),resumed.predict(x),rtol=1e-10,atol=1e-10)

    def test_grouped_full_pair_model_selection_and_cached_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config,ref=fixture(root)
            with patch("src.cloud_validation.exclusion_indices",return_value=(np.array([],dtype=np.int32),{"scope":"synthetic"})):
                report=validation(root,config,"synthetic")
                self.assertEqual(report["reserved"]["overall"]["entities"],20)
                self.assertEqual(set(report["comparisons"]),{"gbdt","logistic"})
                self.assertEqual(report,validation(root,config,"synthetic"))
            model=final_fit(root,config,"synthetic")
            info=json.loads((model.parent/"complete.json").read_text())
            self.assertEqual(info["rows"],600)
            outputs=score_model(root,model,"test",np.ones(100,dtype=bool),"test",config)
            self.assertEqual(sum(len(read_parquet(p,"ri")["ri"]) for p in outputs),600)
            self.assertEqual(outputs,score_model(root,model,"test",np.ones(100,dtype=bool),"test",config))

    def test_histogram_thresholding_matches_pair_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"scores.parquet"
            y=np.array([1,0,1,0],dtype=np.uint8); g=np.array([0,0,1,2],dtype=np.int32)
            p=np.array([.8,.8,.5,.9],dtype=np.float32); truth=np.array([2,1,0,0]); scope=np.ones(4,dtype=bool)
            write_parquet(path,{"ri":g,"y":y,"p":p})
            scores=tune_scores([path],truth,scope,thresholds=[.4,.8,.9,.95])
            for result in scores:
                expected=aggregate(y,p,g,truth,scope,result["threshold"],result["relative"])
                self.assertAlmostEqual(result["macro_f05"],expected["macro_f05"])

    def test_variants_complete_coverage_and_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            policies=[{"name":"oof_best","threshold":.7,"relative":0},
                      {"name":"precision","threshold":.9,"relative":0}]
            counts=write_variants(["S1-1","S1-2"],[(0,(2<<32)+1,.8),(0,(3<<32)+1,.95)],
                                  np.array([.95,0]),policies,out)
            self.assertEqual(counts["oof_best"]["matched_pairs"],2)
            self.assertEqual(counts["precision"]["matched_pairs"],1)
            self.assertIn("S1-2\t\n",(out/"oof_best/matching_results.tsv").read_text())

    def test_config_and_stale_cache_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config,_=fixture(root)
            config["batch_size"]=0
            with self.assertRaises(ValueError):
                validate_config(config)
            claim_config(root/"guard",{"model":"a"})
            with self.assertRaises(ValueError):
                claim_config(root/"guard",{"model":"b"})
