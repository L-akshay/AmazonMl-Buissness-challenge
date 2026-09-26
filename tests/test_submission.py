import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.predict import write_grouped_outputs,export
from src.streaming import encode_secondary
from src.validate_output import validate_rows


class SubmissionTests(unittest.TestCase):
    def test_persisted_scores_export_without_pandas(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            for folder in ("cache/test_scores","cache/model","cache/sparse_v1_test","reports"):
                (root/folder).mkdir(parents=True,exist_ok=True)
            (root/"cache/test_scores/complete.json").write_text(json.dumps({"pairs":2,"references":2,"sha256":"test"}))
            (root/"cache/model/config.json").write_text(json.dumps({"policy":{"threshold":.5,"relative":0}}))
            (root/"cache/sparse_v1_test/reference_ids.json").write_text(json.dumps(["S1-1","S1-2"]))
            np.savez(root/"cache/test_scores/batch_00000.npz",ri=np.array([1,0],dtype=np.int32),
                     tid=np.array([encode_secondary("S3-9"),encode_secondary("S2-8")],dtype=np.uint64),p=np.array([.1,.9],dtype=np.float32))
            export(root)
            self.assertEqual((root/"output/trained/matching_results.tsv").read_text(),
                             "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-8\nS1-2\t\n")

    def test_export_all_candidates_multiple_matches_and_empty_rows(self):
        codes=[encode_secondary(x) for x in ("S2-1","S2-2","S3-3")]
        edges=[(0,codes[0],.9),(0,codes[1],.7),(0,codes[2],.2),(2,codes[2],.4)]
        with tempfile.TemporaryDirectory() as path:
            out=Path(path)
            result=write_grouped_outputs(["S1-1","S1-2","S1-3"],edges,np.array([.9,0,.4]),
                                         {"threshold":.6,"relative":.5},out)
            with (out/"matching_results.tsv").open() as m,(out/"candidate_pairs.tsv").open() as c:
                validated=validate_rows(csv.reader(m,delimiter="\t"),csv.reader(c,delimiter="\t"),
                                        ["S1-1","S1-2","S1-3"],np.array(codes,dtype=np.uint64))
            self.assertEqual(result,validated)
            self.assertEqual(result["matched_pairs"],2)
            self.assertEqual(result["candidate_pairs"],4)
            self.assertEqual(result["predicted_singletons"],2)

    def test_invalid_submission_fails_loudly(self):
        matching_header=["source1_entity_id","matched_entity_ids"]
        candidate_header=["source1_entity_id","candidate_entity_ids"]
        cases=[([],[]),([["S1-1","S2-1"]],[["S1-1",""]]),
               ([["S1-1",""]],[["S1-1","S2-1,S2-1"]]),
               ([["S1-1",""]],[["S1-1","S3-99"]]),
               ([["S1-2",""]],[["S1-2",""]])]
        for m,c in cases:
            with self.subTest(m=m,c=c),self.assertRaises(ValueError):
                validate_rows(iter([matching_header]+m),iter([candidate_header]+c),["S1-1"],
                              np.array([encode_secondary("S2-1")],dtype=np.uint64))

    def test_export_rejects_duplicate_scored_edges(self):
        with tempfile.TemporaryDirectory() as path,self.assertRaises(ValueError):
            tid=encode_secondary("S2-1")
            write_grouped_outputs(["S1-1"],[(0,tid,.8),(0,tid,.9)],np.array([.9]),
                                  {"threshold":.5,"relative":0},Path(path))
