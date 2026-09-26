import unittest
import duckdb
import numpy as np
from src.streaming import encode_secondary, secondary_batches


class StreamingTests(unittest.TestCase):
    def test_order_boundary_and_absent_labels(self):
        db = duckdb.connect()
        for source, ids in ((2, ["S2-9", "S2-10", "S2-2"]), (3, ["S3-8", "S3-1"])):
            db.execute(f"CREATE TABLE train_source{source}_norm(entity_id VARCHAR, name VARCHAR, address VARCHAR, country VARCHAR)")
            db.executemany(f"INSERT INTO train_source{source}_norm VALUES (?, 'name', '', 'US')", [(i,) for i in ids])
        codes = np.array([encode_secondary("S2-2"), encode_secondary("S3-1")], dtype=np.uint64)
        batches = list(secondary_batches(db, "train", 2, (codes, np.array([7, 4], dtype=np.int32))))
        self.assertEqual([len(b) for b in batches], [2, 2, 1])
        self.assertEqual([(r[0], r[4]) for b in batches for r in b],
                         [("S2-10", -1), ("S2-2", 7), ("S2-9", -1), ("S3-1", 4), ("S3-8", -1)])
        db.close()

    def test_encoding_rejects_collisions(self):
        for value in ("S1-1", "S2-01", "S2-4294967296", "S2--1", "S2-x"):
            with self.assertRaises(ValueError):
                encode_secondary(value)
