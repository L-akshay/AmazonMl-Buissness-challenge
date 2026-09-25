import tempfile
import unittest
from pathlib import Path
import duckdb
from src.data import SOURCE_COLUMNS
from src.evaluate import entity_f05, read_results
from src.run_pipeline import normalize_tables, make_pairs, metrics, export_outputs


class PipelineTests(unittest.TestCase):
    def test_multimatch_missing_fields_country_and_sql_metric(self):
        db = duckdb.connect()
        for split in ("train", "test"):
            for source in (1, 2, 3):
                db.execute(f"CREATE TABLE {split}_source{source} ({', '.join(c+' VARCHAR' for c in SOURCE_COLUMNS)})")
            db.executemany(f"INSERT INTO {split}_source1 VALUES (?,?,?,?)", [
                ("S1-1", "Acme & Sons", "12 Road", "France"),
                ("S1-2", "Royal", "28 Road", "US"),
                ("S1-3", None, None, "India"),
                ("S1-4", "Solo", "1 Avenue", "France")])
            db.executemany(f"INSERT INTO {split}_source2 VALUES (?,?,?,?)", [
                ("S2-1", "ACME and Sons", "12 Road", "France"),
                ("S2-2", "Royal", "28 Road", "India"),
                ("S2-3", None, None, "India"),
                ("S2-4", "Royal", "82 Road", "US")])
            db.execute(f"INSERT INTO {split}_source3 VALUES ('S3-1','Acme and Sons','12 Road',NULL)")
        db.execute("CREATE TABLE truth_links (sid VARCHAR, tid VARCHAR)")
        db.executemany("INSERT INTO truth_links VALUES (?,?)", [("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-2", "S2-4")])
        db.execute("CREATE TABLE folds AS SELECT entity_id AS sid, 0 AS fold FROM train_source1")
        normalize_tables(db)
        for split in ("train", "test"):
            make_pairs(db, split)
        result = metrics(db)
        self.assertAlmostEqual(result["overall"]["macro_f05"], .75)
        self.assertEqual(result["overall"]["candidate_pairs"], 3)
        for sid, score in db.execute("SELECT sid, f05 FROM baseline_entity_scores").fetchall():
            truth = {x[0] for x in db.execute("SELECT tid FROM truth_links WHERE sid=?", [sid]).fetchall()}
            pred = {x[0] for x in db.execute("SELECT tid FROM train_candidates WHERE sid=? AND accepted", [sid]).fetchall()}
            self.assertEqual(score, entity_f05(truth, pred))
        with tempfile.TemporaryDirectory() as tmp:
            export_outputs(db, Path(tmp))
            matches = read_results(Path(tmp)/"output"/"matching_results.tsv")
            candidates = read_results(Path(tmp)/"output"/"candidate_pairs.tsv", "candidate_entity_ids")
            self.assertEqual(len(matches), 4)
            self.assertEqual(matches["S1-1"], {"S2-1", "S3-1"})
            self.assertEqual(matches["S1-2"], set())
            self.assertEqual(candidates["S1-2"], {"S2-2"})
            self.assertEqual(candidates["S1-3"], set())
            self.assertTrue(all(matches[k] <= candidates[k] for k in matches))
        db.close()


if __name__ == "__main__":
    unittest.main()
