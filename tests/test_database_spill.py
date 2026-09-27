import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb
from src.data import connect


class DatabaseSpillTests(unittest.TestCase):
    def test_readonly_database_sort_uses_writable_bounded_spill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'cache').mkdir()
            database = root / 'cache/entities.duckdb'
            with duckdb.connect(str(database)) as db:
                db.execute('CREATE TABLE sentinel AS SELECT 1 AS value')
            before = hashlib.sha256(database.read_bytes()).hexdigest()
            spill = root / 'writable-spill'
            with patch.dict(os.environ, {
                'ER_DB_READ_ONLY': '1', 'ER_DB_MEMORY': '32MB', 'ER_THREADS': '1',
                'ER_DB_TEMP_DIRECTORY': str(spill), 'ER_DB_MAX_TEMP_DIRECTORY_SIZE': '128MB',
            }):
                with connect(root) as db:
                    self.assertEqual(Path(db.execute("SELECT current_setting('temp_directory')").fetchone()[0]), spill.resolve())
                    with self.assertRaises(duckdb.InvalidInputException):
                        db.execute('CREATE TABLE forbidden AS SELECT 2')
                    # ~40 MB sortable payload exceeds the deliberately small RAM
                    # budget. Fetch in chunks so the regression stays bounded.
                    cursor = db.execute('SELECT i, repeat(md5(i::VARCHAR), 4) AS payload FROM range(300000) r(i) ORDER BY payload, i')
                    self.assertTrue(any(spill.iterdir()), 'Query must exercise disk spilling')
                    count = 0
                    previous = None
                    while True:
                        rows = cursor.fetchmany(1000)
                        if not rows:
                            break
                        for i, payload in rows:
                            key = (payload, i)
                            if previous is not None:
                                self.assertLessEqual(previous, key)
                            previous = key
                        count += len(rows)
                    self.assertEqual(count, 300000)
            self.assertEqual(hashlib.sha256(database.read_bytes()).hexdigest(), before)
