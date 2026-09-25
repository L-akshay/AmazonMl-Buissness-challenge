"""Strict TSV ingestion into a bounded-memory local database."""

from pathlib import Path
import csv
import duckdb

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def connect(root):
    cache = Path(root) / "cache"
    cache.mkdir(exist_ok=True)
    connection = duckdb.connect(str(cache / "entities.duckdb"))
    connection.execute("SET memory_limit='6GB'")
    connection.execute("SET threads=4")
    connection.execute("SET preserve_insertion_order=false")
    return connection


def load(connection, root):
    for split in ("train", "test"):
        names = [f"{split}_source{i}" for i in (1, 2, 3)]
        if split == "train":
            names.append("train_ground_truth")
        for name in names:
            path = Path(root) / "dataset" / split / f"{name}.tsv"
            expected = TRUTH_COLUMNS if "truth" in name else SOURCE_COLUMNS
            with path.open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle, delimiter="\t"))
            if header != expected:
                raise ValueError(f"Unexpected schema in {path}: {header}")
            if connection.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", [name]).fetchone()[0]:
                continue
            print(f"Loading {name}", flush=True)
            connection.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv(?, delim='\t', header=true, all_varchar=true, nullstr='', strict_mode=true)", [str(path)])
    connection.execute("""CREATE TABLE IF NOT EXISTS truth_links AS
        SELECT source1_entity_id AS sid, unnest(string_split(matched_entity_ids, ',')) AS tid
        FROM train_ground_truth WHERE matched_entity_ids IS NOT NULL""")


def rows(connection, sql, params=None):
    cursor = connection.execute(sql, params or [])
    names = [x[0] for x in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]
