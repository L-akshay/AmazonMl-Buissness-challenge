"""Reproducible full-data integrity and distribution audit (E00)."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from src.data import connect, load, rows


def audit(root):
    start = perf_counter()
    db = connect(root)
    load(db, root)
    report = {"sources": {}, "integrity": {}}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            table = f"{split}_source{source}"
            print(f"Auditing {table}", flush=True)
            fields = []
            for field in ("business_name", "business_address", "country"):
                fields.append(f"count(*) FILTER (WHERE {field} IS NULL OR trim({field})='') AS {field}_missing")
            for field in ("business_name", "business_address"):
                fields.extend([f"avg(length({field})) AS {field}_mean_chars",
                               f"quantile_cont(length({field}), 0.5) AS {field}_median_chars",
                               f"quantile_cont(length({field}), 0.95) AS {field}_p95_chars",
                               f"avg(array_length(regexp_split_to_array(trim({field}), '\\s+'))) AS {field}_mean_tokens"])
            summary = rows(db, f"SELECT count(*) AS rows, count(*)-count(DISTINCT entity_id) AS duplicate_ids, count(*) FILTER (WHERE entity_id IS NULL OR NOT starts_with(entity_id, 'S{source}-')) AS invalid_ids, {', '.join(fields)} FROM {table}")[0]
            summary["path"] = str((root / "dataset" / split / f"{table}.tsv").resolve())
            summary["schema"] = rows(db, f"DESCRIBE {table}")
            summary["countries"] = rows(db, f"SELECT country, count(*) AS rows FROM {table} GROUP BY country ORDER BY country")
            if source == 1:
                summary["collisions"] = {}
                for field in ("business_name", "business_address"):
                    summary["collisions"][field] = rows(db, f"SELECT count(*) AS distinct_colliding_values, coalesce(sum(n),0) AS affected_entities FROM (SELECT {field}, count(*) n FROM {table} WHERE {field} IS NOT NULL AND trim({field})<>'' GROUP BY {field} HAVING count(*)>1)")[0]
            report["sources"][table] = summary
    report["truth_match_counts"] = rows(db, "SELECT coalesce(array_length(string_split(matched_entity_ids, ',')),0) AS matches, count(*) AS entities FROM train_ground_truth GROUP BY matches ORDER BY matches")
    report["truth_source_mix"] = rows(db, "SELECT CASE WHEN matched_entity_ids IS NULL THEN 'singleton' WHEN contains(matched_entity_ids,'S2-') AND contains(matched_entity_ids,'S3-') THEN 'mixed' WHEN contains(matched_entity_ids,'S2-') THEN 'S2-only' ELSE 'S3-only' END AS source_mix, count(*) AS entities FROM train_ground_truth GROUP BY source_mix ORDER BY source_mix")
    checks = {
        "truth_duplicate_s1": "SELECT count(*)-count(DISTINCT source1_entity_id) FROM train_ground_truth",
        "missing_truth_s1": "SELECT count(*) FROM train_source1 ANTI JOIN train_ground_truth ON entity_id=source1_entity_id",
        "unknown_truth_s1": "SELECT count(*) FROM train_ground_truth ANTI JOIN train_source1 ON entity_id=source1_entity_id",
        "duplicate_truth_links": "SELECT count(*) FROM (SELECT sid,tid FROM truth_links GROUP BY sid,tid HAVING count(*)>1)",
        "unknown_truth_targets": "SELECT count(*) FROM truth_links ANTI JOIN (SELECT entity_id FROM train_source2 UNION ALL SELECT entity_id FROM train_source3) s ON tid=entity_id",
        "secondary_ids_with_multiple_s1": "SELECT count(*) FROM (SELECT tid FROM truth_links GROUP BY tid HAVING count(DISTINCT sid)>1)",
        "cross_split_id_overlap": "SELECT count(*) FROM (SELECT entity_id FROM train_source1 UNION ALL SELECT entity_id FROM train_source2 UNION ALL SELECT entity_id FROM train_source3) a JOIN (SELECT entity_id FROM test_source1 UNION ALL SELECT entity_id FROM test_source2 UNION ALL SELECT entity_id FROM test_source3) b USING(entity_id)",
    }
    for key, sql in checks.items():
        print(f"Checking {key}", flush=True)
        report["integrity"][key] = db.execute(sql).fetchone()[0]
    report["runtime_seconds"] = round(perf_counter()-start, 2)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    total = report["sources"]["train_source1"]["rows"]
    singleton = next((r["entities"] for r in report["truth_match_counts"] if r["matches"] == 0), 0)
    lines = ["# Full-data audit (E00)", "", "One source row is one business record; truth has one row per reference S1 entity.", "", "| Source | Rows | Duplicate IDs | Missing name | Missing address |", "|---|---:|---:|---:|---:|"]
    for name, info in report["sources"].items():
        lines.append(f"| {name} | {info['rows']:,} | {info['duplicate_ids']:,} | {info['business_name_missing']:,} | {info['business_address_missing']:,} |")
    lines += ["", f"Training singletons: {singleton:,}/{total:,} ({singleton/total:.2%}).", "", "## Integrity checks", ""]
    lines += [f"- {key}: {value:,}" for key, value in report["integrity"].items()]
    lines += ["", "## Interpretation and scope", "", "Name/address collisions are expected in entity resolution: exact name or address alone is not sufficient identity evidence. Their full counts and country distributions are in data_audit.json.", "", "Country is an open set. France performance cannot be measured from training labels. No event timestamps are supplied; freshness and temporal drift cannot be assessed.", "", "Raw UTF-8 fields are retained. MacOS resource forks were excluded at extraction. No external identities or lookup services were used.", "", "Inspect the exact queries in src/audit.py and notebooks/data_audit.ipynb. Input ZIP paths, members, byte sizes and CRCs are in input_manifest.json.", "", f"Runtime: {report['runtime_seconds']:.2f} seconds."]
    (reports / "data_audit.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    critical = [k for k,v in report["integrity"].items() if v and k not in ("secondary_ids_with_multiple_s1", "cross_split_id_overlap")]
    if critical or any(s["duplicate_ids"] or s["invalid_ids"] for s in report["sources"].values()):
        raise ValueError(f"Data integrity failures; inspect report: {critical}")
    print(json.dumps({"singleton_rate": singleton/total, "integrity": report["integrity"], "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    audit(parser.parse_args().root)
