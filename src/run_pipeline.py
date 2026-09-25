"""E01: untuned exact name-and-address baseline with full output validation."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from src.data import connect, load, rows
from src.normalize import conservative


def sql_path(path):
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def normalize_tables(db):
    db.create_function("conservative_text", conservative, ["VARCHAR"], "VARCHAR")
    for split in ("train", "test"):
        for source in (1, 2, 3):
            table = f"{split}_source{source}"
            if db.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", [table+"_norm"]).fetchone()[0]:
                continue
            print(f"Normalizing {table}", flush=True)
            db.execute(f"""CREATE TABLE {table}_norm AS SELECT entity_id,
                conservative_text(coalesce(business_name,'')) AS name_norm,
                conservative_text(coalesce(business_address,'')) AS address_norm,
                conservative_text(coalesce(country,'')) AS country_norm
                FROM {table}""")


def make_pairs(db, split):
    print(f"Generating {split} candidates", flush=True)
    db.execute(f"""CREATE OR REPLACE TABLE {split}_candidates AS
        SELECT a.entity_id AS sid, b.entity_id AS tid,
            (a.country_norm=b.country_norm OR a.country_norm='' OR b.country_norm='') AS accepted
        FROM {split}_source1_norm a JOIN
            (SELECT * FROM {split}_source2_norm UNION ALL SELECT * FROM {split}_source3_norm) b
        ON a.name_norm=b.name_norm AND a.address_norm=b.address_norm
        WHERE a.name_norm<>'' AND a.address_norm<>''""")
    duplicates = db.execute(f"SELECT count(*) FROM (SELECT sid,tid FROM {split}_candidates GROUP BY sid,tid HAVING count(*)>1)").fetchone()[0]
    if duplicates:
        raise ValueError("Duplicate candidate links")


def metrics(db):
    db.execute("""CREATE OR REPLACE TABLE baseline_entity_scores AS
        WITH truth AS (SELECT sid, count(*) AS truth_count FROM truth_links GROUP BY sid),
        preds AS (SELECT sid, count(*) AS candidate_count, count(*) FILTER (WHERE accepted) AS pred_count FROM train_candidates GROUP BY sid),
        hits AS (SELECT c.sid, count(*) AS covered, count(*) FILTER (WHERE c.accepted) AS tp
            FROM train_candidates c JOIN truth_links t USING(sid,tid) GROUP BY c.sid),
        counts AS (SELECT s.entity_id AS sid, s.country, coalesce(t.truth_count,0) AS truth_count,
            coalesce(p.candidate_count,0) AS candidate_count, coalesce(p.pred_count,0) AS pred_count,
            coalesce(h.covered,0) AS covered, coalesce(h.tp,0) AS tp
            FROM train_source1 s LEFT JOIN truth t ON s.entity_id=t.sid
            LEFT JOIN preds p ON s.entity_id=p.sid LEFT JOIN hits h ON s.entity_id=h.sid)
        SELECT *, CASE WHEN truth_count=0 THEN CASE WHEN pred_count=0 THEN 1.0 ELSE 0.0 END
            ELSE 5.0*tp/(5*tp + truth_count-tp + 4*(pred_count-tp)) END AS f05 FROM counts""")
    summary_sql = """count(*) AS entities, avg(f05) AS macro_f05,
        sum(tp)*1.0/nullif(sum(pred_count),0) AS link_precision,
        sum(tp)*1.0/nullif(sum(truth_count),0) AS link_recall,
        sum(covered)*1.0/nullif(sum(truth_count),0) AS candidate_recall,
        avg(CASE WHEN covered=truth_count THEN 1.0 ELSE 0.0 END) AS all_true_matches_covered,
        avg(CASE WHEN truth_count>0 THEN CASE WHEN covered=truth_count THEN 1.0 ELSE 0.0 END END) AS nonsingleton_all_matches_covered,
        avg(CASE WHEN truth_count=0 THEN CASE WHEN pred_count>0 THEN 1.0 ELSE 0.0 END END) AS singleton_fp_rate,
        sum(candidate_count) AS candidate_pairs, avg(candidate_count) AS mean_candidates,
        quantile_cont(candidate_count,0.5) AS median_candidates,
        quantile_cont(candidate_count,0.95) AS p95_candidates"""
    result = {"overall": rows(db, f"SELECT {summary_sql} FROM baseline_entity_scores")[0],
              "by_country": rows(db, f"SELECT country, {summary_sql} FROM baseline_entity_scores GROUP BY country ORDER BY country"),
              "by_match_count": rows(db, f"SELECT CASE WHEN truth_count=0 THEN 'zero' WHEN truth_count=1 THEN 'one' ELSE 'multiple' END AS match_count, {summary_sql} FROM baseline_entity_scores GROUP BY match_count ORDER BY match_count"),
              "by_fold": rows(db, f"SELECT fold, {summary_sql} FROM baseline_entity_scores JOIN folds USING(sid) GROUP BY fold ORDER BY fold")}
    secondary_count = db.execute("SELECT (SELECT count(*) FROM train_source2)+(SELECT count(*) FROM train_source3)").fetchone()[0]
    result["overall"]["reduction_ratio"] = 1-result["overall"]["candidate_pairs"]/(result["overall"]["entities"]*secondary_count)
    return result


def export_outputs(db, root):
    out = root / "output"
    out.mkdir(exist_ok=True)
    # Inner joins construct candidates exclusively from valid source IDs. Check
    # persisted tables again so stale or manually modified caches cannot bypass it.
    invalid = db.execute("""SELECT count(*) FROM test_candidates c
        LEFT JOIN test_source1 a ON c.sid=a.entity_id
        LEFT JOIN (SELECT entity_id FROM test_source2 UNION ALL SELECT entity_id FROM test_source3) b ON c.tid=b.entity_id
        WHERE a.entity_id IS NULL OR b.entity_id IS NULL OR NOT (starts_with(c.tid,'S2-') OR starts_with(c.tid,'S3-'))""").fetchone()[0]
    if invalid:
        raise ValueError("Unknown or invalid candidate IDs")
    for filename, column, predicate in [("candidate_pairs.tsv", "candidate_entity_ids", "true"),
                                         ("matching_results.tsv", "matched_entity_ids", "accepted")]:
        # Both exports use the final scored table; accepted links are necessarily
        # a subset. No additional pruning is performed after this point.
        db.execute(f"""COPY (SELECT s.entity_id AS source1_entity_id, coalesce(p.ids,'') AS {column}
            FROM test_source1 s LEFT JOIN (SELECT sid, string_agg(tid, ',' ORDER BY tid) AS ids
                FROM test_candidates WHERE {predicate} GROUP BY sid) p ON s.entity_id=p.sid
            ORDER BY s.entity_id) TO {sql_path(out / filename)} (HEADER, DELIMITER '\t', QUOTE '')""")


def write_errors(db, root):
    errors = rows(db, """SELECT sid, country, truth_count, pred_count, covered, tp, f05,
        CASE WHEN covered<truth_count THEN 'blocking' ELSE 'decision' END AS first_failure
        FROM baseline_entity_scores WHERE f05<1 ORDER BY md5('42|' || sid) LIMIT 100""")
    with (root / "reports" / "baseline_errors.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(errors[0]) if errors else ["sid"], delimiter="\t")
        writer.writeheader()
        writer.writerows(errors)
    cause = rows(db, """SELECT
        count(*) FILTER (WHERE t.tid IS NOT NULL AND c.tid IS NULL) AS missed_by_blocking,
        count(*) FILTER (WHERE t.tid IS NOT NULL AND c.tid IS NOT NULL AND NOT c.accepted) AS rejected_true_candidates
        FROM truth_links t LEFT JOIN train_candidates c USING(sid,tid)""")[0]
    (root / "reports" / "error_analysis.md").write_text(
        "# Baseline error analysis\n\n" + "\n".join(f"- {k}: {v:,}" for k,v in cause.items()) +
        "\n\nThe exact joint-name/address blocker deliberately establishes a precision-oriented floor. "
        "It cannot retrieve most typo, abbreviation, partial-address, transliteration or reordered-text pairs. "
        "The next experiment is name character retrieval followed by independent address retrieval, measuring added recall and volume. "
        "The 100-row deterministic error sample is baseline_errors.tsv; it is not used to manually edit predictions.\n", encoding="utf-8")


def run(root):
    start = perf_counter()
    audit_path = root / "reports" / "data_audit.json"
    if not audit_path.exists():
        raise ValueError("Run python -m src.audit before modelling")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    allowed = {"secondary_ids_with_multiple_s1", "cross_split_id_overlap"}
    if any(v for k,v in audit["integrity"].items() if k not in allowed):
        raise ValueError("Resolve audit integrity failures first")
    if any(s["duplicate_ids"] or s["invalid_ids"] for s in audit["sources"].values()):
        raise ValueError("Resolve source ID integrity failures first")
    db = connect(root)
    load(db, root)
    normalize_tables(db)
    db.execute("""CREATE TABLE IF NOT EXISTS folds AS SELECT entity_id AS sid,
        (row_number() OVER (ORDER BY md5('42|' || entity_id))-1)%5 AS fold FROM train_source1""")
    db.execute(f"COPY (SELECT sid AS source1_entity_id, fold FROM folds ORDER BY sid) TO {sql_path(root / 'experiments' / 'folds.tsv')} (HEADER, DELIMITER '\t')")
    for split in ("train", "test"):
        make_pairs(db, split)
    result = metrics(db)
    write_errors(db, root)
    export_outputs(db, root)
    db.close()
    validator = subprocess.run([sys.executable, "-X", "utf8", str(root / "utils" / "validate_submission.py"),
        "--matching", str(root / "output" / "matching_results.tsv"),
        "--candidate", str(root / "output" / "candidate_pairs.tsv"),
        "--test-dir", str(root / "dataset" / "test"), "--check-ids"],
        capture_output=True, text=True, encoding="utf-8")
    (root / "reports" / "baseline_validator.txt").write_text(validator.stdout+validator.stderr, encoding="utf-8")
    print(validator.stdout, flush=True)
    if validator.returncode or "WARNING:" in validator.stdout:
        raise RuntimeError("Validator failed or warned; inspect reports/baseline_validator.txt")
    result["runtime_seconds"] = round(perf_counter()-start,2)
    result["evaluation_scope"] = "Full labeled training population; fixed rule specified before labels, no fitting or threshold tuning. Not an OOF-trained model or unseen-test performance."
    result["policy"] = "Candidate: nonempty conservative name AND address exact. Accept if countries agree or either missing."
    result["run_id"] = datetime.now(timezone.utc).strftime("E01_%Y%m%dT%H%M%S%fZ")
    (root / "reports" / "baseline_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (root / "experiments" / f"{result['run_id']}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    record = {"experiment": "E01", "split": "full-train-fixed-rule", "blocker": "exact-name-and-address",
              "matcher": "untuned-country-consistency-rule", **result["overall"], "runtime_seconds": result["runtime_seconds"]}
    log = root / "experiments" / "results.csv"
    exists = log.exists()
    with log.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(record))
        if not exists:
            writer.writeheader()
        writer.writerow(record)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root.resolve())
