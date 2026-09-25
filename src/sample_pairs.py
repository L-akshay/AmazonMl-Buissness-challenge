"""Sample labeled positives; diagnostic tags are heuristics, not ground truth."""

import csv
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import unicodedata
from src.data import connect, rows
from src.normalize import name_views, address_views


def tags(a_name, a_address, b_name, b_address):
    a, b = name_views(a_name), name_views(b_name)
    x, y = address_views(a_address), address_views(b_address)
    name = SequenceMatcher(None, a["normalized"], b["normalized"]).ratio()
    address = SequenceMatcher(None, x["normalized"], y["normalized"]).ratio()
    found = []
    if name >= .95 and address >= .95:
        found.append("near_exact")
    if a["core"] == b["core"] and a["suffix"] != b["suffix"]:
        found.append("suffix_variation")
    if .75 <= name < .95:
        found.append("possible_typo")
    if a["normalized"] != b["normalized"] and sorted(a["normalized"].split()) == sorted(b["normalized"].split()):
        found.append("token_reorder")
    if x["normalized"] != y["normalized"] and x["standardized"] == y["standardized"]:
        found.append("address_abbreviation")
    scripts_a = {unicodedata.name(c, "").split()[0] for c in a["normalized"] if c.isalpha()}
    scripts_b = {unicodedata.name(c, "").split()[0] for c in b["normalized"] if c.isalpha()}
    if scripts_a and scripts_b and not scripts_a & scripts_b:
        found.append("possible_transliteration")
    if name < .5 and address > .85:
        found.append("weak_name_strong_address")
    if name > .85 and address < .5:
        found.append("strong_name_weak_address")
    if name < .5 and address < .5:
        found.append("possible_severe_corruption")
    return ",".join(found or ["other_variation"])


def main():
    root = Path(__file__).resolve().parents[1]
    db = connect(root)
    sample = rows(db, """WITH sample AS (
        SELECT * FROM truth_links ORDER BY md5('42|' || sid || '|' || tid) LIMIT 2000)
        SELECT t.sid, t.tid, a.country AS s1_country, b.country AS secondary_country,
            a.business_name AS s1_name, a.business_address AS s1_address,
            b.business_name AS secondary_name, b.business_address AS secondary_address
        FROM sample t JOIN train_source1 a ON t.sid=a.entity_id
        JOIN (SELECT * FROM train_source2 UNION ALL SELECT * FROM train_source3) b ON t.tid=b.entity_id
        ORDER BY md5('42|' || sid || '|' || tid)""")
    counts = Counter()
    for row in sample:
        row["heuristic_tags"] = tags(row["s1_name"], row["s1_address"], row["secondary_name"], row["secondary_address"])
        counts.update(row["heuristic_tags"].split(","))
    path = root / "reports" / "positive_pair_sample.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(sample)
    result = {"sample_size": len(sample), "seed": 42, "method": "lowest MD5 of seed|S1|target across all positive links", "tags": dict(counts), "limitation": "Tags are overlapping string-similarity heuristics, not verified linguistic causes."}
    (root/"reports"/"positive_pair_sample.json").write_text(json.dumps(result,indent=2), encoding="utf-8")
    print(json.dumps(result,indent=2))
    db.close()


if __name__ == "__main__":
    main()
