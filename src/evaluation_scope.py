"""Keep previously inspected labeled examples out of the reserved matcher fold."""

import csv
import json
import numpy as np
from src.data import connect


def exclusion_indices(root):
    folder=root/"cache"
    path=folder/"holdout_exclusions.npy"
    report_path=folder/"holdout_exclusions.json"
    if path.exists() and report_path.exists():
        return np.load(path),json.loads(report_path.read_text())
    db=connect(root)
    db.execute("SET memory_limit='2GB'")
    db.execute("SET threads=1")
    # %1000 includes the actual %2000 pilot and the retriever's default pilot.
    # The first 1000 baseline-order entities conservatively cover the 100
    # inspected E01 errors; assert that fact against the original local sample.
    excluded=db.execute("""WITH exposed AS (
        SELECT DISTINCT sid FROM truth_links WHERE hash(tid)%1000=0
        UNION SELECT sid FROM (SELECT sid FROM truth_links
            ORDER BY md5('42|' || sid || '|' || tid) LIMIT 2000)
        UNION SELECT entity_id AS sid FROM (SELECT entity_id FROM ml_train_references
            ORDER BY md5('42|' || entity_id) LIMIT 1000))
        SELECT r.ri,r.entity_id FROM ml_train_references r JOIN exposed e ON r.entity_id=e.sid
        ORDER BY r.ri""").fetchall()
    db.close()
    known={r[1] for r in excluded}
    for name in ("positive_pair_sample.tsv","baseline_errors.tsv"):
        sample=root/"reports"/name
        if sample.exists():
            with sample.open(encoding="utf-8",newline="") as f:
                inspected={r["sid"] for r in csv.DictReader(f,delimiter="\t")}
            if not inspected<=known:
                raise ValueError(f"Reserved-fold exclusion rule does not cover prior sample {name}")
    indices=np.array([r[0] for r in excluded],dtype=np.int32)
    report={"excluded_reference_entities":len(indices),
        "rule":"Owners of hash(secondary_id)%1000=0 pilot links; 2000 seed-42 positive-pair diagnostic examples; first 1000 seed-42 baseline-order entities (verified to include the 100 inspected errors).",
        "scope":"Excluded only from the reserved matcher evaluation sample. These entities remain eligible for development or final model training as their saved folds allow.",
        "limitation":"Earlier full-training baseline and retrieval aggregates were viewed; the reserved result is a supervised matcher holdout, not a completely untouched end-to-end pipeline holdout."}
    np.save(path,indices)
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return indices,report
