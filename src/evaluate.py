"""Exact entity-level macro F0.5, including singleton credit."""

import argparse
import csv
import json


def entity_f05(truth, prediction):
    truth, prediction = set(truth), set(prediction)
    if not truth:
        return float(not prediction)
    tp = len(truth & prediction)
    return 5 * tp / (5 * tp + len(truth - prediction) + 4 * len(prediction - truth))


def read_results(path, column="matched_entity_ids"):
    result = {}
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", column]:
            raise ValueError(f"Invalid header in {path}: {reader.fieldnames}")
        for row in reader:
            sid = row["source1_entity_id"]
            ids = row[column].split(",") if row[column] else []
            if sid in result or len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate row or target IDs: {sid}")
            result[sid] = set(ids)
    return result


def evaluate(truth, prediction):
    if truth.keys() != prediction.keys():
        raise ValueError("Truth and predictions must contain exactly the same S1 IDs")
    if not truth:
        raise ValueError("Cannot evaluate an empty population")
    scores = [entity_f05(t, prediction[sid]) for sid, t in truth.items()]
    tp = sum(len(t & prediction[sid]) for sid, t in truth.items())
    predicted = sum(map(len, prediction.values()))
    actual = sum(map(len, truth.values()))
    singletons = [sid for sid, t in truth.items() if not t]
    return {
        "entities": len(truth), "macro_f05": sum(scores) / len(scores),
        "link_precision": tp / predicted if predicted else None,
        "link_recall": tp / actual if actual else None,
        "singleton_fp_rate": sum(bool(prediction[s]) for s in singletons) / len(singletons)
        if singletons else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(read_results(args.truth), read_results(args.predictions)), indent=2))
