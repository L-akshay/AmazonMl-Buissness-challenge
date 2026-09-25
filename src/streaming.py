"""Sorted, fixed-size input batches with compact training-label lookup."""

import numpy as np


def encode_secondary(entity_id):
    prefix, number = entity_id.split("-", 1)
    if prefix not in ("S2", "S3") or not number.isascii() or not number.isdecimal():
        raise ValueError(f"Unexpected secondary ID format: {entity_id!r}")
    value = int(number)
    if value >= 2**32 or str(value) != number:
        raise ValueError("Secondary ID must have a canonical unsigned 32-bit suffix")
    return (int(prefix[1]) << 32) | value


def label_lookup(db, root):
    folder = root / "cache" / "label_lookup_v1"
    folder.mkdir(exist_ok=True)
    codes_path, owners_path = folder / "codes.npy", folder / "owners.npy"
    if not (codes_path.exists() and owners_path.exists()):
        invalid = db.execute("""SELECT count(*) FROM truth_links WHERE
            NOT regexp_full_match(tid, 'S[23]-(0|[1-9][0-9]*)')
            OR try_cast(substr(tid,4) AS UBIGINT) >= 4294967296""").fetchone()[0]
        if invalid:
            raise ValueError("Truth labels contain unsupported ID formats")
        values = db.execute("""SELECT
            ((substr(t.tid,2,1)::UBIGINT << 32) + substr(t.tid,4)::UBIGINT) AS code,
            r.ri::INTEGER AS owner
            FROM truth_links t JOIN ml_train_references r ON t.sid=r.entity_id
            ORDER BY code""").fetchnumpy()
        if np.any(values["code"][1:] <= values["code"][:-1]):
            raise ValueError("Secondary ownership must be unique")
        np.save(codes_path, values["code"])
        np.save(owners_path, values["owner"])
    return np.load(codes_path, mmap_mode="r"), np.load(owners_path, mmap_mode="r")


def attach_labels(records, labels=None):
    owners = np.full(len(records), -1, dtype=np.int32)
    if labels is not None:
        codes, reference_owners = labels
        queries = np.array([encode_secondary(r[0]) for r in records], dtype=np.uint64)
        positions = np.searchsorted(codes, queries)
        valid = positions < len(codes)
        indices = np.flatnonzero(valid)
        indices = indices[codes[positions[indices]] == queries[indices]]
        owners[indices] = reference_owners[positions[indices]]
    return [tuple(r) + (int(owner),) for r, owner in zip(records, owners)]


def secondary_batches(db, split, batch_size=10000, labels=None):
    if split not in ("train", "test"):
        raise ValueError("Unknown split")
    pending = []
    # All S2 IDs sort before all S3 IDs. Preserve the original UNION ORDER BY
    # ordering, including the batch that crosses the source boundary.
    for source in (2, 3):
        cursor = db.execute(f"SELECT * FROM {split}_source{source}_norm ORDER BY entity_id")
        while True:
            records = cursor.fetchmany(batch_size - len(pending))
            if not records:
                break
            pending.extend(records)
            if len(pending) == batch_size:
                yield attach_labels(pending, labels)
                pending = []
    if pending:
        yield attach_labels(pending, labels)
