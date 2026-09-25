# Problem and Data

## Problem

Business identity data arrives from three independent sources without a shared identifier.

- **S1:** clean, deduplicated reference source; one row per business.
- **S2/S3:** noisy business records that may correspond to S1 businesses.

For every S1 entity, return all matching S2/S3 records. The answer can contain zero, one, or many records.

## Expected files

All files are TSV and must be read with `sep="\t"`.

```text
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

If the files are elsewhere, locate them first. Do not modify originals.

## Schema

Expected source columns:

- `entity_id` — prefix S1-/S2-/S3- identifies the source.
- `business_name` — may contain abbreviations, legal suffix variants, typos, transliterations, punctuation and order changes.
- `business_address` — may be partial, reordered, landmark-based, abbreviated or missing components.
- `country` — train contains US/India; test additionally contains France.

Country is open-set. Do not hardcode the model to US/India.

Ground truth has:

- `source1_entity_id`
- `matched_entity_ids`

Empty match list means singleton/no match.

## Required audit

Before modelling, calculate:

- absolute paths;
- row counts and schema;
- duplicate IDs;
- missingness;
- name/address length and token statistics;
- country counts;
- singleton percentage;
- one-match and multi-match counts;
- S2-only, S3-only and mixed true matches;
- whether any S2/S3 ID maps to multiple S1 IDs;
- identical-name collisions across different S1 entities;
- identical-address collisions across different S1 entities.

Sample positive training pairs reproducibly and classify examples into:

- near exact;
- suffix variation;
- typo;
- token reorder;
- address abbreviation;
- transliteration;
- weak-name/strong-address;
- strong-name/weak-address;
- severe corruption.

## Do not assume

Do not assume:

- same name means same business;
- number mismatch means non-match;
- every S1 has a match;
- every S2/S3 belongs to S1;
- secondary IDs are unique to one S1 until labels confirm it;
- country labels are always consistent.
