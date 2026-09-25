# Decided Approach — V3

Ask before replacing the overall architecture. Experiments may refine components, but changes must be justified by validation.

## 0. End-to-end baseline first

Build a complete simple pipeline before advanced modelling:

```text
load -> normalize -> simple blocking -> simple similarity rule -> outputs -> validator -> validation score
```

Purpose: verify data flow, scorer, output format and establish a floor.

## 1. Multi-view normalization

Never overwrite raw text.

Maintain:

- raw view;
- conservative view: Unicode NFKC, casefold, `& -> and`, punctuation to spaces, whitespace cleanup;
- accent-folded view;
- suffix-stripped core-name view with suffix stored separately;
- address-standardized view.

Possible suffix families include Inc/Incorporated, Corp/Corporation, LLC, Ltd/Limited, Pvt/Private, SARL, SAS, SA and EURL.

Keep every address number.

## 2. Candidate generation

Generate only S1<->S2 and S1<->S3 candidates.

Use multiple retrievers whose failure modes differ.

### Name char retriever

Character n-gram TF-IDF on normalized/core business name.

Test `char`, `char_wb` and multiple n-gram ranges.

### Address char retriever

Independent character n-gram TF-IDF on address.

### Token/BM25 retriever

Test token TF-IDF and BM25 as independent retrieval channels.

### Rare-token retriever

Use high-information tokens as keys.

Do not let IDF grow without bound. Test clipped/floored IDF so a one-off typo does not dominate.

Concept:

```text
clipped_idf = min(raw_idf, max_idf)
```

The clipping value is a hyperparameter, not a fact.

### Numeric retriever

Use generic address-number evidence where useful.

Numbers are retrieval clues, never absolute identity rules.

### Optional multilingual dense retriever

Only after sparse retrieval is strong.

Use only a verified license-safe model. Evaluate dense retrieval independently.

## 3. Adaptive candidate budget

Fixed top-K is the baseline, but also test adaptive retention.

Concept:

```text
keep at least min_k
keep additional candidates while score remains sufficiently close to best_score
never exceed max_k
```

Possible hyperparameters:

- min-K;
- max-K;
- score ratio to top candidate;
- score-drop threshold;
- absolute score floor.

Goal: preserve candidate recall while reducing useless candidates and inference cost.

## 4. Retrieval fusion and metadata

Test:

- union;
- normalized score fusion;
- Reciprocal Rank Fusion.

For every candidate retain:

- retrievers that found it;
- raw score per retriever;
- normalized score;
- rank;
- reciprocal rank;
- number of retrievers supporting the pair;
- best rank;
- rank disagreement.

This metadata becomes matcher evidence.

## 5. Blocking objective

Before increasing matcher complexity, track:

- link-level candidate recall;
- fraction of S1 entities with all true matches covered;
- reduction ratio;
- mean/median/p95 candidates per S1;
- incremental recall per retriever.

Blocking sets the recall ceiling.

## 6. Pairwise features

### Name

- raw exact;
- normalized exact;
- core-name exact;
- Jaro-Winkler;
- normalized Levenshtein;
- Damerau-Levenshtein;
- token Jaccard;
- Dice;
- token containment;
- token sort ratio;
- token set ratio;
- char n-gram cosine;
- longest common substring;
- prefix/suffix similarity;
- acronym match;
- token-count ratio;
- length ratio.

### Frequency / rare-token evidence

- shared rare-token count;
- maximum shared IDF;
- sum shared IDF;
- clipped maximum shared IDF;
- clipped sum shared IDF;
- IDF-weighted Jaccard.

### Address

- normalized exact;
- char cosine;
- token overlap;
- containment;
- edit similarity;
- rare-token agreement;
- numeric-set Jaccard;
- exact numeric-set equality;
- ordered numeric overlap;
- postal-like token agreement;
- longest common substring;
- length ratio.

### Soft contradiction features

- high name similarity + conflicting address number;
- generic same name + conflicting number;
- matching postal-like token;
- conflicting postal-like token;
- one address missing numbers.

Never use one numeric mismatch as an unconditional rejection.

### Country

Generic relational features only:

- same;
- different;
- one missing;
- both missing.

### Missingness

Explicit indicators for missing name/address/country on either side.

### Cross-field interactions

Examples:

- strong name + strong address;
- strong name + severe numeric conflict;
- weak name + very strong address;
- acronym match + address support;
- dense-high/sparse-low;
- sparse-high/dense-low;
- country mismatch + otherwise strong evidence.

## 7. Matcher

Primary matcher: LightGBM or XGBoost.

Also train logistic regression as a sanity/interpretable baseline.

Train on candidate pairs from the real blocker distribution, not mostly random negatives.

## 8. Typed hard-negative mining

Create explicit negative buckets:

1. same/near-same name, different address;
2. same/near-same address, different name;
3. rare-token collision;
4. number conflict;
5. acronym collision;
6. high sparse-score false positive;
7. high dense-score false positive;
8. generic business-name collision;
9. model's highest-confidence false positives.

Use a controlled mixture of positives, hard negatives, semi-hard negatives and a small amount of easy negatives.

Cap negatives per S1.

## 9. Iterative OOF hard-negative mining

1. train v1;
2. make OOF predictions;
3. collect highest-scoring false positives by bucket;
4. add or reweight them;
5. train v2;
6. keep v2 only if validation improves.

Do not mine from in-sample scores.

## 10. Calibration

Using OOF scores, compare:

- no calibration;
- Platt scaling;
- isotonic regression.

Keep calibration only if it improves F0.5 or threshold stability.

## 11. Precision-first entity decision layer

For each S1 derive:

- best score;
- second-best score;
- top1-top2 margin;
- number of candidates above score bands;
- candidate count;
- retrieval-channel support of top candidate;
- contradiction strength;
- score-distribution ambiguity/entropy.

Example:

```text
0.96 vs 0.40 = relatively clear
0.96 vs 0.95 = ambiguous
```

Do not rely only on absolute pair probability.

## 12. Match/no-match policies to compare

A. global pair threshold;

B. threshold + relative margin gate;

C. optional separate any-match/singleton classifier using S1-level features.

Select by entity-level macro F0.5.

## 13. Secondary-ID conflict handling

First inspect training truth.

If secondary IDs never belong to multiple S1 entities, test a uniqueness policy.

If one S2/S3 ID is accepted for several S1s:

- compare scores;
- compare margins;
- compare retrieval support;
- keep only a clearly dominant claim;
- reject ambiguous conflicts.

Start with simple greedy/margin logic. Add graph optimization only if validation proves value.

## 14. Aggregation

Collect accepted S2/S3 IDs per S1, deduplicate, preserve every S1 row, and write empty string for no match.

## 15. Optional embedding branch

Only after sparse + GBDT is strong.

Use multilingual embeddings as:

- extra retrieval branch;
- extra cosine feature.

Embeddings never make the final decision alone.

Keep only if they improve candidate recall, macro F0.5 or country-held-out robustness.

## 16. Out of scope unless explicitly requested

Do not add by default:

- generative LLM matchers;
- large cross-encoder rerankers;
- neural fine-tuning;
- ANN infrastructure if exact dense search is feasible;
- graph neural networks;
- Docker/CI/CD/APIs;
- large config frameworks.

Complexity must earn its place.
