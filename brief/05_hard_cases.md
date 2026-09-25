# Hard Cases and Expected Signals

These examples are illustrative, not challenge records. Use them for normalization tests, feature design, hard-negative buckets and error analysis.

## Legal suffix variants

`Acme Robotics Inc.` vs `Acme Robotics Incorporated` vs `Acme Robotics`

Signals: suffix extraction, core-name match, char similarity.

## Indian suffix forms

`Sharma Textiles Pvt Ltd` vs `Sharma Textiles Private Limited`

Signals: suffix normalization, core-name equality.

## French suffix forms

`Boulangerie Dupont SARL` vs `Boulangerie Dupont`

Signals: suffix extraction, accent-aware/folded views, core-name similarity.

French normalization is auxiliary; the architecture should not collapse without it.

## Punctuation and word order

`Smith & Sons Plumbing` vs `Plumbing Smith and Sons`

Signals: `& -> and`, token Jaccard, token sort/set ratios.

## Address abbreviation

`500 Market St` vs `500 Market Street`

Signals: address normalization, number agreement, token overlap.

## Typo

`Acme Robotics` vs `Acme Robotix`

Signals: Jaro-Winkler, Levenshtein, char n-grams.

## Transliteration / spelling variation

`Shree Ganesh Traders` vs `Shri Ganesh Traders`

Signals: char n-grams, edit similarity, address support.

## DBA / trade name

`ABC Foods Private Limited` vs `Sunrise Cafe`

Possible same business if address evidence is very strong. Treat as difficult; do not match from semantic intuition alone.

## Generic-name collision

`Royal Traders` vs `Royal Traders` at different addresses.

Exact name is insufficient. Use frequency-aware name evidence, address and number contradictions.

## Number conflict

`Royal Traders, Shop 28` vs `Royal Traders, Shop 82`

Often different, but number mismatch is soft evidence only.

## Partial / landmark address

`Near SBI ATM, Karol Bagh` vs a full Karol Bagh address.

Signals: containment, locality overlap, strong name evidence.

## Acronym

`Tata Consultancy Services` vs `TCS`

Signals: acronym extraction plus address support.

## Same address, different business

`Alpha Dental Clinic` and `Beta Pharmacy` in the same building.

This is a typed hard negative. Address alone is not enough.

## Dense semantic false positive

`City Dental Care` vs `Urban Dental Clinic`

Embeddings may score these highly despite being different businesses.

Protection: rare lexical evidence, address contradiction and sparse-vs-dense disagreement.

## Sparse lexical false negative

`ABC Hospitality Pvt Ltd` vs `The Riverside Bistro`

Trade-name mismatch may have little lexical overlap.

Recovery: address retrieval, numeric agreement, optional dense retrieval.

## Rare-token overconfidence

A misspelled token may be unique and receive huge IDF.

Protection: clipped IDF and supporting address/name evidence.

## Ambiguous top candidates

`0.96 vs 0.95` is less trustworthy than `0.96 vs 0.40`.

Signals: margin, score entropy and retrieval-channel support.

## Unseen-country example

`Boulangerie Dupont SARL, 12 rue de la Paix` vs `Boulangerie Dupont, 12 R. de la Paix`

Signals: char n-grams, accent folding, suffix-stripped name, numeric agreement, generic address normalization.

## Error taxonomy

Every false prediction must first be labeled as:

1. blocking failure — true pair never entered final candidates;
2. matcher failure — true pair was a candidate but scored badly;
3. decision failure — score was reasonable but threshold/margin/conflict logic failed.

Each class requires a different fix.
