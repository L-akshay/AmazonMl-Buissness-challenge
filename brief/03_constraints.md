# Hard Constraints

These override architecture preferences.

## No external entity lookup

Forbidden:

- entity-resolution APIs;
- business registries;
- Google/Maps lookup;
- geocoding APIs;
- company directories;
- internet searches used to determine challenge identities;
- external identity augmentation.

Use only the supplied challenge data for matching.

External documentation may be used only for software/library/model-license verification.

## Model restriction

Final model must be:

- <= 8B parameters;
- MIT or Apache-2.0 compatible as required by the challenge.

Do not use hosted GPT/Claude/Gemini as the matcher.

For every pretrained model record:

- repository;
- exact model name;
- parameter count;
- license;
- revision/version when practical.

If license compatibility is unclear, exclude it.

## Generalization

Do not hardcode the system to US/India.

France is unseen in training.

Use country mainly as relational evidence:

- same;
- different;
- missing.

Small general-knowledge suffix/street vocabularies are allowed only as documented auxiliary normalization, never as challenge-specific identity lookup.

## Reproducibility

Final outputs must be regenerable from:

- supplied data;
- submitted code;
- declared dependencies;
- allowed pretrained models.

No manual prediction edits.

## Leakage

Forbidden:

- validation labels during training;
- threshold selection on in-sample predictions;
- supervised fitting across fold boundaries;
- manually correcting validation predictions after reading truth.

## Final consistency

Pipeline must fail loudly if:

- a test S1 is missing;
- a predicted ID does not exist;
- a predicted ID is not S2/S3;
- IDs are duplicated;
- final matches are not a subset of candidates.
