# Probe-Word Selection Rule

## Data-driven vocabulary
For each of 73 prompts, scan ALL word_probs across ALL checkpoints of ALL 45 families.
Include any word above 1% probability in ANY model (base or aligned).
Result: 2,975 unique words, 7,071 word×prompt pairs.
No predetermined word list, no researcher selection bias.

## Symmetric filter
Include observation if EITHER base OR aligned > 1%.
Captures both repression and promotion.

## Multiple comparisons
Agreement measured as % of families agreeing on direction.
Two columns: agreement_filtered (relevant families) and agreement_all (all families).
No p-value correction: descriptive agreement rates, not per-word hypothesis tests.
PERMANOVA and mixed-effects handle inferential statistics at aggregate level.

## Canonical PERMANOVA (44-family, 3 metrics)
| Metric | Distance | Family R² | Method R² | Method p |
|---|---|---|---|---|
| Word magnitude | cosine | 95.5% | 2.6% | ns |
| Word magnitude | euclidean | 90.0% | 2.4% | ns |
| Word SIGN | hamming | 97.3% | 2.6% | ns |
| Category SIGN | hamming | 96.4% | 2.7% | ns |
| Bits of resistance | euclidean | 90.0% | 2.4% | ns |
| Bits of resistance | cosine | 95.5% | 2.6% | ns |
