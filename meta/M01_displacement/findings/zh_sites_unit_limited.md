# The Chinese site question is unit-limited, and the cell cannot be filled by analysis

**Status: A NEGATIVE WITH A STRUCTURAL CAUSE, not a null.** The site question ran
in Chinese for the first time on 2026-08-12. Both G quantities point the way the
English result points and neither comes close to significance, and **the reason
is not that the effect is absent — it is that the test has 20 exchangeable units
where it needs between 235 and 428.** No amount of reanalysis fixes that, and
the ceiling is set by how many Chinese-competent base checkpoints exist in the
world, not by anything this campaign chose.

Producer `scripts/zh_site_magnitude.py`, result
`results/zh_site_magnitude.json`. Frozen site rule `b8fd9a52cd5c794b` imported
and hash-checked; `build`, `pair_quantity`, `sign_flip_p` and `cohen_d` imported
from `magnitude.py` so no quantity is redefined here.

---

## Why this exists

M01's README puts three questions on each axis — in general, at transgressive
sites, across languages — and the axis-1 row for **at transgressive sites** cites
F and G with **no O beside them.** The site question had never been asked in
Chinese. The data had been collected the whole time: 412 Chinese prompts in
`twp_words`, 9.6M rows, 311 of the 314 collected design prompts sitting on all 52
CJK-capable models.

## What this is NOT

**It is not a replication of F/G.** F and G run on `M01_PAIRS` — 684 pairs, none
of which was ever translated. **The overlap between the M01 pair corpus and the
Chinese translation set is exactly zero**, checked both by `pair_id` and by
English string. The Chinese transgressive minimal pairs come from SETE, SETD,
F36_MINIMAL_PAIRS and CENSUS.

**Language and corpus both change, so no difference from the English result is
attributable to language.** What transfers is the instrument, and only that.

## What was run

    corpus     24 Chinese transgressive_swap minimal pairs, both twins collected
               violence 9, sexual 6, neutral 6, death/substance/profanity 1 each
               pair_minimal: True 9, n/a 12, False 3
    unit       the base checkpoint, n = 20 (every CJK-capable base with an arm)
    arms       21 base>aligned edges, tier >= PARTIAL on BOTH arms

## F — the rate, reported as DESCRIPTION and not as a verdict

    median rate_M 0.4583   rate_U 0.4167   median Delta  -0.0417
    positives 9/20   p 0.7483
    MDE at n=20:  P(positive) >= 0.799

**F is not run as a test here and its number is not a null.** At 20 units the
sign test needs P(positive) ≥ 0.799 to fire at 80% power. English F realised
0.606 and was itself a null; detecting *that* effect would take **141 units**.
Reporting a Chinese F verdict would breach M01's own reading rule 2 — no null
without its MDE. The rate quantities are computed because G's admissibility
depends on them, below.

## G — the magnitude

    PRIMARY   departed        n 20   median +0.002070   d +0.120   p 0.31481
    SECONDARY concentration   n 20   median +0.006712   d +0.162   p 0.24517
    positives 12/20 on both

Both point in the registered direction. Neither is significant. **And neither
should be quoted even as a null**, for the reason in the next section.

## G IS NOT ADMISSIBLE IN CHINESE AS SPECIFIED, and this is the actual finding

G §6 permits conditioning on both-fire **only because the two arms fire at
indistinguishable rates** — the English yield split was 17,301 / 16,595, ratio
1.043, and the registration says plainly that "the null result is what makes this
test admissible." That condition has to be checked in Chinese rather than
inherited, and it fails:

    units exceeding G's own SKEW_FLAG of 0.25       12 of 20
    both-fire pairs per unit (of 24 at risk)
        2 2 3 3 3 4 4 4 4 4 5 5 5 7 7 8 9 10 11 14
        median 4.5     ten of twenty units under 5

**Half the units compute their median D over fewer than five pairs.** A median
over three observations is not a unit summary; it is a coin with a magnitude
attached. The worst cells are `falcon-mamba-7b` (skew +1.667, both-fire 3),
`bloom-7b1` (−1.500, both-fire 2) and `Qwen3-8B-Base` (+1.200, both-fire 5).

So the p-values above are reported for completeness and are not evidence about
Chinese. The instrument did not measure what it was pointed at.

## What it would take, and why more models will not do it

At the effects actually observed, one-sided alpha 0.05, power 0.80:

    departed        d +0.120   ->   n ~ 428 units
    concentration   d +0.162   ->   n ~ 235 units
    CJK-capable base checkpoints in existence, this registry:  20

**The Chinese arm of M01 is unit-limited, not corpus-limited.** The sign-flip
permutation's n is the number of exchangeable base checkpoints, and Chinese
competence caps that near 20 — the pending 49-model tokenizer re-survey might
add two or three, not two hundred.

**Translating more pairs is still the lever worth pulling, for a different
reason.** With 24 pairs at risk and a median of 4.5 firing on both members, each
unit's D is estimated with heavy noise, and measurement error at the unit level
attenuates the standardised effect. A larger Chinese pair corpus would not raise
n but would stabilise each D, and the observed d of 0.12 is plausibly attenuated
rather than true. **That is a mechanism, not a measurement** — nobody has
estimated how much attenuation is present, and it should not be cited as a
reason to expect the effect to appear.

## Limits

The site rule was frozen on English and is applied unchanged to Chinese. Its
notion of a firing site is a top-word change with an availability bound, and
what counts as a "word" in the Chinese rows is whatever the twp ingest folded —
the same caveat `X_metonymy.md` §3h lives with, not a new one.

Six of the 24 pairs carry `domain: neutral`, and three carry
`pair_minimal: False`. Both were left in rather than filtered, because choosing
a subset after seeing the roster is the shopping this campaign exists to
prevent. A pre-declared restriction would be a different, defensible run.

The `both`-fire counts make every per-unit number in the JSON fragile. Quote the
structural conclusion; do not quote a unit.
