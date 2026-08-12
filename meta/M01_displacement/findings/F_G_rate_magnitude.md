---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-07
role: finding
description: "Registered letters F/G: rate null (n=33 pair-sites, p 0.148) and magnitude confirmed (d 0.748, p 0.00006) -- alignment displaces HARDER at transgressive sites, not more often. F's null TRIPLY validated 2026-08-12 (argmax flip, faller count, faller share agree; f_mass_rate.py, 46 reps); frequency framing exhausted. Result artifacts cite pre-re-freeze registration hashes -- provenance note travels."
scripts: [f_mass_rate.py]
---
# Findings F / G: alignment does not displace more often at transgressive sites; it displaces harder

Split out of `C_to_O_registered_letters.md` on 2026-08-12 (RH's commission), rewritten the same day to be readable on its own: the hypotheses are stated in plain terms rather than by registration number. `REGISTRATIONS.md` remains authoritative for every number; the registration files hold the frozen statistical detail.

## What was asked

The pair of registrations that set the campaign's shape. Within minimal pairs (one transgressive member, one neutral twin), **F** asked about FREQUENCY: does the transgressive member fire displacement more OFTEN? Its measure was a rate — the fraction of at-risk pairs whose marked member shows a displacement event versus the fraction whose unmarked member does. **G**, commissioned by RH after F's answer came back, asked the same question by MASS: never mind how often, how much probability moves when it moves?

## What was found

**F: no rate difference.** Over 33 pair-sites, the marked member does not fire displacement significantly more often than its neutral twin (p 0.148). **G: a large magnitude difference.** Measured by departed mass — the total probability that leaves words at the site — the transgressive member loses far more (effect size d 0.748, p 0.00006). The sentence the pair licenses: **alignment does not displace more often at transgressive sites; it displaces harder.**

## F's null, validated three ways (added 2026-08-12)

F defines "displacement happened" as an argmax flip: the model's single top word changed, and the new top word was already in the base model's top 20. That is a threshold-free but fragile-sounding criterion — a hair's-width nudge at a near-tie counts, a large migration that leaves the top word standing does not. The obvious objection was that the null is an artifact of that fragility, and it had never been tested. `scripts/f_mass_rate.py` re-asked F's question with the campaign's mass criterion instead (a word counts as falling only if it had real probability, at least 0.003, and lost at least half of it), on 46 lineage representatives rather than 33:

    fallers per cell    transgressive  15.33   neutral twin  15.04   (+2.0%)
    words scored        transgressive 155.56   neutral twin 151.75   (+2.5%)
    faller SHARE        transgressive 0.1034   neutral twin 0.1036   (-0.2%)

The count of falling words and its denominator rise together, so the per-word probability of falling is identical at the two site types. **Three operationalisations of the frequency question now agree — argmax flip, faller count, faller share — so F's null is not an artifact of how "an event" was defined.**

And the frequency framing looks exhausted rather than merely unresolved. Raw counts inherit a bookkeeping asymmetry (the transgressive member simply has ~2.5% more words scored, an artifact varying 12-fold across checkpoints), which mass does not. With finding T-14 — falling words are few and 3.8x larger, rising words many and small — a count of events discards exactly the shape that exists. **G was always the instrument this question needed.**

This validation is not a re-run of F: F's unit, corpus and statistic are untouched and its registered null stands as written.

## The limit that travels

Both result artifacts cite registration hashes from before a re-freeze (`8ff56206…`, `efbab158…`); the `REGISTRATIONS.md` row documents the mismatch and the results stand with that provenance note attached.
