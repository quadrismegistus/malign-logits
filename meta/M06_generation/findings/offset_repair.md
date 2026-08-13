---
status: draft
grade: ungraded  # single pass, no cross-seat audit; per [5503] nothing here is audit-grade until a second seat reproduces from results/offset_repair.json
date: 2026-08-13
role: finding
topics: [self-surprisal, forced-arms, repair, chain]
description: "RH's offset repair for the construction defect that withdrew opening_matched: drop the undisturbed arm's first word so both arms read prompt + one unscored word + scored continuation. R1 confirmed and exceeded -- the withdrawn -0.03..-0.05 does not merely collapse, IT REVERSES SIGN. Repaired estimates run +0.008 to +0.035 (DAMAGE direction, an imposed word leaves a harder continuation than a sampled one), strongest in BASE. The surviving positive is NOT yet separable from an opening-probability difference between arms, which this plan fenced in advance and does not claim past."
---
# The offset repair: the sign reverses

Plan: `plans/plan_offset_repair.md`, committed before this producer, with R1
declared. Producer `scripts/m06_offset_repair.py`; results
`results/offset_repair.json`. No new compute. Single pass; [5503] applies.

RH's fix, on reading the [5811] withdrawal: if forced rows are scored after
`prompt + forced_word` and undisturbed rows after `prompt` alone, then **drop
the first scored word from the undisturbed arm and treat it as if it had been
forced**. Both arms then read `prompt + one unscored word + scored
continuation`, and the only structural difference left is whether that word
was SAMPLED or IMPOSED -- which is the question the offset made unaskable.

## k is measured, not assumed

The dropped unit is a WORD; the array is TOKENS. Measured with each model's
own tokenizer over 2,000 sampled openings (`add_special_tokens=False` -- an
automatic BOS made every word look multi-token in my first pass and read
Llama at 0.0%):

    pythia-6.9b 91.9% | Llama-3.1-8B 92.8% | Olmo-3 93.2% | Qwen2.5 94.2%

So k=1 is right for ~92-94% of rows. The primary drops one token, the
sensitivity drops two, and both are reported.

## R1 confirmed, and exceeded: the effect reverses

    (positive = the FORCED continuation is HARDER, i.e. DAMAGE;
     negative = forced more predictable, the withdrawn direction)

    arm            role     WITHDRAWN    k=1              k=2
    faller         aligned    -0.0342   +0.0077 p .039   +0.0038 p .154
    faller         base       -0.0260   +0.0348 p 3.6e-08 +0.0336 p 2.8e-09
    matched        aligned    -0.0551   -0.0019 p .636   -0.0038 p .268
    matched        base       -0.0299   +0.0153 p .0034  +0.0172 p .0095
    riser_matched  aligned    -0.0477   +0.0131 p .017   +0.0153 p .081
    riser_matched  base       -0.0394   +0.0124 p .108   +0.0174 p .108

**Every cell moves from clearly negative to zero or positive.** The
"compensation" of the withdrawn finding was the offset, entirely: one extra
word of unscored conditioning was worth roughly 0.04-0.09 nats, which is
larger than anything the withdrawn finding reported as its result.

## What remains, and what it is not yet

After the repair the direction is DAMAGE: an imposed word leaves a
continuation the model finds harder than a word it sampled itself. It is
strongest for the FALLER in the BASE arm (+0.035, 36 of 39 pairs, p 3.6e-08)
and weakest -- null -- for `matched` in the aligned arm.

**This is NOT yet a result about imposition, and the plan said so before the
run.** The remaining difference between arms is not only provenance: a
sampled first word is by construction a draw from the model's own
distribution at that site, while a forced word comes from the arms table. If
forced words sit systematically lower in the model's own probability at their
sites than sampled words do, a harder continuation follows without any
imposition effect. The plan declined to match on opening probability because
the two arms' opening probabilities are DIFFERENT OBJECTS -- `logprobs[1]` is
a token logprob, the arms table's `q` is a word probability -- and the
campaign has paid for that confusion before.

**So the honest statement is: the withdrawn finding's effect was the offset;
under repair the sign is positive; and separating imposition from opening
probability requires a single-token-restricted instrument that does not yet
exist.** No arm ordering and no role comparison is claimed -- those need
their own paired tests ([5805]), which are not run here.

## Fences

- Undisturbed rows restricted to word-like openings, as before, so the
  dropped unit is a word and not punctuation or a fragment.
- k=1 vs k=2 agree on every sign and on all six cells' significance except
  `faller aligned` (p .039 to .154), so the aligned faller estimate is the
  one sensitive to tokenization granularity and should not be leaned on.
- Arm-vs-arm and role comparisons: not tested, not claimed.
- Nothing here revisits arm-vs-arm findings, which the offset never touched.
