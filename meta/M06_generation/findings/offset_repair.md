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

**This is NOT yet a result about imposition.** The remaining difference
between arms is not only provenance: a sampled first word is by construction
a draw from the model's own distribution at that site, while a forced word
comes from the arms table. If forced words sit systematically lower in the
model's own probability at their sites, a harder continuation follows with no
imposition effect.

**CORRECTED ([5818], [5819]): the plan's stated GROUNDS for that fence were
wrong, and the fence survives on different ones.** I wrote that opening
probability could not be matched because `logprobs[1]` is a token logprob and
the arms table's `q` is a word probability -- DIFFERENT OBJECTS. They are the
same object. `twp.expand` softmaxes over the full vocabulary and books
unselected mass as tail/drop, so nothing is renormalised: on a real record,
word masses 0.347764 + tail 0.259579 + drop 0.392657 = 1.000000. For a
single-token word -- 91.9-94.2% of openings by this finding's own audit --
`log q` and a token logprob are the same quantity. **The match is runnable.**

What survives as the actual caution: it has not been run; `q` is under the
ALIGNED model, so the undisturbed side must come from aligned-scorer rows;
and the single-token restriction is itself a selection whose ~8% remainder is
longer and rarer, so a null there bounds common openings and says nothing
about the tail. **A fence resting on a false premise fails the moment someone
checks the premise** (malign's line), which is why the grounds are replaced
rather than the fence removed.

A live trap recorded with it: **this repo has two functions computing "word
probability" and only one returns a probability.**
`core.py:score_words_from_logits` renormalises over the candidate set;
`twp.expand` does not. The arms table comes from the twp path. Any future `q`
must be attributed to its PRODUCER before being compared to anything.

## And the comparison this corpus was actually built for

RH, on reading the above: *"why don't we trust our faller-matched and
riser-matched baselines? We spent $60 getting a corpus of them for this exact
purpose."*

Correct, and it reframes this whole line of work. **The undisturbed arm was
never the designed control.** `matched` is: a non-mover held at the faller's
own aligned probability, built so that faller-vs-matched isolates DEMOTION
with probability held fixed. That comparison is arm-vs-arm, so the offset
defect never touched it, and it was already answered on two instruments:

    faller - matched, SELF-surprisal (A|A, self_surprisal.md S3)
        aligned -0.0053 p 0.154 | base -0.0199 p 0.00029
    faller - matched, THIRD-PARTY surprisal (GPT-2, f15_on_passages.md F3b)
        aligned -0.0213 p 0.066 | base -0.0337 p 0.0014

**Both instruments, both roles, the same sign: negative.** Against the
control the corpus was built to provide, forcing a demoted word makes the
continuation MORE predictable, not less. On RH's original pair of theses that
is COMPENSATION and not damage -- and it never needed the undisturbed arm.

The forced-vs-undisturbed comparison asks a different and structurally
messier question (forcing at all, against free generation), and it is the one
that broke twice. The designed control was intact throughout.

## Fences

- Undisturbed rows restricted to word-like openings, as before, so the
  dropped unit is a word and not punctuation or a fragment.
- k=1 vs k=2 agree on every sign and on all six cells' significance except
  `faller aligned` (p .039 to .154), so the aligned faller estimate is the
  one sensitive to tokenization granularity and should not be leaned on.
- Arm-vs-arm and role comparisons: not tested, not claimed.
- Nothing here revisits arm-vs-arm findings, which the offset never touched.
