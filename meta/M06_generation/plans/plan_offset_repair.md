---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades
date: 2026-08-13
role: plan
topics: [self-surprisal, forced-arms, repair, chain]
description: "Plan: THE OFFSET REPAIR — RH's fix for the construction defect that withdrew opening_matched. The forced word is unscored conditioning context, so drop the first scored word from the UNDISTURBED arm and treat it as if it had been forced. Both arms then read prompt + one unscored word + scored continuation, and the only remaining difference is whether that word was sampled or imposed. Restores the damage-vs-compensation question on a valid construction."
---
# Plan: the offset repair

RH, 2026-08-13, in session, on reading the withdrawal:

> *"if surprisal starts scoring at `prompt + forcedword` and undisturbed is
> `prompt`, then slice off the first scored word from undisturbed, pretending
> that the first word of each undisturbed sequence was forced"*

That is the repair. The defect withdrawn at [5811] was an OFFSET: forced rows
carry one extra word of unscored conditioning that undisturbed rows do not.
Dropping the undisturbed arm's first word restores the symmetry rather than
controlling for it.

    BEFORE (invalid)   forced      prompt + W | score w2 w3 w4 ...
                       undisturbed prompt     | score w1 w2 w3 ...

    AFTER (this plan)  forced      prompt + W | score w2 w3 w4 ...
                       undisturbed prompt + w1| score w2 w3 w4 ...

Both arms are now `prompt + one unscored word + scored continuation`. **The
only remaining structural difference is whether that word was SAMPLED from
the model's own distribution or IMPOSED by the harness** -- which is the
question RH's original design was built to ask, and which the offset made
unaskable.

## Construction

`gen_scores`, `corpus='passage'`, `model = scorer`, `scorable=1`,
`n_nan=0`, SmolLM2 excluded, deepseek fenced.

    y_forced      = mean over ALL logprobs (the forced word is not in them)
    y_undisturbed = mean over logprobs[1+k ...]  where k = the token count of
                    the first word

**k IS NOT ASSUMED TO BE 1.** The primary drops k=1; the sensitivity drops
k=2; and the producer MEASURES the multi-token share of first words on a
sample of models with their own tokenizers and reports it before the
contrasts. If most first words are one token the two variants agree and the
question is closed; if they diverge, the tokenizer-exact version is owed and
this plan says so rather than picking the convenient one.

Unit: cell = (pair, prompt, role, arm) mean; paired per (pair, prompt);
pair-grain sign test over <= 41 pairs as the conservative unit, cell grain
beside it.

## Directions, declared before any number

  R1 (directional, and it is the whole point of the repair): the WITHDRAWN
     finding reported forced passages as more predictable than undisturbed
     ones by roughly -0.03 to -0.05 nats. **If that gap was entirely the
     offset, it collapses toward zero under this repair.** R1 says it
     collapses: |effect| falls by more than half.
  Q1 (open, no direction, the surviving question): whatever REMAINS after
     the repair. Positive = DAMAGE (an imposed word leaves a harder
     continuation than a sampled one), negative = COMPENSATION, zero = a
     sampled word and an imposed word are interchangeable as context, which
     is what malign's blindness argument at [5810] predicts and what the
     model's own architecture requires if nothing else differs.
  Q2 (open): whether Q1's residual differs by arm or by role. Arm-vs-arm
     contrasts are the tests, not an ordering of three point estimates
     ([5805]).

## Fences, and one that is new

- **The opening-probability match is NOT restored here.** For undisturbed
  rows the opening's logprob is `logprobs[1]`, a TOKEN logprob; for forced
  rows the analogous quantity is the arms table's `q`, a WORD probability.
  Those are different objects when a word is several tokens, and the
  campaign has paid for that confusion before. This plan therefore compares
  UNMATCHED on opening probability and says so; a matched version needs the
  single-token restriction and is a separate instrument.
- Everything the withdrawal established stands: arm-vs-arm comparisons were
  never affected, and this plan does not revisit them.
- The undisturbed first word is restricted to word-like openings as before,
  so the dropped unit is a word rather than punctuation or a fragment.

Producer: `scripts/m06_offset_repair.py`. Results:
`results/offset_repair.json`.
