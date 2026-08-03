# AMENDMENT A to `m01_within_pair_registration.md` @ `8ff56206deac048e`

**STATUS: DRAFT FOR SIGNATURE. No verdict has been computed under any candidate
pair of declarations. The design arguments below were written before any
crossed verdict existed, per [3072] as extended by [3080].**

Ordered at [3068].b after [3067] found that the computation's unit was the
`(base, aligned)` entry while the registration names the LINEAGE. This amendment
declares the two decisions [3079] separated, and nothing else.

---

## §1 WHAT IS BEING CORRECTED

    registered unit     THE LINEAGE
    computed unit       one entry per ALIGNED MODEL
    66 entries rest on  35 distinct base checkpoints / 30 connected components
    DECLARED UNIT       the BASE CHECKPOINT (§2)
    DECLARED EDGE       the dpo arm, non-reasoning, median if >1 (§3)
    DECLARED n          34   (Falcon-H1-7B HELD OUT by name, §3.5)

The primary as read reported **p 0.0232 at n=65**. It is not withdrawn and it is
not citable: it counted 65 units where the declarations give 34.

---

## §2 DECISION ONE — THE UNIT

### §2.1 DECLARED: the BASE CHECKPOINT (`model_to_base`), n = 35.

### §2.2 The argument — the component OVERRIDES A RECORDED DISTINCTION

**An earlier draft of this section declared the connected component (n = 30) and
was wrong. RH refuted it and the refutation is a fact about the map, not a
preference.**

The map records each base explicitly:

    Olmo-3-1025-7B     stage base   base = ITSELF   5 arms, all Olmo-3-7B-*
    Olmo-3-1125-32B    stage base   base = ITSELF   3 arms, all Olmo-3.1-32B-*
    ...and BOTH carry lineage id  allenai/Olmo-3-1125-32B

**Every arm correctly declares its own base. The base relation is sound. The
CONNECTED-COMPONENT step merges the two anyway** — and it merges them across a
size difference AND a version difference: the 32B arms are **Olmo-3.1** hanging
off an **Olmo-3** base, and there is **no Olmo-3.1-32B base checkpoint in the map
at all.**

**THE CAVEAT DOES NOT LICENSE THIS.** *"A more complete registry gives FEWER
lineages and a LARGER p"* governs models that ARE related and LACK a recorded
edge — it licenses conservatism about MISSING information. **It does not license
overriding a distinction the registry RECORDS.** Here the map states two
different bases and the component contradicts it. That is not conservatism; it
is an artifact of connected components over an unevenly populated relation.

The same shape holds elsewhere: `Falcon3-10B-Base` absorbs four declared bases
(1B/3B/7B/10B) and `Qwen/Qwen2.5-7B` absorbs two (0.5B, 7B). **Different sizes of
one release are separate pretraining runs that share data and recipe — related,
and not one run.**

### §2.3 THE DIRECTION OF THIS CHANGE DEMANDS SCRUTINY AND WE STATE IT

**This declaration moves n UP, from the component's 30 to 35 before §3's
exclusions, and a larger n makes the test EASIER to pass.** That is the direction a reader should distrust, so it is named
here rather than left to be noticed.

**What defends it is that the argument is about what the map RECORDS and not
about what the verdict does.** The unit was declared before any verdict at n=35
was computed, and the refutation that produced it came from the corpus's author
pointing at two model names, not from a p-value.

**AND THE PATTERN THIS MAKES VISIBLE IS NOT THE REASON FOR IT.** At the base
unit the Olmo 7B and 32B become two units pointing opposite ways ([3079].3),
where the component averaged them. **That is a CONSEQUENCE of the declaration,
never its justification** — the size gradient remains a hypothesis for its own
registration with its own null, and it did not choose this unit.

**Reported as declared sensitivity, never as the claim: the same test at the
CONNECTED-COMPONENT unit, n = 29** — 30 components less phi-4, which the §3
edge rule removes under EITHER unit. Both post together, and the component
figure is the conservative bound the map's caveat points at.

**CORRECTED from "n = 30" after [3099]: the component sensitivity must also
have the edge rule applied, and 30 was the count BEFORE phi-4 left.** Surfaced
by malign's independent reproduction, which rebuilt both unit columns from
custody and made the base-vs-component difference impossible to gloss.

---

## §3 DECISION TWO — WITHIN-UNIT REDUCTION

### §3.1 DECLARED, one rule, RH's word:

> **THE EDGE IS THE BASE'S `dpo` ARM(S), EXCLUDING REASONING-TRAINED ARMS.
> WHERE MORE THAN ONE ARM QUALIFIES, THE UNIT'S Δ IS THEIR MEDIAN.**

### §3.2 Why an EDGE and not a median over all arms

**The median over ALL of a base's arms mixes DOSES.** A base's arms sit at
different stages — `sft` is one step of alignment, `dpo` and `rlvr` are more —
so a base holding only a dpo arm is measured at one dose while a base holding
sft+dpo+rlvr is measured at an average of three. **Which dose a unit receives
would then depend on which arms that lab happened to release.** The units would
not be measuring the same treatment.

**`dpo` IS UNIVERSAL ON THIS ROSTER, SO THE EDGE COSTS NO UNITS:**

    dpo    35 / 35 bases   100%
    sft    16 / 35          46%
    rlvr    5 / 35          14%
    slic / kto / ppo   1 each

**Every base has a dpo arm. Equal dose at no loss of n** — which is why the edge
is available here and would not be on a roster with patchier coverage.

**COST, STATED: 31 of the 66 measurements go unused in the primary.** Every sft
and rlvr arm sits out. **They are the natural corpus for a DOSE registration**,
which is a better home for them than being averaged into a number meaning "some
amount of alignment".

### §3.3 Why REASONING-TRAINED ARMS ARE EXCLUDED — measured, not asserted

The proposed mechanism was that reasoning models emit `<think>` and would
dominate a next-word measurement. **That is NOT what happens and it was checked:
across all 1,368 prompts, the top word is a special/think-ish token 0.00% of the
time**, for `phi-4-reasoning` and for both Olmo `Think` arms. These are scored
in `mode=raw` — the bare prompt, no chat template — and the thinking behaviour
is a template effect.

**The real signature is CONCENTRATION ON DISCOURSE CONNECTIVES:**

    microsoft/phi-4              'the' 17.3%   'then' 10.9%   184 distinct tops
    microsoft/phi-4-reasoning    'then' 27.9%  'the' 17.1%    124 distinct tops

    Olmo-3-7B-Instruct-DPO       'then'  5.4%
    Olmo-3-7B-Think-DPO          'then' 15.6%

**Reasoning training collapses the raw next-word distribution onto step-marking
connectives.** The site rule fires when THE TOP WORD CHANGES, so a model that
systematically prefers `then` regardless of content carries a fire rate shifted
for reasons unconnected to transgression. **That is a confound for this
instrument specifically, and it is measured rather than supposed.**

### §3.4 The three consequences, and one of them costs a unit

**OLMO** `Olmo-3-1025-7B` has two dpo arms; `Think-DPO` is excluded; the unit is
`Instruct-DPO` alone.

**LLAMA** `meta-llama/Llama-3.1-8B` has two qualifying dpo arms —
`allenai/Llama-3.1-Tulu-3-8B-DPO` and `meta-llama/Llama-3.1-8B-Instruct` — and
the unit is their MEDIAN. **A first-party rule was considered and rejected: both
are multi-stage pipelines labelled `dpo`** (Tulu is SFT→DPO; Meta's
post-training is undocumented and multi-stage), the map records both
star-shaped off the base, **and neither is the cleaner instance of the stage.**
Median within one stage does not mix doses, so §3.2's argument is untouched.

**PHI-4 IS EXCLUDED, AND FOR A DIFFERENT REASON THAN FALCON-H1. Its only dpo
arm is `phi-4-reasoning`; nothing qualifies.**
Consistency forces it: if reasoning-training disqualifies an arm where a
replacement exists, it disqualifies one where none does. **"Excluded when
convenient and kept when not" is a rule plus an exception shaped like
convenience.** Named in the ledger with the reason. **n 35 → 34.**

**phi-4 is excluded ON THE RULE; Falcon-H1-7B is HELD OUT ON TIMING.** A later
reader must not read them as one category: one had no qualifying arm, the other
has one and its value was not yet knowable.

### §3.5 DECLARED n = 34, AND ONE UNIT HELD OUT BY NAME

    bases with admitted data                35
    less phi-4 (no qualifying dpo arm)      34
    DECLARED n                              34

**`tiiuae/Falcon-H1-7B-Base` IS THE HELD-OUT UNIT.** Its cells were empty from
an fp16 overflow in the runner (both configs declare bfloat16; the vendor's own
doc says *"always use `torch.bfloat16` instead of `torch.float16`"*). The bf16
repair works — 73 to 158 word rows per cell, zero empty — and it lands about a
day after this freeze.

**IT IS HELD OUT, NOT DROPPED, AND THE DISTINCTION IS THE POINT.** By
[3085].2's sequencing it does not retro-enter after the read. **Its value was
UNKNOWN AT FREEZE and will be unknown until the primary is fixed** — which
makes it the only genuinely blind, out-of-sample unit this design will produce.

### §3.5.1 Why 34 and not 35 — the bar does not move

    n=34   critical k 23   achieved size 0.0288   MDE 0.725
    n=35   critical k 23   achieved size 0.0448   MDE 0.707

**The critical count is 23 at BOTH.**

**AND THIS ARGUMENT IS UNIT-DEPENDENT, WHICH THE DOCUMENT MUST SAY.** At the
component unit the bar is 20 (n=29) or 20 (n=30) — a different pair of numbers,
and whether the coin-flip argument survives there would have to be RECOMPUTED,
never inherited. It is stated for the DECLARED unit only. **If the unit is ever
re-opened, §3.5.1 does not travel with it.** So the extra unit helps only if positive
and hurts if negative, against an unchanged threshold — **a coin flip on the
verdict, bought for 22-25 hours and $23-25.**

**And the argument that decides it independently of cost: a result that turns on
one coin flip is not a result.** If the verdict would flip on this unit, the
conclusion is that the evidence is too thin to settle the question, not that the
flip should be purchased. **Waiting to see which way a single declared unit
falls, on a test whose verdict alternates, is the shape of shopping even when
the unit was declared in advance.**

**Reported when it lands, as a replication and never folded into the primary:**
`Falcon-H1-7B-Base`'s Δ, its sign, and whether it agrees with the verdict.

---

## §4 WHAT THE AMENDED READ REPORTS

**Primary and depth BOTH**, at the declared unit and reduction, plus:

1. **THE ALTERNATION TABLE, as the reporting form** ([3071].2 / [3080]).
   The verdict ALTERNATES rather than degrades as n falls, because the critical
   count moves in integer steps while the positives move with it. **A single p
   conceals that; the table is the honest object**, computed around the
   DECLARED n and spanning down to the component's 30.
2. **n as a CEILING, with the map's caveat quoted.** Every future improvement to
   the registry gives fewer units and a larger p, never smaller. **One-directional,
   written in advance, and pointed at a result with no margin.**
3. **The achieved size at the realized n**, never the nominal 0.05.
4. **The CONNECTED-COMPONENT sensitivity (n = 30)** beside the declared
   result — the conservative bound the map's caveat points at.
5. **The diagnostic column unchanged** — tail_share and tail_excess sign.

---

## §5 THE UNIT ASSERTION ([3068].d, extended by [3073].4)

The producer gains, and no read posts without it:

    assert n_distinct(unit_ids) == n_units
    and the output NAMES THE FIELD IT COUNTED:
        "units=34 field=model_to_base edge=dpo/non-reasoning
         components=30 entries=66 excluded=phi-4(no-qualifying-arm)"

**Three artifacts in one night got the arithmetic right and the word wrong**
([3067], [3069], and the texture script written after the correction). **A
count printed beside the name of the field it counted cannot make that error.**

---

## §6 WHAT THIS AMENDMENT DOES NOT DO

- **It does not change the statistic, the sidedness, the corpus, the site rule,
  the exclusions, or the diagnostic.** Only the unit and the reduction.
- **It does not re-open the read.** One collapsed read, audited, then cited.
- **It does not adjudicate the size gradient**, which needs its own registration
  and its own null.
