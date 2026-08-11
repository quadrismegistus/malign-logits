# Plan C: the reference class -- what does alignment do generally, and where does the institutional effect sit in it?

**STATUS: A PLAN.** Written 2026-08-11 by the lacan seat on RH's proposal:
*"let's do an all-other-prompt comparison to answer: do these institutional
prompts displace more than your average prompt? more than sexual, violent,
general transgressive etc"*, and *"we should do a simple word by word
probability subtraction... not rely entirely on semantic fields"* (the latter
already landed in plan B and is reused here).

RH's framing is adopted exactly: **this is a SEPARATE REPORTING, not a
confound control.** Plan B's result is not in doubt pending this; plan C asks
what scale it sits on.

## 1. QUESTION

Plan B established, on two independent populations, that alignment moves the
institutional side of an arm contrast further than the individual side:

    M03 kernel, 252 texts, 5,796 paired cells   +0.00823  41/46  p=4.4e-08
    F21's own 24 mirrored scenarios             +0.02901  44/46  p=3.1e-11

**Neither number has a denominator.** `JS = 0.073` on an institutional prompt
is uninterpretable until we know what alignment does to a prompt about nothing
in particular, and a `+0.029` arm gap is uninterpretable until we know how it
compares to the gap between a violent scene and its neutral twin. Plan C
supplies both.

Three questions, in order of what they license:

1. **The general baseline.** How far does alignment move a next-word
   distribution on an ordinary prompt? This is the UNMARKED half of every
   matched pair -- 748 scenes with the transgression removed.
2. **The transgressive increment.** MARKED minus UNMARKED within a pair, the
   scene held fixed. This is F01's displacement measure and it is the natural
   yardstick.
3. **Placement.** Where does the institutional material fall against 1 and 2,
   and does it fall differently by domain -- more than sexual, more than
   violent, more than taboo?

## 2. INPUT -- ENUMERATED FROM THE JSON, WHICH IS AUTHORITATIVE (RH, 11 Aug)

    matched pairs, ACTIVE en, exactly 1 MARKED + 1 UNMARKED,
    both members scored on all 92 models

    violence         158 pairs      betrayal         102 pairs
    taboo            120 pairs      animal            50 pairs
    power            118 pairs      sexual            38 pairs
    property         104 pairs      other             27 pairs
    contradiction     21 pairs      institutional      7 pairs
    death 1, substance 1, profanity 1
    ------------------------------------------------------------
    748 pairs = 1,496 texts

    plus, from plan B and unchanged:
    F21 institutional                 38 texts   (24 arm-labelled + 14 SETE)
    M03_SPEAKER_KERNEL               252 texts
    ------------------------------------------------------------
    1,786 texts x 92 models = 164,312 cells, ALL ALREADY SCORED

**Dropped and counted, never silently:** 15 `pair_id`s are not clean pairs
(member counts 3, 5, 6, 10 -- these are groups, not pairs) and are excluded;
10 otherwise-clean pairs lose a member to sub-92 coverage (violence 2, other 5,
power 2, sexual 1). 7 texts appear in more than one pair and are counted once
per pair, which is stated because a text-level and a pair-level count of this
population differ by 7 and both are correct.

### 2a. THE POPULATION CANNOT BE A DATABASE PREDICATE, AND THIS IS THE FINDING THAT FORCED IT

Plan B selected its population with a subquery against `prompt_catalogue` so
that no prompt text ever entered a query literal. **That method does not work
here, because the two catalogues disagree about the fields it would select on:**

    distinct ACTIVE en texts     JSON 2,299        ClickHouse 2,201
    domain `other`               JSON   185        DB     0
    domain `literary`            JSON     0        DB    97
    source `OTHER`               JSON    64 texts  DB     4 texts
    source `QUINTUPLETS`         JSON    42 texts  DB     0 rows

RH has ruled the JSON authoritative. So the population is enumerated from the
JSON and **hashed over its texts**, per [5146]-[5150]; "a hash of a wrong
population is a wrong population with a receipt" cuts the other way here --
the hash is over the labels we were told to trust.

**And no text goes into SQL anyway.** The fetch selects on MODEL only, pulls
the prompts back as DATA, unescapes them, and does the membership test in
Python. This keeps plan B's doctrine intact in the direction that matters and
adds the direction plan B was missing (see 2b).

The JSON/DB divergence itself is NOT plan C's to fix. It is recorded here and
belongs to whoever owns ingest.

### 2b. THE TSV ESCAPING DEFECT IS A PRECONDITION, NOT A FOOTNOTE

ClickHouse's `FORMAT TSV` escapes apostrophes on output: `can't` returns as
`can\'t`, verified with `od -c` against a store holding zero backslashes.
Plan C joins EVERY row to JSON labels, so the defect that cost plan B three of
F21's 38 texts would cost plan C far more -- `it's`, `can't`, `won't`,
`didn't` are common in this corpus in a way they are not in the speaker kernel
(which joined 11,592/11,592 only because no kernel prompt has an apostrophe).

`b_twp_institutional.tsv_unescape` is committed (`c91bbb9b`) and plan C's
producer imports it rather than reimplementing it. **The producer must REFUSE
on any unjoined row**, as `b_analysis.joined` does; a plan whose population is
defined by a local join has no business proceeding past a join failure.

## 3. INSTRUMENT

Identical to plan B, so the two are on one scale -- this is the whole point and
any divergence would make the placement uninterpretable:

    js          Jensen-Shannon, base vs aligned twp, theta=0.001,
                RESIDUAL KEPT AS A BIN (C1's settled default)
    movement    malign_logits.movement.CANONICAL -- risers tested against the
                renormalisation null, fallers a bare ratio rule. THE ASYMMETRY
                TRAVELS: fallers may never be called "beyond renormalisation".
    fields      every lexicon fields.available() reports, coverage printed per
                cell and per arm
    word delta  aligned - base per word, no lexicon, per plan B's b_word_delta

    diagnostics `exact_null` (False by construction here -- twp is truncated)
                and `residual_share` on every row.

**THE UNIT IS THE LINEAGE, n = 46.** Not 92 models, not 1,786 prompts, not
164,312 cells. Every reported number is a per-lineage median first and a sign
test over lineages second.

## 4. OUTPUT

    results/c_pair_prompt.jsonl      (lineage, prompt, js, movement, diagnostics)
    results/c_pair_contrast.csv      (lineage, pair_id, domain, d_js = M - U)
    results/c_placement.csv          the ladder: every stratum on one axis
    results/c_word_delta_by_word.csv per domain, reusing plan B's producer

## 5. ANALYSIS

### PRIMARY -- the within-pair displacement ladder, by domain

Per (lineage, pair): `d = JS(MARKED) - JS(UNMARKED)`. Per-lineage median, then
sign test over the 46 lineages, then reported **per domain with the pair count
beside it** -- never pooled across domains, because a pooled displacement
number over 158 violence pairs and 1 profanity pair is a violence number.

### SECONDARY 1 -- the general baseline as a LEVEL

Median JS on the 748 UNMARKED halves. This is "what alignment does to an
ordinary sentence" and it is the denominator plan B lacks.

### SECONDARY 2 -- PLACEMENT, and the asymmetry that must be printed beside it

**THE INSTITUTIONAL MATERIAL HAS NO MARKED/UNMARKED PARTNER**, except for the
7 SETE pairs. So it enters the ladder as a LEVEL (median JS) and as an ARM
contrast (inst - indiv), and **neither of those is the same quantity as
MARKED - UNMARKED**:

    MARKED - UNMARKED    the same scene with a transgression added
    inst - indiv         two different scenes from two social positions

Putting all three on one chart is the deliverable and is also the trap. The
axis is "how far did alignment move the distribution", which they share; the
contrast is not, which they do not. **Any figure carries all three series with
their construction named in the legend, or it carries one.** A ladder that
reads as one measurement when it is three is exactly the "same units is not
same comparison" failure ([5026]).

### SECONDARY 3 -- fields and word deltas by domain

Plan B found the institutional risers flat-valence, dominance-marked and
abstract, and found `ensure / prioritize / document / involve / engage` rising
in both arms and further in the institutional one. Secondary 3 asks whether
that vocabulary is institutional or is simply **what alignment promotes
everywhere** -- which is the word-level form of RH's whole question, and the
one the fields cannot answer alone.

**RID IS REPORTED LAST AND WITH ITS COVERAGE.** On plan B's population RID
produced the largest field differences and covered 40% of the vocabulary. A
share over 40% coverage is a composition over a small non-random subset.

### PRIORS, BOTH BRANCHES

- **The institutional level sits ABOVE the transgressive increment.** Then
  alignment treats an institutional grievance as a bigger intervention than a
  violent scene, which would be the strongest form of the M03 claim and would
  need the level/contrast asymmetry in Secondary 2 stated in the same sentence
  every time it is quoted.
- **It sits INSIDE the ordinary range.** Then plan B's arm effect is real but
  small against what alignment does routinely, and the finding becomes
  "institutional prompts are ordinary prompts on which the arm matters",
  which is a weaker and more defensible claim. **This is the branch I expect**,
  on the prior that the F21 arm gap (+0.029) is of the same order as plan B's
  F21 stratum median JS (0.080) rather than of the same order as displacement.
- **The domain ordering is flat.** Then "transgressive" is not a scale that
  alignment tracks, which would be a result about the displacement literature
  and not about M03 -- and it would put F01's own increment in question, so it
  gets checked against F01's published numbers before being believed.

## 6. WHAT THIS CANNOT DO

- **It cannot measure agency**, for the reasons in plan B §5. The addendum's
  *"Agency RISES in every family... do not narrate submission"* binds any
  narration of plan C's output as it binds plan B's.
- **It cannot make the institutional material paired.** No amount of analysis
  turns 24 mirrored scenarios into minimal pairs; the SETE 7 are the only
  paired institutional data and 2 of those 7 swap the institution for a person
  rather than adding a transgression.
- **It cannot settle the JSON/DB divergence**, only route around it.

## 7. COST

Zero new compute. 164,312 cells are already scored at theta=0.001. The fetch
is ~15M rows and must run PAIR AT A TIME (46 iterations, 2 models each) rather
than as one query, or it will not fit in memory -- plan B's single fetch was
26,680 cells and this is 6x. Estimated wall clock 20-40 minutes.

## 8. FOLLOWUPS, RECORDED SO THEY ARE NOT REDISCOVERED

- **Ladders** (RH, 11 Aug): models with 2+ rungs, base -> sft -> dpo. Breaks
  plan B and C's unit -- the unit here is one base->aligned pair per lineage --
  so it is a separate plan, and it overlaps M05_emergence's territory.
- **Ablations** (RH, 11 Aug): `tulu-no-safety`, the archangels. Same base,
  swapped post-training: an arm comparison, not a base->aligned one, and again
  a different unit.
- **zh.** 13 F21 zh rows exist and English-only is the stated convention.
  Unasked rather than excluded.
