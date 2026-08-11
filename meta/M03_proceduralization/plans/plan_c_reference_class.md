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

**AND ALL THREE ARE ASKED ON THREE INSTRUMENTS, NOT ON JS ALONE** (RH,
11 Aug). A JS ladder answers *how far* and cannot answer *toward what*, which
is the question plan B's field and word results actually raised. Findings T
has already shown that the same fields move under alignment generally, so a
plan C that compared only magnitudes would leave its most answerable question
untouched. See §1a.

## 1a. WHAT FINDINGS T ALREADY ESTABLISHES, AND WHY IT REFRAMES THIS PLAN

`meta/M01_displacement/findings/T_category_flow.md`, findings 10-14, runs at
the EDGE unit -- one alignment edge, one vote, 43 edges, all 2,190 active
English prompts, no threshold and no manufactured pairs. **Two of its results
are direct priors for plan C and they point opposite ways.**

**T §13: the substitution is NOT transgression-specific.** *"Alignment removes
the violent word only where there is one. It adds the deliberative word
everywhere, and if anything slightly more where there was nothing to remove."*
The withdrawal is larger in the marked twin on every violence category
(wordnet `contact` -0.0864 marked against -0.0490 neutral; framenet
`Cause_harm` -0.0311 against -0.0113); the addition is if anything larger in
the NEUTRAL twin (`perception_cognition` +0.0462 marked against +0.0625
neutral; rid `sensation` +0.0526 against +0.0680).

**T §12: the rising vocabulary on ALL prompts is plan B's vocabulary.** USAS
survivors across the whole edge population include `X2.4 Investigate, examine,
test, search` at **43 of 43 edges**, `S1.1.2 Reciprocity` 42/43, `A1.7
Constraint` 40/43, `A1.3 Caution` 40/43, `Q2.2 Speech acts` +0.0157. T's own
sentence: *"a tagset built for corpus linguistics thirty years ago returns the
vocabulary of alignment among its 41 rising fields."*

    SO THE FIRST PRIOR IS: plan B's `ensure / prioritize / document /
    carefully / gather / conduct` is GENERAL ALIGNMENT VOCABULARY, and the
    institutional arm merely gets more of it. Under this prior plan B's
    word-level result survives as a magnitude and dies as a characterisation.

**T §11 pulls the other way, and it already looked at these very strata.** Its
eleven strata include `m03_inst`, `m03_indiv`, `inst_authority` and
`inst_individual`. Five categories reverse between strata -- rising in the
narrative twins and in `violence`, `sexual` and `neutral`, and **falling in all
four institutional strata**. T states the consequence plainly: *"this is also
why WordNet `cognition` is not significant pooled while being a significant
riser in eight strata: the institutional prompts cancel it. Report this
stratified. The pooled number hides the finding rather than summarising it."*

    SO THE SECOND PRIOR IS: the institutional strata are where the general
    pattern BREAKS, and plan B's institutional risers are not more of the
    general vocabulary but a different one.

**These two cannot both be right about the same categories, and resolving them
is plan C's sharpest contribution.** T could not resolve it: its strata are
unpaired marginals, so it can say a category rises in eight strata and falls in
four but cannot hold a scene fixed. Plan C has the paired MARKED-UNMARKED
contrast and the arm contrast on the same lexicons.

**WHAT PLAN C MUST NOT DO WITH T.** T's numbers are at 43 EDGES on the
2026-07 roster; plan C's are at 46 LINEAGES on the current one. T's own §2
already forbids quoting its two halves as one body of evidence, and the same
discipline applies across documents: **T's figures are priors to be tested,
never a baseline to subtract.** No plan C number is computed by differencing
against a T number.

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
    results/c_fields_by_stratum.csv  field share x {general, transgressive,
                                     institutional}, coverage on every row
    results/c_word_delta_by_word.csv per stratum, reusing plan B's producer,
                                     carrying `pattern` and SPLIT ESTIMATORS

## 5. ANALYSIS

**THREE PRIMARIES, ONE PER INSTRUMENT.** They are co-equal; the JS one is
listed first because it is the simplest, not because it is the finding.

### PRIMARY 1 (MAGNITUDE) -- the within-pair displacement ladder, by domain

Per (lineage, pair): `d = JS(MARKED) - JS(UNMARKED)`. Per-lineage median, then
sign test over the 46 lineages, then reported **per domain with the pair count
beside it** -- never pooled across domains, because a pooled displacement
number over 158 violence pairs and 1 profanity pair is a violence number.

### PRIMARY 2 (FIELDS) -- is the rising vocabulary general or institutional?

**This is the T §12/§13 versus T §11 question, and it is the one plan C exists
to settle.** For every field in every lexicon, three quantities on one scale:

    general      share of risers in field F, on the 748 UNMARKED halves
    transgressive  d(share) = MARKED - UNMARKED, within pair, scene fixed
    institutional  d(share) = inst - indiv, from plan B, unchanged

**The decisive comparison is the SECOND against the THIRD.** If plan B's
institutional risers are general alignment vocabulary (prior 1), then the
fields that rise in the institutional arm also rise on the UNMARKED halves and
the arm contrast is a magnitude on a shared direction. If the institutional
strata break the pattern (prior 2), then some fields rise generally and FALL
in the arm contrast, and the sign flip is the finding.

**Named in advance, so the check cannot be assembled after the fact:**
`X2.4 Investigate` (43/43 edges in T), `S1.1.2 Reciprocity` (42/43), `A1.7
Constraint` (40/43), `A1.3 Caution` (40/43), and WordNet `cognition` -- which
T reports as the specific category the institutional prompts cancel. Those
five are the pre-named cells; everything else is exploratory and is labelled
so.

**Coverage is printed per arm and per source with every count.** On plan B's
population RID gave the largest field differences at 40% coverage. Anything
led by RID is reported last and with its coverage in the same line.

### PRIMARY 3 (WORDS) -- the same question with no lexicon

`aligned - base` per word, per plan B's `b_word_delta`, run on the MARKED and
UNMARKED halves and on the institutional strata. **The specific test: do
`ensure`, `prioritize`, `document`, `involve`, `engage`, `handle`, `gather`,
`conduct`, `carefully` -- plan B's institutional risers -- rise on the UNMARKED
halves too?** T §13 predicts yes, and predicts the rise is if anything larger
where there was no transgression to remove.

A word list read off one population and applied to another is exactly the C2
defect, so the direction is one-way: plan B's words are the HYPOTHESIS being
tested on plan C's population, never the instrument selecting plan C's words.
Plan C's own top words are derived independently and reported beside them.

**Report the pattern, not the contrast.** Plan B's word table had to be rebuilt
because a large contrast can mean "rises further here" or "falls less here",
and 228 of 702 words additionally had the paired and marginal estimators
disagreeing in sign. Plan C inherits `b_word_delta.pattern` and its
`SPLIT ESTIMATORS` flag; no directional block admits a flagged row.

### SECONDARY 1 -- the general baseline as a LEVEL

Median JS on the 748 UNMARKED halves. This is "what alignment does to an
ordinary sentence" and it is the denominator plan B lacks.

### SECONDARY 1b -- the withdrawal/substitution asymmetry, re-tested

T §13's headline claim at a second population and a different unit: is the
FALLER side larger in the marked twin while the RISER side is not? T carries
this as *"the withdrawal asymmetry is supported at two seats, significant at
one summary of two. The substitution asymmetry is unresolved, bounded under
the size we claim but never tested against it at adequate power."* Plan C has
748 pairs at 46 lineages against T's 43 edges, so it may have the power T
lacked. **Both summaries are reported (top-of-site and summed), because T
records that the direction depended on the summary choice.**

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

**On magnitude (Primary 1):**

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

**On fields and words (Primaries 2 and 3), which is where T makes this
falsifiable rather than merely descriptive:**

- **The general vocabulary rises everywhere and the arm is a magnitude on it.**
  T §12 and §13's branch. Then the honest sentence about plan B is *"alignment
  promotes a deliberative-procedural vocabulary on every prompt, and gives the
  institutional side more of it"* -- and plan B's characterisation
  ("bureaucratic register against advice register") is withdrawn as a
  characterisation while its magnitudes stand. **I expect this branch for the
  word-level result specifically**, because T measured `X2.4 Investigate` at
  43 of 43 edges on all prompts, which does not leave room for it to be
  institution-specific.
- **The institutional strata reverse.** T §11's branch, which T saw on WordNet
  `cognition` in all four institutional strata. Then the finding is sharper
  than plan B stated: alignment does something to institutional prompts that
  it does the opposite of elsewhere, and the pooled campaign-wide numbers have
  been hiding it. **This is the branch that would make M03 a finding rather
  than a magnitude**, which is exactly why it gets the pre-named cells in
  Primary 2 and no post-hoc category selection.
- **Both, on different categories.** Entirely possible and probably the real
  answer: the deliberative fields general, the person- and contact-fields
  reversing. **If so, the deliverable is the PARTITION -- which categories are
  general and which are institutional -- and not a single direction.** Named
  here so that a mixed result reads as the finding rather than as a failure to
  get a clean one.

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
