---
status: WITHDRAWN
grade: ungraded
date: 2026-08-13
role: finding
topics: [self-surprisal, chain, syntagmatic, forced-arms, withdrawn]
description: "WITHDRAWN AT CONSTRUCTION LEVEL ([5811]), not amended. The comparison never matched openings: the forced word conditions the generation but is absent from BOTH the prompt and the scored text, so forced rows are scored on a continuation carrying one more word of context than undisturbed rows. Every forced-vs-undisturbed number here -- main effect, profile, ANCOVA, word-like-opening control, POS and Zipf matching -- compares sequences with different amounts of conditioning. The context asymmetry is a sufficient explanation for the whole effect and requires nothing of the models."
---
# WITHDRAWN: opening-matched (the comparison was never opening-matched)

**Withdrawn 2026-08-13 at construction level. Nothing below is a result.**
The text is kept because the docket carries it ([5803], [5808], [5809]) and
because the way it failed is worth more than the way it read.

## The defect

The forced word conditions generation but is in NEITHER the prompt NOR the
scored text, and is absent from the logprob array. Three measurements, made
after malign's [5810] §3 named the exact condition that would collapse their
own structural argument:

    text starts with the forced word     0.0008  of 904,544 forced rows
    prompt ends with the forced word     0.0000
    logprobs[1] within-site sd           1.749 mean / 1.677 median over
                                         56,509 sites -- a forced word's
                                         logprob at a fixed site is
                                         DETERMINISTIC, so this is the first
                                         SAMPLED token after it
    len(logprobs) vs tokens(text)        equal exactly (pythia-6.9b: 256/256,
                                         256/255, 256/256)

Example: `forced_word='moved'`, prompt tail *"scooped the goldfish out of its
bowl and"*, text head *" into another room, carrying it with her"*. The word
`moved` appears nowhere.

**So `x` was not a matched opening.** For undisturbed rows it is the model's
own first sampled token; for forced rows it is the first sampled token AFTER
an unmeasured imposed word. Forced rows carry one more word of conditioning
context than undisturbed rows at the same nominal position, and more context
makes a continuation more predictable. That is a sufficient explanation for
the entire "compensation" effect and it requires nothing of the models.

## Why every control survived

The controls -- prompt fixed effects, word-like openings, contextual POS
matching, Zipf-band matching -- were elaborations on top of a broken
comparison. **None of them touched the asymmetry**, because each matched a
property of the word in position 1 while the defect was an entire word
sitting OUTSIDE the measured window. A control can only remove what it
compares.

## What this does not touch

Checked, not assumed: everything ARM-vs-ARM, because all forced arms share
the same structure. `self_surprisal.md`'s S3 and S4 stand. F3a compares an
aligned-base gap within forced-matched against the same gap within
undisturbed, so the asymmetry is common to both roles and cancels. F3b, F3c,
I5, the ascent branch and I7 are arm-vs-arm; F15 and I6 are
undisturbed-only.

## The lesson, which is not the one I was collecting

I ran four controls on this comparison and each survival raised my
confidence. **All four were downstream of a defect in what the two arms ARE**,
and no amount of covariate matching reaches that. The question I never asked
was the cheapest one available: *what exactly is in the sequence being
scored?* malign asked it from outside, about a producer neither of us wrote,
and it took three queries to answer.

---

*Original text follows, superseded in full.*

# Opening-matched: the chain compensates for being handed a word

Plan: `plans/plan_opening_matched.md`, committed before this producer, with
BOTH signs declared. Producer `scripts/m06_opening_matched.py`; results
`results/opening_matched.json` + per-pair parquet. No new compute. Single
pass; [5503] applies.

RH's design: instead of fencing off the undisturbed comparison as
confounded, **match on the opening's surprisal on the fly**, which is what
forcing a single matched word cannot do.

## The two theses, and which one the data picks

    RESIDUAL = forced passage's mean surprisal after the opening
               MINUS what an opening-matched undisturbed passage shows

    T1 DAMAGE        forcing breaks the chain          residual POSITIVE
    T2 COMPENSATION  the syntagmatic absorbs it        residual NEGATIVE

**T2. Every arm, both roles, three estimators, no exceptions.**

    PRIMARY (binned at 0.5 nat on the opening logprob; pair grain)
      faller         aligned -0.0342  9/31  p 6.8e-04 | base -0.0260  7/32  p 7.0e-05
      matched        aligned -0.0551  5/35  p 1.4e-06 | base -0.0299  8/30  p 4.7e-04
      riser_matched  aligned -0.0477  7/32  p 7.0e-05 | base -0.0394  7/31  p 1.2e-04

    SENSITIVITY (linear fit on undisturbed rows, arm mean residual)
      all six cells negative, p from 1.9e-10 to 5.5e-17

    CONTEXT CONTROL (ANCOVA, prompt fixed effects, within-prompt slope)
      faller         aligned -0.0298  5/34  p 2.4e-06 | base -0.0108 13/25 p 0.073
      matched        aligned -0.0378  2/36  p 5.4e-09 | base -0.0334  7/30  p 1.9e-04
      riser_matched  aligned -0.0321  6/32  p 2.4e-05 | base -0.0263  6/31  p 4.1e-05

Common support is not a corner: 234 of 240 (pair, role, arm) cells qualify,
median 20-22 qualifying bins per pair, over an x range of -15.5 to 0 nats.

**Read plainly: hand the model a word and what follows is EASIER for it than
its own free continuation from an equally improbable start.** Free sampling
that wanders into the tail keeps wandering; an imposed opening is
recuperated.

## Two controls, because two alternatives predicted the same sign

Both were run before this was written up, and the finding would have been
wrong without them.

CONTEXT ENTROPY. At a given opening logprob, undisturbed rows are drawn
preferentially from HIGH-ENTROPY contexts -- that is why their sampled token
was improbable -- and entropy propagates (first-token-to-rest correlation
+0.365). That alone predicts the compensation sign. Removed by holding the
PROMPT fixed (ANCOVA above): the effect survives, smaller in the base arm.

OPENING IDENTITY. At equal logprob an undisturbed opening is a TAIL-SAMPLED
token, often a fragment or punctuation, while a forced opening is a curated
content word. Removed by restricting the undisturbed arm to rows whose first
whitespace word is alphabetic and >= 2 characters (211,152 of 238,400
qualify). The effect survives and every primary estimate GREW slightly.

## Q1: NOT ESTABLISHED -- the ordering claim is WITHDRAWN ([5805])

**This section previously read "imposition, not demotion" and made the arm
ordering the headline. That claim was not tested and is withdrawn.** Caught
by malign at [5805]; their arm-vs-arm contrasts reproduce exactly under my
own recomputation from the parquet.

Every test in this finding asks *does this arm compensate* -- arm against
ZERO. The ordering claim is about arms against EACH OTHER, and no such test
existed. Run now, paired over pairs:

    aligned  faller - matched         +0.0117  14-/26+  p 0.081
             faller - riser_matched   +0.0065  18-/21+  p 0.749
             matched - riser_matched  +0.0071  17-/22+  p 0.522
    base     faller - matched         +0.0090  13-/25+  p 0.073
             faller - riser_matched   +0.0210  14-/24+  p 0.143
             matched - riser_matched  +0.0130  14-/24+  p 0.143

**No contrast reaches significance and two are at coin-flip. The three-rung
ordering the section was named after is not distinguishable from no ordering
at all.** Reproducing three medians in an order is not testing the order.

**AND IT INVERTS UNDER THE OTHER ESTIMATOR, in exactly the arm the claim
elevated:**

    by MEDIAN  matched -0.0551 > riser_matched -0.0477 > faller -0.0342
    by MEAN    riser_matched -0.0192 > faller -0.0155 > matched +0.0055

`matched` is first by median, last by mean, and changes sign. The
mean/median divergence was fenced below as a readers' note about heavy
tails; it is not a note, it is the ordering. The median is the right
traveller for each arm's OWN compensation (35 of 40 negative), per [5762] --
but an ordering compares MAGNITUDES, and magnitude is what heavy tails
destabilise.

**"NOT DEMOTION" IS ALSO WITHDRAWN, as an accepted null the data cannot
bound.** The faller-matched interval is 95% bootstrap [-0.0030, +0.0301],
which spans zero and does not exclude an arm effect the size of one already
detected elsewhere: M04's ladder finds faller-matched on D at -0.0673,
32/10, p 0.0009. Different quantity, same arms, same corpus. So movement
class is an UNDETECTED arm difference on this instrument, not an absent one.

What may be said: **being handed a word produces compensation, robustly, in
every arm and both roles, and this instrument cannot resolve whether the
word's movement class matters.** One hint, recorded as a hint: the faller
compensates LEAST in both roles at p 0.081 and 0.073, the same direction
twice, uncorrected -- not evidence, and not evidence for the withdrawn
reading either.

## WHERE THE COMPENSATION LIVES, and the mechanism I predicted is REFUTED

RH asked how the model could know about an imposition it cannot detect. It
cannot, so the difference must be in the WORD occupying position 1. The
hypothesis written into `m06_opening_profile.py` before the run: matching on
CONDITIONAL logprob does not match on MARGINAL frequency, forced words are
curated content words while tail-sampled openings are rare tokens, so
**compensation should GROW as the opening gets less probable**.

**It does the exact opposite.** Aligned, per 1-nat bin of opening logprob:

    x bin     faller              matched             riser_matched
    -1.0    -0.0880  9/30 p.001  -0.1059  6/33 p<.001 -0.0988  8/31 p<.001
    -2.0    -0.0733  6/34 p<.001 -0.0901  3/37 p<.001 -0.0772  3/37 p<.001
    -3.0    -0.0533  6/34 p<.001 -0.0624  5/35 p<.001 -0.0626  7/33 p<.001
    -4.0    -0.0203 15/25 p.154  -0.0428 12/28 p.017  -0.0496  7/33 p<.001
    -5.0    -0.0028 20/20 p1.00  -0.0319 17/22 p.522  -0.0431 12/27 p.024
    -6.0    -0.0016 19/21 p.875  -0.0178 18/21 p.749  -0.0036 19/19 p1.00
    -7.0 and below: null or positive, no consistent sign

Base has the same shape (strongest at -2 to -4, gone by -7).
**Compensation vs opening logprob: Spearman -0.683 (p 0.0002) over 25 bins,
and -0.966 restricted to x >= -7 where the effect lives.** The effect is
concentrated where the imposition is MILDEST and vanishes where the imposed
word is most improbable.

**That kills the reading of compensation as repair-after-disruption.** If
forcing damaged the chain and the model repaired it, the effect would grow
with how improbable the imposed word is. It shrinks.

And the frequency measurement inverts too. Mean Zipf of the opening word:

    x bin   undisturbed   forced (matched)   gap
    -1.0       5.515          4.573         -0.941
    -3.0       5.448          4.736         -0.711
    -5.0       5.072          4.764         -0.308
    -7.0       4.635          4.806         +0.171
    -10.0      4.089          4.808         +0.719
    -14.0      3.463          4.842         +1.379

**At high opening probability the UNDISTURBED opening is the commoner word
by nearly a full Zipf point; only deep in the tail do forced words become
commoner.** So my hypothesis had the frequency relation backwards in exactly
the region where the effect exists, and the two profiles are strongly
coupled in the direction opposite to it: **compensation vs the Zipf gap,
Spearman +0.678 overall and +0.974 for x >= -7** -- the compensation is
largest where the forced opening is LESS common than what the model would
have said.

**REVISED CANDIDATE MECHANISM, stated as a candidate:** at a high-probability
slot the model's own near-modal choice is often a very common word (Zipf
5.5, function-word territory) that defers specification, while the arms
table supplies a content verb (Zipf 4.6) that names the predicate and
CONSTRAINS what can follow. The forced continuation is more predictable
because it was given a more specific word, not because anything was repaired.
Compensation would then be a fact about lexical specificity, not about
imposition at all -- which is consistent with its absence in the tail, where
both arms' openings are specific.

## The specificity candidate: HALF CONFIRMED, AND THEN REFUTED

Run on RH's instruction, using `taxonomy.get_pos(words, prompt)` -- CONTEXTUAL
tagging at the prediction site, which matters because an out-of-context tagger
calls `fall break kiss punch` nouns at exactly these sites. Producer
`scripts/m06_opening_pos.py`, results `results/opening_pos.json`.

**P1 CONFIRMED.** The function-word share of undisturbed openings tracks x
exactly as the candidate requires -- and forced openings are 0% function words
at every bin, by construction:

    x bin      -1    -2    -3    -5    -7    -10   -13
    undist   0.222 0.341 0.295 0.161 0.119 0.063 0.013
    forced   0.000 0.000 0.000 0.000 0.000 0.000 0.000

So where the compensation lives (x >= -4) roughly a third of the model's own
openings are function words, and where it vanishes almost none are.

**P2 REFUTED, and this is what kills the candidate.** Same rows, same 1-nat
bins, POS COLLAPSED against POS MATCHED (verb against verb):

    arm            role      POS collapsed   VERB-vs-VERB
    faller         aligned      -0.0318         -0.0226
    matched        aligned      -0.0462         -0.0442
    riser_matched  aligned      -0.0497         -0.0509
    faller         base         -0.0126         -0.0210
    matched        base         -0.0363         -0.0310
    riser_matched  base         -0.0310         -0.0145

**Matching part of speech removes almost none of it** -- the aligned arm
retains 71% to 102% of the effect, and every within-POS cell stays
significant. Nor does adding a frequency band: matched on POS **and** a
0.5-Zipf band, so a rare verb is never compared against a common one
(~1,700-1,800 cell comparisons per arm and role):

    faller aligned -0.0244 p 0.039 | base -0.0282 p 2.4e-06
    matched        -0.0569 p 3.4e-07 |      -0.0460 p 1.2e-04
    riser_matched  -0.0336 p 0.006   |      -0.0475 p 2.9e-04

**So the compensation is not a part-of-speech effect and not a frequency
effect.** Two candidate mechanisms have now been tested and killed: repair
after disruption (the profile inverts) and lexical specificity (survives
POS and frequency matching). The function-word gradient in P1 is real and
explains none of it.

**WHAT REMAINS UNEXPLAINED, stated as such rather than filled in.** At the
same prompt, the same opening probability, the same part of speech and the
same frequency band, a passage whose opening was IMPOSED is more predictable
to the model than one whose opening it SAMPLED. The remaining structural
difference between the arms is not a property of the word at all: it is that
one opening was drawn from the model's own distribution and the other was
not. What that could do mechanically -- given the model cannot see the
difference -- is the open question, and this seat does not have a candidate
it believes.

## Q2: not alignment-specific

DiDs at the primary estimator: faller p 0.20, matched p 0.034,
riser_matched p 1. Under the context control: faller p 0.073, matched
p 0.324, riser_matched p 1. **The two estimators disagree about WHICH arm's
DiD is nominal while agreeing completely about the main effect**, and no DiD
survives Bonferroni over the three arms (0.0167). The base compensates too.
Nothing here is alignment-specific and nothing should be read as such.

## Fences

- Self-surprisal is not comparable across models; every contrast is within
  (pair, role) and, in the context control, within (pair, role, prompt).
- The binned estimator weights bins equally within a pair, so it is a
  median-of-bins and not a passage-weighted average; the linear sensitivity
  is passage-weighted and agrees in sign, which is why both are reported.
- Mean and median diverge in several aligned cells (e.g. matched primary,
  median -0.0551 against mean +0.0055): the bin-level deltas have heavy
  tails. **The median travels**, per [5762], for each arm's OWN
  compensation. It does NOT travel for a comparison of magnitudes between
  arms -- see the withdrawn Q1 above, where this divergence turned out to
  BE the claim rather than a note about it.
- The forced arms remain SECONDARY population per plan A Amendment 1.
- AGGREGATION LAYER SECOND-SEATED ([5804]): all six contrasts, the sign
  counts, the imposition ordering (matched -0.0551 > riser_matched -0.0477
  > faller -0.0342) and the mean/median divergence all reconstruct to the
  digit from the parquet. **The ANCOVA and word-like-opening controls live
  UPSTREAM of that parquet and remain single-pass with the producer** --
  which is to say the two controls this finding most depends on are the
  part nobody has independently rebuilt.
- NOT A FRESH WITNESS RELATIVE TO S4 (`self_surprisal.md`): different
  quantity -- a residual against opening surprisal rather than a level --
  but the same corpus, the same arms, the same collection. One collection,
  two readings.
