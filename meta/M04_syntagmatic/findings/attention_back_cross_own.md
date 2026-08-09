# Attention-back at the forced site: the mechanism is indifferent, the production is not

**PROVISIONAL. ONE PAIR, ONE PROMPT, n = 24.** Not registered, no spec frozen,
and `registrations/plan_attention_back.md` is not amended. This exists so the
design can be argued with before it is scaled, and because three of its results
are about the instrument rather than about alignment, which is cheaper to learn
now than after 1,044 cells.

Producers: `scripts/attn_back.py`, `scripts/attn_delta.py`,
`scripts/attn_select_arms.py`, `scripts/attn_curve.py`. Results under
`results/attn_*.json`. Commits `60147b2c`, `a3f0ce7b`, `5ce8d4e2`, `e060964a`,
`0c8f3be2`.

---

## 1. The contrast, and that it needed no new data

The Y run forced each word in **both** arms: 1,044 of 1,044 (pair, prompt, word)
cells have a base and an aligned member. So

    D[L, H, j] = attn_back(aligned, word) - attn_back(base, word)

is computable on sequences that already exist, one teacher-forced pass each. The
plan proposed a within-checkpoint faller/riser/non-mover design and ruled the
base-aligned contrast out (§3) to avoid F31's 97.8% family variance. That
reasoning holds for a level compared across families and not for a within-pair
difference, and ruling it out is what left the design with no way to cancel
token identity — `cock` and `popcorn` differ as words, not only as movers.

Inside `D`, token, context and slot position are identical. Both the
token-identity confound and the slot-probability confound the third arm existed
for cancel exactly.

**Two modes, and they are the finding.**

    cross   both models over THE SAME token sequences. Only the weights vary.
    own     each model over its own generations, as Finding A's geometry.

The cell: SmolLM2-360M -> SmolLM2-360M-Instruct, `sexual_explicit_1`, with a
naturally probability-matched three-arm draw.

    word     P base   Q aligned   Delta      role
    penis    0.0623      0.0086  -0.0537     FALLER
    thumb    0.0889      0.0894  +0.0005     NON-MOVER  (P within 1.4x of penis)
    cock     0.2009      0.2047  +0.0038     RISER

## 2. Alignment raises attention-back, and does so indifferently

    attention-back, norm-weighted, cross      base   aligned        D
    penis   (faller)                        0.1037    0.1222   +0.0185
    thumb   (non-mover)                     0.0754    0.0950   +0.0197
    cock    (riser)                         0.1130    0.1348   +0.0218

Not a norm-weighting artifact: raw alpha rises +0.0017 to +0.0020 on a base of
0.011 to 0.016, an 11-18% relative increase, while the ||v|| ratio aligned/base
contributes a further 5.5-8.7%. Both are real.

The faller's rise is the **smallest** of the three. Paired across 480 heads:
faller-minus-non-mover p = 0.35 raw / 0.89 norm-weighted; riser-minus-non-mover
0.32 / 0.48; faller-minus-riser 0.036 / 0.21, and that one nominal hit has a
median of -0.00004 and would not survive correction across six tests.

**Bounded, not merely null.** A bootstrap CI on the faller-minus-non-mover median
puts the demotion-specific effect at **4.4% (raw) / 2.0% (norm-weighted) of the
general alignment effect**. The non-mover arm is what makes that sayable: it
establishes there is an effect to fail to find specificity in.

## 3. The two modes disagree completely, and that is the result

    faller minus non-mover, paired across 480 heads, norm-weighted

               cross (one text, two models)   own (each model on its own text)
    j=0        +0.0011  p=0.018               -0.0110  p=1.1e-19
    j=1        -0.0007  p=0.009               -0.0071  p=6.4e-12
    j=3        +0.0004  p=0.086               -0.0105  p=5.3e-21
    j=7        -0.0004  p=0.0006              -0.0039  p=9.8e-13
    j=15       +0.0001  p=0.41                +0.0007  p=2.8e-08
    j=31       -0.0000  p=0.33                +0.0006  p=6.8e-12

`cross`: medians of +/-0.001 with the sign alternating position to position.
That is 480 correlated heads producing small p-values around a null.

`own`: large and ordered. Relative to the non-mover, the aligned model's
continuation binds **less** to a demoted word early and **more** late.

The only difference between the modes is whether each arm writes its own
continuation. So the effect is not in how the aligned model attends to a given
text; it is in what text it writes.

## 4. How to read that

**The mechanism is indifferent; the production is not.** Given identical text,
alignment's attention does the same thing to a demoted word as to an unmoved
one. Given its own text, it binds measurably less to the demoted word early.

**This lands on Weatherby in the shape a friendly amendment wants.** His
technical claim (*Language Machines* ch. 5) is that attention realizes the
poetic function and computes Saussurean value: purely relational, no normative
dimension, so attention-back should be indifferent to alignment status. On this
cell `cross` says he is right at the level he made the claim. What fails is the
scoping claim at introduction note 2 (p. 213) — that RLHF is "downstream from
the core model" — because the alignment effect is real, specific to demoted
words, and lives in production. He keeps the mechanism and loses the bracket
that let him not look at alignment.

**And the honest deflation, which belongs in the same paragraph.** "The
continuation attends less to the forced word" may be a roundabout way of saying
"the continuation changed the subject." If so, attention-back is a *measure* of
routing-around rather than an *account* of it, which the plan's §8 already
concedes. Its one advantage over a topic-continuity measure is that the
weighting is the model's own rather than an external judgment. That is worth
something and it is not the mechanistic authority the vocabulary implies; the
write-up should not borrow that authority.

**The reversal is the part worth chasing.** Early negative, late positive: the
aligned continuation defers the demoted word and returns to it. That is
specific, non-obvious, and the one thing here a topic-continuity measure would
probably not find on its own. It is also exactly the shape that appears once and
vanishes, so it needs ten cells before it goes in a paragraph.

## 5. The attention effect is local; Finding A's disturbance is not

    D(nw) by DISJOINT bin, cross, 480 heads, n=16, window 200

    word      j=0-9    10-24    25-49    50-99   100-199
    penis   +0.0289  +0.0148  +0.0069  +0.0041  +0.0011
    thumb   +0.0317  +0.0152  +0.0089  +0.0039  +0.0023
    cock    +0.0326  +0.0203  +0.0079  +0.0035  +0.0016

A 25x fall to essentially zero, and still no separation between arms.

**The aggregations had to be matched before the profiles could be compared.**
Finding A sweeps CUMULATIVE 1-N; these are disjoint bins, and a cumulative sweep
of a front-loaded effect falls and then flattens by construction — which is the
shape "nothing dies" is read off. In Finding A's own aggregation:

    Finding A, Y surprisal:  falls ~2.5x to 1-100, then FLAT to 256
    attention-back here:     falls  3.2x to 1-100, keeps falling, 5.8x by 1-200

Distinguishable in the same aggregation. Two phenomena: the attention effect is
local to the joint and gone by ~100 tokens; the surprisal disturbance flattens
and persists.

## 6. Instrument facts, including two that correct the plan

**Head concentration is ~7x, not 17x.** The plan cites a one-prompt OLMo probe at
mean 0.021 / max 0.356 as the reason the head must be the unit. Measured on
SmolLM2-360M over 60 undisturbed sequences: raw mean 0.0128 / max 0.0882 (6.9x),
norm-weighted 0.0850 / 0.6407 (7.5x), top 5% of heads carrying 20-23% of the
mass. The head-level design survives at 4-5x enrichment over uniform. The 17x is
another model and should not be quoted for these.

The head ranking is **stable**: n=8 against n=60, top-5% mass 0.206 -> 0.203 raw
and 0.245 -> 0.234 nw, 4 of 5 top heads shared. `L8.H9` leads raw at both sizes,
`L27.H5` and `L23.H10` lead norm-weighted.

**`eager` attention is required.** sdpa and flash do not materialise the
attention matrix and return `None` for `output_attentions` — a null that reads
as data. The producer exits rather than reporting it.

**Tokenizer equivalence is asserted per run.** Running one arm's ids through the
other arm's model is only meaningful if the tokenizers agree, and on this roster
they sometimes do not: zephyr double-encodes a leading space, internlm2 base
omits a BOS both its aligned arms add.

**The undisturbed sequences cannot supply the nuisance floor.** 18,000 of them
exist and the idea was a calibration curve — attention-back against slot-token
probability, over tokens the model chose itself. Within a file the prompt is
fixed, so **logprob is a 1:1 function of the token**: 25 tokens, 25 distinct
logprobs, no exceptions in six files. A permutation test on that cannot separate
"tracks probability" from "differs between `cock` and `popcorn`". Pooling
prompts does not repair it — the two SmolLM2 prompts share **zero** slot tokens.
`attn_curve.py` is kept, and refuses its own headline number at the point of
printing it, because the token-level structure it does establish is a
precondition for `D`: if attention-back were flat across tokens there would be
nothing for alignment to move.

## 7. Limits

One pair, one prompt, one family, n = 24 (n = 16 at window 200). Every number
above is a pilot value and none should be quoted as a rate.

`cross` and `own` answer different questions and neither is the "correct" one.
`own` is the geometry Finding A uses and the one in which the theoretical
question is posed; `cross` is the one that isolates the mechanism. Reporting
either alone would misstate the result.

The `own` result rests on the double difference cancelling generic text
differences between arms. That is what the non-mover is for and it is the
assumption most worth attacking.

Attention is not explanation (Jain & Wallace 2019; Serrano & Smith 2019). This
measures whether a continuation binds to a token, which is descriptive, and does
not establish that attention causes incorporation or its absence. The causal
version is an ablation and is a separate run.

Frequency is not controlled. The three arms share a site and a position; they do
not share token frequency.

## 8. Four errors caught here, all one class

Recorded because the class is the point: **reasoning from a summary instead of
the artifact.**

- The plan's 17x head sparsity, quoted from a one-prompt probe on another model.
  Measured: 7x.
- Top heads selected by the faller's own |D|, then the faller reported as high at
  them. Re-selecting on the base arm alone, the **non-mover** is highest.
- "Truncate to the common J rather than pad" took `min()` over the pool, so one
  short sequence set the window for a whole word — J was 140 for `penis`, 200
  for `thumb`, 103 for `cock` in one run, and the per-word means were compared
  across different windows. Now ragged, with the count per position kept.
- A concern that Finding A carried the `own`/`cross` confound. It does not: it
  carries the full `A|A` / `A|B` / `B|A` / `B|B` grid explicitly, and its robust
  asymmetry is `A|A` against `A|B`. The worry came from the plan's summary of the
  finding rather than the finding.

## 9. What scaling needs

Compute, not data: 1,044 forced cells, both arms present in all of them, 47 of
49 models cached locally.

Before it runs: the plan needs amending to make `D` the primary and to state the
`cross`/`own` split as the design rather than a robustness check; the non-mover
needs its selection rule declared (`attn_select_arms.py` reports the yield grid
rather than a chosen threshold — 3 to 154 cells across two parameters, and the
binding constraint is the base-probability match, not the non-mover threshold);
and the reversal in §4 needs a pre-declared position split, since "early" and
"late" are currently read off the data.
