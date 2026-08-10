# Attention-back at the forced site: a large pair-specific effect, and no effect of alignment status

Alignment changes how much a continuation binds back to the slot by a
PAIR-SPECIFIC amount -- up to +24% and -19% on the model's own chosen token,
with pair identity predicting the shift at Kruskal-Wallis p = 0.0019 over 6
pairs. It does NOT change that binding differently for a word it demoted, a word
it promoted, or a word it left alone: 14 of 28 cells on the primary contrast.

An earlier version of this file said alignment did nothing to attention. That was
wrong twice: the normalisation put the effect in a denominator, and pooling
across pairs whose signs oppose cancelled what was left. See section 3e.

**PROVISIONAL. 6 PAIRS, 5 PROMPTS, 28 CELLS. The central prediction is
REFUTED at the cell level (section 3c); sections 2 and 3 are superseded by 3b
and 3c wherever they disagree.** Not registered, no spec frozen,
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

## 2. Alignment raises attention-back at this cell, and does so indifferently

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

### Forcing does NOT distort the level. A correction, then its retraction.

RH asked whether the undisturbed slot could serve as the control. It cannot for
fallers -- the aligned arm chose `penis` once in 50 draws against `cock` twelve
times, which is what "demoted" means. On the words both arms choose often enough
it appeared to show that forcing distorts the level, in opposite directions by
arm: base -11.4% / -3.0% / -23.3%, aligned +10.9% / +22.5% / +6.1%, six of six
consistent in sign, with head profiles preserved at r = 0.95 to 0.995. That was
written up here as a correction to section 2's level claim.

**RH then asked how a model could know it had been forced. It cannot, and the
apparent effect is noise.**

The producer builds the forced prompt as `slot["prompt"] + " " + w`
(`scripts/vllm_y_run.py:119`). So a chosen `cock` and a forced `cock` are the
IDENTICAL token sequence up to and including the slot; the model's state there is
the same and the continuation distribution is the same. Chosen and forced
sequences of one word are draws from one distribution and the difference is zero
by construction.

Tested rather than argued -- split-half null, resampling the forced pool into a
group of size n_chosen against the rest:

    arm      word     observed   split-half null sd      p
    base     cock      -0.0125         0.0070          0.087
    base     thumb     -0.0056         0.0095          0.561
    aligned  cock      +0.0103         0.0095          0.322
    aligned  thumb     +0.0151         0.0141          0.319

Not one clears. Every observed difference sits inside the between-sequence
variation of the forced pool itself, and the six-of-six direction consistency
was six correlated observations across two words sharing one prompt.

**So section 2's level claim stands as measured** and the correction is
withdrawn. What this exercise did establish is worth keeping: **forcing is benign
for this measurement**, by construction and consistent with the data, and the
head profiles are preserved across conditions at r > 0.95.

One check this argument depends on and that should be run when scaling: the
prefix identity requires `tokenize(prompt + " " + w)` to equal
`tokenize(prompt) + tokenize(" " + w)`. It holds on the pilot cell -- plen 14 =
13 + 1, and `full_ids[plen-1]` decodes to the word -- but a merging boundary in
another tokenizer would break it silently. `attn_delta.py` currently asserts
base-against-aligned agreement, not boundary integrity.

Producer: `scripts/attn_forcing_check.py`.

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

### It is specific to demotion, and the riser is what shows it

RH asked whether this is about the word having been DEMOTED or about any word
having been FORCED. All three arms are forced, so faller-against-non-mover
already holds forcing constant — but the riser settles it, because it moves the
other way.

    own, D(nw) head-mean          j=0      j=1      j=3      j=7     j=15
    penis   FALLER             +0.0452  +0.0463  +0.0311  +0.0126  +0.0513
    thumb   NONMOVER           +0.1469  +0.1159  +0.0795  +0.0351  +0.0080
    cock    RISER              +0.1152  +0.0826  +0.1036  +0.0847  +0.0396

    paired across 480 heads, median  (* p<0.001, : p<0.05)

    FALLER - NONMOVER   -0.0109*  -0.0071*  -0.0105*  -0.0039*  +0.0007*
    RISER  - NONMOVER   -0.0047*  -0.0005:  +0.0010:  +0.0035*  +0.0018*
    FALLER - RISER      -0.0031*  -0.0059*  -0.0148*  -0.0124*  -0.0004

From j ~ 3 the ordering on THIS cell is FALLER < NONMOVER < RISER, monotone in
alignment status, with the two contrasts against the non-mover significant and
opposite in sign. None of it appears in `cross`.

**THAT ORDERING IS RETRACTED. See below.**

### Retraction: the ordering was confounded, and the discriminating cell refuses both accounts

The three words' base probabilities are `penis` 0.062, `thumb` 0.089, `cock`
0.201. **The observed ordering is exactly the ordering by base probability**, so
on this cell "ordered by alignment status" and "ordered by how probable the word
was" are indistinguishable. `thumb` is matched to `penis` within 1.4x, which is
what the plan asked for; `cock` sits at 3.2x the faller and was matched to
nothing.

43 cells in the corpus have P(faller) > P(riser), where the two accounts predict
OPPOSITE orderings. Running the cleanest small one -- OLMo-2-0425-1B -> DPO,
`sexual_explicit_3`, where the faller is 33x more probable than the riser:

    word              P base   D_prob   attn-back D   base level
    cock    FALLER    0.1136   -0.0967      -0.0250      0.1302
    dick    NONMOVER  0.0034    ~0          -0.0289      0.1319
    manhood RISER     0.0041   +0.0696      -0.0098      0.0368

    observed:              NONMOVER < FALLER < RISER
    alignment predicts:    FALLER < NONMOVER < RISER
    probability predicts:  NONMOVER ~ RISER < FALLER

Neither. The faller is in the middle where the alignment account puts it lowest,
and the high-probability faller is not highest where the probability account
puts it there.

**Two further defects this cell exposes.**

D is NEGATIVE throughout here -- alignment LOWERS attention-back -- where on
SmolLM2 it was positive throughout. So section 2's "alignment raises
attention-back" is a fact about one pair, not a fact.

**D as an absolute difference is confounded with the base level**, and no choice
between absolute and ratio was ever declared. `manhood` sits at 0.037 against
`cock` at 0.130, so equal proportional changes give very different absolute D.
In ratio terms: cock -19.2%, dick -21.9%, manhood -26.6%, which inverts the
ordering a third time.

What survives from section 3 is only the `cross`/`own` split itself: `cross` is
flat and `own` is not, on both cells. What the `own` effect is ordered BY is
open.


**One reading of the question this design cannot answer.** Attention-back to a
forced word against a word the model CHOSE would isolate forcing itself. Finding
A rules that comparison invalid by design for surprisal -- the undisturbed arm
has committed to nothing, and `plen` offsets the scored positions -- and the
first objection applies here unchanged. Forced-versus-chosen is not available
from this corpus.

## 3b. Normalising by each arm's own undisturbed slot

RH's proposal: take the first token of the undisturbed generations as the
control and read the forced words against it. Per (pair, prompt, arm),

    U(arm)        attention-back to the model's OWN first generated token,
                  mean over 50 undisturbed sequences
    D_norm(word)  log[attn(aligned,word)/U(aligned)] - log[attn(base,word)/U(base)]

**This fixes a real defect.** Absolute D is confounded with the base level -- on
the OLMo cell `manhood` sits at 0.037 against `cock` at 0.130, so equal
proportional changes give very different absolute differences and the ordering
inverted depending on which was used. D_norm is a log ratio of ratios,
scale-free per head and per arm, and it settles the absolute-vs-ratio question
that was left open rather than answering it by preference.

It does NOT give a token control -- U is a mixture over whatever the model
chose. But the reference is the same for every word in a cell, so it cancels
exactly in any word-vs-word contrast. It buys the arm normalisation only.

    SmolLM2-360M -> Instruct, explicit_1, 474 of 480 heads kept

    word              D_norm      pairwise, paired by head
    penis   FALLER    +0.0990     penis - thumb  -0.0996  p=1.6e-15
    thumb   NONMOVER  +0.2281     penis - cock   -0.0934  p=1.2e-16
    cock    RISER     +0.2117     thumb - cock   +0.0230  p=0.048

The faller sits distinctly below both; non-mover and riser are close. **That is
demotion-specific and it is NOT the probability ordering** -- probability runs
penis 0.062 < thumb 0.089 < cock 0.201, D_norm runs penis < cock < thumb. The
normalisation removed the confound that made the raw ordering uninterpretable in
section 3.

    OLMo-2-0425-1B -> DPO, explicit_3, 256 of 256 heads kept

    cock    FALLER    +0.0198  p=0.37     cock - dick     +0.0243  p=0.034
    dick    NONMOVER  -0.0384  p=0.74     cock - manhood  -0.0138  p=0.203
    manhood RISER     -0.0962  p=0.22     dick - manhood  +0.0732  p=0.443

Nothing, and the ordering runs the other way.

**One cell with a strong, correctly-shaped, probability-independent effect; one
cell with nothing.** Two cells is two cells, and three claims in this file have
already been retracted for being called a pattern at this n. What is established
is the INSTRUMENT: D_norm is the right quantity and the earlier tables should be
read as superseded by it wherever they disagree.

Producer: `scripts/attn_norm.py`.

## 3c. The sweep: the prediction is refuted

28 cells, 6 pairs, 5 prompts, arms auto-selected by the rule declared in
`attn_norm_sweep.py`, which was committed at `3de8c16c` BEFORE the sweep
returned so the prediction has a timestamp preceding its own test.

Predicted: FALLER below, NONMOVER and RISER together.

    FALLER - NONMOVER    median +0.0174   14 of 28 negative   p=0.171
    FALLER - RISER       median -0.0278   16 of 28 negative   p=0.194
    NONMOVER - RISER     median -0.0478   18 of 28 negative   p=0.0247

`FALLER - NONMOVER` is an exact coin flip with the median on the wrong side. The
only nominally significant contrast is the one predicted to be near zero.

**The generating cell does not replicate inside its own pair.** SmolLM2
`explicit_1` gave F-N = -0.039; SmolLM2 `explicit_3` gives **+0.188 at
p = 2e-32**. Same models, same instrument, opposite sign, both overwhelming.

**That is the methodological lesson and it applies to everything above.** The
per-cell p-values run to p = 0 and p = 2e-32 in both directions because they are
computed across 250 to 480 heads that are massively correlated. They are not
evidence about anything, and section 3b's `p = 1.6e-15` is one of them. The
CELL is the unit. At the cell level, 28 of them, there is nothing.

So D_norm is a better instrument than raw D -- it is scale-free and it broke the
probability confound -- and what it measures does not vary with alignment status.

## 3d. The tokenizer guard, tested only after I claimed to have tested it

`stablelm-2-zephyr-1_6b` went into the sweep specifically to make the guard
fire. It did not, and the guard was fine: that model shares a tokenizer with its
base exactly, 0 of 4 strings differing. **I selected the test case by matching
the word "zephyr" in a name.** The recorded defect is in
`HuggingFaceH4/zephyr-7b-beta` against Mistral, an unrelated model, and that
pair is not in the Y corpus at all.

Exercised on the real pair, the guard's condition behaves:

    stablelm-2-zephyr-1_6b   0 of 4 strings differ   -> proceed
    zephyr-7b-beta           3 of 4 strings differ   -> SKIP PAIR
                             ' cock'  [12408] vs [28705, 12408]

So it refuses when the condition holds and proceeds when it does not. It was
untested until after the sweep, and the sentence claiming it had been tested was
written before any of this was checked.

## 3e. Alignment DOES change attention-back. I normalised it away and then pooled it away.

RH: "So alignment doesn't do anything to attention? Remember how you've given up
too early so many times?" Correct on both counts.

**The first error is in the instrument.** D_norm divides each arm by its own
undisturbed baseline U, so any effect that moves forced words and self-chosen
words TOGETHER is zero by construction -- which is the most likely shape for a
general effect. The normalisation was built to fix a real confound (section 3b)
and it does, but it was also blind to the largest plausible signal, and I read
its silence as absence.

The decomposition it hid:

    log[attn(aligned,w)/attn(base,w)] = log[U_a/U_b] + D_norm(w)
              TOTAL on the word          BASELINE      WORD residual

    median |total|     0.1015
    median |baseline|  0.1633      <- larger than the total
    median |residual|  0.0951
    corr(total, baseline) = 0.709

**The second error is the pooled test.** Across 28 cells the baseline shift is
+0.0531, 17 of 28, p = 0.227 -- null. Per pair it is not:

    stablelm-2-1_6b     +0.2179   IQR 0.0187   |med|/IQR 11.62    +24%
    Falcon3-1B-Base     +0.1883   IQR 0.1017    1.85              +21%
    OLMo-2-0425-1B      -0.2157   IQR 0.1343    1.61              -19%
    TinyLlama-1.1B      +0.0594   IQR 0.1474    0.40
    SmolLM2-360M        +0.0386   IQR 0.0666    0.58
    Qwen2.5-0.5B        -0.0197   IQR 0.1439    0.14

Three pairs shift by around a fifth, tightly; three barely move; the signs
oppose, so pooling cancels them. **Kruskal-Wallis across the six pairs on the 28
prompt-level values: p = 0.0019.** Which pair you are in predicts the shift.

That is consistent with F31's 97.8% family variance rather than in tension with
it. A pair-specific effect is the expected shape here and a pooled test is the
wrong instrument for it.

**One test that does NOT support this, reported because I ran it first and got
it wrong.** Modal-sign agreement across prompts within a pair: 22 of 28. Against
a binomial null of 0.5 that is p = 0.0019, and I nearly reported it. But the
modal sign is defined by the majority, so under random signs a 5-prompt pair
already agrees 3.44 times. The correct null expects 19.24 and gives **p = 0.077**.
The Kruskal-Wallis is the test that holds; the sign count is not independent
support for it.

Limits: 6 pairs, 5 sexual prompts, U measured on undisturbed generations from
those prompts only. Nothing here says the shift is general rather than a property
of this domain, and nothing predicts its direction.

Producers: `scripts/attn_decompose.py`, `scripts/attn_norm_sweep.py` (now storing
U and the per-head levels; the earlier 28-cell file did not, which is why this
needed a re-run).

## 4. How to read that

**Alignment changes attention-back; alignment status does not.** The pair-level
shift is real and large (section 3e). What does not vary is the effect of a
word's MOVEMENT. Given identical text, alignment's
attention treats a demoted word exactly as it treats an unmoved one -- bounded
at 2.0 to 4.4% of the general effect. Given its own text the arms separate
strongly in every cell, but across 28 cells that separation does not order by
alignment status: 14 of 28 on the primary contrast, median on the wrong side.

**What survives is the `cross`/`own` split itself.** It is large, it reproduces,
and it is a fact about the design rather than about alignment: an effect present
when each arm writes its own continuation and absent when both read the same
one, lives in what gets written, not in how it is attended to. That is worth
carrying because it constrains any future attention-based claim in this
campaign, including Finding A's, which is in the `own` geometry.

**On Weatherby, this yields less than I first wrote, and in his favour.** His
technical claim (*Language Machines* ch. 5) is that attention realizes the
poetic function and computes a purely relational value with no normative
dimension, so attention-back should be indifferent to alignment status. `cross`
says exactly that, and now `own` does too at the cell level. **So this
instrument produces no evidence against his scoping claim** at introduction
note 2 (p. 213) that RLHF is "downstream from the core model." An earlier
version of this file said the scoping claim failed here. It does not. The
amendment the paper wants will have to come from somewhere other than
attention-back at a forced site.

**And the deflation stands regardless.** "The continuation attends less to the
forced word" was always at risk of being a roundabout way of saying "the
continuation changed the subject", which the plan's SS8 concedes. That worry is
now moot for the alignment question, since nothing varies with alignment status,
but it remains the reason not to reach for the mechanistic register if this is
revived.

**What would be worth chasing, if anything.** The `own` effect is real and large
per cell; it simply is not ordered by movement. Something orders it, and base
attention-back level, token frequency and prompt identity are the unexamined
candidates.

Two of those are cheap and one is not, which an earlier version of this
paragraph got wrong. `attn_norm_sweep.json` stores only the medians and the word
names: prompt identity and token frequency are a join away, but the base level
was computed and discarded, so testing it needs the whole sweep again. The
producer now stores `U`, the per-head levels, the per-head D_norm and the
selected words' P and Q, which is ~750 KB against a 25-minute re-run. The 28
cells on disk predate that change.

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

Two pairs, two prompts, n = 24 (n = 16 at window 200). Every number above is a
pilot value and none should be quoted as a rate. The two cells disagree on the
SIGN of the general effect -- SmolLM2 positive, OLMo-2 negative -- so even the
direction is a per-pair fact so far.

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

## 8. Seven errors caught here, six of them one class

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
- A "forcing distorts the level" correction, written up and committed, that was
  noise at n_chosen = 6 to 12 and died to a split-half null. RH's question --
  how would a model know it had been forced -- is what prompted the test.
- Per-cell p-values over 250 to 480 correlated heads reported as evidence, twice,
  including a `p = 1.6e-15` presented as a strong result. Not this class: this
  one is a statistical error rather than a reading error, and it is the most
  expensive of the seven because it made three separate nulls look like
  findings.

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
