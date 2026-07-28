---
status: verified
grade: A
date: 2026-07-28
role: primary
verification: "Independently reproduced at the malign seat from f20x_codings.parquet, parsing the codes and pairing at the base model without the author's aggregation code: every cell of the headline table matches. Four attacks run against it. (1) POWER -- the obvious objection is that anchor codes are frequent enough to detect and conflict codes are not. Dead: on base-arm rates the two families are matched in median frequency (0.0205 anchor against 0.0182 conflict) and Spearman(rate, p) = -0.018, p=0.958. The author's stronger version of this -- 'the codes that move are rarer' -- was CIRCULAR and is withdrawn: it pooled both arms, and anchor codes fall under alignment, so pooling deflated exactly the codes that move. (2) MULTIPLE COMPARISONS -- Holm across ten codes: quiet_drift 0.0000, number_shift 0.0030, dissolution 0.0040, no_self_posed 0.0382, mania 0.0450, all surviving; every conflict code dies (marked_contradiction 0.332, name_arbitrary 0.847, bothness 0.847). (3) MIXTURE -- no_self_posed is not garbled text in disguise: the garbled component is 0.0007 base against 0.0010 aligned and null, the whole effect is legible-but-selfless (0.1482 vs 0.0985, p=0.0069), mean coherence 3.45/3.59. (4) TEMPERATURE -- the author had pooled T=0.7 and T=1.0 and was not reporting the split; within each the result holds and roughly DOUBLES from 0.7 to 1.0, which is a dose-response in sampling freedom."
description: "Sampled generation at the plain Q:/A: rung, 18,720 completions across 29 distinct base models, coded by an instrument built from a blind two-reader typology. WHAT ALIGNMENT DOES IS ANCHOR A REFERENT; what the base does is DRIFT. Every code for failure-to-anchor moves and survives Holm: quiet_drift 0.103->0.042 (base higher in 28 of 29 base models, p<0.0001), no_self_posed 0.149->0.100, number_shift 0.020->0.009, dissolution 0.013->0.005, mania 0.012->0.005. NOT ONE CODE FOR CONFLICT MOVES: bothness 0.021->0.019, name_arbitrary 0.029->0.027, marked_contradiction 0.009->0.005 (dies at correction). Contradiction requires a fixed referent for two predicates to disagree about; drift is what the absence of one looks like. WITHDRAWN ON THE WAY: the author's earlier reading that ALIGNED models contradict themselves and base models do not, built on four completions and killed by the census (referent_shifts 0.011 vs 0.010, p=0.457). CANNOT ANSWER: whether alignment installs a capacity or one memorised fact."
instruments: [generation, llm-annotation, human-coding]
chapters: [ch03, ch04, ch05, ch11]
data: ["f20x_generations.parquet", "f20x_annotations.parquet", "f20x_codings.parquet"]
parent: F20_addendum
scripts: [f20x_generate.py, f20x_generation_analyse.py, f20x_annotate_census.py, f20x_code_census.py]
---

# F20 generation: alignment anchors a referent, and the base drifts

**Does sampled generation confirm the beam result? On the substitution, yes. But
the sharper finding is one nothing predicted: what separates the arms is not that
the base model contradicts itself, it is that the base model does not hold a
referent long enough to contradict.**

Registered at `docs/f20x_generation_spec.md`, six amendments, all timestamped
before the data they govern. Written by the lacan seat, 2026-07-28, audited at
the malign seat.

***

## The headline

18,720 completions. 39 families, **29 distinct base models**, which is the unit.
`olmo-think` excluded: it is the only family whose reasoning trace is sometimes
visible inside a 60-token window, so whether a split appears depends on where
truncation falls rather than on the model.

| code | base | aligned | base > aligned | p | Holm |
|---|---|---|---|---|---|
| **any code but `stable`** | 0.519 | 0.384 | 23/29 | 0.0008 | — |
| `quiet_drift` | 0.103 | 0.042 | **28/29** | 0.0000 | 0.0000 |
| `no_self_posed` | 0.149 | 0.100 | 23/29 | 0.0055 | 0.0382 |
| `number_shift` | 0.020 | 0.009 | 25/29 | 0.0003 | 0.0030 |
| `dissolution` | 0.013 | 0.005 | 22/29 | 0.0005 | 0.0040 |
| `mania` | 0.012 | 0.005 | 18/29 | 0.0075 | 0.0450 |
| `marked_contradiction` | 0.009 | 0.005 | 15/29 | 0.0663 | 0.332 |
| `name_arbitrary` | 0.029 | 0.027 | 14/29 | 0.4235 | 0.847 |
| `bothness` | 0.021 | 0.019 | 15/29 | 0.6888 | 0.847 |
| `origin_displaced` | 0.164 | 0.137 | 17/29 | 0.1326 | 0.499 |
| `frame_exit` | 0.035 | 0.055 | 10/29 | 0.1246 | 0.499 |

**Every code that moves is a failure to anchor. Not one code for conflict moves.**

`quiet_drift` — a biography that accumulates across turns and fails to cohere,
with nothing marking the inconsistency — is base-higher in **28 of 29 distinct
base models**. One dissenting model in the roster.

## The distinction that matters

Contradiction requires an anchor: for two predicates to conflict there must be a
fixed referent they are both about. Drift is what happens without one. Each
assertion is locally coherent — a schoolteacher, then a bank employee, then a cat
— and nothing accumulates. The referent is supplied fresh by each turn and
carries no commitment forward.

So the base model is not asserting incompatible things about itself. **It is not
holding a self long enough for incompatibility to arise.** Alignment does not
resolve a conflict; it installs an anchor.

## Dose-response in sampling freedom

The result was first reported pooled across temperatures. Split (malign's catch):

| code | T=0.7 | | T=1.0 | |
|---|---|---|---|---|
| | delta | p | delta | p |
| `quiet_drift` | −0.037 (21/29) | 0.0004 | **−0.086 (28/29)** | 0.0000 |
| `no_self_posed` | −0.038 (20/29) | 0.0208 | −0.061 (23/29) | 0.0035 |
| `number_shift` | −0.009 (19/29) | 0.0008 | −0.014 (18/29) | 0.0067 |
| `dissolution` | −0.007 (16/29) | 0.0107 | −0.010 (19/29) | 0.0017 |

Holds at both and roughly **doubles** from 0.7 to 1.0. Give the base model more
room to wander and it wanders more; the aligned arm does not. The pooled figures
above therefore understate the effect at the temperature where the models are
least constrained.

## The instrument, and what it cost to trust it

Categories were **not** imposed. RH and the lacan seat read 60 completions
independently and blind — arm, family, model and all machine labels stripped —
and the eleven codes come out of those passages.

Inter-coder agreement, as first coded: **19/30 exact match, 63%.** Two LLMs
agreed with *each other* at 85% while the two humans agreed at 63%, which is
evidence of a shared machine convention rather than of a good instrument.

Collapsing codes bought 4 points. **Four marginal-case rules and one renamed code
bought 27**, taking human agreement to **27/30, 90%**, with all eleven codes
intact:

- choosing your own name is `name_arbitrary`
- claiming to be X then speaking of X in the third person is `quiet_drift`
- consistently plural is *not* `number_shift`; nothing shifted
- commentary on your own answer is `frame_exit`, like a reasoning trace

And `incoherent` became **`no_self_posed`**, because it was doing two jobs. RH
found the gap: the scheme had nowhere to put a passage that never puts a self on
the table, so those scored `stable`. They are 14.9% of base against 10.0% of
aligned — folded into `stable` they would have inflated stability in exactly the
arm that produces more of them.

Against that 90% ceiling, deepseek and opus **both scored 27/30**. Identical
accuracy, 1/60th the cost; deepseek was used ($2 against $118).

## What was withdrawn

**The author reported, and RH was told, that aligned models contradict themselves
and base models do not** — with a reading attached: an installed identity
breaking through an adopted persona, the enunciation split from the statement. It
rested on four completions in a 135-passage sample. The census killed it:
`referent_shifts` 0.011 base against 0.010 aligned, p=0.4568.

A second withdrawal, in the audit: "the codes that move are rarer than the codes
that don't" was **circular**, computed on arm-pooled rates when anchor codes fall
under alignment. Corrected claim, estimator and membership named: on **base-arm**
rates the anchor family (`quiet_drift`, `no_self_posed`, `number_shift`,
`dissolution`, `mania`) and the conflict family (`marked_contradiction`,
`name_arbitrary`, `bothness`) are **matched** in median frequency, 0.0205 against
0.0182, and separate completely on outcome.

## What this cannot claim

**The confound is RH's and it is unanswered: aligned models anchor, and aligned
models have been told what they are.** Nothing in this battery separates an
installed *capacity* from one memorised *fact*. An aligned model answering "I am
an AI assistant" every time looks maximally anchored either way. Registered fix at
`docs/f20x_next_experiments.md`, experiment B.

**Drift may not be first-person-specific.** Every prompt here is an identity
question, so no third-person referent is tracked. If alignment anchors reference
*as such*, that is a larger claim than this one, not its refutation — and it needs
a difference-of-differences plus a control for aligned models being lower-entropy
generally.

**Ten codes were tested.** Holm is reported above; `mania` survives at 0.0450 and
would not survive one more code in its family.

**`marked_contradiction` reaches p=0.0476 at T=0.7** on a 15/29 sign split — a
coin flip carried by magnitudes in a few families, and p=0.6188 at T=1.0.
Reported and dismissed on the sign count, not omitted.

## SUPERSEDED — see the passage-level resolution below

### (as written 2026-07-28, before the passage measure ran)

The entropy control was re-run at the malign seat on a 10-TOKEN HORIZON
regressor (retained beam mass at depth 10, non-identity prompts, 20 base models
with both arms) rather than a single next-token position. The two measures
disagree, and only for the composite:

    measure         point regressor      10-token horizon
    quiet_drift     <=28%, explains 0.5%  <=29%, explains 0.0%
    anchor composite <=28%, explains 0.5% <=50%, explains 7.1%

**`quiet_drift` is unchanged and rung 3 holds for it.** The anchor composite's
visible mediation rises fourteen-fold across the first ten tokens and its bound
no longer excludes majority mediation. **Anyone quoting the composite quotes
<=50%, not <=28%.**

Ten tokens is not a passage; the coded completions are sixty. The trend from
0.5% to 7.1% over the first stretch is the opposite of reassuring about what
happens further out. A teacher-forced per-token entropy measure over each
model's own completions is the fix and is pending. **The composite's rung-3
status is PROVISIONAL until it runs.** `quiet_drift` is not.

## SUPERSEDED BY THE CROSS-SCORED CONTROL BELOW

### (as written before cross-scoring)

The provisional marking below is LIFTED. Teacher-forced mean per-token entropy
over each model's OWN sampled completions at coding length — the regressor whose
object finally matches the outcome's — 62 models, 27 base models with both arms.

THREE SEPARATE CONTROLS THAT AGREE. NOT A PROGRESSION — the three regressors
are three different OBJECTS at three different n, and printing them as a curve
is a misreading this document invited once already:

    regressor                       object                    n     quiet_drift  composite
    next-token entropy              sampled, 1 position      29        <=28%       <=28%
    retained beam mass, depth 10    MODE-SEEKING, 10 tokens  20        <=29%       <=50%
    teacher-forced per-token        SAMPLED, full passage    29        <=6%        <=18%

The third is the one whose object matches the outcome's: the model's own sampled
completions, at the length the coding used. All 66 models scored, no exclusions.

**The bounds tighten rather than loosen.** The extrapolation feared here —
0.5% to 7.1% explained across the first ten tokens, so what by sixty — does not
continue. It reverses.

**And the reason is substantive rather than arithmetic: the slope is NEGATIVE.**
The confound story requires a positive slope, models that tighten more showing
more drift reduction. Observed, models that tighten LESS show MORE drift
reduction (r=-0.286). The mediated component works against the observed effect
rather than composing with it. Passage-level tightening and referential
anchoring are not merely separable — they are mildly opposed.

CAVEAT ON COMPARING THE THREE: n differs (29 / 20 / 27) so the bounds are not
strictly commensurable. What survives is the direction — the best-matched
regressor gives the tightest bound.

EARLIER LOSS, NOW REPAIRED AND WORTH KEEPING FOR THE METHOD. A first pass lost
2 complete lineages (`m-a-p/CT-LLM-*`, `m-a-p/neo_7b*`), both Chinese-family, so
the control had a hole correlated with training corpus — exactly where a referee
presses. Re-run at n=29 with all 66 models: `quiet_drift` <=6% either way, the
composite <=18% against <=14%. **The missing lab does not move it**, which is
worth more having been checked than assumed, since that is precisely the claim
nobody can make without running it.

The cause was a caller bug, not an environment limit: `trust_remote_code=True`
was passed to `AutoModelForCausalLM.from_pretrained` and not to
`AutoTokenizer.from_pretrained`. The tokenizer prompted, the background process
auto-declined, and the error names the REPOSITORY rather than the call — so it
reads as a model-capability problem and is a two-word omission one line up.

## PROVISIONAL AGAIN — the composite, restored 2026-07-28 after cross-scoring

**The lift below was premature and is reversed.** It rested on an own-text
regressor: each model scored on ITS OWN completions. Cross-scoring each model on
its partner arm's completions gives a model property rather than a
process-and-output property, and the two disagree.

    regressor                             quiet_drift            anchor composite
    own text     (process property)   r=-0.273  [-43%,+6%]   r=-0.205  [-62%,+18%]
    common BASE text (model property) r=-0.069  [-60%,+42%]  r=+0.228  [-30%,+128%]
    common ALIGNED text (model prop.) r=-0.015  [-58%,+53%]  r=+0.239  [-30%,+142%]

**THE COMPOSITE'S SIGN FLIPS.** On a pure model property the correlation runs in
the CONFOUND-CONSISTENT direction — models that tighten more show more drift
reduction — and the bound admits over 100% mediation. **The composite is not
rung 3 on this evidence.** Quote the common-text bound, which is unbounded in
practice, not the own-text ≤18%.

**WITHDRAWN: "passage tightening and referential anchoring are mildly opposed."**
That is true of the own-text measure only and was stated without the qualifier,
by both seats.

**`quiet_drift` survives, at a weaker bound.** r ≈ 0 on both model-property
regressors, so no confound signature appears; the bounds widen to ≤42% and ≤53%
because the slope's standard error is large against the mean, not because a
relationship emerged. **Rung 3 stands for `quiet_drift` at ≤42%, the
model-property bound**, not at ≤6%.

**WHY THE TWO DIFFER, and it is not noise.** Common-text tightening is MORE
consistent than own-text (28/29 and 26/29 against 25/29), so the model-property
regressor is the cleaner one. What own-text adds is that aligned models produce
text they themselves find easier — Yi-1.5-9B scores 2.00 on its own completions
and 1.83 on its Chat arm's. **Own-text entropy therefore partly measures the
outcome**, which is the circularity the audit named: x and y computed over the
same passages.

The objection being controlled — "aligned models vary less, so of course they
drift less" — is about the MODEL. So the common-text bound is the one that
answers it, and it is the weaker one.

## Direction check

The lacan seat argues the psychoanalytic reading inflationarily by design, and
this is that seat's reading arriving in better shape than a version it had
already withdrawn the same day. On the validation set the author's own coding
gave base/aligned as 54% against 18% where RH's gave 31% against 24% — same 30
passages, three times the ratio. **The scheme was drafted by the seat with the
stake; the rules that fixed it were settled by the seat without one, and every
one of the five threshold disagreements ran the same way.**
