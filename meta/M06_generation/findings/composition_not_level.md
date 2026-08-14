---
status: draft
grade: ungraded  # single pass, one seat; nothing here is audit-grade until a second seat reproduces
date: 2026-08-13
role: finding
topics: [mediation, M01, M06, composition, level, displacement, paradigmatic]
description: "M01 and M06 are one operation at two scales. Alignment's passage-level surprisal drop is COMPOSITION (-1.310 of -1.285 nats/word) and not LEVEL (+0.025): it changes WHICH words appear, not what they cost in context -- selection, not combination. Composition tracks M01's DIRECTIONAL displacement (partial rho -0.276, 36/36 pairs) and NOT volatility (+0.008, 18/36, exact chance). THE CLAIM ON COST IS A CONTRAST, NOT A LEVEL: on base text the aligned model finds nearly everything costlier and words M01 never measured costliest of all (+0.614 at 71.7% of tokens), so fallers sit BELOW the corpus average; what survives is that AT MATCHED ALIGNED PROBABILITY, words pushed DOWN cost 0.347-0.444 nats more in context than words pushed UP (34/35 and 35/35 pairs) -- the demoted-versus-merely-improbable separation A_post_utterance_shock.md declared impossible for want of a matched corpus. NOT damage to the syntagm: the chain is held constant by construction. Withdrawn: the +0.284/+0.320 net_fall-vs-level correlations, all classified-mover numbers, and the 33-pair figures."
---
# Composition, not level: M01's displacement carries M06's passage effect

RH's question, and the one a reviewer asks first: **alignment lowers passage
surprisal — which words carry that drop, and are they the words M01 found
alignment moved?** If they are, the campaign has one operation observed at two
scales. If not, it has two findings that co-occur, and the paper must not imply
a mechanism it never tested.

## The instrument, which needed no new generation

Every undisturbed passage is scored by BOTH members of its pair — 1,142,944
self rows and 1,142,944 cross rows in `passage`. So for one text:

    s_base(T)      the base model's per-token surprisal
    s_aligned(T)   the aligned model's, ON THE SAME TOKENS

**Composition is held fixed BY CONSTRUCTION, not by matching.** Every word is
its own control: same word, same context, two distributions. RH pointed at
these scores; the first draft of the plan was going to re-score with gpt2 and
would have been strictly worse.

Nothing is re-tokenized either. `gen_sequences.token_ids` records the
generation (`length(token_ids) = n_tokens` on all 238,400) and `logprobs` aligns
1:1. Re-tokenizing was tried and rejected: it drops 20-30% of texts, and drops
them non-randomly, since a text that fails to round-trip differs in content.

## Composition and level, defined

Mean surprisal per word is `sum_w f(w) s(w)`. Two ways aligned passages can be
cheaper than base ones, and they exhaust the possibilities:

    COMPOSITION   sum_w (f_a - f_b) sbar(w)     different words are used
    LEVEL         sum_w fbar(w) (s_a - s_b)     the same words cost less

**COMPOSITION IS SELECTION AND LEVEL IS COMBINATION** — the paradigmatic and
syntagmatic axes, in a form that can be measured and that sums exactly.

## Result 1: it is composition, near-entirely

36 pairs, self-surprisal, nats per word:

    Delta (aligned self - base self)   -1.2849
      composition (symmetric)          -1.3098
      level       (symmetric)          +0.0249
    order 1  comp -1.7748  level +0.4899
    order 2  comp -0.8448  level -0.4400
    GATE R residual                     0.0000

    by relation   base_to_superego n=14  Delta -1.5835  comp -1.6543
                  dpo_of           n=20  Delta -1.0047  comp -1.0047
                  ppo_of           n= 2  Delta -1.9965  comp -1.9502

**Both decomposition orders are reported and neither is the headline.** They
disagree substantially (level swings +0.49 to -0.44), so composition and level
are entangled and the symmetric mean is a summary, not a fact. A single order
would be an undeclared choice, and this campaign has had one flip a refutation
into agreement.

**Alignment changes which signifiers appear, not how the chain coheres.** That
converges with the propagation result from the same series: ~99% of an imposed
word is absorbed within a few tokens, so the syntagmatic axis is exactly where
alignment does not reach.

## Result 2: composition tracks M01's DIRECTIONAL displacement

No classification and no threshold. Every mover flag admitted `the`, which is
not a threshold defect: **`the` moves constantly and goes nowhere** — non-still
in 30.5% of its 1,575 Llama cells, more volatile than `hit` or `hurt`, with a
direction of -3.3%. Volatility and direction are two dimensions and a binary
flag multiplies them into one. So two continuous scores per (pair, word):

    pct_moved       100 * (n_fall + n_rise) / n_cells
    dir_when_moved  100 * (n_fall - n_rise) / (n_fall + n_rise)
    net_fall        their product

    kill  51.8% moved, +100 direction     scream  41.9% moved, -44
    the   30.5% moved,   -3.3             hurt    16.7% moved, -92

Against the composition change (f_aligned - f_base):

    net_fall        rho -0.285  36/36 negative  p 2.9e-11   partial -0.276
    dir_when_moved  rho -0.269  36/36           p 2.9e-11
    logratio(theta) rho -0.308  36/36           p 2.9e-11
    pct_moved       rho +0.008  18/36           p 0.479     NULL

**DIRECTION PREDICTS, VOLATILITY DOES NOT**, and 18/36 is exact chance. `the`
disqualifies itself arithmetically instead of being excluded by hand. A generic
"unstable words behave differently" effect would show in `pct_moved`; it does
not. Top decile of fallers: -212 occurrences per 10k tokens; top risers +58.

## Result 3: displaced words cost more RELATIVE TO RISERS — the contrast, not the level

**TWO DIFFERENT QUANTITIES ARE BOTH CALLED "LEVEL" AND EARLIER DRAFTS OF THIS
FILE CONFLATED THEM.** RH caught it by asking how, if the effect is all
composition, the same words can cost differently.

    DECOMPOSITION level   s_aligned|aligned-text - s_base|base-text
                          each model on ITS OWN generations.  ~ 0 (+0.0249)
    CROSS-SCORER level    s_aligned(T) - s_base(T)
                          two scorers, ONE fixed text.  LARGE and positive

They are not the same term and they support different claims. Results 3 and 4
use the second.

**And the cross-scorer level is dominated by a generic off-policy shift, not by
displacement.** Occurrence-weighted, on base-generated text:

    class         tokens   share   mean level   contribution
    fall         1293354    7.6%     +0.3806      +0.0288
    rise          596374    3.5%     +0.1443      +0.0050
    still        2948340   17.3%     +0.2574      +0.0444
    unmeasured  12242204   71.7%     +0.6137      +0.4399
    TOTAL                                         +0.5181

**Everything is more costly to the aligned model on base text, and the LARGEST
effect is on words M01 never measured.** Base text is simply foreign to the
aligned model. Against that backdrop fallers at +0.381 sit BELOW the corpus
average, so "displaced words cost more in context" is true and NOT DISTINCTIVE.

The distinctive quantity is the CONTRAST between directions, on a fixed text
where the generic shift applies equally to both:

    fall +0.3806  vs  rise +0.1443                     raw, all cells
    median(level|fall) - median(level|rise)  +0.3471   common support, 34/35
                                             +0.4435   aligned-gen, 35/35

The correlations that were reported here as the result (+0.284 / +0.320 against
net_fall) inherit the same problem: they are computed against a level whose bulk
is off-policy shift. **The contrast at matched probability is the claim; those
correlations are not.**

### What "costs more" means, and what it does NOT mean

Surprisal is `-log p(word | the words before it)`, in nats, so the cross-scorer
level is **two models' opinions about ONE FIXED token sequence**. +0.35 nats
means the aligned model finds that word about 1.4x less probable, in that exact
context, than the base model finds it. Nothing about the text changes; only the
reader does.

**THIS IS NOT DAMAGE TO THE SYNTAGM, AND THE DESIGN COULD NOT SHOW DAMAGE IF
THERE WERE ANY.** The chain is held constant by construction — the same tokens
are scored twice — so a claim about the passage cohering less has no purchase
here. Two measurements in this series say the chain is in fact intact:

- **Decomposition level is +0.0249.** When each model generates AND scores its
  own passage, a word costs about the same to whichever model wrote it.
  Aligned passages are not less coherent to their own author.
- **Propagation slope ~+0.008 nats-per-bit.** An imposed improbable word damages
  the continuation in direction but ~1.3% in magnitude; the chain absorbs it
  within a few tokens.

What the contrast registers is **DISAGREEMENT, NOT INJURY**: at every position
where a displaced word appears, the aligned model's distribution says *this is
not the word I would have reached for*. A standing paradigmatic disposition,
legible wherever the word occurs, which does not become damage along the chain.

That agrees with `A_post_utterance_shock.md`'s own channel-3 result from the
other side — the charge on the demoted signifier is LOCAL, "paid at the word,
bleeding one token past it, then nothing." Alignment operates on selection and
leaves combination alone.

**Limit:** because level is measured only where the word was emitted, and always
on a fixed text, this instrument cannot address chain damage at all. The
propagation study is the instrument for that question.

## Result 4: demoted, not merely improbable — the M04.A confound, separated

`A_post_utterance_shock.md` names this exactly:

> *"The faller is by construction low-probability **under aligned** ... the next
> token inherits that mechanically. Separating them requires a word matched on
> improbability-under-aligned but NOT demoted by alignment. No collected corpus
> has one."*

It wanted a matched PAIR. Continuously, `p_aligned` is a covariate and no
matched corpus is needed. **Per context** — 848,453 cells, 36 pairs, 203
prompts, each a word's cost in passages generated from prompt P conditioned on
its aligned probability at P's OWN slot:

                          raw     partial | log p_aligned AT SLOT   pairs
    level   (base-gen)   +0.171            +0.135                   36/36  p 2.9e-11
    level_a (aligned)    +0.188            +0.172                   36/36  p 2.9e-11
                                 partial | log p_base AT SLOT
    level                                 +0.172
    level_a                               +0.183

The control absorbs some and eliminates nothing, and it is close to the harshest
constructible: under CANONICAL a faller at a slot is partly DEFINED by its
aligned probability there, so covariate and predictor are entangled by rule.

### The partial had a common-support problem; the contrast does not

@malign flagged ([5882]) that a covariate holding current probability cannot
separate a word pushed DOWN from one pushed UP to the same place. The predictor
here IS direction, so the design encodes it -- but the objection lands for a
reason neither of us stated: **fallers and risers barely share a p_aligned
range**, so conditioning on it linearly extrapolates rather than compares.

    log p_aligned bin        fall    rise   minority share
    (-3.001, -2.963]        30601      0     0.000
    (-2.963, -2.734]        15300      0     0.000
    (-2.734, -2.464]        14676    625     0.041
    (-2.464, -2.145]        10128   5172     0.338
    (-2.145, -1.841]         4521  10779     0.295
    (-1.841, -1.465]         1941  13359     0.127
    (-1.465, -0.0009]         450  14851     0.029

45,901 fall cells sit in bins holding ZERO risers. Restricted to the common
support band (-2.464 .. -1.465), as a direct contrast rather than a partial:

    median(level | fall) - median(level | rise)
      base-generated      +0.3471   34/35 pairs positive   p 1.2e-10
      aligned-generated   +0.4435   35/35                  p 5.8e-11

**At matched aligned probability, a word alignment pushed DOWN costs ~0.35-0.44
nats more in context than one it pushed UP to the same place.** That is the
ladder's own contrast -- rungs at one probability -- obtained observationally,
and it is the form to quote. The partial correlations above stand only as the
weaker, extrapolating version.

**Raw values here are NOT comparable to Result 3's** — the predictor changed
from a continuous per-word net score to a categorical per-slot direction. Only
raw-to-partial within a grain compares.

This was only possible because of the prompt repair: the passage corpus stored
prompts truncated to 60 characters, destroying exactly this join, and
`prompt_full` (560e44a2) restored it.

## Mechanism: one candidate measured and REFUTED

RH asked why a demoted word should still cost more once its aligned probability
is matched. The obvious candidate: **demotion generalises across contexts and
promotion does not**, so equalising probability at one slot would leave the
demoted word lower everywhere else. Measured, and it is false — the asymmetry
runs the other way.

Consistency = |n_fall - n_rise| / (n_fall + n_rise) per (pair, word), over words
with >= 20 mover cells, DISTINCT rows:

                      consistency mean   median    words
    fall-dominant           0.533         0.526    45,724
    rise-dominant           0.580         0.613    49,211
    per-pair fall - rise   -0.0361   n=152 pairs  p 0.0027  [59/152 positive]

**PROMOTION IS THE MORE CONSISTENT OPERATION.** A word alignment promotes tends
to be promoted wherever it appears; a word it demotes is demoted more
selectively. So the mechanism is dead, and it dies in a way that makes the
result harder rather than easier to explain: a site-specific demotion should
leave the word near its normal level away from that site, hence LESS surprising
elsewhere, not more.

**UNTESTED, and flagged as speculation rather than a reading:** the scope of a
demotion may be the SCENE rather than the site or the lexicon. Passages continue
the very prompts where movement was measured, so their contexts stay inside that
scene; a demotion inactive in unrelated scenes would look inconsistent across
prompts while holding throughout a passage grown from a scene where it is
active. That would reconcile the two observations, and it has not been tested.
The obvious discriminator — whether the contrast decays as a passage drifts from
its prompt — sits close to the passage-position territory this campaign closed,
so it wants a design rather than the nearest available query.

**Fences on this section specifically**: exploratory, no direction declared in
advance, and computed over ALL 152 movement pairs rather than the 36 of the
mediation — it characterises `movement`'s structure, not this population.

## Superseded en route, so nothing stale is quoted

- **The classified mover sets.** Bare majority gave movers 72% of all tokens;
  supermajority-plus-magnitude still admitted `the` in 12 of 33 pairs. The unit
  was wrong, not the threshold. All M1/M2/M3 numbers from those passes are VOID,
  including a strong-looking 31/33 composition result.
- **The 33-pair figures** (rho -0.278). An `is_fast` gate rejected 4 of 42 pairs
  for offset mapping this producer stopped using; 3 return, Teuken fails the
  real check.
- **"The tautology objection fails" (b8ab701a).** Premature: the level term
  shows the effect is NON-LOCAL, which is a smaller claim. Result 4 is what
  answers it.

## Fences

- **Single pass, one seat, no second-seating.** Nothing here is audit-grade.
- **CANONICAL's fallers are NOT null-tested** — risers are, fallers are a bare
  ratio rule. Nothing here may describe fallers as though they were tested.
- **Roster**: 46 declared, 42 with passage generations, 36 surviving text
  reconstruction. The 6 drops are DROP-RETOK; Aquila has a genuine tokenizer
  divergence. **Independence is NOT a limit here**: verified against
  `data/lineage_map_models.json`, the 42 passage pairs are 42 distinct lineages
  and the 36 are 36, with no lineage contributing more than one pair (@malign,
  [5882]). Contrast [5872], where the same-sounding worry was real because a
  representative BASE in `movement_edges` can carry several aligned children —
  different table, different unit, opposite answer.
- **LEVEL IS MEASURED ONLY WHERE THE WORD WAS EMITTED**, which is selection on
  the outcome. Displaced words appear less often, so the occasions where they
  still surface may be contexts that demand them unusually strongly. The forced
  ladder does not have this problem because it makes the model emit the word;
  this instrument cannot, and no amount of data fixes selection on the outcome.
  **This is the limitation to carry, in place of the independence one.**
- **No presence/increment control.** [5880] separates site selection from
  coupling (x4.7 versus x1.44 on kill/scream); this instrument has no equivalent
  and would be strengthened by one.
- **`p_aligned` is floored at theta** (0.001) before log, because twp is
  theta-truncated and p=0 means "below theta", not zero. An epsilon of 1e-9
  turned `murder`'s 44 zero cells into a fabricated 10.5 nats.
- Two tests were proposed and dropped, both correctly: the DOMAIN SPLIT
  (displacement is not domain-sensitive, so a flat profile refutes nothing — the
  test is asymmetric) and a POSITION-BINNED test (walks back into the
  passage-opening trap this campaign already closed).

## Producers

    meta/M06_generation/scripts/m06_mediation.py        stage 1 (--by-prompt)
    meta/M06_generation/scripts/m06_mediation_read.py   decomposition
    meta/M06_generation/scripts/m06_mediation_corr.py   continuous correlation
    meta/M06_generation/scripts/m06_mediation_ctx.py    per-context control
    meta/M06_generation/scripts/m06_mediation_contrast.py
        PRODUCER DEBT, DISCHARGED 2026-08-14. The level-by-class table, the
        common-support diagnostic, the contrast on common support (this
        finding's surviving claim) and the consistency asymmetry were all
        computed inline with `uv run python -c` and existed in no script. Per
        RH's ruling relayed at [5890], a missing producer makes a number
        UNAUDITABLE, which outranks the work that surfaced it. Written the same
        night and verified to reproduce every published figure to the digit
        rather than merely to resemble it.
    plans/plan_mediation.md   committed d79b6c0f, BEFORE any producer existed
