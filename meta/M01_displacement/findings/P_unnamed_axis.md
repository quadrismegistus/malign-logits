---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-12
role: finding
topics: [semantic-norms, prediction]
description: "The PREDICTION study split from K: held out by word, none of the eighteen rated norms predicts movement direction; word identity carries +0.121 AUC headroom (87% of variance within-word across sites); GloVe recovers 21% of headroom vs the norms' 7%; every name for the axis is a minority share (register survives frequency residualisation in narrowed form, [5604]); zh replicates on an unrelated seam; Tulu ablations show no safety-specific effect."
---
# Findings P: the unnamed axis

**Status: THERE IS A WORD-LEVEL DIRECTION ALIGNMENT SORTS ON, IT IS NOT IN OUR
DESCRIPTIVE VOCABULARY, AND THE UNNAMED RESIDUAL OUTPREDICTS EVERY NAME WE HAVE
TRIED. Held out by word, none of the eighteen rated norms predicts which way
alignment moves a word. A measured ceiling shows this is not a dead design --
word identity carries +0.121 AUC of headroom over base probability -- and 300
unsupervised GloVe dimensions recover 21% of that headroom against the rated
norms' 7%. The direction replicates in Chinese on an unrelated historical seam.
Every named component of it (corpus register, concreteness, length) is a
minority share, and what is left after removing them predicts better than any of
them.**

Split from `K_word_properties.md` on 2026-08-12. K is a MEASUREMENT study: an
instrument, its reliability, and what it correlates with in sample. This is a
PREDICTION study: held-out generalisation, a measured ceiling, and the
decomposition of a direction. **P supersedes K's interpretation and leaves K's
measurements intact.** K's correlations are real and survive their own
permutation nulls; they do not generalise to unseen words, which is a different
claim and does not make them wrong.

Producer chain: `k_predict.py`, `k_ceiling.py`, `k_embed.py`,
`k_predict_embed.py`, `k_axis.py`, `k_register.py`, `k_brooke.py`,
`k_confound.py`, `k_length.py`, `k_concreteness.py`, `k_scale_solo.py`,
`k_dpo_corpus.py`, `k_tulu_ablation.py`, `k_residual.py`.

---

## 1. The rated scales do not predict

The case is a CELL -- one (word, prompt, base, aligned) -- and the held-out unit
is the WORD, five-fold `GroupKFold`, so a model that has memorised `murder falls`
scores nothing for it. English lexical verbs only, so part of speech is not a
confound. 100,958 cells over 4,075 words.

    increment over the SAME model without the norms, and over the SAME model on
    ratings shuffled across words

                       over same class      over shuffled
      additive         +0.0021 / +0.0035    +0.0034 / +0.0054
      interactions     -0.0016 / -0.0017    +0.0035 / +0.0041
      trees            +0.0038 / +0.0029    +0.0102 / +0.0057   <- see below

**EVERY `trees` ROW IN THIS DOCUMENT CARRIES A SPREAD OF ROUGHLY 0.003 AND SHOULD
NOT BE COMPARED TO A LINEAR ROW AT THE THIRD DECIMAL.** `HistGradientBoosting` is
thread-nondeterministic (see §3), which affects `k_predict`, `k_predict_embed`
and `k_register`; `k_concreteness`, `k_length`, `k_scale_solo`, `k_confound`,
`k_axis` and `k_tulu_ablation` are linear or rank-based throughout and reproduce
exactly. So the trees row above at +0.0057 per-site is NOT distinguishable from
the additive row at +0.0054 -- do not read the tree model as having found
something the linear one missed here. The conclusion of this section does not
depend on which of them is larger: both are near zero.

**Adding the eight Warriner norms does not help**; on the 1,042 verbs Warriner
covers the additive model is WORSE than nuisance alone. **Chinese, run
separately, agrees** -- and in its covered subset `norms only` scores 0.478
against a shuffled 0.525, so the real ratings predict worse than random ones.

THE EFFECTIVE n IS 4,075, NOT 100,958. Eleven of twelve features are constant
within a word; only `log p_base` varies across a word's cells. The extra cells
inform the outcome, not the predictors. This is not a power problem: the learning
curve is flat from 652 training words to 2,608, which is also why leave-one-out
was not run -- it adds 20% to a curve that stopped moving three doublings before.

THE SCALAR OUTCOME AGREES AND IS THE BETTER INSTRUMENT. `log10(p_aligned/p_base)`
keeps magnitude and needs no exclusion of non-movers, where the binary outcome is
thresholded on a p_base-relative quantity and therefore selects on its own
predictor. It also shows what that selection costs: **within a site, p_base
predicts magnitude at Spearman +0.001**, against a per-site AUC of 0.66 in the
binary version. The nuisance variable's apparent within-site power is
manufactured by the binarisation.

## 2. There is word-level signal, so the failure is the instrument

Split a word's own cells in half and use the fall rate in one half to predict the
other. That is the best any function of the word alone can reach, since it uses
the word's identity, which strictly dominates any finite feature set.

    population                  oracle   p_base   HEADROOM     ICC   words
      all words, all sites      0.7065   0.5755    +0.1311   0.140   6,395
      verbs, all sites          0.7025   0.5818    +0.1207   0.131   2,760
      verbs, verb-eliciting     0.6971   0.5792    +0.1178   0.125   2,269
      Chinese verbs             0.7261   0.6182    +0.1079   0.182     836

**THE CEILING IS STABLE ACROSS ALL FOUR**, moving by 0.013 between the widest and
narrowest, so the "% of the headroom" figures below do not depend on which
population is used as the denominator. The verb-eliciting row is the one matching
§8's population; the verbs/all-sites row is the denominator quoted elsewhere in
this document.

THE SCRIPT WAS NOT DETERMINISTIC UNTIL 2026-08-12 AND THE FIX IS RECORDED HERE
BECAUSE THESE NUMBERS ARE CITED ELSEWHERE. Two identical invocations returned
headroom +0.1186 and +0.1174: `words` and the cells inside each word were built
by insertion from a ClickHouse result set with no ORDER BY, and the site key used
`hash()` on strings, which Python randomises per process. A fixed seed does not
make a run reproducible when the data order is not fixed. Both levels are sorted
now and repeated runs agree exactly; the deterministic recomputation reproduces
the previously committed English and Chinese values to four decimals, so nothing
downstream moves.

**The eighteen rated norms buy about 7% of that headroom.** And the ceiling is
itself low: ICC(1) = 0.131 in English, 0.182 in Chinese, so **82-87% of the
fall/rise variance is WITHIN a word across the sites it appears at** and is
unreachable by any word-level feature. Movement is a property of the word AT A
SITE. A perfect word-level theory tops out near AUC 0.70.

## 3. A distributional embedding recovers three times as much

    what beats its OWN shuffle, pooled      share of the +0.1207 headroom
      18 rated norms, trees      +0.0083                7%
      GloVe 300d, trees, k=50    +0.0229              ~19%   (range 18-21%)
      bge-m3 1024d, trees        +0.0208               17%

**THE GLOVE FIGURE IS A MEAN OVER FIVE RUNS, NOT A POINT ESTIMATE, AND AN EARLIER
VERSION OF THIS TABLE QUOTED THE TOP OF ITS RANGE.** `HistGradientBoosting`
parallelises through OpenMP and its result is thread-order dependent regardless
of `random_state`: five identical invocations of `k_predict_embed` returned k=50
tree increments of +0.0256, +0.0223, +0.0216, +0.0218 and +0.0231 pooled, a
spread of 0.0040 on a mean of 0.0229. The logistic rows are byte-identical across
the same runs, which is how the cause was localised. This was NOT the ClickHouse
row-order defect that `k_ceiling` had; adding `ORDER BY` to the fetch changed
nothing here.

So the headline is **about a fifth of the headroom, 18-21%**, and the comparison
that carries the finding is unaffected: the rated norms sit at 7%, outside any
plausible band around the embedding. **Do not quote 21%**; it was one draw and
happened to be the largest of five.

Two encoders that agree about nothing else land within 0.005 of each other. GloVe
is near-isotropic (median pairwise cosine 0.037) and bge-m3 is not (0.529); one
is 300 dimensions and the other 1024.

**CHINESE RECOVERS MORE THAN ENGLISH**, 26,692 cells over 2,005 words, bge-m3:
trees add +0.0246 to +0.0270 per-site over their own shuffle across k = 10 to
200, which against the Chinese headroom of +0.1079 is about **24%** where English
is 21%. Read the tree rows only: the Chinese logistic increment climbs from
+0.0317 to +0.0767 across the sweep, but the real model is flat (0.6546 to
0.6505) while the SHUFFLE collapses (0.6229 to 0.5738) -- the same widening-gap
artifact as English, where an increment grows because both of its terms fall.

ENCODERS ARE GATED ON BARE WORDS BEFORE USE, because docket [459] gate-checked
bge-m3 on `prompt + " " + word`, a SENTENCE, and a gate passed for one use is not
evidence about another. `k_embed` refuses to write if near-synonyms do not
separate from unrelated pairs on bare input.

    encoder        synonym gap   anisotropy   dims
    GloVe 300d        +0.400        0.037      300     <- primary
    bge-m3 en         +0.138        0.529     1024
    bge-m3 zh         +0.319        0.472     1024

**So the rated dimensions are not a coarse version of the right axis. They are
substantially the wrong basis.**

## 4. The direction

The GloVe direction predicting FALL, extracted with `log p_base` and `log fpm` in
the model as nuisance, so it is not the frequency axis:

    falls  drop fucks slams fuck whacked screw throws dump blow drink suck dies
           lick dunked ate stabbed puts eats peel bury sues
    rises  enhance collaborate explore standardize interact research automate
           formalize validate reinforce utilize elucidate analyze mobilize

and the nearest words to the axis in the FULL GloVe vocabulary, which is not
restricted to our verbs and so cannot be flattered by our sample:

    falls  'em ass gonna hey motherfucker dunk guts gotta crap grandma sucks
           okay fucking pub butt
    rises  interactions enhancement perceptual cross-cultural alliances pathways
           computer-based heterogeneous behaviors contexts mechanisms
           capabilities interventions interpersonal strategies cultures

**Coder `charge` correlates with this direction at +0.009.** The campaign's
headline construct is orthogonal to the direction that predicts. The sharpest
single case: **`exploit` and `manipulated` RISE while `fuck` and `stabbed` FALL**
-- both pairs carry transgressive content, and what separates them is not
transgression.

THE AXIS FAILED ITS OWN PRE-DECLARED STABILITY GATE. Minimum pairwise cosine
between the five fold axes was 0.841 in English against a gate of 0.9 set before
the run; en/bge is 0.826 and zh/bge 0.834. Consistent across all three, so not
selective, and too low to have passed a line that does not move afterwards.
**Every named direction here is provisional in that same way.**

### The gate failure is an n problem, not a site-noise problem

Refitted on the verb-eliciting minimal pairs only -- the population §8 shows is
where the per-norm question is well posed -- **stability gets WORSE**: min
pairwise cosine 0.816 against 0.841, on 83,182 cells and 3,379 words against
100,861 and 4,062. The restriction costs 18,000 cells and buys no stability,
which rules out site noise as the cause and points the remedy at more lineages or
more prompts rather than cleaner ones. The refitted axis is also MORE entangled
with the nuisance: log frequency +0.282 against +0.220, p_base +0.298 against
+0.273.

**But the naming shifts, toward concreteness:**

    falls   whacked fuck smacked stabbed ate ripped blow eats dumped suck
            drank hit shove smack shoot kills died bought fetch
    rises   enhance explore observe validate identify evaluate analyze describe
            demonstrates summarized verified reinforce cooperate visualized

Concrete physical action against abstract cognitive operation, where the
unrestricted poles read vernacular against institutional. `concreteness` becomes
the top-correlating scale at **+0.311**, ahead of `register_level` at -0.209;
`charge` is +0.010, unchanged. **The best description of the axis depends on
which sites it is fitted on** -- RH's collinearity objection arriving from the
population side rather than from lexical history.

## 5. Chinese replicates on an unrelated historical seam

    falls  讨厌 没命 住 厌食 厌烦 该死 掌 捡 叫 拖 弄 饿 抢 吓 闭 睡 蹲 锁
    rises  创作 探究 探寻 思考 探索 分析 反思 写作 应对 表达 撰写 阐述

Single-character physical colloquial words against two-character literary
compounds. English splits Germanic from Latinate; Chinese splits monosyllabic
native words from bisyllabic literary compounds. **Two unrelated seams, the same
behaviour, so etymology is ruled out as the explanation.** `charge` correlates
-0.084, `register_level` -0.149, concreteness +0.143.

The Chinese axis is markedly LESS entangled with the nuisance: log frequency
+0.053 and p_base +0.140, against English's +0.220 and +0.273. It is also about
four times stronger predictively (+0.068 over floor against +0.017) on a quarter
of the cells.

## 6. What the direction is made of, and none of it is most of it

Each construct given one CODER measure and one INDEPENDENT measure, on its own
coverage. Verb-eliciting sites for the prediction column.

    component                 n      R2 of axis   cos(axis,dir)   removing it costs
    SUBTLEX/acad register  5,735       0.1994        +0.373        HALF the prediction
    Brysbaert concreteness 2,916       0.1183        +0.241        a quarter
    coder concreteness     6,084       0.0921        +0.267        a quarter
    word length (en/bge)   6,120       0.1386        -0.213        NOTHING
    word length (en/glove) 6,084       0.0921        -0.211        NOTHING
    coder register_level   6,084       0.0470        -0.161        NOTHING

**AND THE TWO LEADERS CONVERGE ON THE VERB-ELICITING AXIS.** Decomposed against
the refit rather than the original, register's advantage nearly disappears:

    component                  unrestricted axis   verb-eliciting axis
      SUBTLEX/acad raw               0.1994             0.1998
      SUBTLEX resid on freq          0.1641             0.1529
      Brysbaert concreteness         0.1183             0.1400
      coder concreteness             0.0921             0.1105
      coder register_level           0.0470             0.0510

Register-after-frequency led concreteness by 0.046 on the original axis and by
**0.013** on the refitted one. The ordering does not flip, but "the largest
single named component" is a much weaker claim on the population where the
question is best posed, and should not be quoted without this row.

**LENGTH IS RULED OUT.** Projecting it out rotates the axis by under 13 degrees
(cos 0.977-0.990) and costs nothing predictively; in Chinese it slightly helps.
Length alone predicts +0.0017 in English.

**CONCRETENESS IS A REAL MINORITY COMPONENT.** Two measures agreeing at rho 0.79,
and Brysbaert's HUMAN norms give LARGER numbers than our coder, so it is not an
artifact of the rating instrument. But the concreteness-free axis retains three
quarters of the prediction and outpredicts concreteness alone by 2x to 20x.

**REGISTER SPLITS BY INSTRUMENT AND THE TWO DISAGREE.** The corpus genre ratio is
the largest single named ingredient found -- removing it costs half the axis --
while the coder scale is nearly orthogonal and removing it costs nothing. They
agree with each other at only |rho| 0.431.

**RUN 2026-08-12 AND IT SURVIVES**, commit `407fc503`, at registrar's ask
[5601] §3. The SUBTLEX index is a ratio of two frequencies and the axis
correlates with log frequency at +0.220, so the register naming was held PENDING
on whether its share was frequency wearing a genre ratio. Residualised on
`log coca_fic` -- deliberately the same frequency already in the nuisance block,
so the residual is measured against what the model already controls for:

    measure                    n      R2 of axis   rho w/ axis   cos(axis, dir)
      SUBTLEX/acad index    5,735       0.1994        +0.435        +0.373
      SUBTLEX resid on freq 5,734       0.1641        +0.397        +0.372

    removing it from the axis, over its own shuffle, verb-eliciting sites
      raw index         +0.0087 / +0.0075
      residualised      +0.0087 / +0.0095

Frequency accounts for about 18% of the index's share. The geometric alignment is
unchanged to three decimals and the predictive cost of removing it is the same.
**The largest named component of the axis is not frequency.**

THE QUOTABLE SENTENCE, as narrowed and booked at [5606]: *a corpus-measured
register ratio is the largest single named component of the axis, and it is not
frequency* -- with the stratum caveat below welded to it. "The axis is register"
unqualified remains a MAY-NOT-SAY, because the collinearity decay stands.

RH'S COLLINEARITY OBJECTION, and it is right. In the history of English the Latin
and Norman borrowings arrived polysyllabic and abstract while the Germanic core
stayed monosyllabic and concrete, so register, abstractness and length are one
stratum with several names. Near-synonym pairs do NOT hold abstractness fixed:
on Brysbaert the informal member of a `Choose the Right Word` pair is +0.337 more
concrete, Wilcoxon p = 3.3e-11. Under progressive matching the register index
goes 77.9% (253 pairs) to 75.6% (same syllables) to 72.2% (concreteness within
0.5) to **67.5% on the 40 doubly-matched pairs, 2.2 sd**. It survives and it
decays at every step.

ETYMOLOGY WAS NEVER ACTUALLY TESTED. The Latinate suffix flag correlates +0.10 to
+0.17 with the axis, the register index, concreteness and syllables -- outside a
cluster whose members correlate +0.34 to +0.83. A real Latinate marker would be
inside it, so the "orthographic form" result was mostly word length, and testing
Latinateness needs an etymological resource rather than a suffix regex.

## 7. External validation of the register construct

Brooke, Wang & Hirst (2010), hand-curated material only; their induced lexicon is
LSA plus frequency ratios and is circular for us.

    CTRW near-synonym pairs        pairwise      seed separation (AUC)
      coder register_level            97.1%          0.974
      register index                  76.8%          0.955
      orthographic form               74.1%          0.939
      glove axis                      76.6%          not scorable

**The coder scale measures register accurately and predicts movement not at all**
(last of ten in the per-norm solo, -0.0004). Those are compatible and together
they say the axis's predictive power does not run through that scale.

97.1% is on 68 of 399 pairs -- where both words are in our movement vocabulary --
so it is not head-to-head with the register index's 76.8% on 358.

THE CHINESE BROOKE SEED LIST IS CONTAMINATED AND WAS NOT USED. Twelve of its 49
"formal" entries are internet slang (酷毙 美眉 小强 酱紫 帅呆 弓虽 狂顶 东东 恐龙
菜鸟 大虾 马屁). As labelled, a quarter of the formal seeds are extreme informal
items and the natural conclusion would have been that register does not replicate
in Chinese. Trimming another group's data after seeing its effect on our result
is not a repair, so the file is reported unusable and left alone.

## 8. Sites must elicit verbs, and it changes the answer

Restricting the WORDS to lexical verbs is not enough. At "She slowly took off her
___" the verbs in the top-50 are long shots competing against nouns, and pooling
them with verbs at "He began to ___" makes the within-site ranking
incommensurable. The M01 minimal-pair corpus is verb-eliciting by design.

    TEN CODER NORMS, verb-eliciting sites, 83,257 cells, 3,392 words
    per-site increment over its own shuffle

      concreteness            +0.0055      valence_extremity       +0.0007
      bodily_harm             +0.0037      concreteness_extremity  +0.0004
      charge_extremity        +0.0018      charge                  +0.0004
      vulgarity               +0.0013      register_level          -0.0004
      transgressiveness       +0.0008
      valence                 +0.0008

On ALL sites the eighteen norms scatter symmetrically about zero and nothing is
distinguishable from noise. On verb-eliciting sites **nine of ten are positive
and concreteness leads at a 5x margin over the ~0.001 resolution.** The site
restriction is therefore a defect in every ranking in K, which was computed over
2,187 prompts including noun-eliciting ones.

THE EIGHTEEN-NORM VERSION HAS NO POWER HERE AND ITS RANKING IS NOT QUOTED. At 807
words `concreteness` scores pooled +0.0114 and per-site -0.0068, `n_concreteness`
+0.0068 and -0.0100 -- the same norm with opposite signs on the two metrics.
Warriner coverage is what costs the power.

## 9. Two causal tests, and neither supports a harm-targeting mechanism

**HH-RLHF preference direction.** `chosen` and `rejected` share their prefix, so
only the diverging suffix (25-33% of the string) is counted; counting the whole
string dilutes the contrast to nothing.

    subset          n words   rho w/ axis
    all                2162       +0.038
    harmless-base       765       -0.246
    helpful-base        878       +0.051

Harmlessness tracks the axis direction, helpfulness does not, and **pooling turns
-0.246 into +0.038**. But it does not predict out of sample (+0.0016 pooled), and
its poles are harm content plus refusal vocabulary, not register -- so the
-0.246 is most plausibly the overlap between vernacular and profanity. HH-RLHF is
also Anthropic data almost none of our lineages saw.

**TULU 3 SFT ABLATIONS**, four slices removed one at a time by AI2, all against
the full mix so the base is identical:

    removed     rho w/ axis   specific component (minus mean of the other three)
    safety           +0.227          +0.036   p 0.31    rank 3 of 4
    math             +0.152          -0.055   p 0.13
    persona          +0.135          -0.066   p 0.068
    wildchat         +0.110          +0.022   p 0.54

**Safety has no specific effect. But all four are positive**, so removing ANY
slice moves axis-positive words back up: every ablation partially undoes the
displacement, roughly in proportion to the slice's contribution. That is evidence
AGAINST harm-specific suppression and mildly FOR the axis being a property of the
instruction mix as a whole.

An earlier pairwise version of this test is superseded: it used one ablation as
another's control, which injected the math ablation's numeral effect, and its
contrasts were incomparable because the movement rule is asymmetric in base and
aligned.

## What not to quote

**The sufficiency comparison.** The one-axis model scores 0.6831 against the full
300-d model's 0.6670, but at different penalties (C=1.0 against C=0.1), so that
is regularisation and not dimensionality.

**The third decimal of any prediction increment.** Between two runs of
`k_register` the nuisance floor moved 0.0004 purely because the covered
population changed by 326 cells.

**`k_residual`'s register direction.** Built as the mean of 385 near-synonym
difference vectors, it is orthogonal to the seed-centroid version (cos -0.061)
and uncorrelated with the coder scale (rho -0.008); its "informal" pole is `get
going come got put go want know think`. It is a frequency direction. Committed as
a recorded negative.

**Any en/zh comparison that crosses encoders.** GloVe is English-only, so
cross-language claims must be bge-to-bge.

## The claim, stated plainly

We built a psycholinguistic instrument to name the property that determines which
words alignment suppresses -- 47,896 words, seven scales, validated against human
norms at 0.88 on concreteness and 97.1% on register -- and **it does not
predict**. Three hundred unsupervised dimensions that name nothing predict three
times better, in two languages, on unrelated historical seams. Every name we have
tried for the direction takes a minority share of it, and the unnamed residual
outpredicts all of them.

That is not a failure of the experiment. It is the result: **the dimension
alignment sorts on is not in the descriptive vocabulary we have for words.**

## The distinction this rests on, and its one named precedent

**Surviving a permutation null and generalising to unseen cases are different
claims, and a finding can do the first while failing the second.** Everything in
K does exactly that. The nulls in K are real: ratings are shuffled across words,
the coupling between words competing at a site is preserved, and the observed
statistic clears the shuffled distribution. None of that is evidence that the
scale predicts anything about a word the model has not seen.

**The campaign's one documented instance is F12, and it is a better citation than
this abstract phrasing.** Its original evaluation leaked training prompts into
the held-out set. Corrected, closure fell from 77% and 20% to **61% on Pythia and
4% on OLMo and Llama**, with train closure at 37-92% showing the gap was
generalisation rather than capacity. The second question was asked there, late,
and the answer took most of the finding with it.

Recorded because the pattern was nearly overstated: [5600] asserted "at least
three earlier results that were never asked the second question", [5601] booked
it as a debt, and [5602] withdrew it after checking -- **F13 and F03 turned out
to be counter-examples that asked the question properly**, F13 holding a finding
number open "for want of an out-of-sample corpus" and F03 reporting that its
r=0.43 is driven by cross-family differences with within-family correlations near
zero. One instance, named and checkable, is what the distinction rests on. It
does not need to be a campaign-wide pattern to be worth having.
