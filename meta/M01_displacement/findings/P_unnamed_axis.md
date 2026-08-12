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
      trees            +0.0038 / +0.0029    +0.0102 / +0.0057

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

    English verbs, 2,760 words, 792,549 scored cells
      oracle (word identity)          0.7025
      log p_base alone, same cells    0.5818
      headroom for any word feature  +0.1207

    Chinese verbs, 836 words
      oracle 0.7261, p_base 0.6182, headroom +0.1079

**The eighteen rated norms buy about 7% of that headroom.** And the ceiling is
itself low: ICC(1) = 0.131 in English, 0.182 in Chinese, so **82-87% of the
fall/rise variance is WITHIN a word across the sites it appears at** and is
unreachable by any word-level feature. Movement is a property of the word AT A
SITE. A perfect word-level theory tops out near AUC 0.70.

## 3. A distributional embedding recovers three times as much

    what beats its OWN shuffle, pooled      share of the +0.1207 headroom
      18 rated norms, trees      +0.0083                7%
      GloVe 300d, trees, k=50    +0.0256               21%
      bge-m3 1024d, trees        +0.0208               17%

Two encoders that agree about nothing else land within 0.005 of each other. GloVe
is near-isotropic (median pairwise cosine 0.037) and bge-m3 is not (0.529); one
is 300 dimensions and the other 1024.

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

UNTESTED AND IT COULD UNDO THE REGISTER READING: the SUBTLEX index is a frequency
ratio and the axis correlates with log frequency at +0.220, so part of that
R2 0.199 may be frequency rather than register. Residualise the index on
frequency before decomposing. **This is the highest-value unrun analysis in P.**

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
