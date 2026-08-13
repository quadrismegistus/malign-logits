---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-12
role: finding
topics: [semantic-norms, prediction]
description: "The PREDICTION study split from K: held out by word, none of the eighteen rated norms predicts movement direction; word identity carries +0.121 AUC headroom (87% of variance within-word across sites); GloVe recovers 18-21% of headroom vs the norms' 7%; every name for the axis is a minority share (register survives frequency residualisation in narrowed form, [5604]); the nameable face has a provisional name, INTERIORITY / enacted->represented, worth ~a quarter (7b); zh replicates on an unrelated seam; Tulu ablations show no safety-specific effect."
---
# Findings P: the unnamed axis

**Status: THERE IS A WORD-LEVEL DIRECTION ALIGNMENT SORTS ON, IT IS NOT IN OUR
DESCRIPTIVE VOCABULARY, AND THE UNNAMED RESIDUAL OUTPREDICTS EVERY NAME WE HAVE
TRIED. Held out by word, none of the eighteen rated norms predicts which way
alignment moves a word. A measured ceiling shows this is not a dead design --
word identity carries +0.121 AUC of headroom over base probability -- and 300
unsupervised GloVe dimensions recover 18-21% of that headroom against the rated
norms' 7%. The direction replicates in Chinese on an unrelated historical seam.
Every named component of it (corpus register, concreteness, length) is a
minority share, and what is left after removing them predicts better than any of
them. Its nameable face now has a PROVISIONAL NAME -- INTERIORITY, or enacted ->
represented (section 7b): blind-rateable at rho 0.26-0.36 on held-out words, 77%
colinear with concreteness by the structure of the lexicon, sharpened past
concreteness by the perception wedge, replicated in generated prose by M02, and
worth roughly a quarter of the direction. The majority remains unnamed.**

**AND A SECOND INSTRUMENT THAT NEVER TOUCHES THE MOVEMENT RULE FINDS THE SAME
DIRECTION.** A classifier reading base from aligned out of word probabilities
reaches AUC 0.956 held out by ORG, its coefficient vector is stable at min cos
0.944 where the movement axis fails its own gate at 0.841, and the two agree at
Spearman -0.386. Section 6. It also resolves the apparent tension in this
document: no individual word's movement is predictable from its meaning, and the
same word probabilities identify the arm nine times in ten, because **the
signature is distributed and redundant rather than located in any nameable
word**.

**AND "WORD-LEVEL" IS THE WEAKEST WORD IN THE HEADLINE.** A univariate per-word
AUC -- one number per (word, in-context tag), computed identically for all 4,106
features, and the vector that agrees with the axis best at Spearman -0.461 --
reproduces on LITERARY prompts at only **+0.238**, with 423 genuine sign flips
(`right/noun` runs -0.36 to +0.28). A word's arm-diagnosticity is mostly a
property of the prompt population, not of the word. That is ICC 0.131 arriving by
a route that never touches the movement rule, and it bounds how much any
word-level claim in this document can be asked to carry. Section 6.

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
`k_dpo_corpus.py`, `k_tulu_ablation.py`, `k_residual.py`,
`k_armclf.py`, `k_armclf_binary.py`, `k_armclf_minimal.py`.

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
§9's population; the verbs/all-sites row is the denominator quoted elsewhere in
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
is 18-21%. The Chinese figure is a single draw and inherits the same OpenMP
nondeterminism, so 24% against 19% is inside the English band and the ordering is
not established. Read the tree rows only: the Chinese logistic increment climbs from
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

### 3b. A CELL-level feature, and the ceiling is not a wall

Everything above assigns one vector per WORD, so everything above is bounded by
section 2's oracle. **`V(prompt + word) - V(prompt)` is not** -- the same word
gets a different vector at every site. RH's proposal; three scripts, cheap check
first.

THE CONSTRUCT WAS PROBED BEFORE ANYTHING WAS SPENT (`k_delta_probe`). Bare-word
bge separates near-synonyms from unrelated pairs by 0.1382 at anisotropy 0.5569;
the within-prompt delta by **0.6027 at 0.1610**, beating GloVe's 0.4001 reference
with the encoder that was this study's worst instrument. The synonym cosine
barely moves (0.677 to 0.732); the UNRELATED cosine collapses, 0.539 to 0.129.
Subtracting the prompt removes the shared component, and anisotropy is a shared
component. Chinese 0.319 to 0.715.

    held out by WORD / by PROMPT      k_delta_predict, 1,487,514 cells, 4,064 words
      log p_base alone       0.5046   0.5172
      + site vector          0.5377   0.5382    <- the control
      + DELTA                0.6158   0.6517
      oracle headroom on this population  +0.1638

**68% and 82% of the headroom, against GloVe's 18-21% and the norms' 7%.** The
site control matters and it passes: a site vector alone buys +0.033, the delta
clears it by +0.078 to +0.114, and the delta scores HIGHER under the
prompt-disjoint split -- a model living on site identity cannot transfer to
prompts it never saw. The oracle here is computed on these rows, not read from
section 2, after a first version compared a 4,064-word population against a
2,760-word ceiling.

**IT DOES NOT BEAT THE ORACLE (0.6517 against 0.6821), so the ceiling stands as a
ceiling.** And the delta contains word identity, so this is mostly evidence that
it is an excellent WORD feature.

THE NARROW TEST IS THE ONE THAT MOVES SECTION 2. Hold the word fixed and ask
whether the projection orders that word's OWN falling cells above its own rising
ones. A word-level feature scores exactly 0.5 by construction.

    within-word, 2,372 words with both classes and >=10 cells
      delta projection   median 0.5254   weighted mean 0.5402   61% of words >0.5

**So the 87% is reachable, and barely.** Section 2's "unreachable by any
word-level feature" is exactly true and slightly misleading: it is a ceiling on a
CLASS, not on the variance. `log p_base` is not a floor for this test despite
varying within a word -- the movement rule defines a faller relative to p_base,
so it selects on its own predictor, as section 1 says of the binary outcome.

AND IT HAS NO NAME EITHER. Word-mean projection correlates with length **-0.353**
(the strongest NAMED correlate, and section 7 shows length is removable from the
axis at zero cost), concreteness +0.244, frequency +0.205, bodily_harm +0.177,
register -0.138, and **charge +0.002** -- the third independent instrument to
find the campaign's headline construct orthogonal to what predicts.

### 3c. Four instruments, one convention, and the vocabulary is not the finding

`k_instrument_poles` puts every word-scoring instrument on one orientation
(fall/base = high; the ARM outcome runs backwards, which is why agreement with it
is reported as a negative correlation) over the 1,708-word shared vocabulary.

    pairwise Spearman        0.465 to 0.649
    top-100 pole overlap     22 to 46 of 100

No two instruments share more than half their poles, including two versions of
the same axis differing only in encoder at 46/100. **THAT IS NOT A DISAGREEMENT,
AND READING IT AS ONE WAS AN ERROR THIS SECTION ORIGINALLY MADE.** Under a
single-latent-direction model -- two variables correlated at rho, both ranked,
top-100 of 1,708 -- the expected overlap is:

    pair               rho    predicted        observed fall/rise
    arm   GloVe       0.465   24.0 +- 4.0        30 / 23
    arm   bge         0.508   26.2 +- 4.0        28 / 22
    arm   delta       0.495   25.6 +- 3.9        32 / 26
    GloVe bge         0.643   36.0 +- 4.1        46 / 32
    GloVe delta       0.562   29.8 +- 4.0        30 / 23
    bge   delta       0.649   36.2 +- 4.2        38 / 39

**Every observed overlap meets or exceeds its prediction; none falls below.** The
tails of a noisy measurement are its least stable part, so a quarter to a half is
simply what top-100 agreement looks like at these correlations. The instruments
are consistent with measuring ONE direction with different noise, and the word
lists diverge by as much as the arithmetic requires and no more.

The semantic character is identical in all four: bodily and vernacular action against
procedural deliberation. `delta` fall = `punching stabbing kick threw killed
smacked`; `delta` rise = `qualify contemplated engage evaluate assessed ensure`;
GloVe fall = `drop fuck whacked dump lick bury shoot`; GloVe rise = `explore
automate prioritize communicate respond verified`.

**The characterisation is the invariant; the vocabulary is not.** A specific
hundred-word list is an unstable SAMPLE of a real direction -- unstable because
tails are, not because the instruments disagree -- so a figure quoting one as the
finding quotes the sampling noise along with it. One cluster is delta-specific and worth its
own look: manner-of-speaking verbs on the rise side (`murmured whispered mumbled
snarled snorted`), which the arm AUC also carries (`whispered 0.858, muttered
0.787, barked 0.768`) and GloVe does not.

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

## 6. A second instrument, sharing no machinery, finds the same direction

### 5b. Chinese, and the aligned pole is the one that crosses languages

The per-word AUC runs on zh: 13,691 cells, **34 models over 17 lineages and 13
orgs**, against English's 92/46. A third of the n, and every number below is the
noisy side of the comparison.

    1,212 features   AUC 0.343 / 0.460 / 0.540 / 0.623 / 0.737   (5,25,50,75,95)
                     observed median 0.540 | arm-flip null 0.516

    BASE-SIDE   自己 0.194  喜欢 0.208  去 0.208  已 0.211  再 0.214  哭 0.225
                往 0.228  不 0.239  到 0.242  打电话 0.251  叫 0.256  能 0.256
    ALIGNED     关注 0.870  编写 0.863  需要 0.862  寻求 0.853  采取行动 0.844
                质疑 0.843  采取 0.830  准备 0.827  无法 0.824  进行 0.820

**It agrees with the Chinese movement axis at Spearman -0.578, stronger than
English's -0.461**, and the reading is the English one in an unrelated lexicon:
deixis, motion and bodily action against institutional-procedural verbs.

THE POS DECISION DOES NOT DRIVE IT, WHICH HAD TO BE CHECKED BECAUSE IN CHINESE
THE FEATURE IS NOT WELL DEFINED. Our word unit is `twp.py`'s tokenizer boundary,
not a Chinese lexicon, so pkuseg isolates the candidate in only **71.6%** of
132,318 pairs and the failure is structured -- 93.7% at one character, 21.1% at
three, and **46.0% of the 26,887 word types are never isolated at any site**.
Dropping the unisolated ones selects on candidate length, which correlates with
word class. So all three readings were run and they agree at Spearman 0.984 to
0.995: `word_auc_zh_nopos.tsv` (no tag, every word), `word_auc_zh_exact.tsv`
(tagged, isolated only), `word_auc_zh.tsv` (tagged, all). The axis agreement is
-0.578 / -0.607 / -0.567 across them. **Quote the no-pos table**; the tag adds
nothing here and costs half the vocabulary to defend.

**DO THE TWO LANGUAGES' ALIGNED POLES OCCUPY THE SAME REGION OF ONE SPACE? ON THE WEAK NULL YES, ON THE STRONG NULL NOT DEMONSTRATED.**
`k_pole_bridge.py`, bge-m3 for both languages (checked to be one `model` string,
and each language centred on its own mean first -- multilingual encoders separate
languages far more strongly than anything within one, so raw cosines would make
the test unfailable):

    cos(zh aligned, en aligned)  +0.5451     cos(zh aligned, en base)  +0.1771
    cos(zh base,    en base)     +0.3025     cos(zh base,    en aligned) +0.1554

    contrast, aligned  +0.3680     uniform permutation p < 0.001
    contrast, base     +0.1471     uniform permutation p = 0.0505

**AND THAT NULL IS TOO EASY TO BEAT, WHICH THE STRONGER ONE SHOWS.** A uniform
permutation builds semantically INCOHERENT zh sets whose centroids partly cancel,
so the contrast can be rewarded for coherence rather than correspondence. The
second null draws COHERENT sets not selected by arm -- a random seed word and its
100 nearest zh neighbours -- and also absorbs the register objection, since some
seeds land in institutional-procedural vocabulary:

    COHERENT null, 500 draws
      aligned contrast +0.3680 beaten by 5.0% of them    p = 0.0519
      base    contrast +0.1471 beaten by 38.6%           p = 0.3872
      strongest coherent draw +0.4233, seeded on 期待 "expect"

**So the cross-language pole correspondence does not survive its own control, and
the claim is withdrawn to a much weaker one.** A randomly seeded coherent Chinese
cluster reaches this correspondence one time in twenty, and one of them beat it
outright. What can be said is that the zh aligned pole sits in a region of the
shared space that leans toward the en aligned pole at roughly the 95th percentile
of coherent Chinese clusters -- suggestive, one-sided, and not a result. The base
pole shows nothing at all (38.6%).

The coherence measurement that motivated the second null is itself worth keeping,
because it argues against the easy dismissal too: centroid norms are aligned
0.1237, base 0.1625, random-100 0.1015. **The BASE pole is the more coherent of
the two and has the WEAKER contrast**, so whatever is happening on the aligned
side is not simply that our selection produced a tight cluster.

WHAT WOULD SETTLE IT. The coherent null may be too conservative in the other
direction: a seed-plus-neighbours set is maximally tight, while an AUC-selected
pole is not selected for proximity at all, so the comparison asks our pole to beat
sets built to be coherent. A matched-dispersion null -- coherent draws constrained
to the observed pole's centroid norm -- is the test that would separate these, and
it has not been run. The cross-lingual question is instead answered below by a
route that does not use embedding geometry at all.

### 5c. The bridge as a declared instrument: blinded translation

Every multilingual embedding space is TRAINED to place translation equivalents
together, so a cross-language geometry test partly measures its own aligner. The
replacement makes the bridge explicit and auditable: four blinded translators,
one pole each, shuffled, no scores and no pole label, one equivalent per item --
then look up each translation's arm AUC in the other language's table. Lists,
blinded copies, the prediction recorded BEFORE any translation existed, and the
result are in `pole_lists.md`.

    pole          ->    n    median    dev from centre    predicted   got
    en base       zh    63   0.4498      -0.0900            LOW       low
    en aligned    zh    28   0.6972      +0.1574            HIGH      high
    zh base       en    57   0.3757      -0.1309            LOW       low
    zh aligned    en    69   0.5144      +0.0078            HIGH      ~zero

**AND THE FIRST TWO NULLS WERE BOTH CONFOUNDED BY LENGTH**, which is worth
recording because the confound is the same shape as the coverage problem. AUC
correlates with character length in both tables (+0.253 zh, +0.272 en) and the
surviving translations differ in length by pole, so a random-word null and a
label-permutation null each destroy pole and length together and cannot separate
them. Controlling for length directly:

    OLS  AUC ~ pole + chars
      A/B -> zh (n=91)    pole +0.1912 (t +5.46)   chars +0.0772 (t +2.37)
      C/D -> en (n=126)   pole +0.0851 (t +2.96)   chars +0.0608 (t +6.85)
    gap vs LENGTH-MATCHED random draws: p = 0.0001 both directions

**The pole term survives, so translation carries information the tokenizer
artifact does not explain.** On the en side length is the larger term (t 6.85
against 2.96), so for zh->en length explains more than pole does even though pole
is real.

WHAT IS QUOTABLE AND WHAT IS NOT. The pre-declared coverage floor BINDS on the en
aligned list at 28%: the length control shows the 28 survivors behave as
predicted and says nothing about the 72 that never entered, which are
systematically the multi-character compounds that pole is made of. And `zh
aligned -> en` is +0.008 on 69% coverage -- a real null, not a coverage failure --
so the C/D pole coefficient is carried almost entirely by the BASE side.

**THE ASYMMETRY IS THE FINDING AND IT IS NOT NOISE.** By majority in-context tag
the en aligned pole is 89% verbs; the zh aligned pole is 38% verbs and 33% NOUNS,
and the nouns are concrete -- 面包 bread, 木头 wood, 香料 spice, 糖 sugar, 盐 salt,
鹿 deer, 长椅 bench. Translated into English those land at the English centre,
which is exactly the +0.008. **The two aligned poles are not the same object**:
English's is a near-pure class of institutional-procedural verbs, Chinese's is
that class plus a large concrete-noun component with no English counterpart. So
the correspondence is PARTIAL AND DIRECTIONAL, and a single symmetric statistic
would have averaged a real effect with a null.

The cleanly supported claim is the BASE one, which inverts the withdrawn
geometric result -- consistent with bare-word bge being the weaker instrument
(English synonym gap 0.138 against GloVe's 0.400) rather than with either pole
being special.



**Everything above depends on the canonical movement rule.** The rule thresholds
a p_base-relative quantity, and several defects corrected in this document came
from exactly that. So the direction wants corroboration from an instrument that
has never heard of the rule. `k_armclf`, `k_armclf_binary`, `k_armclf_minimal`.

THE DESIGN. Rows are (model, prompt) cells, columns are the top-N content words
by pooled mass, values are that word's probability, and the label is the ARM.
Nothing is paired, nothing is thresholded, no word is excluded for sitting below
a floor.

THE UNIT, STATED BEFORE THE RESULTS BECAUSE IT IS THE SAME TRAP AS SECTION 1.
204,438 cells but **92 independent labels** -- the label is exactly constant
within a model -- in 46 lineages. And the lineage is NOT the right holdout: 21 of
46 lineages share an ORG with another (tiiuae 5, allenai 4, six more with 2), so
holding one out leaves siblings in training and the classifier can recognise the
org rather than the arm. Held out by org, 33 groups. **The leak was real and
small: 0.9560 by lineage, 0.9371 by org.** All figures below are model-level, over
the 92.

### Most of it is one number, and it is not a word

    top-1 mass          0.8885        entropy         0.7368
    support size        0.8284        stored mass     0.4664   (chance)
    all four together   0.8563   <- WORSE than top-1 alone

**The probability of the single most likely next token identifies an aligned
model at AUC 0.889.** The four-feature nuisance block is worse than that one
feature, because stored mass is at chance and drags the fit. A hundred content
words reach 0.9371, misclassifying 8 models of 92 against 13.

### Removing the confound by construction beats controlling for it

Binary top-N BY RANK: each cell contributes exactly N ones, so mass, top-1 and
support size are constant and cannot signal. Collapsed to per-model fractions --
92 rows, the honest unit -- top-20 at k=50 reaches **AUC 0.9560, 92.4%**, above
the probability version's 0.9371 on the same org holdout.

**AND THAT FIGURE IS ONE CELL OF AN 18-POINT GRID THAT NOTHING PRE-DECLARED.**
The sweep is depth of binarisation (top-N in 5, 20, 50) crossed with feature
count (k in 5 to 200), and no registration, spec or docstring names a primary
cell -- checked, not assumed. Across the grid model-level AUC runs 0.8147 to
0.9575, and top-20/k=50 is neither the best nor the worst:

    per-model AUC        k=5     10      25      50      100     200
      top-5            0.8147  0.8544  0.9026  0.9362  0.9438  0.9390
      top-20           0.8530  0.8686  0.9286  0.9560  0.9537  0.9480
      top-50           0.8889  0.9036  0.9376  0.9575  0.9565  0.9353

**So quote the plateau, not the maximum.** Everything at k>=25 lies in
0.903-0.958 with median 0.941, and the depth of binarisation barely matters
there; the honest single sentence is "above 0.9 model-level for any feature set
of 25 words or more, best 0.958". A reader given 0.9560 alone is being handed a
selected cell, which is the same defect as the sufficiency comparison this
document already refuses to quote. So the signature is about WHICH WORDS ARE IN CONTENTION,
not about how peaked the distribution is.

"Present in the stored table" would NOT have worked in place of rank: the stored
count is threshold-determined and predicts the arm at 0.828 on its own, so a
presence-binary smuggles sharpness back in through the denominator.

### How few words: the count is a claim, the list is not

The five commonest content words -- `said found took put began` -- reach 0.8998
label-blind, misclassifying 10 of 92. Selection inside each training fold does
better at tiny k (0.9249 at k=3) and WORSE at large k. But fold agreement is
Jaccard 0.23-0.33 at k<=5: one word is chosen by all five folds at k=5, three at
k=10. **The signature is redundant, many small subsets work, and publishing "the
ten diagnostic words" would misrepresent it.**

### The coefficients, and their agreement with the axis

k=50, per-model, leave-one-org-out. **Stability min cos 0.958, median 0.994** --
the most stable direction in this campaign, where the movement axis fails its own
0.9 gate at 0.841.

    ALIGNED  whispered +0.41  let +0.34  felt +0.32  added +0.25  tried +0.23
             set +0.21  began +0.17  smiled +0.17  stood +0.16  held +0.15
    BASE     know -0.31  sent -0.30  threw -0.29  kill -0.28  went -0.28
             told -0.25  go -0.24  put -0.24  get -0.24  kissed -0.22

**These are NOT the marginal differences.** `said` has the largest marginal gap
(30.8% of prompts in base against 24.3% aligned, -6.5) and no top-ten
coefficient: `told`, `went`, `go` and `put` absorb it. Reporting marginals as
coefficients would have named the wrong words.

    Spearman(coefficient, word's axis position)   -0.385
    cos(coefficient direction, movement axis)     -0.236

**Negative IS agreement** -- a word that falls under alignment should be less
present in the aligned arm -- and it is stronger than the probability version's
-0.285 / -0.203. Two instruments sharing no machinery, one built on the movement
rule and one that has never heard of it, point the same way.

### Where in the lexicon it lives, and what the prompts decide

Ranking each part of speech by pooled mass and walking down it in bands of 50,
scored per-model against each band's OWN within-lineage flip null. 240 bands.

    IN-CONTEXT POS       hits      AUC     NULL       gap
      verb    0-50       6.18   0.9598   0.5028   +0.4570
      verb    500+      0.003   0.7901   0.5000   +0.2826   median of 90 bands
      noun    0-50       0.25   0.9286   0.5165   +0.4121
      noun    500+      0.001   0.7133   0.4771   +0.2329   median of 90
      adjective 500+     0.000   0.5000   0.5000   +0.0000   median of 14
      adverb    500+     0.000   0.5000   0.5000   +0.0000   median of 6

**The signature reaches rank 5000 in verbs and nouns.** Words appearing in three
of a model's 2,220 cells against none of another's still clear their null by
+0.23, because a rate over 2,220 prompts is precise even for rare events. This is
the strongest form of the redundancy result: it survives not only dropping to
five words but restricting to fifty RARE ones.

**THE ADJECTIVE AND ADVERB FLOOR IS VOCABULARY EXHAUSTION, NOT A POS EFFECT.** In
context there are 5,000+ verb and noun (word, tag) types and only 1,156
adjectives and 769 adverbs. Their deep bands are the tail of a small vocabulary,
so "dies past rank 500" and "runs out past rank 500" are not separable here. An
earlier version of this section reported the adverb column as an unplanned
negative control. It is not one.

THE POS IS IN-CONTEXT AND HAS TO BE. `fields._byu()` returns the most frequent
reading of a word FORM, and over all 365,892 (prompt, word) pairs its "noun"
label is **41.2% verbs** and its "adjective" label only **40.5% adjectives**
(verb 97.3% and adverb 94.8% hold). The first version of this sweep had a "noun"
band containing `fall break kiss punch strike stroke touch change work sign dance
tear love` -- verbs at sites like "She began to ___". The feature is therefore a
(word, in-context tag) PAIR, so `kiss`-as-verb and `kiss`-as-noun are separate
columns and each band is a population of usages rather than of forms.
`results/k/pos_context_en.tsv`, spaCy, keyed by sha16 of the prompt text.

**AND THE PROMPTS DECIDE WHICH WORDS CAN APPEAR AT ALL, which conditions every
word list in this section.** Measured over the top-20 slots:

    per top-20            LITERARY (novel snippets)   M01_PAIRS (designed)
      verb                     2.98                       11.88
      noun                     4.06                        0.37
      adjective                1.18                        0.13
      adverb                   1.62                        1.42

The designed minimal pairs are **59% verbs and 2% nouns** in contention and they
outnumber the literary cells 14:1. So `whispered felt threw kissed` being the
signature words partly reflects that these sites elicit verbs. **The
classification results are unaffected** -- they never used POS -- but any claim
about WHICH words carry the signature is a claim about this prompt mix.

**RUN ON THE LITERARY PROMPTS ALONE, THE POS ORDERING REVERSES:**

    median gap over that POS's bands   LITERARY        ALL PROMPTS
      verb                             +0.1371 (26)    +0.2916 (100)
      noun                             +0.2049 (30)    +0.2423 (100)
      adjective                        +0.1267  (9)    +0.2504  (24)
      adverb                           +0.2864  (6)    +0.2022  (16)

Adverbs go from weakest to strongest and verbs from strongest to weakest. Nouns
are the only class stable across both.

**SO THE POS ORDERING IS NOT A PROPERTY OF THE MODELS AND THIS SECTION SHOULD NOT
BE READ AS SAYING WHERE IN THE LEXICON THE SIGNATURE LIVES.** It reorders under a
change of prompt set, so the question is answerable only relative to a prompt
mix, and the two mixes available disagree. What survives both is the LEVEL: every
POS clears its null by a wide margin in both sweeps.

The literary arm restrains itself further. Its 97 prompts per model against 2,220
leave the nulls visibly unstable -- 0.6248, 0.6522, 0.4164, 0.4121 across the
bands above, where the full-corpus nulls sit near 0.500 -- so individual bands
are not readable there and the adverb median rests on six of them.
`armclf_bandsctx_en_literary.jsonl`.

TWO CONFOUNDS WERE FOUND AND FIXED IN THIS SWEEP, both recorded in the superseded
files rather than deleted. Fixing the top-N at 20 makes each CELL contribute 20
slots, but the SHARE of those landing inside the candidate set still varied by
model -- base 13.656, aligned 13.923 -- and separated the arms at AUC 0.750 on
its own, which every band inherited (`armclf_bands_en.CONTAMINATED.jsonl`). Rows
are now compositions. And there was no per-band null, so a band could not be told
from its own chance level (`armclf_bands_en.NONULL.jsonl`).

**The fix for the first disabled the diagnostic that found it.** `mean_hits` was
computed after normalisation and silently began reporting a share rather than a
count, so the contradiction that exposed the scale confound -- coverage 0.08 at
AUC 0.949 -- would not have been visible on the next run.

### The band coefficients are not one vector, and what is

The sweep fits 240 separate models, and **their coefficients cannot be compared
across bands.** Each fit spends a fixed L2 budget across its own fifty features,
so a band holding little signal still spends it; correlated features inside a
band split their weight, so a redundant band pays each of its words less than an
independent one does; and standardisation is within band within fold, so the
units differ. Normalising a band's vector to unit norm buys "share of THIS band's
direction" -- readable as relative importance inside the band, still not as effect
size across bands. Concatenating the 240 would be a category error, and the single
global fit is unavailable at 15,832 features against 92 models.

What IS comparable is a univariate statistic computed identically for every
feature: the per-model share of that model's top-20 slots, then the AUC of that
one number across the 92 models. Bounded, no penalty, no feature set to be
relative to. `k_word_auc.py`; `results/k/word_auc_en.tsv`, all 4,106 features
present in at least 20 models, and `word_auc_en_literary.tsv`.

    4,106 features   AUC 0.308 / 0.423 / 0.503 / 0.587 / 0.715   (5,25,50,75,95)
                     900 (22%) separate the arms by more than 0.15

    MOST BASE-SIDE          MOST ALIGNED-SIDE
      went/verb     0.104     provide/verb    0.920
      told/verb     0.110     provided/verb   0.906
      kill/verb     0.112     inform/verb     0.901
      put/verb      0.118     discuss/verb    0.878
      back/noun     0.121     avoid/verb      0.872

**This is the vector to quote, and it agrees with the axis at Spearman -0.461
over 1,798 features** -- better than the norms, the coefficients, or anything
else in this document. The base pole is deixis and bodily action in the past
tense; the aligned pole is a near-closed class of institutional-procedural
infinitives. Median |AUC - 0.5| by POS: verb 0.093, adverb 0.095, noun 0.071,
adjective 0.071.

RECOMPUTED 2026-08-13 AFTER A DEFECT IN THE INPUT, and the numbers above are the
corrected ones. `twp_words` sorts on `(model, prompt, word, SOURCE)`, so `FINAL`
collapses the storage key and leaves the analysis unit: every producer here read
it raw, a repeated word took two of the twenty slots, and **17.76% of cells held
fewer than 20 distinct words in their top-20** (mean 17.99, worst cell one word
in all twenty). It was ARM-LINKED -- base 17.88 distinct against aligned 18.10,
Mann-Whitney p=0.031 -- so it was a live confound in an arm contrast, and one
that row-wise compositional normalisation cannot reach because it changes WHICH
words hold the slots rather than the scale. Docket [5657], [5659].

**The correction moved the numbers and not the conclusion**: old against new is
Spearman 0.985 over the 3,977 shared features, mean |delta| 0.0136, and the
top-100 lists overlap 86 and 90 of 100. The agreement with the axis got slightly
STRONGER, -0.451 to -0.461, which is the direction I declined to predict before
measuring. The mechanism is legible after the fact: the 16,237 (prompt, word)
pairs that dedup promoted into the top-20 are **66.5% verbs**, and verbs carry
the most arm signal of any POS here, so the defect had been diluting the
strongest part of the vector.

**0.5 IS THE NULL FOR ONE FEATURE AND NOT FOR THE CENTRE OF THE TABLE.** Rows sum
to 1, so one arm concentrating on a minority of words pushes every other word
slightly toward the other arm. On the full prompt set that see-saw is quiet
(median 0.503, arm-flip null 0.507). On LITERARY prompts it is not: median 0.576,
and the tilt grows with how widely a word is shared (0.532 for features in 20-39
models, 0.685 for those in 80+). The arm-flip null there is 0.496, so the tilt is
arm-linked rather than mechanical, and the reason is a single scalar -- per-model
concentration (row Gini) separates the arms at AUC 0.288 on literary prompts and
0.764 on the full set. **The direction of that scalar reverses between the two
prompt populations**: aligned models are the concentrated ones on designed
prompts, base models on literary continuations. Where the median is off-centre,
only the RANKING is quotable.

### And it is not frequency, which is a slice rather than a run

**Frequency needs no band sweep here.** The per-word AUC is univariate and already
computed for every feature, so frequency is a `GROUP BY` on an existing table;
re-running the statistic inside bands would make it worse, because the
compositional normalisation would go band-local and reintroduce the see-saw fifty
times over. What the MULTIVARIATE band sweep buys is the thing a univariate
cannot -- joint signal, a band classifying at 0.95 whose best single word is 0.62
-- and that is section 6's claim, which does need the fit.

Sliced (COCA fiction, 3,946 of 4,106 features carry a frequency):

    Spearman(log fpm, AUC)      -0.228     direction: commoner words are base-side
    Spearman(log fpm, |AUC-c|)  +0.090     strength: flat

    rarest quintile   n=789   median AUC 0.526   median |AUC-c| 0.073
                        789              0.540                  0.083
                        789              0.506                  0.081
                        789              0.490                  0.085
    commonest         n=790              0.453                  0.093

**Direction carries a weak frequency tilt and strength carries none**: over six
orders of magnitude the median |AUC - centre| moves 0.073 to 0.093, so
arm-diagnosticity is available at every frequency, which is why the sweep found
classifiable bands at rank 3000 as well as rank 50. The -0.228 is the share
frequency already owns -- `go get say know put back` against `escalate prioritize
reconsider` -- but both poles span the range, `nipples checkbook rupees sterling`
being rare and base-side, and the same vector agrees with the movement axis at
-0.461, twice the frequency correlation. On LITERARY prompts the tilt vanishes,
as it should if the Latinate-institutional pole carrying it is what
those prompts never solicit.

One detail cuts against an artifact reading. Rarer features are measured on fewer
models and should show LARGER spurious |AUC - c|; they show smaller. The mild
positive slope is attenuation at the rare end, not inflation.

### The same words, on literary prompts, are largely different words

Read against its own centre, the literary table is a weak echo of the full one.

    shared features                     1,034
    Spearman(ALL, LITERARY)             +0.238
    sign agreement about the centre     59.1%
    genuine sign flips                  310 base->aligned, 113 aligned->base

    HOLDS IN BOTH   base      back/noun, all/adverb, went, told, go, know,
                              first/adj, then/adverb, would, front/noun, was,
                              bought, have, head/noun, said, die, just, so
                    aligned   seemed, suddenly, speak, shared, felt, consider,
                              everything/noun, offered, vanished, talk, abruptly,
                              scattered, slowly, everyone/noun, forgot, need

    FLIPS           right/noun -0.36 -> +0.28   think/verb -0.31 -> +0.29
                    left/adj   -0.31 -> +0.27   say/verb   -0.35 -> +0.17
                    must/verb  +0.14 -> -0.24   disappeared +0.16 -> -0.20

The stable core is real and small: deictic and bodily past-tense verbs on the base
side, a manner-adverb and mental-state band on the aligned side. Everything else
is prompt-population specific. **A word's arm-diagnosticity is not a property of
the word**, which is the same fact section 2 records as ICC 0.131 and reaches here
by a route that never touches the movement rule. The literary arm's 97 prompts per
model against 2,220 make it substantially noisier, and its instability is what
0.238 measures as much as any real difference; it bounds the agreement from below
rather than estimating it.

### What this does to the rest of P

**RE-RUN 2026-08-13 UNDER THE DEDUP FIX** (see the block above; the raw read hit
this section hardest, because it contaminated BOTH the candidate-column selection
-- `sum(p)` double-counts a duplicated word into the top-N vocabulary -- and the
per-cell values). Per band the mean |delta| is 0.0457, three times the per-word
table's 0.0136. **Everything survived and the sweep got slightly stronger**: mean
band AUC 0.7315 -> 0.7376, real-minus-null +0.2395 -> +0.2475, arm classifier
0.9589 -> 0.9560, coefficient stability 0.9582 -> 0.9442, agreement with the axis
-0.3855 -> -0.3863. The POS medians moved most where n is smallest -- adjective
0.625 -> 0.683 on 24 bands, adverb 0.769 -> 0.734 on 16 -- which is one more
reason the POS ordering here is not quotable, as this document already says.

It corroborates the axis from a design that cannot inherit the rule's artifacts,
and it does so with a direction that is STABLE where the axis is not. It also
resolves an apparent tension rather than creating one: no individual word's
movement is predictable from its meaning (sections 1 and 9), and the same word
probabilities identify the arm nine times in ten. Both hold because **the
signature is distributed and redundant** -- present in the joint profile, absent
from any word one could name. That is the claim of this document reached from the
other side.

LIMITS. Only 43% of each cell's top-20 falls inside the 200 candidate columns, so
this is a slice of the vocabulary and widening it is untested. Accuracy figures
use a threshold chosen on the same predictions and are optimistic; the AUCs are
the honest numbers. And 92 models is a small n for a 5-point accuracy difference:
13 wrong against 8 is five models.

## 7. What the direction is made of, and none of it is most of it

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

## 7b. The provisional name, and exactly how much it covers: INTERIORITY

The direction now has a provisional name, arrived at by three routes that were
run in order and are kept in that order, because each fences the next.

**FIRST, THE NAME A READER WOULD GIVE IS REAL AND NOT A RORSCHACH.** 300 words
drawn at random from the shared vocabulary -- not from the poles -- were rated
blind on the poles' characterisation (unplanned/bodily/vernacular against
deliberative/procedural), rater seeing no scores, ranks or purpose
(`nameability_en.json`). The rating predicts words it was never derived from:

    vs axis/GloVe   rho +0.364      vs arm AUC   +0.311      vs delta   +0.261

**SECOND, THAT NAME IS 77% CONCRETENESS** (rho -0.769 against the coder scale on
the same words), and partialling concreteness out leaves +0.09 to +0.16. So the
surface nameability is mostly the one component human intuition is built to see,
and it was already in section 7's ledger at about a quarter of the prediction.

**THIRD, THE TAXONOMIES SHARPEN IT PAST CONCRETENESS** (`k_field_poles`, 112
fields, size-matched nulls, all four instruments one orientation, BH over 448
tests; 109 survive q<0.05 against ~22 expected):

    FALL / BASE                          RISE / ALIGNED
    wordnet:contact      z +8.77 n=307   wordnet:communication  -5.02 n=273
    matter/objects       +4.69           inquiry/education      -4.90
    consumption          +3.73           cognition (x2 sources) -4.40 / -4.37
    motion               +2.83           perception             -3.58
    body/health          +2.70           evaluation/modality    -3.44

**The wedge is `perception` rising**: perception verbs are concrete -- done with
the body -- and they rise with cognition. A pure concrete->abstract axis cannot
produce that row. The rise pole is the class of MENTAL PREDICATES (cognition,
perception, evaluation, communication), for which INTERIORITY is the working
name.

WHY NO FINER ADJUDICATION IS POSSIBLE, AND WHY THAT IS A FINDING RATHER THAN A
GAP. Interiority and abstraction are colinear by the construction of the
concreteness norm (mental events score low BY INSTRUCTION) and by the history of
the language (English packed mental, evaluative and institutional vocabulary
into one Latinate stratum). Residualising one on the other chases the thin
subpopulation where the lexicon lets them apart. What CAN be done is to ask
which name covers the measured exceptions, and each simple name fails on one
subpopulation that rises anyway:

    concreteness    fails on perception verbs (concrete, mental -- they rise)
    interiority     fails on the designed-prompt pole (provide, inform,
                    escalate: abstract, institutional, fully exterior)
    latinateness    fails on the literary-prompt pole (felt, seemed, wished,
                    dreamt: Germanic, and maximally interior)

The description surviving elimination is **ENACTED -> REPRESENTED**: what falls
is immediate bodily doing, what rises is action that passes through
representation -- a mind, a speech act, or an institution. That is M01's
deliberation-replaces-action (six lexicons, 684 pairs) reached by elimination.
And the Chinese replication (-0.578, section 5b) does specific work here: an
etymological-seam reading predicts NO replication in a language with an
unrelated lexical history; the functional reading predicts exactly what is
observed -- the same function deposited in a different vocabulary.

CONVERGENT, FROM GENERATED PROSE RATHER THAN LOGITS: M02's field analysis of
54,080 continuations over 26 pairs reads, in its own words, "off the concrete
particulars of narrative and onto interiority" -- 39 of 79 fields surviving,
DOWN personal names, places, objects, bodies, time; UP cognition, emotion,
communication, evaluation, and again perception
(`M02/field_signature_not_contradiction_specific.md`). Two grains, one
signature.

AND TWO FENCES FROM M02 THAT KEEP THE NAME FROM OVERREACHING. The field shift is
NOT contradiction-specific (0 of 79 with a form-matched control), while the
second-order ascent at contradiction is real (3.37x, controls 1.00) -- so the
interiority shift is ambient REGISTER, not a triggered response to content. And
the two do not covary across the 26 pairs (Spearman +0.080, p 0.695, MDE 0.391,
disjoint passages; `M02/results/ambient_vs_ascent.json`): **the vocabulary shift
and the metarepresentational operation are separately installed.** The name
covers the register, not the operation.

WHAT THE NAME DOES NOT COVER IS STILL THE MAJORITY. The blind rating's
predictive content is a quarter-share; the delta residual after every named
component still outpredicts the names; and the within-word 87% is untouched by
any of this, interiority included -- interiority is a property of words, and
most of what alignment does to a word is not about the word. The claim below is
AMENDED by this section, not replaced: the direction has a nameable face, the
face is interiority (enacted -> represented), it is worth roughly a quarter, and
the body remains unnamed.

## 8. External validation of the register construct

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

## 9. Sites must elicit verbs, and it changes the answer

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

## 10. Two causal tests, and neither supports a harm-targeting mechanism

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
alignment sorts on is mostly not in the descriptive vocabulary we have for
words.** Section 7b amends this claim by a quarter: the direction's nameable
face is INTERIORITY -- enacted bodily process falls, represented (mental,
perceptual, communicative, evaluative) process rises -- named blind, fenced by
its collinearity with concreteness, and covering roughly a quarter of the
prediction. The remaining three quarters, and 87% of the within-word variance,
answer to no name that has been tried.

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
