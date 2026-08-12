# Findings K: which properties of a word predict how alignment moves it

**Status: FOUR SCALES CLEAR A CALIBRATED NULL, AND THREE OF THEM ARE
SITE-SPECIFIC. The route here matters as much as the destination — four earlier versions of this analysis gave four
different answers, including two with the wrong sign, and one of them failed its
negative control outright.**

**AMENDED 2026-08-12 BY A PREDICTION TEST, AND THE AMENDMENT IS LARGE. Every
result above this line is correlational and in-sample. Held out by WORD, the
seven scales do not predict which way alignment moves a word: they add +0.003 to
+0.011 AUC over base probability, frequency and part of speech, and adding the
eight Warriner norms does not help. A measured ceiling shows this is not a dead
design -- word identity has +0.121 AUC of headroom -- and GloVe recovers 21% of
that headroom against the rated scales' 7%. The direction GloVe finds runs from
vernacular to institutional vocabulary, and it correlates with coder `charge` at
+0.009. Whether to call that direction REGISTER is unsettled: register,
abstractness and syllable length are historically fused in English and the axis
survives matching on the other two only at 2.2 sd on 40 pairs. See "PREDICTION,
THE CEILING, AND THE REGISTER AXIS" below. The sections above are not withdrawn;
they are correlations that survive their own nulls and fail to generalise to
unseen words.**

Producer chain: `scripts/rate_charge_v1.py` (the rater),
`scripts/k_population.py`, `k_bulk.py`, `k_analysis.py`, `k_by_prompt2.py`.
Prediction chain: `k_predict.py`, `k_ceiling.py`, `k_embed.py`,
`k_predict_embed.py`, `k_axis.py`.
Instrument frozen at `results/k/INSTRUMENT.txt`, sha `5b59a44c…`. Plan at
`registrations/plan_k_charge_annotation.md`.

---

## What was rated

47,896 words — 27,242 English rating units and 20,654 Chinese — each scored on
seven 1–7 scales by `deepseek/deepseek-v4-flash` at temperature 0, **one call
per word**, so blindness to movement is structural rather than promised: the
coder is never shown a direction it could condition on.

    vulgarity  register_level  transgressiveness  charge
    valence    bodily_harm     concreteness

Inter-annotator agreement against `claude-haiku-4-5` on 475 English words is in
`results/k/iaa.json` and **travels with every number here**: valence 0.90,
vulgarity 0.88, bodily_harm 0.88, charge 0.87, transgressiveness 0.83,
concreteness 0.83, register_level 0.60.

Calibration against human norms, which is what licenses the Chinese half:
coder valence against Warriner **0.81**, concreteness against Brysbaert
**0.88**, and Chinese concreteness against the 9,877-word Chinese norms
**|0.81|** (the published scale runs 1=concrete, so the raw figure is −0.81 and
reporting it unsigned would read as the instrument failing).

## THE RESULT

Per prompt, the unit inside a prompt being the WORD, movement averaged over 46
lineage representatives, ratings residualised on base probability and
register-matched corpus frequency (`coca_fic`), null by permuting ratings across
words, **null band from ten independent shuffles**:

    scale                REAL   null mean    null range      z    direction
    concreteness          412       100.8      72 - 173    11.0   86% FALLING
    charge                291       102.8      72 - 143     8.8   60% rising
    bodily_harm           284       122.9      68 - 168     5.9   90% FALLING
    transgressiveness     219       104.1      70 - 135     5.2   94% FALLING
    valence               212       135.9      96 - 250     1.7   —
    register_level        153       123.4      81 - 183     1.0   —
    vulgarity              82       102.2      51 - 170    -0.6   —

of 2,187 English prompts. **Mass leaves words that are concrete, harmful and
transgressive.** Transgressiveness fires at a tenth of all sites and points the
same way at 94% of them.

**RH's question was "do transgressive WORDS fall more due to alignment?" The
answer has two halves and the second one is the important one.**

**Yes, transgressive words fall** -- at 219 sites, 94% in that direction, and
only visible when the question is asked per site rather than pooled.

**And the SITE does modulate it, for three scales but not for
transgressiveness itself.** Paired within stem across 745 minimal pairs,
concreteness and bodily_harm are stronger at the transgressive twin at
p = 0.00005 and vulgarity at p = 0.0004, while transgressiveness (p 0.48) and
charge (p 0.70) are flat. See the site-specificity section.

## THE FULL FRAME, 18 SCALES, AND THE QUALIFICATION THAT MATTERS MOST

`results/k_frame_en.csv` (2,187 prompts x 18 scales) and `k_frame_zh.csv` (389 x
11), summarised in `k_summary_*.csv` and `k_summary_by_group_*.csv`. Coder scales
beside HUMAN NORMS -- Warriner valence/arousal/dominance and Brysbaert
concreteness for English, the 9,877-word set for Chinese -- each with its
extremity twin on the repo's own lexicon-mean centring.

    scale                 sig    exp   mean rho   z_scale   falling
    transgressiveness     221  109.4    -0.0415    -23.7      94%
    bodily_harm           288  109.4    -0.0414    -23.1      91%
    n_concreteness        335  109.2    -0.0460    -18.8      83%
    register_level        157  109.4    +0.0291    +17.5      11%
    concreteness          416  109.4    -0.0401    -17.1      86%
    valence               213  109.4    +0.0285    +15.2      15%
    n_arousal             134   32.1    -0.0691    -15.1      93%
    n_valence_extremity    62   32.1    -0.0314     -7.7      76%
    vulgarity              77  109.4    +0.0018     +1.5      48%
    n_dominance_extremity  47   32.1    -0.0016     -0.7      53%

**THE CODER AND THE HUMAN NORMS CONVERGE INDEPENDENTLY.** Coder concreteness
z -17.1 against Brysbaert's -18.8, both ~86% falling; coder valence +15.2
against Warriner's +8.7, both ~12% falling. A 2026 LLM rating words out of
context and human norms from 2013 agree on sign and rough magnitude.

**`n_arousal` is the largest single effect and the coder has no equivalent.**
z -15.1, 93% falling. `charge` was the attempt at that construct and returns
z +5.9 with the OPPOSITE sign, so charge is not arousal -- exactly what the 0.54
calibration warned and it should never be reported as one.

**Registration C's de-extremification finally has a word-level test, and it
fires:** `n_valence_extremity` z -7.7, 76% falling. Extreme-valence words fall
regardless of direction. **And dominance is dead here as C found it dead**, both
dominance columns being the two smallest |z| in the table -- an unplanned
negative control the design gets for free.

### SITE-SPECIFICITY: THREE SCALES YES, TRANSGRESSIVENESS NO

**CORRECTED 2026-08-12, and the first version of this section was wrong in the
direction that matters.** It reported that nothing was site-specific, on the
basis of comparing GROUP MEANS of marked against unmarked prompts. That discards
the pairing the minimal-pair design exists to supply. Paired within stem, 745
pairs, `partial(MARKED) - partial(UNMARKED)`:

    scale                pairs    median      t         p
    concreteness           745   -0.0159   -6.34   0.00005   site-specific
    bodily_harm            745   -0.0148   -5.24   0.00005   site-specific
    vulgarity              745   -0.0029   -3.57   0.00040   site-specific
    n_valence               70   +0.0480   +3.42   0.00080   stronger at the NEUTRAL twin
    n_valence_extremity     70   +0.0287   +2.14   0.037     stronger at the NEUTRAL twin
    transgressiveness      745   -0.0018   +0.71   0.479     no difference
    charge                 745   -0.0006   +0.39   0.695     no difference

**Concreteness and bodily harm are stronger at the transgressive twin than at
its matched neutral control, at p = 0.00005.** So is vulgarity, the scale
written off twice elsewhere in this document as a floor effect. This agrees with
the 684-pair lineage-unit test (concreteness 33/46 lineages p 0.0045,
bodily_harm 32/46 p 0.0114), which the withdrawn version contradicted.

**What is NOT site-specific is `transgressiveness` itself (p 0.48) and `charge`
(p 0.70)** -- and those are the two scales the withdrawn version happened to
check by eye before generalising to all seven.

**`n_valence` runs the other way**: human-normed valence predicts movement MORE
at the neutral twin (+0.048, p 0.0008). Nobody proposed that direction.

So the site DOES modulate the effect, for harm, concreteness and vulgarity. The
correct summary of RH's question is:

    transgressive words fall              yes, 219 sites, 94% one direction
    they fall MORE at transgressive sites  not by their transgressiveness --
                                           but harmful, concrete and vulgar
                                           words do

**The withdrawn claim is kept visible rather than deleted** because it was
posted to the docket at [5581] and another seat was preparing to test it as a
prediction.

## CHARGE IS NOT A CONFOUND, IT IS A PRECONDITION

**The strongest result in this document, and the only one that got stronger
under scrutiny rather than dissolving.** `scripts/k_charge_control.py`.

K tests every scale MARGINALLY -- residualised on base probability and corpus
frequency, never on another scale. `transgressiveness ~ charge` is +0.65, the
highest collinearity in the set, so the obvious objection is that the
transgressiveness result is charge wearing a costume. It is not. Both effects
get STRONGER when the other is controlled:

    transgressiveness MARGINAL        -0.0414   z -14.7
    transgressiveness | charge        -0.0508   z -18.0    stronger
    charge MARGINAL                   +0.0129   z  +5.5
    charge | transgressiveness        +0.0358   z +12.2    nearly 3x stronger

That is MUTUAL SUPPRESSION -- two correlated predictors with opposite-signed
effects, whose shared variance was masking both. Controlled symmetrically, so
neither scale keeps the shared part by fiat.

### THE INTERACTION IS THE FINDING

Split each prompt's vocabulary at its OWN median charge and compute the
transgressiveness effect in each half:

    trns @ HIGH-charge words   -0.0916   shuffled +0.0045   z -22.1
    trns @ LOW-charge words    -0.0029   shuffled +0.0004   z  -3.1
    INTERACTION hi minus lo    -0.0887   shuffled +0.0041   z -20.8

    1,974 prompts, split on real charge, ONLY transgressiveness permuted.

**Transgressiveness predicts falling among charged words and does nothing among
flat ones.** -0.092 against -0.003. A word must be BOTH affectively charged AND
transgressive to fall; neither property alone moves it.

**This retroactively explains the day's most confusing number.** Pooled,
transgressiveness read +0.012 and "flat" -- because the pooled corpus is
dominated by low-charge vocabulary, which is exactly where the effect does not
exist. The pooled null was not a small effect, it was an effect averaged with
the population in which it is absent.

**And it is what `kill -> scream` has been doing all along.** X_metonymy's
flagship substitution holds charge fixed at 6 and drops transgressiveness 6 -> 1
and bodily_harm 7 -> 1. The campaign's central example is a within-charge
contrast and nobody had tested charge as a variable.

### THE CONTROL FAILED FIRST, AND THE FAILURE WAS MINE

The first version permuted charge as well as transgressiveness, so the median
split ran on SHUFFLED charge and the two halves were not the same words. Its
null came back asymmetric -- hi +0.0095 against lo -0.0147, where both should
sit at zero -- which is what flagged it. Holding the split on real charge and
shuffling only transgressiveness puts both baselines at ~0.00 and leaves the
interaction where it was. **The result survived a control that was itself
broken, and was only quotable after the control was fixed.**

### LIMITS

The split is per-prompt, so "high charge" is a different absolute level at
different prompts; a continuous product term is the proper version and has not
been run. And `charge` is the scale whose calibration against Warriner's arousal
is only 0.54 -- internally reliable (IAA 0.87, rank stability 0.88 when isolated)
but NOT the human construct it most resembles. **This is a finding about the
coder's charge scale, not about arousal**, and `n_arousal` behaves differently
in the main table (z -15.1, 93% falling, where coder charge is +5.9 rising).

## PREDICTION, THE CEILING, AND THE REGISTER AXIS

**Everything above this section is a correlation measured in-sample. This section
asks whether any of it predicts, and the answer changes what the document is
about.** `k_predict.py`, `k_ceiling.py`, `k_embed.py`, `k_predict_embed.py`,
`k_axis.py`, `k_register.py`, `k_brooke.py`, `k_confound.py`.

### The scales do not predict

The case is a CELL -- one (word, prompt, base, aligned) -- and the held-out unit
is the WORD, five-fold `GroupKFold`, so a model that memorised `murder falls`
scores nothing for it. Outcome fall vs rise under the canonical rule. English
lexical verbs only, so part of speech is not a confound; 100,958 cells over 4,075
words.

    increment over the SAME model without the features, and over the SAME model
    on ratings shuffled across words

                        over same class      over shuffled
      additive          +0.0021 / +0.0035    +0.0034 / +0.0054
      interactions      -0.0016 / -0.0017    +0.0035 / +0.0041
      trees             +0.0038 / +0.0029    +0.0102 / +0.0057

**Adding the eight Warriner norms does not help**; on the 1,042 verbs Warriner
covers (22% of them, not the 48% it covers of the whole vocabulary) the additive
model is WORSE than nuisance alone. **Chinese, run separately, agrees**: coder
scales add +0.010, and in the covered subset `norms only` scores 0.478 against a
shuffled 0.525 -- the real ratings predict worse than random ones.

The scalar outcome `log10(p_aligned / p_base)` agrees and is the better
instrument, because the binary outcome is thresholded on a p_base-relative
quantity and so selects on itself. It also exposes what that costs: **within a
site, p_base predicts magnitude at Spearman +0.001**, against a per-site AUC of
0.66 in the binary version. The nuisance variable's apparent power in the binary
analysis is manufactured by the binarisation.

**THE EFFECTIVE n IS 4,075, NOT 100,958.** Eleven of the twelve features are
constant within a word; only `log p_base` varies across a word's cells. The extra
cells inform the outcome, not the predictors. This is not a power problem -- the
learning curve is flat from 652 training words to 2,608 -- which is also why
leave-one-out was not run: it adds 20% to a curve that stopped moving three
doublings earlier.

### But there is plenty of word-level signal, so the failure is the instrument

Split a word's own cells in half and use the fall rate in one half to predict the
other. That is the best any function of the word alone can reach, since it uses
the word's identity.

    English verbs, 2,760 words, 792,549 scored cells
      oracle (word identity)          0.7025
      log p_base alone, same cells    0.5818
      headroom for any word feature  +0.1207

**The eighteen rated norms buy +0.008 of that 0.121, about 7%.** And the ceiling
is itself low: ICC(1) = 0.131, so **87% of the fall/rise variance is WITHIN a
word across the sites it appears at** and is unreachable by any word-level
feature. Movement is a property of the word AT A SITE.

### A distributional embedding recovers three times as much

    what beats its OWN shuffle, pooled       share of the +0.1207 headroom
      18 rated norms, trees      +0.0083                7%
      GloVe 300d, trees, k=50    +0.0256               21%
      bge-m3 1024d, trees        +0.0208               17%

Two encoders that agree about nothing else -- GloVe is near-isotropic at 0.037
median pairwise cosine, bge-m3 sits at 0.529 -- land within 0.005 of each other.
Encoders are gated on BARE WORDS before use, because docket [459] gate-checked
bge-m3 on `prompt + " " + word`, a sentence, and a gate passed for one use is not
evidence about another. GloVe separates near-synonyms at +0.400 against bge-m3's
+0.138 and is the primary on that basis.

**So the rated dimensions are not a coarse version of the right axis; they are
substantially the wrong basis.**

### The direction, and how far it can be named

The GloVe direction predicting FALL, extracted with `log p_base` and `log fpm` in
the model as nuisance:

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
-- both pairs carry transgressive content and what separates them is not
transgression.

Confirmed by instruments sharing no inputs with the embedding. A genre log-ratio
`log10(SUBTLEX-US / coca_acad)` correlates +0.435 with the axis; COCA and BNC
spoken-over-academic give +0.410 and +0.414. Against Brooke, Wang & Hirst's
`Choose the Right Word` near-synonym pairs, the register index orders 76% of 326
correctly and word form alone 74%.

**BUT THE NAME IS NOT ESTABLISHED, AND RH'S OBJECTION IS WHY.** In the history of
English the Latin and Norman borrowings arrived polysyllabic and abstract while
the Germanic core stayed monosyllabic and concrete, so register, abstractness and
length are one stratum with several names. Our own numbers show it: axis with
concreteness +0.29, with syllable count +0.31, register index with concreteness
+0.49. Matching the near-synonym pairs progressively:

    all pairs with Brysbaert on both      253   77.9%   8.9 sd
    + same syllable count                  78   75.6%   4.5 sd
    + concreteness within 0.5             126   72.2%   5.0 sd
    + BOTH matched                         40   67.5%   2.2 sd

**And near-synonymy does NOT hold abstractness fixed**, which the matched test had
assumed: on Brysbaert the informal member of a CTRW pair is +0.337 more concrete,
Wilcoxon p = 3.3e-11. So: the axis runs along the vernacular/institutional
stratum; register contributes something beyond length and abstractness; that
something is 2.2 sd on 40 pairs and is a hypothesis.

**Etymology was never actually tested.** The Latinate suffix flag correlates
+0.10 to +0.17 with everything, sitting outside a cluster whose members correlate
+0.34 to +0.83. A real Latinate marker would be inside it, so the "form alone"
result was mostly word length, and testing Latinateness needs an etymological
resource rather than a suffix regex.

### What this does to `register_level`, and it is a correction

**The one scale of the seven pointing at this axis is the one this document
declares not established**, at line 292: "inter-coder agreement 0.60, rank
stability 0.619, z = 1.0 against its null. Treat it as not established."

The first two grounds stand. **The third is the wrong statistic.**
`k_frame_summary.py:27` states in its own docstring that `z_scale` is the summary
statistic to read, because the count ratio treats every prompt as a coin flip and
ignores effect size. On `z_scale`, `register_level` is **+17.5, the fourth
strongest of eighteen scales, 157 sites against 109 expected, 89% RISING** --
exactly the direction the axis predicts. Its marginal coefficient at line 309 is
+0.044, the largest quoted there.

And against an external human gold standard it is the best measure we have:

    CTRW pairwise                  seed separation (Mann-Whitney AUC)
      coder register_level  97.1%    0.974
      register index        76.8%    0.955
      orthographic form     74.1%    0.939

on 68 of 399 pairs, so the coverage is selected and the figure is not
head-to-head with the others. **The scale is VALID and NOISY, which are different
things**: IAA measures agreement on fine 1-7 gradations, this measures whether it
tells `smooch` from `kiss`. Low reliability attenuates toward zero, so +17.5 was
shown despite the measurement, not because of it. We read "we measured this
badly" as "this construct is absent."

### Limits, and one thing not to do

The axis failed its own pre-declared stability gate: minimum pairwise cosine
between the five fold axes was 0.841 against a gate of 0.9 set before the run.
High enough that the direction is not noise, too low to have passed, and the line
does not move afterwards.

**Do not quote the sufficiency comparison.** The one-axis model scored 0.6831
against the full 300-d model's 0.6670, but at different penalties (C=1.0 against
C=0.1), so that is regularisation and not dimensionality.

**Do not read the third decimal of any prediction increment.** Between two runs
of `k_register` the nuisance floor itself moved 0.6663 to 0.6659 purely because
the covered population changed by 326 cells. Run-to-run variation of these
quantities has not been characterised.

The Chinese register seed list from Brooke is **contaminated** -- twelve of the 49
"formal" entries are internet slang (酱紫, 弓虽, 东东, 菜鸟, 大虾, 美眉) -- and was
left unrepaired rather than trimmed, since trimming another group's data after
seeing its effect on our result is not a repair. Chinese has no embedding axis
yet; `k_embed zh bge` has not completed.

## THE FOUR ANSWERS THIS ANALYSIS GAVE BEFORE THIS ONE

Each was wrong in a way that looked like a result. They are recorded because the
pattern is the finding as much as the table above.

**1. Pooled over everything: transgressiveness +0.012, "flat".** Every word's
movement summed across all its cells, neutral prompts included. Concreteness led
at −0.166 and was reported as the headline. Both figures are facts about a
corpus that is mostly neutral, not about displacement.

**2. Thirteen named prompts: vulgarity 12/13 falling, p 0.0034.** I called it
the clearest result in the study. The thirteen were chosen, with no control
class. On the declared corpus vulgarity is 361/323, p 0.157 — nothing.

**3. 684 M01 pairs, PAIR as unit: bodily_harm p 0.00027, concreteness
p 0.00001.** The unit was wrong. 684 partials each computed on movement pooled
across 46 lineages treats 684 correlated numbers as independent draws. At the
LINEAGE unit the p-values inflate about 40x (0.0045 and 0.0114) — the effect was
real, the confidence was not.

**4. Per prompt with the LINEAGE as the unit inside it: 712 prompts for
transgressiveness. THIS FAILED ITS NEGATIVE CONTROL.** With ratings shuffled
across words so no real link could exist, the same test fired at 35–42% of
prompts against an expected 5%, and **for five of seven scales the shuffled
count exceeded the observed one**. Mean p under the shuffle 0.279 where 0.500 is
calibrated.

The flaw was pseudo-replication one level below where the campaign's unit
discipline guards. All 46 lineages at a prompt see THE SAME WORDS with THE SAME
RATINGS; if the rating vector correlates with movement in that word sample,
every lineage shows it, and a sign-flip over lineages counts one shared draw as
46. **The effective n is the word sample.** I had spent the afternoon fixing
this at the model level and rebuilt it at the word level.

`results/k/failed_control_v1.log` is kept so the failure is inspectable.

## Why the pooled and per-site answers differ in SIGN

Not attenuation — cancellation. Over 13 charged prompts, mean effect −0.0026
against mean |effect| 0.0792, a **30x ratio**. `bodily_harm` alone is 10.3x with
signs 6+/7−. Pooling averages scenes whose directions oppose, and two scales
come out with the wrong sign rather than a small one.

This is `V_embedding_regions` §5's scene-locality and the campaign's own "the
relation is local" — six geometric instruments failed because each scene has its
own direction. K is a seventh, failing identically when pooled.

**The campaign has a doctrine for the model unit and none for the prompt unit.**
F/G take the base checkpoint with the pair corpus pooled inside it; U takes the
rung with 2,182 prompts pooled inside it. Pooling across prompts was never
decided, it fell out of taking the checkpoint as the unit. Proposed rule: *a
result reported only pooled across prompts has not been shown to exist at any of
them.*

## What the scales are, as measurements

Rated alone versus inside the seven, same 428 words (`scale_isolation.log`):

    transgressiveness r 0.927 | valence 0.911 | bodily_harm 0.911
    charge 0.880 | concreteness 0.878 | register_level 0.619 | vulgarity 0.283

Charge and concreteness shift in LEVEL when isolated (2.29→2.58, 3.83→4.06) and
hold their ORDER. Every statistic here is rank-based, so the level shift is
irrelevant — but it means these values are properties of the frozen instrument
and outputs from different versions must never be pooled.

**Register is the one unstable scale**, and it is weakest on three independent
measures: inter-coder agreement 0.60, rank stability 0.619, z = 1.0 against its
null. Treat it as not established.

> **CORRECTED 2026-08-12. The verdict above is wrong on its third ground and the
> conclusion does not hold.** `z = 1.0` is the count statistic; on `z_scale`,
> which `k_frame_summary.py:27` says is the one to read, `register_level` is
> +17.5 and 89% RISING. Against Brooke, Wang & Hirst's human-ordered near-synonym
> pairs it scores 97.1%, the best of every measure tried. The scale is noisy at
> fine grain and valid at coarse discrimination, and it is the only one of the
> seven pointing at the axis that predicts. See "PREDICTION, THE CEILING, AND THE
> REGISTER AXIS". The sentence is left standing rather than deleted because it
> was the operative judgement for two weeks.

**Vulgarity is a SPARSE INDICATOR, not a failed scale.** It has variance on 463
of 27,242 words. Its correlation with register is **+0.000 among the 26,779
words with no coarseness and −0.571 among the 113 coarse ones** — the two are
one construct at the coarse end and independent everywhere else. Its 0.283
isolation figure and 0.28 pilot IAA are floor effects, not disagreement, and
"fires at 82 prompts" should not be read as a null.

## What the design deliberately does NOT do

The seven scales are **never entered into one model**. Each is residualised on
base probability and frequency only, never on another scale, and each has its
own null band. Where they overlap semantically that overlap is left intact.

A joint regression was run once, to show what would be lost: register's
coefficient shrinks 95% (+0.044 → +0.002) and valence 53%. Those are not
revelations of spuriousness, they are shared variance assigned arbitrarily among
correlated predictors, and a reader would take +0.002 as "register does
nothing."

## Limits

**Effect sizes are small.** The largest partial anywhere is −0.166, which equals
the base-probability nuisance floor `X_metonymy` records. No scale exceeds the
floor; what they have is consistency of direction across hundreds of sites.

**Chinese is not established either way.** At 17 CJK-capable lineages the
sign-flip permutation needs d ≥ 0.85 against English's 0.44, so Chinese must
show twice the effect to register. On the 24 shared transgressive pairs no
Chinese scale clears while English shows vulgarity t −4.20 and charge t −2.80.
Five of seven signs oppose across the languages, and **that reversal should not
be written up as a language difference** — the Chinese side cannot distinguish
it from noise. Same unit ceiling as `zh_sites_unit_limited.md`, second route.

**One unresolved discrepancy.** Vulgarity is the top English effect on the 24
translated pairs (t −4.20) and null on the 684 M01 pairs (p 0.157). Same
language, instrument and unit; different corpora. Either the 24 are
unrepresentative or the M01 corpus is, and this design cannot say which.

**FDR controls a rate, not a list.** Roughly 5% of any named prompt is expected
spurious. The counts are usable; a single prompt is not, until re-run held out.

**The `animal` domain** returns large effects (valence t −5.13) that no theory
predicted, on the smallest cell. Treat as a property of that sub-corpus until
shown otherwise.
