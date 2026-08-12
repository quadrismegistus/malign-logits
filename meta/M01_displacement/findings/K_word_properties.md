# Findings K: which properties of a word predict how alignment moves it

**Status: FOUR SCALES CLEAR A CALIBRATED NULL, AND THREE OF THEM ARE
SITE-SPECIFIC. The route here matters as much as the destination — four earlier versions of this analysis gave four
different answers, including two with the wrong sign, and one of them failed its
negative control outright.**

Producer chain: `scripts/rate_charge_v1.py` (the rater),
`scripts/k_population.py`, `k_bulk.py`, `k_analysis.py`, `k_by_prompt2.py`.
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
