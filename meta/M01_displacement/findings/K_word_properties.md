# Findings K: which properties of a word predict how alignment moves it

**Status: FOUR SCALES CLEAR A CALIBRATED NULL, and the route here matters as
much as the destination — four earlier versions of this analysis gave four
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

**This answers RH's question — "do transgressive WORDS fall more due to
alignment?" — as: yes, at 219 sites, and only visible when you ask per site.**

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
