# Y: the superego is diegetic — alignment moralises inside the scene it keeps writing

lacan seat, 2026-08-09. Producer `meta/M01_displacement/scripts/y_diegetic.py`,
log `results/y_diegetic.log`. Corpus: registration Y, 41,596 pass-A parsed
passages, 32 base>aligned pairs, manifest `af79083c675aae7f`.

Extends `Y_superego.md` §4, which established that the superego measures rise.
This asks the next question: **rise instead of what?** The composites that answer
it were run on 32 pairs and never reported.

---

## The claim

**Alignment's dominant response to sexual content is not refusal, deflection, or
leaving the frame. It stays inside the fiction and attaches guilt, hesitation and
moral comment to a scene it goes on writing.**

The trained behaviour does not move. The untrained one does, and by about four
times as much.

**And it survives the strongest available control.** Force the identical
transgressive word into both arms and the scene becomes identical by measurement
-- same rate of sex, same rate of consummation, same rate of leaving the frame --
while the guilt is still added, +5.0 points on `X_metonymy.md` §3g's own scene
(p=7.2e-08). The moralisation is not a by-product of alignment selecting milder
words. See §6 and §7.

## 1. The extra-diegetic response is flat

| measure | base | aligned | delta | p | pairs |
| --- | --- | --- | --- | --- | --- |
| `EXIT` (refusal OR frame_exit) | 26.48% | 27.80% | +1.03pp | 0.61 | 17/32 |
| `frame_exit` | 26.43% | 26.92% | +0.66pp | 0.81 | 17/32 |
| `sexual_scene` | 53.85% | 50.01% | −0.22pp | 0.30 | 16/32 |
| `assistant_refusal` | 0.10% | 1.14% | +0.22pp | 0.0002 | 18/32 |

`EXIT` is a coin flip. `assistant_refusal` rises eleven-fold and is still one
passage in a hundred; the multiplier is real and the magnitude is negligible.
**And the sex still happens** — the sexual-scene rate does not move.

RLHF preference data rewards the assistant declining. That is the behaviour under
direct training, and at this corpus's scale it is close to absent.

## 2. The diegetic response is where the movement is

| measure | base | aligned | delta | p | pairs |
| --- | --- | --- | --- | --- | --- |
| `SUPEREGO_IN_SCENE` | 8.58% | 11.18% | **+2.67pp** | 0.0071 | 22/32 |
| `CLEAN_SCENE` | 45.22% | 38.26% | **−4.74pp** | 0.0009 | 24/32 |

    SUPEREGO_IN_SCENE = sexual_scene AND (moralisation_in_scene OR guilt_or_shame
                        OR consent_hesitation) AND NOT assistant_refusal
    CLEAN_SCENE       = sexual_scene AND none of those AND NOT refusal

+2.67pp against `assistant_refusal`'s +0.22pp. **The in-scene effect is an order
of magnitude larger than the refusal effect in absolute terms**, and it is the
one nothing in the training objective names.

## 3. Conditioning on the sex makes it airtight, and larger

Unconditionally the composite could rise because more passages contain sex. They
do not (§1), but conditioning removes the question. Restricted to the 21,858
passages where a sexual scene occurred:

| measure | base | aligned | delta | p | pairs |
| --- | --- | --- | --- | --- | --- |
| `CLEAN_SCENE` | 84.72% | 76.68% | **−6.12pp** | 4.6e-05 | **27/32** |
| `SUPEREGO_IN_SCENE` | 15.18% | 21.60% | **+4.30pp** | 5.8e-04 | 24/32 |
| `consent_hesitation` | 11.00% | 16.34% | +3.99pp | 2.0e-04 | 24/32 |
| `guilt_or_shame` | 3.55% | 5.79% | +1.27pp | 1.2e-03 | 23/32 |
| `moralisation_in_scene` | 2.37% | 3.67% | +0.81pp | 0.056 | 19/32 |

**Given that the model writes the sex, it is a fifth less likely to write it
clean.** 27 of 32 pairs, the strongest sign agreement anywhere in this corpus.

## 4. Why this is the interesting reading

Two accounts of what alignment does to a generative model:

- **Filter.** A gate at the output. Blocks, deflects, declines. Predicts `EXIT`
  rises and `sexual_scene` falls.
- **Disposition.** The normative pressure is in the generative process, so
  content polices itself and no gate is needed. Predicts the scene proceeds and
  arrives with its own moral apparatus attached.

`EXIT` flat, `sexual_scene` flat, `CLEAN_SCENE` down 6.12pp on 27 of 32 pairs.
**The filter account predicts the two things that do not move.**

No preference pair says *write the sex scene, but have the character feel bad
about it*. The training signal concerns what the assistant does, not what happens
to characters inside a fiction the assistant is producing. So this is a
generalisation out of the refusal objective and into the content.

**And it is a rate effect on a structure the base model already has.**
`Y_superego.md` §4 establishes that guilt's form does not change — span length
identical, onset identical, explicit writing resumes after it at the same rate.
Onsets confirm the sequence and its stability: sexual content at 0.22–0.27 of
the way through a passage, guilt at 0.54 in **both arms**. The sex-then-guilt
order is in pretraining. Alignment does not install the apparatus, does not move
it, does not lengthen it. It fires it more often.

That is the shape "no emancipatory outside" predicts: not a moral layer added on
top, but a selection from moral material the base model already held.

## 5. What is new here and what is not

**Not new**, and this document depends on it: `Y_superego.md` §4 already reports
`<consent>` +1.32pp, `consent_hesitation` +2.80pp, `<guilt>` +0.80pp at 22/32,
`guilt_or_shame` +0.87pp, `<moral>` +0.39pp, and the form-invariance of guilt.
Independently re-derived here to the digit from the coded corpus.

**New**: the composites (run on 32 pairs per `Y_statistics.md`, never written
up); the `EXIT`-flat contrast that makes the diegetic reading a contrast rather
than an observation; the conditional-on-`sexual_scene` panel, where the effect is
largest and the sign agreement strongest; the undisturbed-against-forced
dissociation in §6; and the resolution of `X_metonymy.md` §3g's scope question in
§7.

## 6. Two defences: one is removable, the other attaches to the act

RH's question: the sexual words are falling in the aligned model anyway, so does
it simply prefer not to get into the situation? Y can test that, because 6,168
pass-A passages are UNDISTURBED -- no word forced, the model chooses.

**Unconditionally, avoidance is real and it is removable.**

| | base | aligned | delta | p | pairs |
| --- | --- | --- | --- | --- | --- |
| **undisturbed** `sexual_scene` | 53.20% | 47.82% | **−4.30pp** | 0.016 | 20/32 |
| **forced** `sexual_scene` | 53.96% | 50.39% | +0.26pp | 0.43 | 17/32 |

Left to choose, the aligned model enters a sexual scene 4.30pp less often than
its parent. Pin the word into both arms and that difference is gone -- 17 of 32
pairs, a coin flip. The avoidance operates entirely at the CHOICE OF WORD, which
is `X_metonymy.md` §3d's displacement seen from the other side, and forcing the
word is precisely the intervention that abolishes it.

**Conditionally, the superego does not care how the model got there.** Restricted
to passages where a sexual scene occurred:

| | base | aligned | delta | p | pairs |
| --- | --- | --- | --- | --- | --- |
| **undisturbed** `SUPEREGO_IN_SCENE` | 15.23% | 21.85% | +3.67pp | 0.0036 | 22/29 |
| **forced** `SUPEREGO_IN_SCENE` | **15.27%** | **21.69%** | +4.39pp | 0.0011 | 24/32 |
| **undisturbed** `consent_hesitation` | 10.98% | 16.81% | +5.56pp | 0.0004 | 22/29 |
| **forced** `consent_hesitation` | 11.08% | 16.35% | +4.72pp | 0.0008 | 23/32 |
| **undisturbed** `moralisation_in_scene`, base | 2.38% | | | | |
| **forced** `moralisation_in_scene`, base | 2.38% | | | | |

**The base rates agree to two decimal places and so do the aligned rates.** The
route makes no measurable difference. Given a sexual scene, the aligned model
attaches the moral apparatus 43% more often than its parent, and it does so at
the same rate whether it wandered in or was pushed.

This also corrects a reading the unconditional numbers invite. Unconditionally
the superego effect looks slightly smaller undisturbed (+2.11pp) than forced
(+2.66pp), which could be read as forcing provoking it. It does not: the
undisturbed condition simply has fewer sexual scenes for the apparatus to attach
to (47.8% against 50.4%). Per scene the two are the same.

So the structure is not avoid-then-moralise:

    displacement    operates on WHICH WORD.   Removable -- forcing the word
                                              abolishes it entirely.
    superego        operates on THE SCENE.    Invariant to the route. Identical
                                              whether the model chose or was forced.

Two interventions at different levels of the same production, and only one of
them is a decision the model can be prevented from making.

**Read in the register the campaign uses:** the superego attaches to the act, not
to the agency. Being forced into the transgression earns no exemption. The model
handed the word and the model that walked in on its own are moralised at the same
rate, which is the behaviour the theory names and not an obvious property of a
next-token predictor.

## 7. This closes X_metonymy §3g's open scope question

§3g holds the word constant across arms and finds the model adds nothing: −0.8
points on a 0-100 sexual scale, 15 of 30 cells, p=0.918. Its own caveat:

> THE SCOPE IS TEN TOKENS ... Defensible as *within ten tokens of the
> substitution, alignment adds nothing*; not yet as *alignment's intervention is
> exhausted by the substitution*.

Y is the same design at 256 tokens. On §3g's exact scene (`sexual_explicit_1`),
word held constant, 191 (pair, prompt, word) cells:

| measure | base | aligned | p | pairs | MDE |
| --- | --- | --- | --- | --- | --- |
| `sexual_scene` | 66.8% | 65.6% | 0.99 | 101/191 | 4.0 |
| `consummation` | 17.3% | 17.3% | 0.72 | 115/191 | 2.7 |
| `EXIT` | 30.2% | 30.5% | 0.54 | 105/191 | 4.4 |
| `SUPEREGO_IN_SCENE` | 9.0% | **14.7%** | **7.2e-08** | 110/191 | 2.6 |

**§3g's null replicates at 25x the length**, and extends to `consummation`, an
outcome a ten-token window cannot represent. Hand the aligned model the
transgressive word and it writes the base model's scene -- confirmed, and more
strongly than §3g could claim.

**And the intervention is NOT exhausted by the substitution.** One thing survives
the word being held constant: the moral apparatus, +5.0 points on §3g's own
scene.

**§3g could not have found this.** Guilt onset is at 0.54 of the passage, about
token 138 of 256. §3g's entire window was 10 tokens. The effect sits fourteen
times beyond its horizon, so its null was not a weak test of the right thing --
it was a test that could not reach the thing.

This also closes the obvious objection to §§1-5 of this document. Someone could
say the aligned model looks moralising only because it selects milder words and
milder words invite different scenes. Here the word is identical by construction,
the scene is identical by measurement, and the guilt is added anyway.

## 8. The base model is surprised by the aligned model's guilt — and by nothing else it writes

Everything above is counting. This is the surprisal complement, and it is the
one place in the corpus where a moral register costs the base model something.

Per passage: **mean surprisal inside a tag minus mean surprisal in plain story**,
where plain story is `layer1 == 'story' AND layer2 is null` — ordinary narration
with every layer-2 span removed. Means are length-invariant, so no window
matching is required. Unit is the pair; producer
`scripts/y_full_analysis.py --only story` over the two tables.

| tag | written by | pairs | delta | boot 95% CI (mean) | wilcoxon | sign |
| --- | --- | --- | --- | --- | --- | --- |
| **`guilt`** | **aligned** | 26 | **+0.102** | **[+0.005, +0.202]** | **2.6e-02** | **21/26** |
| `guilt` | base | 24 | +0.040 | [−0.077, +0.149] | 0.25 | 15/24 |
| `moral` | either | 19–21 | ~0 | straddles | ns | 9–10 of 19–21 |
| `sexual` | base | 32 | −0.179 | [−0.276, −0.093] | 3.3e-04 | 24/32 |
| `consent` | base | 21 | −0.250 | [−0.352, −0.131] | 1.2e-03 | 19/21 |
| `consent` | aligned | 24 | −0.219 | [−0.298, −0.148] | 6.3e-05 | 21/24 |
| `resist` | base | 28 | −0.288 | [−0.395, −0.195] | 9.0e-06 | 26/28 |
| `resist` | aligned | 25 | −0.164 | [−0.279, −0.071] | 1.9e-03 | 20/25 |

**`guilt` written by the aligned model is the only positive cell in the table.**
Every other tag is easier than plain story, or null. And it holds on the
**aligned scorer independently** (+0.080, p=3.0e-02, 20/26), so it is not an
artifact of which model is doing the reading.

**This is the one thing alignment writes that its parent does not find ordinary.**
Sections 1–7 establish that alignment moralises more often, in the same place, at
the same length, with the same words, and that the base is not estranged from the
result. That last part now has an exception, and it is guilt specifically —
not `moral`, not `consent`, not `resist`.

### The moral cluster is not one thing

Sections 4–6 treat `guilt`, `consent`, `resist` and `moral` together. **They do
not behave alike.** `consent` and `resist` are strongly NEGATIVE in both arms and
both scorers — easier than ordinary narration by 0.16 to 0.31 nats, up to 26 of
28 pairs. That is the formulaic-language result: hesitation and refusal language
is stock phrasing both models predict well (`Y_examples.md` §4). `guilt` runs the
other way. Any claim about "the superego measures" that pools them is pooling an
effect with its opposite.

### Why an earlier version of this reported a null

The first attempt compared the **maximum** divergence in a span against a
length-matched window of story. A maximum is a high-variance extreme-value
statistic and needs the matching precisely because it grows with window length;
a **mean** does not, and is far more sensitive. Using the harder statistic for no
reason produced a null in every cell. Recorded because the choice of summary was
the whole difference between a null and a result, and nothing about the null
looked wrong.

### Multiplicity, and why not Bonferroni

Ten cells per scorer panel (5 layer-2 tags × 2 arms). Correction is
**Benjamini-Hochberg at q=0.05**, which controls the proportion of false
positives among rejections; it rejects six, `guilt/aligned` among them, on both
scorers. Bonferroni is the wrong instrument here: it controls family-wise error,
which is not the target for a descriptive table, and it assumes an independence
these cells do not have, since they share passages, spans and pairs.

The honest statement is the interval, per Y's convention that the CI is the
claim and the p ranks: **+0.102 [+0.005, +0.202]**. It clears, and it clears
narrowly.

## Limits

- **`moralisation_in_scene` alone does not clear** (p=0.056, 19/32). The
  composite is carried by `consent_hesitation` and `guilt_or_shame`, and consent
  carries most of it. "Consent hesitation rises" is the narrow claim; "alignment
  moralises" is the wide one and consent is doing the work.
- **The topical link is untested.** `SUPEREGO_IN_SCENE` requires sex and a moral
  marker in the same passage, not that the guilt is *about* the sex. Onset
  ordering (sex 0.22, guilt 0.54) is consistent with it and does not establish
  it.
- **Heterogeneity is large** (§4): AmberSafe +15.4pp against a median +0.8pp,
  with four negative pairs including both Mamba architectures. The median is the
  claim; the spread is 20 points.
- **Five prompts.** The whole corpus is `sexual_explicit_1/3/5` and
  `sexual_liminal_6/7`, sampled 50 deep across 24 forced-word cells. Depth is
  real, prompt base is narrow, and nothing here transfers to another domain
  without being run there.
- One coder (`deepseek-v4-flash`, task `code_y_superego_v3`). Field reliability
  is the v3 gate's business, not re-established here.
- **§8's interval nearly touches zero** (+0.005) and `guilt/base` is null, so the
  effect exists only for guilt the aligned model wrote. It is a lead worth
  registering, not a number to quote as settled.
