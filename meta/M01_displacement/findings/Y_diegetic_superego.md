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
than an observation; and the conditional-on-`sexual_scene` panel, where the
effect is largest and the sign agreement strongest.

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
