# Registration Z, ladders and regimes: what we found

Positive results only, in the format of `Y_findings.md`: the number, the test, the population. Two questions a base/aligned contrast structurally cannot ask, and one it can be shown to get wrong.

**Corpus**: the `generations` stash, 256,035 passages, filtered to temp=1.0 and to the checkpoints named below. Field counts from `malign_logits/fields.py` (USAS fine, Warriner and Brysbaert norms tertiled on their own distributions, RID). Script `meta/M01_displacement/scripts/z_ladders.py`.

**Unit is the prompt, within a chain.** Every chain has 45 to 73 prompts that all of its stages carry, and a prompt only ever compares to itself. Wilcoxon signed-rank over prompts plus a bootstrap 95% CI on the median, per chain. Four families is too few to test across chains, so families are reported individually and consistency of sign is the only cross-family claim made.

**Two statistics, and they are allowed to disagree.** A cell is starred only when the signed-rank and the median CI both clear. They measure different things: in two Olmo-3 cells 63% of prompts moved up while the minority that moved down moved further (mean |neg| 2.14pp against |pos| 1.49pp), so the median is displaced from zero and the magnitude-ranked signed-rank is not. Both correct. A displaced median with asymmetric tails is not significance and is not quoted as such.

---

## 1. The base/aligned contrast can report nothing where two large effects cancel

The strongest structural result here, and it is methodological rather than about any one field.

Dominance, `N:dominance=dominant`, share of Warriner-scored tokens in the top tertile:

| family | SFT − base | DPO − SFT | total DPO − base |
| --- | --- | --- | --- |
| OLMo-2-1B | **+1.00** p 0.005 * | **−0.78** p 0.004 * | +0.26 p 0.195 |
| Olmo-3-7B | **+2.52** p 0.000 * | **−1.57** p 0.000 * | +1.15 p 0.000 * |

In both OLMo families the two alignment steps move dominance in **opposite directions**, each clearing both tests. In OLMo-2-1B they very nearly cancel: the total is +0.26pp at p 0.195, which a two-checkpoint design reports as no effect. The effect is there twice over, and the composite hides it.

This is the one finding that licenses a general claim about method. Every base/aligned number in this project is a difference across a composite of at least two training operations, and a composite can be flat while its parts are not. n = 73 prompts per family.

## 2. Abstraction rises, and both training steps do about half of it each

`N:concreteness=concrete`, share of Brysbaert-scored tokens in the top tertile, falls in three of four families, and in all three the two steps contribute comparably:

| family | SFT − base | DPO − SFT | total | split |
| --- | --- | --- | --- | --- |
| OLMo-2-1B | −1.66 * | −1.52 * | **−3.12** * | 52 / 48 |
| pythia-6.9B | −1.13 * | −0.93 * | **−2.02** * | 55 / 45 |
| Tulu-3-8B (SFT = no safety data) | −0.75 * | −0.69 * | **−1.66** * | 52 / 48 |
| Olmo-3-7B | +0.12 | −0.38 | −0.53 * | — |

All nine cells in the first three rows clear both tests. The split column is each step's share of the **summed step magnitudes**, not of the total: these are medians of per-prompt deltas and medians are not additive, so the two steps do not sum to the total column and a split computed against the total would be the wrong denominator.

This is a different division of labor from F18's entropy result, where SFT did 66 to 81 percent in three families: here it is close to even. Worth stating as a contrast rather than a contradiction, since the measures are unrelated.

Concreteness is the null control in the residualisation (R² = 0.0003 on valence), so this is not valence arriving under another name.

## 3. Two fields rise across every family that shows an effect

`F:physical_appearance_and_properties` rises in all four families on the total (+0.82 *, +0.57 *, +0.25 ~, +0.89 *), and both steps are positive in all four. `F:emotion_and_arousal` rises in three of four (+1.16 *, +1.04 *, +0.84 *) with both steps positive in each of the three; Olmo-3-7B goes the other way (−0.34 *).

`R:need`, Martindale's need imagery, falls in two families (OLMo-2 −1.62 *, Tulu −1.56 *) and in both the DPO step does more of the work than the SFT step (−1.38 against −0.60; −0.99 against −0.55). The other two families show nothing on it.

`F:logical_modal_and_discourse_operators` splits sign two against two across families (−1.46 *, +0.35 ~, +0.42 *, −1.47 *) and does not support a direction claim.

---

## 4. The closed-model contrast is a system prompt, not alignment

Three frontier models sit in the stash in two conditions, and the suffix is the opposite way round from what it looks like. From `malign_logits/api_generate.py:6-14` and `cli.py:1319-1320`:

- `{model}-raw` has **no** system prompt: native chat behavior, the deployed assistant
- `{model}` has the system prompt `"Continue the following text. Write only the continuation, no commentary or explanation."`

Both conditions are the same RLHF'd weights. **There is no closed-model alignment delta in this corpus and there cannot be**, because no frontier base checkpoint is available. What the contrast measures instead is how far one sentence at inference time moves the register, which is a narrower and still useful question: whatever a system line fails to move is resident in the weights.

Reported as chat minus continuation, so it points the same way as base to DPO. n = 45 or 46 prompts.

| measure | sonnet-4-6 | haiku-4-5 | gpt-4o-mini |
| --- | --- | --- | --- |
| `F:language_and_communication` | **+9.46** * | **+5.54** * | +0.19 |
| `N:dominance=dominant` | **+9.80** * | **+4.68** * | +0.63 |
| `N:concreteness=concrete` | **−6.45** * | **−4.81** * | **−2.49** * |
| `F:sensory_perception` | **−2.03** * | **−1.20** * | **−0.72** * |

Magnitude orders sonnet > haiku > gpt-4o-mini on every row. That ordering is not interpreted here; it is confounded with scale, vendor, and whatever the three vendors' chat defaults are.

**A confound that applies to two rows specifically.** The system prompt names commentary. So `F:language_and_communication` and `F:logical_modal_and_discourse_operators` are partly a compliance check on the instruction and are marked `[NAMED]` in the output rather than read as register discovery. The measures the instruction does not name, concreteness and sensory perception and physical appearance and emotion, carry no such shortcut, and those are the informative rows.

## 5. Open retraining and a closed system line share exactly two dimensions

The direct comparison, and the answer to whether frontier models can be read as a further layer of the same regime. Both sides are **within-model and prompt-matched**: no levels are compared across model families, so scale and pretraining corpus cancel inside each column.

A direction claim needs an effect on both sides to compare. Four of nine measures fail that and are marked not comparable, which is deliberate: without it, a measure that simply did not move on one side gets written up as an inversion.

| measure | open, 4 families | closed, 3 models | verdict |
| --- | --- | --- | --- |
| `N:concreteness=concrete` | −2.99 * −0.40 −2.15 * −1.55 * | −6.45 * −4.81 * −2.49 * | **same direction, 2.6x** |
| `N:dominance=dominant` | +0.05 +1.28 * +1.39 * +0.32 | +9.80 * +4.68 * +0.63 | **same direction, 5.8x** |
| `F:physical_appearance_and_properties` | +0.79 * +0.57 +0.25 +1.07 * | −1.48 * −0.33 −0.47 * | inverts |
| `R:need` | −1.62 * +0.26 −0.35 −1.54 * | +0.92 * +0.52 +0.27 * | inverts |
| `F:logical_modal_and_discourse_operators` | −1.38 * +0.02 +0.39 −1.79 * | +1.03 * +0.26 +0.29 * | inverts `[NAMED]` |
| `F:language_and_communication` | −0.90 * +2.37 * −0.21 +0.02 | +9.46 * +5.54 * +0.19 | inverts `[NAMED]` |
| `F:emotion_and_arousal` | +1.02 * −0.35 * +1.04 * +1.22 * | −0.21 −0.20 +0.12 | not comparable, 0/3 closed |
| `F:sensory_perception` | +0.18 +0.12 +0.25 * +0.33 | −2.03 * −1.20 * −0.72 * | not comparable, 1/4 open |
| `R:sensation` | +0.46 +0.28 −0.44 +0.05 | +0.52 −1.69 −1.85 * | not comparable, 0/4 open |

Open order OLMo-2-1B, Olmo-3-7B, pythia-6.9B, Tulu-3-8B. Closed order sonnet-4-6, haiku-4-5, gpt-4o-mini. Ratios compare median absolute movement and are a magnitude claim only.

**The result.** Two of nine axes agree in direction with effects clear on both sides: abstraction and dominance. On those two, one system line moves a closed model 2.6 and 5.8 times further than the entire SFT plus DPO pipeline moves an open one. Two more invert with both sides clear, and two of the four inversions are the instruction-named rows.

So the closed contrast does not reproduce the open alignment axis. It reproduces two components of it and reverses or ignores the rest. Read against Y, the two it reproduces are the ones Y already identified as the register core: the shift toward abstraction and toward agentive vocabulary is available as a switchable mode, while the emotion rise, the appearance rise, and the need drop are not moved by a system line at all or move against it.

That is a dissociation between a prompt-recoverable component of the register and a weight-resident one, measured on the same nine axes. It does not show that any part of alignment is reversible by prompting: the closed models are RLHF'd in both conditions, so the most this licenses is that the abstraction and dominance components are **available to prompt control in an already-aligned model**, which is weaker and still worth having.

## 6. The projection onto a pooled axis mostly does not work, for a reason worth recording

The first version of section 5 tried to locate frontier output as a fraction along the open base-to-DPO axis. Of nine measures, three get a ratio at all and only two of those are usable: `F:emotion_and_arousal` at 0.46 to 0.85 of the way along, and `R:need` beyond open DPO at 3.4 to 4.0. `N:concreteness=concrete` gets a ratio but the two frontier framings straddle the open base level (+1.27pp continuation, −4.70pp chat), so it locates nothing. The remaining six rows have a pooled open axis under 0.7pp wide and are refused a ratio outright.

The narrowness is the finding. Families move by different amounts and in two cases different directions, so a median across families flattens the axis: concreteness pools to a −1.50pp span against per-family totals of −3.12, −0.53, −2.02, −1.66 on the ladder's own prompt set. A first pass computed the axis as a difference of cross-family median **levels** rather than a pooled **difference**, which discards the pairing the ladder result rests on and collapsed concreteness to −0.97pp, at which point eight of nine rows printed as too flat to project onto. Fixed in `regimes()`; the note is in the source at the `span` assignment.

A ratio also needs its denominator watched. On a 0.26pp axis a 1pp frontier gap prints as "4x open DPO" and reads as an enormous effect, so the output now prints the absolute gap beside every fraction and refuses the fraction below a 1pp span.

## Limits

- **The closed side has no base model and never will.** Section 5 compares retraining against prompting. They are not the same manipulation, and a larger closed number does not mean more alignment.
- **Framing does not match across the open/closed line.** Open generations are bare completions with no chat template; neither frontier condition is that, since both go through the chat endpoint. Section 6 therefore reports the closed side as a range spanned by its two conditions rather than a point, and the width of that range is the framing uncertainty left visible.
- **Tulu's SFT rung is the no-safety-data ablation.** Plain `Llama-3.1-Tulu-3-8B-SFT` has 231 passages in the stash against 5,300 for the ablation, too few for a prompt-matched contrast, so that chain's step 1 is SFT without safety data and is labeled that way throughout. It is arguably the more interesting rung; it is not the same rung as the other three families'.
- **Sections 1 to 3 and section 5 use different prompt populations.** The ladder intersects base, SFT and DPO (73 prompts); section 5 intersects base and DPO only (more prompts). So OLMo-2's emotion total reads +1.16 in section 3 and +1.02 in section 5. Same measure, different population, not a discrepancy.
- **Twelve passages per cell**, sampled by sorted index so the same passages are read on every run. Cells hold 40 to 80; the cap is what keeps this a minutes-long run.
- **No multiplicity correction.** Nine measures times four families times three steps. The starred cells in sections 1 and 2 are far from the 0.05 boundary; the marginal ones elsewhere should be read as exploratory.
