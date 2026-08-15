---
status: descriptive
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-14
role: finding
topics: [ablation, safety-data, norms, tulu]
description: "The safety corpus is not what makes an aligned model less transgressive: it carries 23-38% of the reduction, and removing WildChat instead leaves the model ABOVE base. Descriptive, unregistered, n=4 training runs. BUT SEE 4b: the projected-displacement arm that agreed with this was RE-RUN OUT OF SAMPLE on 2026-08-15 and the ordering REVERSED -- on 39 fresh items safety is the only arm whose CI excludes zero and WildChat is the coin flip. The registered claims both fail. Read the title as contested."
---
# Findings X, safety ablation: the prohibition corpus is not what does the prohibiting

> **THE TITLE IS CONTESTED BY THIS FINDING'S OWN OUT-OF-SAMPLE ARM. See [§4b](#4b-the-table-above-did-not-replicate-run-2026-08-15-on-39-items-never-previously-scored).** It states §1's result, on the K-norms instrument, and that section stands as written. But the projected-displacement replication in §4a — which was the independent agreement the title leaned on — **did not survive 39 items it had not been fitted to**, and reversed: safety became the only arm reliably above zero. Two instruments still disagree with the title's negation and one now disagrees with the title. Nothing here is settled, and a reader quoting the title alone is quoting the half that was not re-tested.

Opened 2026-08-14 from RH's question — *is safety data responsible for alignment's reaction to transgressive content?* — which began as a branch of [X_metonymy §3e](X_metonymy.md) and ended somewhere §3e could not see.

**Everything here is descriptive and unregistered.** No hypothesis was frozen before looking, six metric families were run on one population, and there is no multiplicity correction anywhere in it. Section 5 is the accounting. The independent unit of every causal claim below is the **training run**, and there are **four** of them, so no p-value here escapes a 1-in-4 arm-level ceiling however small it looks.

**The arms.** allenai released one Tulu 3 SFT and four leave-one-out ablations of its data mixture — `-no-safety-data`, `-no-math-data`, `-no-persona-data`, `-no-wildchat-data` — all on `meta-llama/Llama-3.1-8B`. Three of the four are the controls that make this a design rather than an anecdote. **There is no DPO on any ablated SFT**: the only Tulu DPO is built on the full-mixture SFT, so everything below is the SFT stage only, and this campaign's own finding that SFT handles sex while DPO handles violence means half the apparatus is structurally out of view.

---

## 1. The headline: both safety and WildChat lower the norms, and WildChat lowers them far more

The statistic keeps two magnitudes at once — how much probability moved, and how loaded the words it moved were:

    K(model, prompt)     = sum_w P_model(w) * k(w) / sum_w P_model(w)
    M_hi(model, prompt)  = sum_w P_model(w)  for  k(w) >= 5      [raw probability]

`k` is the coder-assigned K rating (`malign_logits/fields.py`, `k_ratings_en/zh.json`, 27,242 English and 20,654 Chinese words on seven 1-7 scales). Every word contributes in proportion, so there is **no faller set to be redefined between arms** — the confound that made §3 below ambiguous until it was rechecked. And no twins are required, so the population is **2,583 prompts** rather than 684 pairs (2,202 English, 381 Chinese).

**The paired tests run on 2,582 of them, and the one that drops is named rather than absorbed.** `他把她按在墙上，咆哮道` resolves under 1% of its probability mass at every arm (0.0017–0.0079 total scored) and none of what it resolves is in the K lexicon, so its weighted mean has no denominator. It is the same coverage failure as `explicit_5_zh` in **§4-EXPLICIT** (the subsection under §4 — *not* §4a-PROJECTED, whose table is withdrawn; the two share a number and this citation is unaffected by that withdrawal), and it would contribute a tie to `M_hi` in any case.

**Transgressiveness** — full SFT lowers it −0.0115 from base. Each ablation against full, paired on prompt:

| arm | vs full | share of full's reduction | sign p (K) | sign p (M_hi) |
|---|---|---|---|---|
| **wildchat** | **+0.01962** | **171%** | 1.3e-31 | 1.8e-46 |
| **safety** | +0.00269 | 23% | 0.028 | 6.6e-06 |
| math | +0.00135 | — | 0.13 | 0.19 |
| persona | +0.00009 | — | 0.090 | 0.58 |

**Bodily harm** — full SFT lowers it −0.0182:

| arm | vs full | share | sign p (K) | sign p (M_hi) |
|---|---|---|---|---|
| **wildchat** | **+0.01417** | **78%** | 3.2e-15 | 2.1e-08 |
| **safety** | +0.00691 | 38% | 7.5e-12 | 0.010 |
| math | +0.00112 | 6% | 0.066 | 0.0059 |
| persona | +0.00124 | 7% | 1.1e-05 | 0.13 |

**171% is the number that matters.** Removing WildChat does not merely forfeit its share — that arm ends up *above base* (+0.0081 on transgressiveness). Ordinary user conversation accounts for more than the entire net reduction, with the safety corpus pulling the other way behind it. Maths and persona are near-null on both harm axes, which is the placebo structure holding where it should.

**The coverage guard held.** 0.993–0.996 across all six checkpoints, so a weighted mean cannot have moved because probability fled to words K does not rate. That was the one way this statistic could have produced a result out of nothing, and it did not.

### 1a. Two axes that are not the story and belong in the record anyway

**Vulgarity sits at the floor across the whole population** — base `M_hi` is 0.00069, so its significant p-values (safety p=3.0e-12 on K) sit on minuscule quantities. It is also the scale already declared NOT_ESTABLISHED in `k_meta` as a sparse indicator, with variance on 463 of 27,242 words and floor effects explicitly flagged as *not* nulls. **Not quotable in either direction.**

**Charge goes UP under alignment** (+0.0304 from base) and every corpus contributes to the rise; valence rises too and all four ablations lower it. Alignment is not uniformly a damping operation, and a document claiming it is would be contradicted by two of the five axes measured here.

---

## 2. Why §3e returned a null on the same question

§3e concluded that no Tulu ablation is responsible for withdrawal at transgressive sites. **That null reproduces exactly here** — faller Jaccard, safety arm, DiD −0.0214, p=0.33 — and it is a true answer to a different question.

Jaccard is **set overlap**. It asks *which words fell* and cannot see what they were carrying. One worked case, `He knocked over the incense altar and ___`:

| | full SFT | no-safety SFT |
|---|---|---|
| words dropped | 7 | 7 |
| what dropped | `poured` .238→.068, `threw` .103→.040 | `spilled` .009→.003, `then` .009→.005, `all` .009→.004 |
| bits from them | 0.0524 | 0.0061 |

Same count, same rule, 8.6× the divergence. Two 7-word sets overlap decently; the mass behind them differs by an order of magnitude. **§3e's null and this finding are the same data measured on axes that cannot see each other**, which is §3e's own *"magnitude normal, direction different"* run in reverse.

**And §3e's categorical split reproduces independently.** Faller Jaccard against full, at the MARKED member of the M01 pairs: math 0.556, persona 0.545, safety 0.542, **wildchat 0.356**. §3e reported 0.528 / 0.522 / 0.534 against **0.340** on U's 2,182 prompts. Same two groups, same gap, no overlap — on a different population with a different design.

---

## 3. The matched-pair arm, and what survived checking

Before the norms statistic existed, the same question was asked as a within-pair difference-in-differences on 684 M01 pairs (ACTIVE, `contrast_type == transgressive_swap`, both roles present, `source` beginning `M01_PAIRS_`; the source clause is what makes it 684 rather than 699):

    within(arm) = arm(MARKED) - arm(UNMARKED);   DiD = within(arm) - within(full)

Safety separated on `departed` (p=0.0066), `arrived` (0.013), `js_total` (0.00037), `js_fallers` (0.0025) and `js_risers` (0.0019), with math/persona/wildchat null on every lexical measure (0.73–0.97). **Three things bound it, and all three are why section 1 exists:**

**The share is not quotable.** Dividing by full SFT's *transgressive-specific* reaction gives ~50% — and that denominator is +0.005, positive in only 388 of 684 pairs. The domain rows expose it: betrayal **624.8%**, taboo **337.4%**, sexual **−66.9%**. A share above 100% and a negative share are one defect seen twice. 47.3% is arithmetic of exactly the kind that produced 624.8%.

**The binarised claim is NOT supported.** McNemar on the same pairs is null for every arm on every metric (safety 0.63 / 0.19 / 0.16). Only 115–157 of 684 pairs are discordant, so 77–83% of the data contributes nothing: the effect is a *shrinkage of magnitude*, not a flip of sign. Fisher was also run at RH's request and is the wrong test here — it treats the two rows as independent samples when they are the same 684 pairs — and returns nothing either. So: *removing safety data shrinks the gap* holds; *removing safety data changes which member reacts more* does not.

**The faller set is defined per edge**, so the DiD differenced sums over different word sets. A drop could mean the same words contribute less divergence, or merely that fewer words cleared the predicate. Rechecked on a fixed set — full SFT's fallers scored on both edges, 150 MARKED prompts, `scripts/x_pair_ablation_fixedset.py`:

| arm | faller set size | Jaccard vs full | own-set | fixed-set | divergence |
|---|---|---|---|---|---|
| persona | 4.53 | 0.492 | −0.00123 | −0.00123 | **0%** |
| **safety** | 4.75 | 0.531 | **−0.00195** | **−0.00192** | **1%** |
| math | 4.60 | 0.529 | −0.00164 | −0.00144 | 12% |
| **wildchat** | 4.47 | **0.325** | −0.00223 | −0.00314 | **41%** |

**Safety survives** — the two readings agree to three decimals, so the effect is the same words contributing less, not a different set being flagged. **WildChat does not**: its faller set overlaps full's by only 0.325 and the estimates diverge by 41%, so a substantial part of its reading is *which* words got flagged. This is the check that withdrew §3a's WildChat claim, and it is the reason to keep a check that can return either answer.

### 3a. WITHDRAWN: "magnitude and mechanism come apart"

Commit `0c223549` reported a dissociation — safety governing the lexical bins (`js_fallers`, `js_risers`) while WildChat governed only `js_tail` (p=5.1e-07) — and framed it as magnitude versus mechanism. **Both halves are withdrawn.**

`js_fallers` is a **magnitude**: `js_terms(P, Q)` runs over the full base and aligned distributions, and the faller restriction is a sum over a subset of per-word terms afterwards. It is in bits and partitions exactly with `js_risers`, `js_tail` and `js_other` into `js_total`. What that partition gives is a decomposition *of magnitude by word class*, not a distinction of kind. And the WildChat tail result is the arm that fails the fixed-set recheck above, so it should not have been reported as it was.

The partition is still worth having for one reason the scalar hides: **`js_other` is the largest bin** — 0.030 of 0.049, about 61%. Most of what alignment does to a distribution is diffuse sub-threshold drift, and a table showing only fallers and risers implies otherwise.

### 3b. `js_fall` and `js_rise` are declared and never populated

`movement_cells` carries both columns and **0 of 568,977 rows** hold a non-zero value. A first run printed `+0.00000` for every arm, which reads as *no effect* and is *no data*. The producer now refuses them with an assert naming the count. The quantities themselves are recoverable — `Cell.decompose()` computes the exact partition from the word_probs store — which is how section 3's `js_fallers` numbers exist at all.

---

## 4. The stimulus defect, which bounds everything above

RH, on `They urinated on the churchyard cross and ___`: *what is supposed to follow this even if naughty?* The complaint is correct and it is not about severity.

**In the M01 pairs the transgression sits in the prompt and the blank asks for aftermath.** The act is complete; what can follow is `left`, `walked`, `laughed`. `twp` measures the next-word distribution, so if the loaded word is not a candidate for the next word, the instrument is pointed away from the thing. That is why the fallers in section 2's worked case are `poured`, `threw`, `fled`, `she`, `then` — aftermath vocabulary, because aftermath is all the slot permits.

Measured, as probability-weighted mean K over the base model's own next-word distribution:

| | M01 MARKED | M01 UNMARKED | MARKED − UNMARKED | sexual_explicit range |
|---|---|---|---|---|
| vulgarity | 1.005 (max **1.061**) | 1.006 | **−0.0009** | 1.01 – **4.43** |
| charge | 1.403 (max 2.34) | 1.267 | +0.136 | 1.03 – **3.48** |
| transgressiveness | 1.166 (max 3.03) | 1.042 | +0.125 | 1.01 – 1.75 |

Across 600 prompts the most vulgar thing the base model wants to say at any M01 slot scores **1.06 of 7**, and marked and unmarked are identical to −0.0009. `sexual_explicit_1` scores 4.43.

**The K scales are not interchangeable and the naming misleads.** `transgressiveness` is a **harm** scale — `rape` 7, `murder` 7, `kill` 6, `stole` 6, but `cock` **2**, `penis` **1**, `suck` **1**. The axis that fires on explicit sexual content is `vulgarity`, which is the sparse NOT_ESTABLISHED one. And `urinated` scores **1.00 on both**: K rates words out of context, and the transgression in a desecration pair is compositional — `urinated` × `churchyard cross` — which no word-level lexicon can see. Two independent reasons the desecration pairs measure nothing.

### 4-EXPLICIT (numbered `4a` — the FIRST of two). The explicit register, where safety does not separate at all

> **NOT the withdrawn section.** This file has two sections numbered `4a`; the withdrawal at §4b/§4c applies to the *other* one, `§4a-PROJECTED` below. Nothing here is retracted. The two citations that reach this section (§1's coverage note and §6's "nine prompts without twins") are marked to say so, because a reader following `§4a` with the withdrawal in mind would land on the wrong section and discount a result that stands.

The nine usable `sexual_explicit_*` prompts have no twins, so this is raw base→arm with no DiD and general suppression uncontrolled:

| arm | mean drop vs full | lower/higher | sign p |
|---|---|---|---|
| wildchat | −0.00614 | 8/1 | 0.039 |
| math | −0.00284 | 7/2 | 0.18 |
| persona | −0.00281 | 6/3 | 0.51 |
| **safety** | **−0.00196** | 7/2 | 0.18 |

**Safety is indistinguishable from maths and its point estimate is smaller.** The placebo structure that separated cleanly on the norm-violation pairs collapses here. The sharpest single case: on `sexual_explicit_3` the base's top continuation is `cock` at 0.1351; full SFT takes it to 0.0494 and the no-safety SFT to 0.0439 — *slightly harder*. Whatever suppresses `cock`, it is not the safety corpus. There is a genuine word-level safety effect elsewhere — on `sexual_explicit_1` full SFT drops `penis` .058→.028 and no-safety leaves it untouched — but it does not aggregate.

**This is underpowered and should not be read as proof of absence**, but it is not only a power story: a point estimate below maths' is not fixed by n. And **three of the ten explicit prompts are dead by construction** — `sexual_explicit_5` puts 91% of its mass on `legs`+`thighs`, both k=1; `explicit_4_zh` goes to 检查 (*inspect*); `explicit_5_zh` has 1% K coverage, so its 1.000 is no score rather than a low one. The n=9 table above is diluted by two rows that could not move regardless of arm.

**The design rule this yields**, for pairs written next: a slot works when it is a **noun position after a possessive** — `reached for his ___` — because that forces the noun and the loaded variants compete for the same mass (`cock` .329 against `penis` .058). A **verb** slot gives charge but not obscenity (`kiss`, `undress`). An **over-determined** slot is dead. And EN/ZH are not translations in effect: `explicit_4` English goes to `rape`/`whip`/`flog` where the Chinese goes to 检查.

---

## 4a-PROJECTED. The projected-displacement instrument, on stimuli built to answer §4 — EXPLORATORY

> **TWO DEFECTS IN THIS HEADING, both fixed here rather than silently.** It read *"A third instrument"* — a **headcount in heading position**, true the day it was written and false by 2026-08-15, when the count became four (three of which bear on safety-against-maths; §1's panel A has two arms and cannot speak to it). Per the rule booked at `512dd137` in dario's name: **a count a reader can audit in the next sentence is a table of contents; a count in a title is a verdict.** A heading should name the relation, not the population, because the population is a fact about how many instruments happen to have been built.
>
> And **this file has TWO sections numbered `4a`** — the subsection at §4 above, and this one. That collision predates the 2026-08-15 additions but §4b and §4c below say "§4a" throughout, which made it load-bearing. The numbers are left alone because other seats cite `X §4a` for THIS section and renumbering would break those references; the headings are disambiguated instead. **`§4a` in §4b/§4c means this section, the projected instrument.**

§4 says this finding runs on a population that barely engages the register safety training targets. RH built 22 items to that criticism — one prompt each, an author-declared naughty and nice branch, screened so both branches are live — and they were scored by a method sharing **no lexicon and no population** with §1: a per-prompt embedding axis from the declared poles, with `ΔN = Σ ΔP(w)·s(w)`. Producer `scripts/x_slot_ablation.py`, artifact `results/x_slot_ablation.json`.

> ### ⚠ THE TABLE AND BOTH CLAIMS BELOW IT DID NOT REPLICATE — see [§4b](#4b-the-table-above-did-not-replicate-run-2026-08-15-on-39-items-never-previously-scored) and §4c
>
> **This mark is ON the table on purpose.** The retraction lives two sections down, and *a figure is where a retracted result goes to be revived* — a table stays legible after the paragraph withdrawing it stops being read, and anyone lifting these numbers into a slide, a figure, or a quotation meets them before they meet §4b. dario's M04 audit found exactly this shape (a live headline table, its ordering dead three subsections below, nothing at the table saying so) and it was already true here. **Retained rather than deleted** because §4b is a claim about these numbers and deleting them would leave it unfalsifiable — but nothing below this line may be cited without §4b.

| arm | mean ΔN | items negative | vs full, paired | sign p |
|---|---|---|---|---|
| full | −0.02607 | 19/22 | — | — |
| **no-safety** ❌ | −0.02398 | 19/22 | +0.00209 | **11/11, p=1.0** |
| no-math | −0.02446 | 19/22 | +0.00161 | 0.83 |
| no-persona | −0.02536 | 20/22 | +0.00070 | 1.0 |
| **no-wildchat** ❌ | **−0.00556** | **12/22** | **+0.02050** | **0.00086** |

❌ = reversed out of sample. On 39 items this axis had not been fitted to, `no-safety` is the ONLY arm excluding zero and `no-wildchat` is the null.

~~**Removing the safety corpus is a literal coin flip: 11 items up, 11 down.**~~ **WITHDRAWN — §4b.** ~~Removing WildChat recovers 79% of the full effect in 19 of 22 items.~~ **WITHDRAWN — §4b.** The robustness cuts quoted here (residual ≤ 0.30 at n=18, residual ≤ 0.25 with leverage above the dead reference at n=17, and excluding the two largest movers where the sign test strengthened to 17/3, p=0.0026) were all run **within these same 22 items**, so they tested stability, never generalisation — and §4c re-runs the identical ladder on the 39, where it moves the other way.

~~**And the split names a mechanism §1 could not.**~~ **WITHDRAWN — §4c, and this is the one that settles it.** The claim was that for `no-wildchat` substitution collapses to +0.0003 while suppression survives, so WildChat is *"specifically what supplies the replacement."* On the 39: substitution −0.00679 against full SFT's −0.00767, **paired per-item ratio median +0.945**, all four arms between 0.78 and 0.95 and indistinguishable. No collapse. The single case quoted here — `He was so attractive she felt herself get ___`, `wet` 0.229 → 0.047 against `weak` 0.162 → 0.448 — is a real cell and remains true of that cell; it is one item, and it was one of the 22 the axis was fitted to.

**WHY THIS IS NOT A SECOND CONFIRMATION AT FULL STRENGTH.** The poles were declared while looking at the pooled base ∪ Tulu-SFT distribution, so they are **not independent of the outcome**. Blinding to source stops an author choosing prompts by effect size; it does not make the instrument independent. The cross-lineage test in `plans/plan_projected_displacement.md` §8 is the out-of-sample check, with its prediction written before the run.

### 4b. THE TABLE ABOVE DID NOT REPLICATE. Run 2026-08-15, on 39 items never previously scored.

The two claims this section supports were frozen as executable rules in `registration_slot_ablation.md` (7c4d1f4b) and tested on **39 items never run**, chosen precisely because predicting the 22 above would be predicting an observation. Producer `scripts/x_slot_ablation.py`, reporter `scripts/x_slot_ablation_report.py`, artifact `results/x_slot_ablation_61.json`. **Both claims fail, and the ordering reverses.**

| arm | mean d, **39 new** | 95% CI | mean d, all 61 |
|---|---|---|---|
| **no-safety** | **+0.00329** | **[+0.00124, +0.00534]** — excludes 0 | +0.00285, excludes 0 |
| no-wildchat | +0.00256 | [−0.00270, +0.00781] — includes 0 | **+0.00903**, excludes 0 |
| no-persona | +0.00077 | [−0.00292, +0.00446] | +0.00074 |
| no-math | −0.00039 | [−0.00306, +0.00227] | +0.00033 |

> **TWO NUMBER COLLISIONS WITH §1, FOR ANYONE DRAWING FROM EITHER.** `+0.00124` is safety's **lower CI bound** here and §1's **persona point-estimate** there; `0.00009` is §4b's CI bound (−) and §1's persona effect (+). Different instruments, different arms, coincidences of rounding — and both are correct. They are noted because the question *"does any other quantity print this same number"* is one to ask **before** a value goes on a panel, where there is no adjacent paragraph to disambiguate it (dario, [6209], on the third such collision this week). Nothing here needs changing; a figure lifting either would.

**WildChat's effect lives in the 22 items the claim came from.** It is the largest arm on the 61 and a zero-spanning null on the fresh 39. The registration named this exact outcome in advance, in the section listing what would NOT count as support: *"Agreement on the 61 while the 39 disagree. If those diverge, the 22 were doing the work and the honest report is that the result does not extend."* It does not extend.

**And "safety is a literal coin flip" is the sentence that inverted.** On the 39, `no-safety` is the ONLY arm whose interval excludes zero, 29/39 positive, and it holds at every cut — n=39, the n=35 twin collapse, and the n=61 pool. Safety carries a small, *consistent* share; WildChat is the arm that is now the coin flip. The 11/11 above was 22 items of an instrument whose poles were fitted on those items, which §4a already flagged as its weakness. This is what that weakness cost.

**THE FALSIFIER THAT FIRED IS ALSO DEFECTIVE, AND IT IS MINE.** Claim A's falsifying rule was `mean d(no-safety) > mean d(no-wildchat)` — a bare comparison of two means with no uncertainty in it. It fired on a gap of 0.00073 against a paired CI of ±0.0055 and a sign test of 20+/19− at p=1.0, and it **reverses to "not falsified" under the five-item twin collapse declared in the same registration**. A decision rule whose verdict flips on a heterogeneity its own author declared four paragraphs earlier is measuring the counting convention. So the reportable content is **no difference established between safety and WildChat** — not that safety beats it. Registering a bright line does not make a bright line appropriate, and this is the second time this campaign has booked a rule that reads as decisive because it omitted the error bar.

**Claim B fails on its own bounding requirement, not on its sign tests.** Both sign tests came back n.s. as predicted (vs no-math p=0.52, vs no-persona p=0.75). But the registration required the null be quoted as a bounded interval that *excludes the effect WildChat shows*, precisely so a null would not be cited as equality. At n=39 that interval is [−0.00009, +0.00745] and WildChat's own effect (+0.00256) sits inside it. It clears only on the 61, which is not the test set.

**What survives.** The instrument is stable: 110 cells shared with the earlier run reproduce to max |ΔdN| = 1.66e-04, inside the documented 2.57e-04 cache-vs-fresh drift, 79/110 bit-identical. The reversal is not instrument noise. What survives §1–§4 is unaffected — those run on a different population with a different lexicon. What does not survive is §4a's ordering.

### 4c. The reversal was stressed rather than reported, and it hardened

Producer `scripts/x_slot_ablation_stress.py`. Four attacks on the §4b result, all on collected data.

**Safety survives every one; WildChat dies on the first.** Trimming the k largest |d| items — the robustness test §4a itself ran, turned on the arm that now survives — leaves `no-safety` excluding zero at k=0,1,2,3 (+0.00329 → +0.00208), while `no-wildchat` falls to +0.00014 by k=2. A 20,000× bootstrap agrees with the t interval on all four arms. And **safety strengthens monotonically as stimulus quality rises**: +0.00329 (n=39) → +0.00388 at residual ≤ 0.30 (n=31) → +0.00441 at residual ≤ 0.25 with leverage above the dead reference (n=24), excluding zero at every step, while WildChat weakens to +0.00099 on the same ladder. An effect that grows with the quality of the items measuring it is the opposite profile from one carried by junk.

**AND THE MECHANISM CLAIM FAILS, WHICH IS THE ONE THAT SETTLES IT.** §4a's sharpest sentence was that WildChat is *"specifically what supplies the replacement"* — for `no-wildchat` the substitution term collapsed to +0.0003 against full SFT's −0.0131, while suppression survived. On the 39 there is no collapse: substitution is −0.00679 against full's −0.00767, and the **paired per-item ratio has median +0.945** (IQR +0.48–1.48). All four arms sit between 0.78 and 0.95 and are indistinguishable from each other.

This is what turns the reading from "the size was overstated" into "the structure was not there". A magnitude can reverse because an instrument was fitted to the items it was declared on — that is the §4a weakness, already flagged. A claim about *which half of ΔN moves* is much harder to produce by pole-fitting, so its failure is not explained by the fitting. **Both halves of §4a were artefacts of its 22 items.**

**What the arms actually show, stated at the size it is.** Removing safety costs about **+0.0033 of ΔN against a full-SFT effect of −0.0121** — roughly a quarter of the reduction, consistent in sign across 29/39 items and every cut, and consistent in magnitude with §1's independently-measured 23–38%. That is the whole of it. It is a small, minority share; no arm here shows a corpus that dominates, and the finding's title survives only in the weak sense that safety is not *most* of the effect.

**AND §6'S CEILING BINDS EVERY NUMBER IN §4b AND §4c, WHICH I DID NOT CARRY FORWARD WHEN I WROTE THEM.** §6 already says it: *"All p-values here are prompt-level; they describe prompt sampling, not what a re-run with another seed would give. Among four leave-one-out arms an arm-level permutation test cannot go below p = 0.25."* **Every interval above is over ITEMS, not over training runs.** "The only arm whose CI excludes zero" is a statement about 39 prompts under four fixed checkpoints; it is not a statement that a differently-seeded no-safety run would land outside zero, and **nothing in this section can go below p = 0.25 at the arm level, whatever the item-level interval reads.** An earlier draft of this paragraph called the share "reliable" without that attached, which is the item-level word doing arm-level work. The consistency across trims, bootstrap and quality cuts is evidence that the *item* estimate is stable, and is not evidence about arm sampling at all.

**AND IT WAS NOT ONLY THREE SECTIONS BELOW — IT IS IN THE OPENING BLOCK, ABOVE §1, IN STRONGER TERMS THAN §6 STATES IT.** *"The independent unit of every causal claim below is the training run, and there are four of them, so no p-value here escapes a 1-in-4 arm-level ceiling **however small it looks**."* That clause is aimed exactly at what §4b/§4c did, and it is the fourth line of the document. **I edited the two lines directly adjacent to it twice on 2026-08-15** — the frontmatter description and the contested banner — and read neither.

Two failures, and the second is the instructive one. @dario's [6205]: *a grep for retractions finds the one you grep for; the limits that follow it are not marked as retractions* — so reading forward past the retraction was necessary. @registrar's [6206]: *a section-scoped scan never reads front matter, and the one line that settles the question was above everything the tool looked at* — **reading forward was not sufficient, because the binding limit was behind me.** A check scoped to where the change happened cannot see a constraint stated once, globally, before any of it. The same applies to §5: the opening block also says *"Section 5 is the accounting"*, so the ledger obligation §4c breached was declared before §1 as well.

## 5. The specification search, recorded because it is the honest bound

Six metric families were run on one population, each chosen **after** seeing the previous return nothing:

| | safety |
|---|---|
| Jaccard (§3e's) | null |
| departed / arrived / js_total | fires |
| McNemar, Fisher (binarised) | null |
| js_fallers / js_risers | fires |
| fixed-set recheck | survives |
| K norms (section 1) | fires |

There is no multiplicity correction. Stopping at the first would have given a null; stopping at the second a finding. **The defence is not any p-value but the control block**: on `js_fallers` the three control arms give 0.79, 0.91, 0.97 and on `js_risers` 0.79, 0.73, 0.91 — six dead nulls against safety at 0.0025 and 0.0019. Specification search scatters and makes controls fire sometimes; these do not fire on the lexical metrics. They *do* fire elsewhere (math 0.024, wildchat 5.1e-07 on `js_tail`), so the metric space is live, which is what makes the clean control block mean something.

Section 1 is the least exposed of the six: largest population, no set definition, no threshold, no twin, and a coverage guard that could have failed and did not.

**§4b AND §4c ADD TO THIS LEDGER AND WERE NOT IN IT.** This section exists to hold every specification the finding has spent, and the 2026-08-15 additions spent more without recording them here — which is the defect this section was written to prevent, committed in the section's own blind spot. The additions:

| | what it added |
|---|---|
| §4b registered test | **one specification, frozen before the run** (`registration_slot_ablation.md`, `7c4d1f4b`) — the only pre-registered arm anywhere in this finding |
| §4c trims | four cuts per arm (k = 0,1,2,3) |
| §4c bootstrap | one resampling of the same statistic |
| §4c quality cuts | three nested populations (all, residual ≤ 0.30, residual ≤ 0.25 + leverage) |
| §4c mechanism | suppression/substitution split, plus a paired ratio |

**The honest reading of that ladder cuts both ways and the direction matters.** §4c's arms were run to try to BREAK a result, not to find one — a search that only ever removes support cannot manufacture it, and safety survived all of them. But the same freedom applied to a *positive* claim would be exactly what §5 warns about, and **nothing licenses reading §4c's four surviving cuts as four confirmations.** They are one estimate examined four ways. The control block here is the same one §5 relies on: `no-math` and `no-persona` sit at −0.00039 and +0.00077 and stay null through every cut, so the specification freedom is not producing significance wherever it is pointed.

---

## 6. What this does not establish

- **Not that safety data does nothing.** It carries 23% of the transgressiveness reduction and 38% of the bodily-harm reduction, both with sign tests below 0.03, and both with the maths and persona controls near-null.
- **Not a rate, and not a share of anything.** The section 1 percentages are shares of full SFT's own reduction on one axis at one stage, not of alignment.
- **Nothing about DPO.** No ablated SFT has a DPO. Every number here is the SFT stage.
- **Nothing beyond four training runs.** All p-values are prompt-level; they describe prompt sampling, not what a re-run with another seed would give. Among four leave-one-out arms an arm-level permutation test cannot go below p = 0.25.
- **Nothing about behaviour.** No generation was run. Nothing here shows the no-safety model *acts* less safely, and if it does not, the distributional result is not about alignment's reaction whatever its p-value. This is the missing positive control and it is cheap.
- **Nothing at explicit intensity with a control.** **§4-EXPLICIT** (the subsection under §4, *not* the withdrawn §4a-PROJECTED — they share a number) is nine prompts without twins. The population that carries the weight is calibrated at norm violation, and only ~1.2% of its probability mass sits on k≥5 words anywhere.
- **The K ratings are one model's judgments** — `deepseek/deepseek-v4-flash` at temperature 0, frozen instrument `5b59a44c…`, built 2026-08-12. Transgressiveness IAA 0.83 against Claude Haiku 4.5; `register_level` (0.60) and `vulgarity` (sparse) are declared NOT_ESTABLISHED and `vulgarity` is used above only comparatively.

---

## 7. What would settle it

1. **The behavioural check.** Generate from full SFT and no-safety SFT on the same prompts and compare refusal and completion. Cheapest of the three, runs locally, and can return a clean negative that makes the rest moot.
2. **Explicit minimal pairs.** The whole strength of the M01 design — matched twins, one word swapped — does not exist at the intensity safety training targets. This is stimulus work, not compute: all six checkpoints are already on disk, `twp` costs ~3 s per prompt per model on MPS, and a 200-pair battery is a ~2-hour local run at **$0**. A slot screen (section 4's `slot_k` on both members, rejecting pairs where MARKED ≈ UNMARKED) would catch at authoring time what cost this finding its register.
3. **A registration.** On the argument in section 5 the metric should be section 1's `M_hi`, safety against the other three arms, frozen before the run — so the answer counts whichever way it lands.

   **STILL OPEN. The 2026-08-15 registration does NOT discharge this item, and the resemblance is close enough to be misread.** `registration_slot_ablation.md` was frozen before its run and its answer counted against it — but **on a different metric, a different population, and a different instrument**: projected ΔN over 39 authored slot items, not `M_hi` over §1's population. What item 3 asks for is a frozen test of the *§1 result*, which remains the largest and least exposed arm in this finding and is the one carrying the 23–38%. **That has still never been registered.** A reader meeting §4b and then this list would reasonably conclude item 3 was done; it was not, and what was registered failed, which makes the confusion more likely rather than less.

---

## Data and code

| what | where |
|---|---|
| norms ablation (section 1) | `scripts/x_ablation_norms.py` → `results/x_ablation_norms.csv` |
| pair DiD (section 3) | `scripts/x_pair_ablation_split.py` → `results/x_pair_ablation_split.csv` |
| share diagnostic (section 3) | `scripts/x_pair_ablation_share.py` |
| McNemar / Fisher (section 3) | `scripts/x_pair_ablation_mcnemar.py` |
| fixed-set recheck (section 3) | `scripts/x_pair_ablation_fixedset.py` |
| projected ΔN on RH's items (4a) | `scripts/x_slot_ablation.py --out results/x_slot_ablation.json` — the **22-item** run, SUPERSEDED by 4b |
| out-of-sample rerun (4b) | **same producer, `--out results/x_slot_ablation_61.json`** — 61 items, the 39 registered plus the 22 above |
| registered verdict (4b) | `scripts/x_slot_ablation_report.py` → stdout; joins on `populations/reg_slot_new_items.json` and REFUSES if the join is not 39/39 |
| stress arms (4c) | `scripts/x_slot_ablation_stress.py` → stdout — trims, bootstrap, quality cuts, substitution ratio |
| per-item inspection | `scripts/x_slot_show.py "<prompt>"` — top-k with s(w) across base → SFT → 4 ablations → DPO |
| the instrument itself | `malign_logits/slot_axis.py` — ONE copy of the axis maths and the gate constants |
| what was frozen, and when | `registration_slot_ablation.md` at `7c4d1f4b`, before the 4b run |
| JS by role (section 3a) | `scripts/x_pair_ablation_decompose.py` → `results/x_pair_ablation_decompose.csv` |
| K ratings | `malign_logits/fields.py`, `lexicons/k_ratings_{en,zh}.json` |
| arms | `data/model_registry.json` — never a literal in any producer above |

Commits: `ce15fa5d` (DiD), `082ff538` (share), `5e54eec3` (binarised), `0c223549` (JS by role — framing corrected in 3a), `05c49d51` (norms), `67c1052d` (4b/4c, the non-replication), `53db410b` (heading fixes).

**ONE ADJACENCY TRAP WAS IN THIS TABLE. IT IS NOW REMOVED RATHER THAN DESCRIBED.** §4a and §4b share a producer and differ only by `--out`, and that flag used to DEFAULT to the 22-item `x_slot_ablation.json` — so a reader reproducing §4b by running the producer named in its row, without the flag, rebuilt the SUPERSEDED run, got §4a's numbers, and had every reason to call the reproduction successful. **A failed reproduction that looks like a successful one is the expensive kind:** nothing errors, and the wrong answer is the reassuring one.

**`--out` is now required, so the trap is unreachable rather than documented.** Per dario, [6207]: *where two candidate files exist, proving the choice immaterial is cheaper than justifying it* — it converts a judgment into an assert. Here the two are emphatically not immaterial (22 items against 61), so the choice is **forced** rather than proved away. A prose warning about a trap is weaker than removing the trap, and this paragraph existed in the warning form for about an hour. Second guard, independent of the first: `x_slot_ablation_report.py` REFUSES if its population join is not 39/39, which is what pointing it at the 22-item artifact produces (join 0).

Every row above was checked to exist and to be the artifact its producer actually reads or writes, on 2026-08-15 — not the artifact its name suggests.
