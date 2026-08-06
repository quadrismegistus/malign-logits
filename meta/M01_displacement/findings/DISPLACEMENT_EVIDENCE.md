# Displacement: the evidence, for and against

Started 2026-08-06 at RH's request, as a running document. **This is the paper-facing view.** Three documents now index this campaign and they do different jobs:

- `ledger.md` is the DETAIL layer, clause by clause, with supersessions and docket citations.
- `README.md` is the AXIS index: what the campaign found, organised by question.
- **this file is the VERDICT view**: what counts as evidence for displacement, what qualifies it, what went against it, and what remains open.

**It inherits README's anti-drift rule and adds one.** No number appears here that does not live in a `findings/` write-up or a registration result. And **every row carries its fence in the row**, because a claim quoted out of a table loses the paragraph that qualified it. Where a fence is long, the row says so and points at the section.

Status vocabulary, used strictly:

    CONFIRMED       ran, cleared its stated test, fence attached
    BOUNDED NULL    ran, no effect detected, MDE stated -- not "no effect"
    NOT SUPPORTED   ran, failed its stated test
    MID-RUN         result exists, population not closed, not yet quotable
    WITHDRAWN       was claimed, then struck; kept visible so it cannot return

---

## FOR: five kinds of evidence, not five instances

The five differ in what they license. Grouping by registration letter hides that; grouping by question shows it.

### 1. That it happens — distributional

| claim | number | status |
|---|---|---|
| Mass leaves the unresolved tail for nameable words | **91.0% of 82,775 cells** (75,340), 90.9% under adversarially corrected input; 34 clusters, 44 edges, 2,199 English stimuli | CONFIRMED |

Registration N, one-sided, direction registered in advance. `results/result_n_primary.json`, `_reading: SUBSTITUTION CONFIRMED`.

**Fence:** quote the sign split, not the Stouffer Z. The Z is ceilinged at 8.3265 for most clusters, so the pooled 47.9 is a floor and identical across both arms to fifteen digits.

### 2. What shape it has — a chain, not a swap

| claim | number | status |
|---|---|---|
| Few large withdrawals, many small uptakes | **206 rising categories against 36 falling**, fallers **3.8x larger** each (-0.01267 vs +0.00334), Mann-Whitney **p = 5.8e-09**, ratio exceeds one in every lexicon | CONFIRMED |

`findings/T_category_flow.md` finding 14.

**This is the finding that earns the word "displacement" rather than "suppression."** Mass leaving one place and spreading across many is what the pattern is. **Fence:** the count claim carries a granularity caveat, stated at T.15.

### 3. Which direction — three instruments, no shared design

| route | claim | status |
|---|---|---|
| taxonomy | Off contact, motion and force onto perception, cognition and speech, on **six lexicons sharing no design**, no category reversing | CONFIRMED |
| geometry | The frequency-residualised axis orders T's categories on all six, **median rho +0.498** against the raw axis's +0.367 | CONFIRMED |
| annotation | Promoted word judged **milder 3.6x as often**, more intense **5x less**, more punishable **3.1x less**; all p < 3e-15, d 0.36-0.40, 510 items, seven annotators | CONFIRMED |

`findings/T_category_flow.md`; `findings/V_embedding_regions.md` section 6; `findings/S_annotation.md` finding 1.

**Why the geometry route matters and what it does not do.** It answers the standing objection that T's ordering is an artefact of how the categories were drawn: an embedding space with no categories in it recovers the same ordering. **It is not independent corroboration of alignment's direction** — the axis is `mean(risers) - mean(fallers)` and T's delta is riser share minus faller share, so a rising category sits at the riser end partly mechanically, and the two share 1,358 of T's 1,361 prompts. The clean quantity is the raw-vs-residualised gap, which the mechanical relation cannot produce. See V.6 and V.7.

**Why the annotation route is not circular.** The annotator is never told which word moved. Judgement is made about two words in a slot and the direction recovered afterwards.

### 4. Cross-lingual — the substitution travels, the affect does not

| hypothesis | reading | within-pair sign agreement, en vs zh |
|---|---|---|
| H1, substitution | **SUPPORTED IN BOTH ARMS** | **83.9%** (2,138 of 2,549 pairs) |
| H2, absolute valence decreases | NOT SUPPORTED, English arm only | 52.5% |
| H3, arousal decreases | NOT SUPPORTED, English arm only | 56.7% |

Registration O, `results/result_o_primary.json`. 9 clusters, 9 edges, 301 pairs, 2,709 cells per arm.

**The contrast between 84% and 53% on the same pairs is the cleanest way to say it.** Substitution crosses languages; the affective story does not.

**Data-quality flag, unresolved:** `bloomz-7b1` has a competence share of **0.109** where every other model in the population sits between 0.54 and 0.68. One of nine edges runs on a model barely performing the task. Check whether it was meant to be excluded before the 9-cluster figure travels.

### 5. What kind of operation — substitution, not deletion

| claim | number | status |
|---|---|---|
| Forcing the aligned model to utter the demoted word costs it essentially nothing | swap_base +0.0106 (MDE 0.0182), swap_algn -0.0002 (MDE 0.0592), dd -0.0108 (MDE 0.0467), constraint cost +0.0055 (MDE 0.0602) | **MID-RUN**, bounded null |

Forced-continuation run. Population: cross-model-recurrent movers, 32 of the 36-pair roster; both Falcon-H1 and both Falcon-Mamba pairs absent on a `selective_scan_cuda` conflict. **Not quotable until pass 2 lands.**

**Conceptually this is the load-bearing one**, because it is the only evidence separating displacement from removal: the word remains fully sayable and is simply not chosen. What changed is the horizon of the expected, not the range of the possible. Estimates have shrunk toward zero as pairs were added while MDEs tightened, which is a true null being approached rather than an effect too small to see.

---

## QUALIFYING: what narrows the claim

**The substitution is not transgression-specific; only the withdrawal is.** T.13's sentence for the paper: *alignment's withdrawal is transgression-specific in how many words it pulls down and in how far it pulls them; its substitution is not transgression-specific in either.* This must be held together with S.2 below or the paper overclaims.

**But the softening IS stronger at transgressive sites.** S.2: the softening runs **2 to 3 times larger** in the transgressive version than its neutral twin, replicated across two disjoint samples. The two are compatible because they measure different things — T.13 whether substitution occurs and how big, S.2 how much milder the substitute is. Say both.

**The suppression is graded by bodily harm rather than by transgression.** Violence six times property; taboo invisible. `findings/S_annotation.md` finding 3.

**Verschiebung in the strict sense was not detected.** The within-item coupling that would constitute it failed at its stated MDE. S's own "what did not hold."

**The substitution reverses sign between frames.** Registration Q. Magnitude confirms at the twin and dies in the corpus.

**Rate null, magnitude confirmed.** F/G: not more often, not more sharply, but larger at sites.

**P's ACT result is narrower than it is usually stated.** The finding is that all three coder families independently read the risen word as an **exclamation** more often than a stationary control drawn from the same faller — not speech-act change in general. **CONFIRMED under LLM coding, explicitly not human validation**, and the agreement statistics are weak: Krippendorff's alpha on the speech-act variable is **0.269**, and `gpt-4o-mini` barely uses the `NOT_COMPARABLE` and `SAME_PITCH` categories the other two use freely (3 and 4 uses against Claude's 77 and 44). The confirmation rests on each family independently showing the effect, not on item-level agreement.

**The direction of displacement is a property of the scene, not of alignment.** V.5: a marked site's displacement vector agrees with its twin's at **0.327** and a random site's at **0.060**, fivefold, 14 of 14 families, p = 0.0001, surviving both the frequency residualisation and the verb restriction. This is why pooled instruments kept failing, and why a reader sees the relation instantly on one item while no aggregate recovers it.

---

## AGAINST: what failed

**The paired relation cannot be located geometrically.** Six grains: four similarity instruments (ledger clause 6), P's REF stratum, and V's regional, set-level, scene-controlled and directional tests. **NOT SUPPORTED throughout.** We can show that risers differ from fallers in aggregate; we cannot show that *this* riser is near *that* faller.

**P's REF stratum: NOT SUPPORTED, single-coder.** One of three coder families showed the effect. This was metonymy's most direct annotation test.

**Registration R's decoy programme closed as a negative with a cause.** Both primaries failed. The mechanical diagnosis: every matched control population carried its own lexical character (the argmax over-drew light verbs) and that character was the effect.

**The regional test is a null with its artefact confirmed.** V.1: no cluster structure to begin with (silhouette 0.046-0.061), and region rise/fall tracks how content-word-y a region is, significant at every resolution from k = 20 up.

---

## SURPRISING: results nobody predicted

**Safety data is not what produces displacement.** Findings U: the operation survives removal of the safety split at 90% strength (full 0.0651, no-safety 0.0583), and removal of math, persona and wildchat data does the same. **No second ablation suite exists in the open-weight world** — this is "cannot be replicated with anything that exists," not "wants replication."

**SFT does the cutting, not preference optimisation.** U: SFT carries 74% of ladder JS; four preference methods produce zero fallers per site.

**The substitution is general and only the withdrawal is marked.** T.13, above. The expectation was that both would be transgression-specific.

**S.2's specificity was never predicted** and replicated across two disjoint samples anyway.

**The frequency control opened a line it was expected to close.** V.6 was predicted, in the script header before the run, to end the geometric programme. It produced the only lexicon-free corroboration T has. The wrong prediction is left standing in the header.

**"Bodily action to cognition" was the wrong caption and the control that found this out was RH's.** V.7: the body-specific norm dimension is *weaker* than the general one (LSN-Hapt -0.235 against MT-Conc -0.361) and they are separable on our verbs (r = +0.486). No published perceptual norm explains much of the axis — the largest is 13% of variance and one concreteness measure gives -0.021, p = 0.61. The axis is semantic in T's sense, not perceptual.

---

## OPEN, AND ONE NOW CLOSED

**Does the resist asymmetry survive where models barely concentrate?** The committed test's fitted asymmetry at zero entropy drop is **-0.0867, 95% CI [-0.1383, -0.0352]**, excluding zero, on 32 pairs. It travels with three qualifications, always: the intercept is driven by the full range; two curvature-permitting forms disagree with the intercept moving toward zero as low-end curvature is allowed (linear -0.0867, log1p -0.0640, sqrt -0.0120 spanning zero); and a fit local to the low-concentration regime gives **-0.0205 with an interval spanning zero**. Residuals are systematically positive at the low end (six of eight lowest-drop pairs above the line), which makes the full-range intercept biased rather than merely precise.

**RESOLVED 2026-08-06, and as the second branch: it is structurally unresolvable, not under-powered.** The roster query ran on a criterion declared before looking (concentrates little = entropy drop < 0.10 nats; aligns strongly = fallers per site at or above the 44-pair median of 13.0; scoreable from `true_word_probs` without new generation). **Of 44 candidate base-to-superego pairs, NONE meet both.**

The five low-concentration pairs show why, and the pattern is more informative than the count:

| pair | entropy drop | displacement |
|---|---|---|
| Qwen2.5-0.5B | +0.0033 | 3.5 fallers/site below median |
| phi-4 | +0.0812 | 10.2 below median |
| pythia-2.8b (x3, archangel arms) | +0.086 to +0.089 | **0.0 — displaces nothing at all** |

**Every pair that concentrates little also displaces little, and three of the five displace nothing.** In the published population entropy drop and displacement strength are coextensive rather than merely correlated, so asking whether the asymmetry survives without concentration is asking whether it survives without the operation that produces it. **The cell cannot be populated. The caveat retires instead of travelling.**

**AND THE COVERAGE IS BETTER THAN A HEDGE ABOUT "THE PUBLISHED POPULATION" SUGGESTS.** RH pushed on this and he is right. The registry holds 120 models: **39 families carrying a base and at least one aligned member, across 30 organisations, and across six distinct post-training objectives** — SFT, DPO, KTO, PPO, SLiC, RLVR. The empty conjunction is not "everyone ran the same recipe so of course they all concentrate." It is six objectives and thirty orgs, and none of them aligns strongly while leaving entropy alone.

**So the caveat was the wrong shape, and it should be read as a result instead.** Concentration and displacement are coextensive across every published way of aligning a model. That is a claim about alignment *as actually practised* — a converged industry, not one method's artefact — and it is more interesting as a finding than as a hole in the roster.

Three limits, all narrower than the original hedge.

- **The lineage count is not citeable.** `family` is not "independent pretraining lineage": `falcon3-1b/3b/7b/10b` is four entries and one pretraining run, and the same collapse applies to the five `olmo*` entries, `qwen`/`qwen-tiny`/`qwen3`, `llama`/`llama-70b`, `smol`/`smol3`. Collapsed it is roughly two dozen. **The collapse rule has never been written down**, which is why this campaign has quoted four different roster counts in one evening. Cite the 39 families and 30 orgs, which are countable, and publish the independence map before citing lineages.
- **The edge is the chain, not the rung.** The query ran on base-to-superego pairs. No family was lost by that — every base-plus-SFT family also has a base-plus-superego member, set difference empty — but the base-to-SFT edge *as such* was not the unit, and findings U has SFT doing the cutting while DPO adds a fifth. The entropy-displacement relation could differ by rung. Unresolved and specifiable.
- **"Aligns strongly" has one operationalisation.** Fallers per site at or above the 44-pair median, declared before looking but untested against alternatives; a JS-divergence or top-k-overlap criterion might admit a pair this one excludes.

**And it makes the committed intercept weaker, not stronger.** The intercept extrapolates to a regime that does not exist in the population, so **the committed number describes a counterfactual model: one that aligns at typical strength while leaving entropy alone.** No such model has been published. This is the fourth companion sentence and it must travel with the other three.

**A method note worth keeping.** The answer was a property of the roster, not of the estimator, and four fits were spent establishing that the estimator could not settle it. When a quantity is read at the edge of the data, ask what populates that edge before asking which curve to fit through it.

**Does the cost live downstream?** The discriminator between chain substitution (no cost anywhere) and full topography (cost in integration) was specified as three channels: containment, frame-break, and renewed displacement after the forced word. **One has run.** Position-resolved damage is flat across all ten continuation positions, LATE minus EARLY -0.0022 at p = 0.98 — but MDEs grow from 0.021 early to 0.110 late, so the instrument is weakest exactly where the competitor predicts the effect. **A model can pay zero probability-cost for the forced word and still steer back to the sanctioned frame, and steering is cheap.** The two remaining channels need no new generation.

**The bound that currently holds on the theory:** this instrument does not confirm the topographic account and refutes only its crudest form — the word buried and unsayable. Freud separates uttering from lifting (*Verneinung*, 1925), so a zero-cost forced utterance does not adjudicate between a cost that lives downstream and no cost at all.

---

## Provenance

    findings/S_annotation.md          annotation: milder/less punishable, site specificity
    findings/T_category_flow.md       category flow, the chain, the specificity split
    findings/U_ladder.md              which training stage, the ablations
    findings/V_embedding_regions.md   geometry: six failures, scene-locality, the axis
    results/result_n_primary.json     the substitution at scale
    results/result_o_primary.json     crosslingual, H1/H2/H3
    results/result_p_primary.json     annotation of the relation, ACT and REF
    ledger.md                         clause-by-clause detail and supersessions
    README.md                         the axis index
