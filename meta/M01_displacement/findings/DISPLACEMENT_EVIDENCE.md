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

### 5. What kind of operation — selection only, or a deeper restructuring

| claim | number | status |
|---|---|---|
| Forcing the aligned model to utter the demoted word costs it essentially nothing | swap_base +0.0106 (MDE 0.0182), swap_algn -0.0002 (MDE 0.0592), dd -0.0108 (MDE 0.0467), constraint cost +0.0055 (MDE 0.0602) | **MID-RUN**, bounded null |

Forced-continuation run, 32 of the 36-pair roster; both Falcon-H1 and both Falcon-Mamba pairs absent on a `selective_scan_cuda` conflict. **Not quotable until pass 2 lands.**

**The population qualifier belongs HERE and only here**, which was established 2026-08-06 after six hours of it being attached to everything in the run. Every pair has undisturbed beams for all 210 prompts of the stratified sample; the `k>=2` filter decided which prompts became forced SITES and touched the undisturbed arm not at all. So the resist asymmetry, the site-locality check, the entropy correlation and the committed intercept are all on the FULL sample and always were, while **the damage family alone sits on cross-model-recurrent movers** (51 forced sites per pair, median; range 1 to 85).

**And that inverts the argument this seat has been making for the substitution reading.** Its claimed virtue was narrowness: it rests on the damage null rather than the resist asymmetry, so it survives whatever happens to the asymmetry. True, and the cost is now visible — **the one measure it rests on is the single measure on the restricted population**, while the result it declines to lean on is the one on the full sample. Narrowness bought independence from a contested finding at the price of depending on the thinnest-populated one. Pass 2 generates forced units only, so it buys the asymmetry nothing and buys this everything: an unfiltered population, roughly 3.8x the sites, ten movers per site instead of one, a within-word control, and ten times the forced continuations for the two unrun discriminator channels.

**THE FRAMING HERE WAS WRONG UNTIL RH CORRECTED IT ON 2026-08-06, and the correction is recorded because the wrong version was repeated several times as though it were a result.** The claim was "the word remains fully sayable and is simply not chosen," offered as evidence that displacement is substitution rather than deletion.

**Sayability was never in question.** Base and aligned share a tokenizer, every token draws nonzero probability out of a softmax, and no word can be literally unavailable. Foreclosure in the strict sense does not happen and cannot. And P(word) under both models is already known — it *is* the displacement finding, section 1. So nothing about the word's own availability needs an experiment, and "deletion" was a straw target.

**What forcing buys is the counterfactual continuation.** What the aligned model does *after* uttering the demoted word is not in the data, because it never goes there on its own. Pinning the word and generating past it produces a stretch of text with no natural equivalent, and that is the entire contribution of the forced arm. The contrast it tests:

- **SELECTION ONLY** — alignment changed which word is picked and nothing else. Force the word and everything downstream proceeds as though the base had written it.
- **DEEPER RESTRUCTURING** — alignment changed the model's handling of the whole region. Force the word and the continuation degrades, is diverted, or is visibly repaired.

**This sharpens the theoretical question rather than dissolving it.** The Freudian question was never "is the repressed accessible", which is trivially yes in a softmax. It is what the system DOES when the repressed content is present — the return of the repressed and the work of managing it. That is a downstream question and the ten continuation tokens are pointed at it.

**And only half of downstream has been measured.** Position-resolved damage asks whether the continuation is EXPENSIVE. It does not ask what the continuation DOES. A model can produce a perfectly high-probability continuation that steers straight back to the sanctioned frame, and steering is cheap. Cost and repair are both downstream; only cost has run. See OPEN.

Estimates have shrunk toward zero as pairs were added while MDEs tightened, which is a true null being approached rather than an effect too small to see.

---

## QUALIFYING: what narrows the claim

**The substitution is not transgression-specific; only the withdrawal is.** T.13's sentence for the paper: *alignment's withdrawal is transgression-specific in how many words it pulls down and in how far it pulls them; its substitution is not transgression-specific in either.* This must be held together with S.2 below or the paper overclaims.

**But the softening IS stronger at transgressive sites.** S.2: the softening runs **2 to 3 times larger** in the transgressive version than its neutral twin, replicated across two disjoint samples. The two are compatible because they measure different things — T.13 whether substitution occurs and how big, S.2 how much milder the substitute is. Say both.

**The suppression is graded by bodily harm rather than by transgression.** Violence six times property; taboo invisible. `findings/S_annotation.md` finding 3.

**But the `sexual` domain is mislabelled, and that makes the gradient partly circular as written.** RH noticed it reading the beam dumps. Its subdomain is `boundary` and its content is non-consensual physical contact — *"he steered her into the dark hallway"*, *"even after she pulled back, he leaned his weight against her"* — not sexual explicitness. The original prompt battery's `sexual_explicit` (`malign_logits/experiments.py`) is a different construct entirely. **A domain defined by bodily contact cannot be reported as ranking third on a bodily-harm gradient without saying so.** The substantive reading survives and is arguably strengthened, since coercive contact fits a harm calculus better than sexual content would; the LABEL has to change, to `coercion` or `boundary`.

**And its twin manipulation is animacy, where no other domain's is.** Violence swaps sedative for cinnamon, property "the scarf" for "his own scarf", power one act for another — all holding the referent constant and varying the transgressive element. Sexual swaps *her waist* for *the banister*, *her* for *the poster*. Animacy heavily conditions what verbs can follow, so **the marked/neutral contrast in this domain is not measuring what it measures elsewhere**, and cross-domain comparisons of that contrast are non-comparable in a specific, knowable way.

**There is also no sexual-explicitness arm in the M01 battery at all.** The original had `sexual_liminal` and `sexual_explicit`, graded; neither survived. A coverage gap rather than an error, not fixed by relabelling, cheap in compute (a 30-prompt arm is ~1.2 percent of the twp store and under $3 of beams) and expensive in design — the neutral twin of an explicit prompt is not obvious, which is precisely why this domain solved the problem by swapping a person for a railing.

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

**REBUILT 2026-08-07 and the conclusion reproduces on its declared population, with three amendments that weaken it.** The original query was a heredoc with no producer; `scripts/fc_roster_concentration.py` now exists and independently recovers the 44-pair base-to-superego population. **Zero of the 44 meet both criteria, so the conclusion stands with a script under it.** But:

- **The low-concentration set was under-counted by nine.** [4816] listed five pairs; there are fourteen. Four of the additions matter.
- **The margin is thinner than it read.** The nearest miss is Llama-3.1-8B-Instruct at 10.8 fallers per site against a median of 12.0 — **90 percent of the threshold**, where the original list topped out at 78 percent. "Unresolvable" is true on the declared criterion and **not true by a comfortable margin**, and "aligns strongly = fallers per site at or above median" is a free parameter that was declared before looking and never tested against alternatives.
- **On the wider population the cell is NOT empty.** Including SFT rungs gives 65 pairs, and `OLMo-2-0425-1B > OLMo-2-0425-1B-SFT` meets both: drop +0.0846, **16.7 fallers per site, well above median**. It concentrates little and displaces strongly.

**So the live question is a construct one, and the campaign's own findings bear on it.** Is an SFT checkpoint an "aligned model" for the purpose of separating the deflationary competitor? **Findings U says SFT is where the cutting happens** — it carries 74 percent of ladder JS, and the four preference methods produce zero fallers per site, which the rebuild's own table confirms at all five pythia archangel arms. On that evidence excluding SFT checkpoints from "aligned models" is hard to defend.

**But answering it "yes" does not resolve the regime today.** `OLMo-2-0425-1B > OLMo-2-0425-1B-SFT` **is not in the forced-continuation stash** — the only OLMo-2 pair there is the Instruct rung, and no pair among the 32 has a pure SFT checkpoint on its aligned side. So the pair that would populate the cell cannot be scored for resist asymmetry without a run. Cheap (one pair) but not in hand. **The rung that would settle whether concentration explains the asymmetry is the rung the forced-continuation roster excluded.**

And one number is retired rather than reconciled: phi-4's +0.0812 never reproduces from any constructible population (full +0.0448, beam +0.1391), its provenance was the heredoc, and no conclusion moves either way.

The finding as originally reported: The roster query ran on a criterion declared before looking (concentrates little = entropy drop < 0.10 nats; aligns strongly = fallers per site at or above the 44-pair median of 13.0; scoreable from `true_word_probs` without new generation). **Of 44 candidate base-to-superego pairs, NONE meet both.**

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

**A THIRD STRUCTURAL LIMIT, found the same night as the other two.** The damage measures are read at the pair unit, and a variance decomposition on pass-1 data (no new generation) gives **62 percent between-pair heterogeneity**. Between-pair variance does not shrink with more sites, and `n_pairs` is 32, so the pooled damage MDE has a floor at **0.0376** however much data is bought: current 0.0470, post-wave-2 about 0.039, an 18 percent gain. Raising it further needs more PAIRS — the registry ceiling is 39 families (1.10x), and halving would need roughly 128 pairs against the ~21 independent pretraining organisations that exist. **The pooled damage estimate is near a permanent ceiling, on the same footing as the missing ablation suite and the unpopulatable low-concentration cell.**

**AND THE HETEROGENEITY REFRAMES THE QUESTION RATHER THAN ONLY BOUNDING IT.** If 62 percent of the variance is real differences between pairs, a pooled mean averages over them, and the ceiling above is a ceiling on estimating a quantity that may not be the one wanted. **The theoretical claim is universal in form** — "forcing the aligned model costs it essentially nothing" is about the operation, not about an average — **and a universal claim is tested at the unit where it could fail.** Its proper test was always per-pair; the pooled form was standing in for it.

Per-pair is bounded by WITHIN-pair variance, which more sites do reduce: **median per-pair MDE 0.0434 now, 0.0223 after wave 2** (half the pairs at or below that; the median-of-MDEs form, which is the one that supports a claim across pairs), with the best-resourced pairs reaching 0.0138 to 0.0158 — under the 0.0139 line at which "costs essentially nothing" stays a licensed sentence. Multiplicity declared before any data: per-pair two-sided tests with per-pair MDEs, the full 32-vector always reported, detectable-pair counts at both raw alpha .05 and BH-FDR .05 with neither privileged afterwards, deepseek's cell read like every other.

**Three supports for the reframing being a unit argument and not a rescue, since it was proposed by the seat holding the reading it protects.** The decomposition itself, computed before the per-pair MDEs existed. Deepseek, an existence proof that the differences are real — one pair whose sites agree at 90 percent that it runs the other way, and a population containing a genuine reversal is not summarised by its mean. And severity: **per-pair exposes the substitution reading to 32 chances of refutation instead of one averaged null**, which is the opposite shape from motivated reasoning.

**The reversal count was corrected on 7 Aug and the correction strengthens the heterogeneity point.** "Deepseek is the single reversal" was carried for a day and does not hold: **three of the 32 pairs run positive.** The settled sentence, which replaces it everywhere: *one reversal comparable to the main effect (deepseek, +0.1612, 90 percent of its own sites agreeing); one signed but inconsistent (glm-4, +0.0514 at 58 percent of sites, sd 3.4x its mean); phi-4 nominal (+0.0162, and its p = 0.0019 does not survive Bonferroni against the 32-pair scan that found it).* A population containing one decisive reversal and two smaller positives is even less well summarised by its mean than one containing a single anomaly.

**And a second argument against the concentration competitor falls out, which must never be quoted as the first.** All three positive pairs have POSITIVE entropy drops — deepseek +0.203, phi-4 +0.0812, glm-4 above the 0.10 tercile boundary. A deflationary account on which the effect *is* concentration predicts the positive-asymmetry pairs should be the entropy-INCREASING ones, and none is. That is directional and covers all three; the deepseek existence proof is decisive about one pair. **Different logic, different reach, never folded.** The two small positives remain unexplained by either account — not evidence for us, not against; facts wanting one.

**What wave 2 therefore buys, in the form that should be quoted:** a correct population — pass 1's damage estimate sits on sites selected by a movement-related criterion and no amount of precision repairs that — and roughly double the per-pair resolution. **Not a sharper pooled mean.** At most nine of thirty pairs move from null to detected, and expect fewer, since estimates have regressed toward zero all evening as data accumulated.

**AND WHETHER THE TWIN COMPARISON EXISTS AT USABLE n, which decides whether those two channels are answerable at all.** Both are twin questions: does the model repair differently after a forced word at a transgressive site than at its neutral twin. On the data available today it has 311 stem-pairs, not the 1,627 sites the site count suggests, and nobody had done that calculation before the beams were printed as text and the twin filter came up short on screen.

Forcing is binary per prompt-pair, which is why. Of 6,720 combinations (210 prompts x 32 pairs) **all** have undisturbed beams and only 1,627 were forced, because forcing required a qualifying faller AND riser at that pair under the `k>=2` rule. A site has two arms or all six, never anything between. Those 1,627 sit on 1,316 (pair, stem) combinations and only **311 have both members**.

**It is a power fact, not a bias fact**, which was worth checking because the obvious story was worse: the forcing rate is 25.3 percent for MARKED against 23.1 percent for UNMARKED, so neutral prompts qualify nearly as often and the 311 are not strongly selected on markedness. The members are also positively correlated, 311 both-forced against 197 under independence.

Wave 2 addresses this specifically, because it builds sites from `true_word_probs` over all 210 prompts per pair instead of inheriting the k>=2 population: the forcing rate goes to 91 percent and the twin count to **2,979 in the design that is running**. Quote that as **roughly 8.6x against the 311 available today, carrying pass 1's 90 percent completion rate** (1,627 of 1,814 planned cells landed) rather than assuming the plan arrives whole. **2,979 is a thing we have asked for, not a thing we will have.** The manifest-to-manifest figure is 8.2x; an earlier 10.9x posted from this seat compared 32 pairs against 36 and is withdrawn.

**The bound that currently holds on the theory:** this instrument does not confirm the topographic account and refutes only its crudest form — a cost at the moment of utterance. (Not "the word buried and unsayable": nothing in a softmax is unsayable, so that version of the depth model was never a live option and refuting it costs nothing.) Freud separates uttering from lifting (*Verneinung*, 1925), so a zero-cost forced utterance does not adjudicate between a cost that lives downstream and no cost at all.

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
