---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-06
role: finding
topics: [semantic-fields]
description: "Embedding geometry fails at the relation and validates something else: the V.6 axis caption is 'off contact, motion and force onto perception, cognition and speech' (never 'bodily action -> cognition', never 'register'). OPEN AUDIT: base probability not controlled in V.6/V.7 under lacan's matched-aggregation definition (d) ([4933]-[4943])."
---
# Findings V: embedding geometry fails at the relation and validates something else

Written 2026-08-06. The plan is `registrations/plan_v_embedding_regions.md`, written before any of this ran and amended once — after the variance measurement and before any clustering — to make bare type embeddings primary. That amendment removed two free parameters rather than adding any, which is the only reason it was legitimate after seeing a number.

**The title said "mostly a negative" until 2026-08-06 and that is now wrong as a description of the contents, while remaining right about the question.** The two came apart when sections 6 and 7 landed, and the distinction is the honest summary:

- **The question fails.** Can embedding geometry locate the relation between a faller and the riser replacing it? No, at six grains, and section 1's null has its artefact confirmed rather than merely uncleared. Nothing here rescues metonymy.
- **The document has three results that will be cited.** Scene-locality (section 5), the axis's agreement with findings T on six lexicons with no lexicon in the instrument (section 6), and field cosine as a grouping route Jaccard cannot replicate (section 2).
- **And two positives too small to lead with**, both of which must travel with their magnitudes: the 4.6 percent relatedness floor (section 3) and the +0.32 percent anti-adjacency against a 2.11 percent scene effect (section 4).

**The original phrase was written as a guard against padding and it did that job.** It is retired rather than reversed, because the failure it described is real and is the first bullet above. The correction is recorded rather than made silently, since the risk in the other direction — each pass finding the last framing too modest and inflating a notch — is this seat's characteristic one.

Section 6 was added the same day, after RH asked for the frequency control that sections 5 and the plan had both specified and left unrun. It is the one part of this document that came out better than predicted, and the prediction it beat is recorded in the script's own header rather than reconstructed afterwards. It still does not make the document a positive: what section 6 corroborates is findings T, using geometry as the instrument, and it says nothing new about alignment on its own. Section 7 then ran RH's concreteness control and CORRECTED SECTION 6'S CAPTION: no published perceptual norm explains much of the axis, but the body-specific dimension is weaker than the general one, so "bodily action" is not the phrase the evidence supports and T's own category language is. **Read section 6 with that substitution.**

## The question, and why it was not already answered

Ledger clause 6 is VERIFIED as an instrument-failure record: four similarity instruments could not locate the relation between a faller and the riser replacing it. Registration P's REF stratum failed metonymy by annotation, 1 of 3. But both concern the **paired** relation — is *this* riser near *that* faller. RH asked a **marginal** question: which neighbourhoods supply fallers, and which supply risers. Those come apart cleanly, and only the paired one was answered.

They no longer do come apart. The marginal version has now failed too.

## 1. The regional test is a null, and the artefact is confirmed

Bare `bge-m3` type vectors for the 2,268 word types with at least 40 movement rows, k-means at k = 10, 20, 35, 50, 75, 100, per region per edge the riser share minus the faller share, family as the unit across 16 families.

**There is no cluster structure to begin with.** Silhouette runs **0.046 to 0.061** across the whole sweep. k-means is slicing a continuum, and "region" is a convenience of the method rather than a property of the space.

**Artefact 1 is confirmed rather than merely uncleared.** The plan named three artefacts before looking; this is the first, that regions are word classes rather than semantic fields.

| k | regions | shift ~ open-class share | shift ~ frequency rank |
|---|---|---|---|
| 10 | 10 | r = +0.531, p = 0.114 | p = 0.65 |
| 20 | 20 | r = +0.447, **p = 0.048** | p = 0.77 |
| 35 | 32 | r = +0.506, **p = 0.0032** | p = 0.52 |
| 50 | 42 | r = +0.369, **p = 0.016** | p = 0.70 |
| 75 | 59 | r = +0.411, **p = 0.0012** | p = 0.58 |
| 100 | 69 | r = +0.411, **p = 0.0005** | p = 0.61 |

Significant at every resolution from k = 20 up, r stable around 0.37–0.53, and **more significant as power grows**. Frequency is clean throughout. A region's rise or fall tracks how content-word-y it is.

**And the load-bearing test could not run.** At k = 50, seven regions survive Bonferroni, **all seven are sinks, and all seven are 100 percent open-class**. There were **no significant sources at all**, so there was nothing to measure adjacency from. Survivor counts also scale with k — 3, 4, 6, 7, 14, 12 — which the plan flagged in advance as a pattern to distrust rather than celebrate.

**A note on how the artefact was caught, because the first pass missed it.** k was initially swept 6–20 and set to 10 by best silhouette. RH asked whether 10 was too few. It was, and the reason is sharper than coarseness: **the artefact control correlates region shift against class share with REGIONS as the unit, so at n = 10 it needed r > 0.63 to reach p < 0.05 and could not fire.** The control's power scales with k. The first run reported the confound as "not cleared" when the test had never been capable of clearing or condemning it. Widening the sweep turned an ambiguous result into a confirmation.

Silhouette was also treated as a choice when all six values were indistinguishable noise. With no cluster structure, k is a resolution dial to report across, not a parameter to optimise.

**Per the plan, this cell reads: the marginal question fails as the paired one did. Record it and stop rather than reaching for another instrument.**

## 2. Field cosine is a third grouping route and it earns its place

Not part of the original plan — RH proposed it after the regional null. Three routes now group the lexicons' fields on three principles: **Jaccard** extensionally (same words), the **agent pass** intensionally (same meaning, blind judgement), and **cosine** distributionally (same region of space, whether or not any words are shared).

It repairs a known blind spot. On the four coarse lexicons the semantic route found 29 cross-resource pairings Jaccard found none of, against 2 the other way, because RID selects by regex, WordNet by supersense and the induced taxonomy by an agent reading types — three membership rules picking different words for one field. Overlap is Jaccard's only signal. A centroid comparison needs no shared membership at all.

**The control passes**, unlike the regional test: cosine distance against difference in open-class share gives **r = +0.264** over 56,280 pairs, so class is about 7 percent of field geometry rather than being it.

**It recovers pairings we would endorse:**

| | | cos d | Jaccard |
|---|---|---|---|
| `induced:locomotion_posture` | `wordnet:motion` | 0.0017 | 0.545 |
| `induced:object_handling` | `wordnet:contact` | 0.0027 | 0.358 |
| `induced:speech_act` | `wordnet:communication` | 0.0033 | 0.466 |
| `framenet:Communication_manner` | `verbnet:manner_speaking` | 0.0079 | 0.579 |

**And 22 percent of the closest 1 percent of cross-resource pairs share no words at all** — the cell Jaccard cannot see by construction. Median Jaccard across those 435 closest pairs is 0.029, so the two routes agree at the very top and diverge quickly below it.

**Caveat that travels with it.** The zero-overlap pairs are weaker than the high-Jaccard ones. `framenet:Self_motion ↔ wordnet:contact` and `rid:sensation ↔ wordnet:motion` are not obviously one field; `Body_movement ↔ aggression` is plausible. This route adds coverage Jaccard lacks and what it adds needs judging rather than trusting. **Finding 16's count of 124 deduplicated fields is an over-count by a knowable amount, and this is how to know it.**

## 3. A relatedness floor exists and is 4.6 percent

RH's second proposal, framed deliberately not as another metonymy test. For each site, mean cosine from its fallers to its own risers, against the same fallers paired with risers from a **different prompt in the same family**.

| rung | families | own | cross-site | gap | p |
|---|---|---|---|---|---|
| base → sft | **14 of 14** closer | 0.3077 | 0.3224 | **−4.57%** | 0.0001 |
| sft → pref | **11 of 11** closer | 0.3113 | 0.3239 | −3.90% | 0.0010 |

Unanimous across families and both rungs. **A site's own risers are genuinely closer to its fallers than a stranger's are** — the floor the adjacency claim always needed and never had.

**And 95.4 percent of the faller-riser distance is the site.** The relation buys 4.6 percent. That is the shape malign found on their forced-continuation JS metric, where the headline was 88 percent instrument.

**This is not metonymy rescued, and the reason is pre-declared in the plan.** Relatedness is not adjacency. If a prompt concerns a knife, both what falls and what rises will be knife-adjacent vocabulary; a shared topic produces this pattern with no substitution relation at all.

## 4. The twin control: displacement moves AWAY from what it replaced

Section 3 could not distinguish relation from topic, and RH supplied a better control than the "same domain" one this document originally specified: **the twin.** Each M01 stem exists as a marked and an unmarked prompt differing in ONE WORD — `hammer`/`clipboard`, `knife`/`flashlight`. Same scene, same syntax, same length. Using the twin's risers as the control holds topic almost perfectly fixed, and it reuses the campaign's own minimal-pair design, which had only ever been applied to the marked/unmarked contrast and never as a null.

Three levels, so the answer decomposes rather than being a binary. `scripts/v_twin_control.py`.

| | base -> sft | sft -> pref |
|---|---|---|
| far, unrelated prompt | 0.3061 | 0.3094 |
| **own** | 0.3006 | 0.3056 |
| **twin** | **0.2997** | **0.3043** |

**The ordering is twin < own < far.**

    THE SCENE     twin - far   -2.11%   14/14 families   p=0.0001
    THE RELATION  own - twin   +0.32%    0/14 negative    p=0.0001

The scene is real and accounts for most of section 3's 4.6 percent. But **within the scene, a site's own risers are FARTHER from its fallers than its twin's risers are** — unanimously, at both rungs, p=0.0001.

**Displacement moves away from what it replaced — ON AVERAGE.** Holding the scene to a single word's difference, what rises at a site sits further from what fell there than what rises at its near-identical twin. That is the opposite of what metonymy predicts, and it is the first **signed** geometric result in the campaign: four grains have reported absences, this one has a direction.

**The qualifier is load-bearing and section 5 is why.** This is an aggregate over the sites within a family, and section 5 establishes that displacement direction is a property of the SCENE. **The result therefore does not distribute: nothing here establishes that any individual site is anti-adjacent**, only that the family-level mean is. The quotable sentence is *the relation is anti-adjacent on average, +0.32 percent against a scene effect of 2.11 percent*, and the size travels with it.

That qualifier reached the docket at [4780] before it reached this file, which is the wrong order — the file is the quotable record and it carried the stronger claim for three commits.

**Quote it with its size.** +0.32 percent of the reference against the scene's 2.11. Unanimous and tiny.

**And this was a cell the outcome map did not contain.** Three were declared — own closer than twin, own equal, all three equal — and the answer was own *farther*. The script's fall-through branch collapsed "not closer" into "topic only" and printed the wrong conclusion until it was corrected. **Plan U's map had the same hole and this one was written knowing that.** An enumeration of outcomes is only as good as its coverage of the sign, and twice now the missing cell is the one that landed.

## 5. Direction: the scene has one, alignment does not

RH's question after section 4: can a faller-to-riser vector be identified and used? Distance had failed at five grains, but **a vector asks which way rather than how far, and the two are independent** — sets can be far apart and consistently offset, which section 4 made a live possibility rather than an idle one. `scripts/v_displacement_vector.py`, run pooled and again restricted to CLAWS `vv*` lexical verbs (RH's suggestion, the same restriction `s_lexicon_crosstab` uses).

**The control fails, and the verb restriction ISOLATED the artefact rather than removing it.**

| | pooled | verbs only |
|---|---|---|
| axis ~ log frequency rank | r = +0.460 | **r = +0.554** |
| axis ~ open-class | r = +0.392 | undefined by construction |

Removing the class variance made frequency *more* visible, not less. And the verb poles are interpretable and confounded in the same breath:

    from   put, got, turn, tell, goes, pull, sat, buy, go, get, call, say, told, push, pay, throw
    to     observed, administered, determined, addressed, informed, cautioned, suggested,
           ignored, wondered, recognized, expressed, accepted, introduced, concluded

**Plain Anglo-Saxon action verbs to formal Latinate ones.** That rhymes with findings T's proceduralisation — `administered`, `cautioned`, `recognized`, `concluded` is the neighbourhood of `Caution`, `Constraint` and `Investigate`. But **Latinate words are rarer**, so "shifts register" and "shifts toward rarer words" predict the same axis and r = 0.554 cannot choose between them.

**There is no global direction.** Mean pairwise cosine between site vectors is **0.059**, against a pairing-shuffled null of 0.046 — the null keeps every faller set and every riser set and destroys only which pairs with which. The gap is real (14/14 families, p = 0.0001) and negligible. Site vectors are near-orthogonal to one another.

**The rungs are not parallel, which cuts against findings U.6.** A family's own `base>sft` and `sft>pref` axes agree at **0.238**, while two *different* families' `base>sft` axes agree at **0.323** — its own rungs less aligned than strangers' same rung, over a range of −0.86 to +0.70. U.6 concluded "one operation at two amplitudes" from word overlap and a field split; the geometric version does not support it. Held loosely, because section 5 has just established that the whole vector measure is weak, but recorded as a negative rather than as neutral.

**And the one robust result is scene-locality:**

| | cosine |
|---|---|
| a marked site's vector vs **its twin's** | **0.327** |
| vs a random site's | 0.060 |

Fivefold, 14 of 14 families, p = 0.0001, and unchanged by the verb restriction. **The direction of displacement is a property of the scene, not of alignment.**

That also explains the frequency axis rather than competing with it. If every prompt displaces in its own direction, averaging thousands of them cancels the scene-specific components and leaves only what they all share — the frequency gradient of the embedding space itself. **The pooled "axis of alignment" is an artefact of pooling directions that are not pooled in nature.**

**One control specified and not run:** residualise log-frequency out of the vectors and recompute. It separates *alignment shifts register* from *alignment shifts toward rarer words*, which are different claims and only one is interesting. Until it runs, the plain-to-formal reading is not available.

That control has now run. Section 6.

## 6. The frequency control: the register reading does not survive, and a better one replaces it

Run 2026-08-06 as `v_displacement_vector.py --verbs --resid`, the same code path as section 5 with one switch, so the two are comparable by construction. Each of the 1,024 dimensions is regressed on log frequency rank across the 1,312 retained verb types and replaced by its residual. Every one of the 1,312 has a real BYU rank, so nothing is imputed. Frequency accounts for 5.7 percent of the embedding variance over those types.

**The control had power, which is the first thing to establish and not the last.** The axis rotated: `cos(raw axis, residualised axis) = +0.862`, and the raw axis lay at `cos = +0.518` from the frequency direction itself. Roughly half of the raw axis was frequency. Had the rotation come back near 1.0 nothing below would have meant anything, since an unmoved axis cannot have its poles rechecked.

**The plain-to-formal reading does not survive.** Every generic high-frequency verb leaves the from-pole once frequency is out, and the Latinate formality at the to-pole goes with it:

    raw    from  put, got, turn, tell, goes, pull, sat, buy, go, get, call, say, told, push, pay
           to    observed, administered, determined, addressed, informed, cautioned, suggested,
                 ignored, wondered, recognized, expressed, accepted, introduced, concluded

    resid  from  put, sat, got, turn, lay, spit, putting, shut, puke, threw, rub, pull, turned,
                 shook, sit, spun, drown, quit
           to    considered, appreciate, discovered, consider, explore, understand, examine,
                 encourage, determine, noticed, realized, identified, understood, established,
                 recommend, realize, acknowledge, discuss

`go`, `get`, `say`, `tell`, `call`, `buy` and `pay` were on the from-pole because they are common. What is left is the body: `sat`, `lay`, `spit`, `shut`, `puke`, `threw`, `rub`, `shook`, `spun`, `drown`. And `administered` and `cautioned` give way to `consider`, `understand`, `examine`, `notice`, `realize`, `acknowledge`, `discuss`. **The axis runs from bodily action to cognition and deliberation, not from plain to formal.** Register was doing less work than frequency was.

### The pole reading is 36 words of 1,312, so it was checked against all of them

Findings T made the same claim with lexicons. Scoring every retained verb by its axis position, averaging within category, and correlating against T's marginal delta gives, at a minimum of five verb types per category:

| lexicon | categories | residualised | raw |
|---|---|---|---|
| induced | 13 | +0.527 (p = 0.064, MDE 0.55) | +0.363 (p = 0.223) |
| wordnet | 14 | +0.538 (p = 0.047, MDE 0.53) | +0.521 (p = 0.056) |
| verbnet | 89 | +0.398 (p = 0.0001, MDE 0.21) | +0.372 (p = 0.0003) |
| usas | 51 | +0.472 (p = 0.0005, MDE 0.28) | +0.216 (p = 0.127) |
| framenet | 58 | +0.397 (p = 0.0020, MDE 0.26) | +0.381 (p = 0.0031) |
| rid | 16 | +0.524 (p = 0.037, MDE 0.50) | −0.088 (p = 0.745) |

**Six of six positive, median rho +0.498 against the raw axis's +0.367.** The induced lexicon's two extremes are its top riser and its top category on the axis: `perception_cognition` at delta +0.0489 and mean projection +0.1910, `bodily_violence` at delta −0.0183 and +0.1108. WordNet's `contact` and `motion`, T's two clearest fallers at p = 0.0011 and p = 0.0002, sit first and third from the bottom.

Raising the type minimum to 3, 10 and 20 gives median residualised rho of +0.470, +0.549 and +0.832 against raw +0.333, +0.390 and +0.443. **The +0.832 is not the finding and should not be quoted as one:** at a minimum of 20 only six or seven categories survive per lexicon, and the ones filtered out are the small disagreeing ones, so the number rises because the exceptions leave. Five is reported as primary because it retains the most categories and shows them.

**The disagreements are real and the largest one is where my pole reading leaned hardest.** WordNet `cognition` has the highest mean projection of any supersense, +0.1867, and T's delta for it is −0.0132 at p = 0.578, which is to say cognition does not rise in T at all. WordNet `body` rises in T at p < 0.0001 and sits near the bottom of the axis. A rho near +0.5 is a moderate agreement with visible exceptions, not a match. The induced lexicon's `perception_cognition` and WordNet's `cognition` carve the same territory differently and only one of them agrees.

### What the comparison can and cannot carry

It is **not** independent corroboration of alignment's direction. The axis is `mean(risers) − mean(fallers)` and T's delta is riser share minus faller share, so a category whose words rise will sit toward the riser end more or less mechanically; and the two instruments share 1,358 of T's 1,361 prompts, with partly overlapping model rosters. The absolute rho should never be quoted as evidence about the world.

What is clean is **the comparison between the two columns**, because the mechanical relation is identical for both and cannot produce a difference between them. Removing frequency moves agreement from +0.367 to +0.498, and moves RID from −0.088 to +0.524. The one case going the other way is VerbNet at a minimum of 10 types, +0.257 residualised against +0.417 raw, and it is recorded rather than smoothed.

And the comparison does answer one standing objection to T. Six hand-built taxonomies are still six human category systems, so "the ordering is an artefact of how the categories were drawn" was live. **A pretrained embedding space with no categories in it recovers the same ordering.** T's result is not a taxonomy artefact. That is a claim about the categories, not about alignment, and it is the right size for what was measured.

### Nothing else moved

Everything else in section 5 was recomputed on the residualised vectors and none of it changed sign or verdict:

| | raw | residualised |
|---|---|---|
| site alignment, `base>sft` | 0.059 vs null 0.046 | 0.048 vs null 0.039, gap +0.0087, 14/14, p = 0.0001 |
| site alignment, `sft>pref` | | 0.060 vs null 0.052, gap +0.0083, 11/11, p = 0.0010 |
| own rungs vs cross-family | 0.238 vs 0.323 | 0.133 vs 0.295 |
| twin vs random site | 0.327 vs 0.060 | **0.313 vs 0.051**, 14/14, p = 0.0001 |

**There is still no global direction**: both levels fell and the gap stayed unanimous and negligible, which is what the pairing-shuffled null predicts, since it preserves both marginals and therefore carries the frequency gradient itself. **The rungs are still not parallel**, and the gap against U.6 widened. **Scene-locality is not frequency**: sixfold now where it was fivefold, on the same 14 families at the same p.

One number is an arithmetic identity and not a result. On the un-renormalised residuals the projection correlates with log frequency at exactly r = +0.000000, because every column of the residual matrix is orthogonal to centred log frequency and therefore so is any projection of it. It is printed as an implementation check: non-zero there means the code is wrong, zero there means nothing about the world. After row renormalisation, which the raw pipeline also does, the residual correlation is r = +0.039 at p = 0.162.

## 7. The concreteness control, which corrects section 6's caption

RH asked whether the body-to-cognition axis is a concreteness axis, then made the sharper objection while it was running: concreteness is not something you can cleanly control for when the fields under test are body and thinking. He is right, and the run is reported here as the narrower thing it is.

**The norms operationalise perceptibility, not bodiliness.** `hammer` is concrete and not bodily; `betray` is abstract and not cognitive. So residualising them cannot separate the two constructs, and this was written into the script header before the run rather than after.

What it can do is kill one specific deflationary story, and the number that does it is not the residualisation:

| the freq-residualised axis against | r | n |
|---|---|---|
| MT-Conc, concreteness (MTurk) | **-0.361** | 1301 |
| LSN-Hapt, haptic | **-0.235** | 1301 |
| MRC-Imag, imageability | -0.138 | 643 |
| LSN-Imag, imageability | -0.117 | 1301 |
| MRC-Conc, concreteness (MRC) | **-0.021**, p = 0.61 | 591 |

**No published perceptual norm explains much of this axis.** The largest accounts for 13 percent of the variance and one concreteness measure accounts for nothing, while the axis agrees with T's semantic categories at rho about 0.5 across six lexicons. Residualising MT-Conc out leaves the ordering at Spearman 0.878 against the freq-only arm on 1,301 shared types, the poles substantially intact, and the lexicon agreement unchanged: median +0.531 against +0.498, all six still positive. Unchanged, not improved; the difference is small and the six measures are correlated.

**And the control corrects section 6's caption rather than confirming it.** If "bodily action to cognition" were right, the body-specific dimension should beat the general one. It is the weaker of the two (-0.235 against -0.361) and they are separable on our verbs (r = +0.486), so that is not one measure counted twice. **"Bodily" was doing work the evidence does not support.** The defensible caption is findings T's own language, off contact, motion and force onto perception, cognition and speech, and section 6 should be read with that substitution.

Two notes on method. The same-population control matters and was initially missing: `--resid freq,conc` narrows the vocabulary as well as removing the variable, so `--pop-conc --resid freq` was added to isolate the manipulation. With the lemma fallback in place the narrowing is 11 types and the two agree at Spearman 0.9993, so the population is not a confound; without the fallback it was 542 types and would have been. And MT-Conc and MRC-Conc agree at r = 0.92 across the full 37,563-word norm set while giving -0.361 and -0.021 here, which is a reason not to rest anything on a single norm column.

## What this leaves

Embedding geometry has now been asked about this operation at six grains: pairwise (clause 6, four instruments, failed), by annotation (Registration P's REF stratum, failed), regionally (section 1, null with a confirmed artefact), set-level within site (section 3, a 4.6 percent floor), scene-controlled (section 4, anti-adjacent), and directional (section 5, no global direction). **The ledger's original verdict survives all six: the faller-riser relation is interpretive, not geometric.**

What geometry did produce is four things, and none is a claim about alignment's direction. Section 2 is a better instrument for grouping the lexicons we already use. Section 4 is a signed result where the others had absences: displacement moves AWAY from what it replaced, small and unanimous. Section 5 is the reason the rest kept failing — **the scene has a direction and alignment does not**, so every pooled measure was averaging over the thing that carries the signal. And section 6 turns the pooled axis, which section 5 had written off as the frequency gradient, into a check on findings T: with frequency projected out it orders T's categories the way T does on all six lexicons, and better than it did before the control. Section 7 fixes what that axis may be called — off contact, motion and force onto perception, cognition and speech, in T's words rather than a body/mind framing the norms do not license. **The pooled axis is not evidence about alignment. It is evidence that T's ordering is not an artefact of how the categories were drawn.**

**Two expectations recorded before section 6 ran were wrong, and are left standing rather than edited away.** The frequency control was expected to close the geometric line; it opened the only lexicon-free corroboration T has. And section 5's reading of the pooled axis as "the frequency gradient of the embedding space" turns out to be half right in a literally measurable sense: half the axis was frequency, `cos = +0.518`, and the rest was the result.

## Data and code

    variance     scripts/v_bge_variance.py          results/v_bge_variance.csv
    regions      scripts/v_regions.py               results/v_regions.csv
    field cosine scripts/v_field_cosine.py          results/v_field_cosine.csv
    relatedness  scripts/v_site_relatedness.py      results/v_site_relatedness.csv
    twin control scripts/v_twin_control.py          results/v_twin_control.csv
    direction    scripts/v_displacement_vector.py   results/v_displacement_vector.csv
                 --verbs for the vv* restriction    v_displacement_vector_verbs.csv
                                                    v_displacement_twin.csv
    concreteness --verbs --resid freq,conc         v_displacement_vector_verbs_resid_freq_conc.csv
      control    --verbs --pop-conc --resid freq   v_displacement_vector_verbs_popconc_resid.csv
                 norms: LSN/MRC/MTurk/Paivio       data.wordnorms_orig.csv (chambers)
    frequency    --verbs --resid                    v_displacement_vector_verbs_resid.csv
      control                                       v_displacement_twin_verbs_resid.csv
                 per-type axis positions            v_axis_projection_verbs{,_resid}.csv
    axis vs T    scripts/v_axis_vs_fields.py        results/v_axis_vs_fields.csv
                 T's marginals it is scored against results/s_everything_marginal.csv
    vectors      results/v_bare_vectors.npz         bare bge-m3, 25% depth, 2,268 types
    movement     results/t_ladder_words.parquet     1,281,413 rows, 16 families
    plan         registrations/plan_v_embedding_regions.md
