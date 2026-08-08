# Channel 3 — renewed displacement after the forced word

**Status: DRAFT, circulated for feedback. Nothing has been run. No number in this file is a result.**

Author: malign. Prompted by registrar [5009]: *"The third channel — renewed displacement after the forced word — is now the tiebreaker, and it sits unrun in beams already collected."*

## 1. What the tiebreaker is for

Findings W's discriminator table has split. Two channels point in opposite directions:

| channel | measures | wave-3 state | supports |
|---|---|---|---|
| 1 — cost | improbability of the forced word itself | +0.0144, p 0.0043, sign 19/24 | depth account |
| 2 — repair | disruption of the continuation after it | flat | chain-substitution |

Cost positive with repair flat is not a result, it is a tie. Channel 3 asks a question neither answers: **once the demoted word is out, does the aligned model steer away from where the base goes?**

- **Ongoing defense** (depth): having been made to say `said`, the aligned model's continuation pulls away from base's. The signifier stays charged and must be managed after utterance.
- **Absorbed** (chain-substitution): aligned continues roughly where base does. The cost is paid at the moment of selection and then the chain runs on.

This is the psychoanalytic content of the tie: is repression a **selection rule** or a **standing pressure**?

## 2. The measurement window already exists and is the right one

`fc_remote.generate_beams` appends `force_ids` to the prompt, so `plen` **includes** the forced word, and `teacher_force` scores from `plen-1`. The stored `scored_by_base` / `scored_by_aligned` arrays are therefore exactly the 10 tokens **after** the forced word. No re-generation is needed and no window choice is being made post hoc — the corpus was built with this window.

**CHECKED ON RECORD, not inferred** (registrar [5011].2 independently, and here on 3,549 forced records of the new-lineage corpus): `len(scored_by_base[0]) == len(scored_by_aligned[0]) == len(tokens) == 10` and `len(full_ids) == prompt_len + 10`, with **0 violations**. Observed `n_forced_tokens` ∈ {1: 2133, 2: 1248, 3: 152, 4: 16} — **multi-token forced words put the whole word inside the prompt**, so the window is 10 tokens after the complete word in every case, not after its first piece.

## 3. PRIMARY statistic

For one site *s* in pair *p*, and one arm *a* ∈ {force_faller, force_riser}:

```
D(s, a) = mean_logprob( aligned's own beams | scored under aligned )
        − mean_logprob( base's beams        | scored under aligned )
```

Mean is over all 100 beams × all 10 continuation positions. `D` is the aligned model's **dispreference for the base model's continuation** relative to its own, in nats/token. Larger D = aligned is going somewhere else and dislikes where base went.

**PRIMARY CONTRAST**

```
Δ(s) = D(s, force_faller) − D(s, force_riser)
```

**Δ > 0 = renewed displacement is specific to the demoted word.**

Per-pair value = median of Δ(s) over that pair's sites. Test = Wilcoxon signed-rank over pairs, two-sided, α = 0.05. Bootstrap CI on the median of per-pair values, 10,000 resamples, seed declared at freeze.

### Why faller-vs-riser and not forced-vs-undisturbed

Registrar [5009].3: forcing **anything** raises E-QUIZ +0.41pp and E-QA +0.33pp, word-agnostically. A forced-vs-undisturbed contrast would recover that mechanical effect and report it as repression. Faller-vs-riser holds "a word was forced" constant and varies only **which** word — which is the only contrast that isolates the demoted signifier.

> **POST-FREEZE AMENDMENT, 2026-08-08, visible per the amendment convention — the frozen text above is left standing and is not rewritten.**
>
> **The cited magnitude claim has been demoted** ([5027]). Registrar's "forcing anything is mildly destabilising" crosses the same two boundaries as this spec's own positive control ([5026], §6 amendment): a **commitment boundary** (uncommitted vs committed state) and a **one-token position offset** (`plen` includes the forced word, so forced arms score sentence positions 2–11 against undisturbed's 1–10). Regex counts rather than logprobs changes nothing about the comparison structure. "Mildly destabilising" does not travel.
>
> **The design choice this paragraph justifies is UNCHANGED, and did not need the magnitude.** A forced-vs-undisturbed primary would compare across both boundaries and report the result as repression. Faller-vs-riser holds commitment *and* scored position constant. That argument stands on its own structure and never required the effect to have been measured at any particular size.
>
> **No outcome-meaning change**: the primary, population, unit, test and decision rule are untouched.

## 4. UNIT, POPULATION, AND WHAT IS EXCLUDED

- **Unit = pair.** Sites are nested within pair and are not independent.
- **Population:** `design == "wave3-lexical"` (read from record VALUES — `design` is not in the key), plus `newlin-lexical` once merged. The function-word wave is excluded on RH's instruction per [5009].
- **A site enters only if all four cells exist**: {base, aligned} × {force_faller, force_riser}. Half-present sites are counted and reported, never imputed.
- **Decoder — FIRST-ORDER PROTECTED, INTERACTION NOT.** An earlier draft said "decoder-immune by construction". **That was an overclaim and registrar [5011].3 is right.** The [4994] protection class covers per-checkpoint ADDITIVE constants; `D` mixes two producers, and Δ's protection requires the decoder effect to be constant ACROSS ARMS. Channel 1 itself supplies the interaction mechanism: a sampling decoder draws from the post-faller distribution, which **differs from the post-riser distribution by design** — that difference IS the cost effect. So the clean-roster column (12 samplers dropped, [4996]) is promoted from "reported anyway" to **ARBITER ON DISAGREEMENT: if ALL and CLEAN differ in sign or CI status, CLEAN governs.**
- **n, declarable now and not an outcome:** 35 pairs (29 wave-3 + 6 new-lineage), 6,485 sites. **n_lineages = 35**, computed against `model_to_base`: every pair sits on a distinct base model, so pairs and lineages coincide here and the D3 census discipline is satisfied rather than merely invoked. Exact usable n after the four-cell filter goes in the freeze.

## 5. SECONDARIES, declared in advance so they cannot be promoted after the fact

1. **Position profile**, Δ at each of the 10 positions separately. Pre-declared expectation, and the two accounts differ here: a **one-shot cost** should spike at +1 and decay to zero; an **ongoing defense** should persist across the window. This is the most informative secondary and it is explicitly NOT the primary, because choosing a position window after seeing the profile is how a null becomes a finding.
2. **Twin moderator**, Δ computed within MARKED and within UNMARKED stems separately. If renewed displacement is about transgression rather than about lexical surprise, MARKED > UNMARKED.
3. **Own-beam divergence**, an alternative operationalisation: token-level disagreement between base's top beam and aligned's top beam. Measures *behavioural* divergence where the primary measures *evaluative* dispreference. Reported beside the primary; **disagreement between them is itself the finding**, not a reason to pick one.
4. **Per-family breakdown**, descriptive only, no test.
*(The mirror contrast Δ' proposed at registrar [5011].4 is **withdrawn as a separate item** and superseded in form by §5b, per registrar [5015]. Δ' remains computable as one contrast among the four terms; it is not reported alone.)*

## 5b. REQUIRED CO-REPORT — THE FOUR-TERM DECOMPOSITION

**Not a secondary. Reported WITH Δ, never instead of it** (lacan [5014].4, ruled by registrar [5015]).

Δ and Δ' are differences of differences. On Y they produced an apparent mirror — Δ −0.0515, Δ' +0.0458, equal magnitude, opposite sign, matching **neither** declared branch — which dissolved completely on decomposition:

| term | text | scorer | Y median (0-10) |
|---|---|---|---|
| A\|A | aligned | aligned (self) | **−0.08077** p 0.0060 |
| A\|B | aligned | base (cross) | **−0.08257** p 0.0001 |
| B\|A | base | aligned (cross) | −0.00392 p 0.376 |
| B\|B | base | base (self) | +0.01102 p 0.979 |

**Only the aligned arm's TEXT moves; base's output is unchanged under both scorers.** `A|A` and `A|B` are one fact entering Δ positively and Δ' negatively — that is the whole of the apparent symmetry. There was no mirror to interpret.

**And the direction of the reading changes with it.** If this were aligned's *dispreference* for the chain region the faller opens (§7's evaluative row), it would appear in `A|A` and not in `A|B`, because base has no such dispreference. It appears in **both**. Two scorers agreeing that the same text is less probable is not an evaluative stance — it is a **production** difference. The faller changes what the aligned model *writes*, not how either model *reads*.

**A composite that can conceal its own decomposition does not travel alone.** The four terms cost nothing — all four arrays sit in every record — so they are required output, not an option.

Note the grain asymmetry this exposes: on Y the *composite* is insensitive at windows where its *components* move. That argues for the decomposition, not against the instrument.

## 6. CONTROLS

- **Positive control:** D(forced) vs D(undisturbed), pooled, run FIRST and reported whichever way it goes.

  **ITS GATING ROLE IS CORRECTED, AND lacan's [5012] IS WHY.** The first draft said a failed control makes the primary uninterpretable. **That was wrong and the error is instructive.** The control asks *does forcing anything move D*; the primary asks *does which word was forced move D*. `D` carries a large model-level constant — how different these two checkpoints are — and **Δ differences that constant out while the control does not**. The two are not nested, so a null control with a non-null primary is coherent rather than contradictory.

  lacan measured exactly this on Y: **the control fires at none of seven windows** (0-10 through 50-256, all p ≥ 0.067), because `D` sits at ~1.15 nats/token everywhere and forcing does not move it — at that grain D measures the checkpoint difference, not the site. Yet the primary at 0-10 is non-null. **A failed control therefore BOUNDS what the instrument detects; it does not void the primary.** If fc's D is likewise ~constant and unmoved by forcing, that is reported as a scope limit on channel 3, not as a reason to discard Δ.

- **Negative control:** Δ computed on the residual/tail token is meaningless and is not computed. Stated so nobody adds it later.

## 6b. PRIOR FROM Y — DECLARED, because this is no longer a blind test

lacan [5012].3 ran this primary on Y (1,087 sites, 32 pairs, 256-token passages) as a declared secondary, and reports a **monotone decay from the forced word with a NEGATIVE sign**:

    0-10   -0.05151  CI [-0.10083, -0.00254]   excludes 0
    0-25   -0.04155  0-50 -0.02729  0-100 -0.01143  0-256 -0.00176

**Δ < 0 is the opposite of renewed displacement**: after a forced faller, the aligned model's own continuation is *less* preferred over its base's than after a forced riser — the demoted word pulls aligned back toward the base's chain rather than away from it.

**This is recorded as a PRIOR, not as support.** fc is a different corpus, a different generator and a ten-token window, so it is a genuine replication target — but I now know the expected sign before running, and a matching fc result must be reported as **replication under a known prior**, never as independent confirmation. Had I run fc first and found Δ < 0, I would have been entitled to more surprise than I am now.

**AND IT IS A PRIOR WITH A SHAPE, NOT A Y RESULT** (registrar [5015], applying the §4 arbiter rule to lacan's own finding). The Y four-term effect **does not clear the clean roster**: at n=25 both `A|A` and `A|B` lose their CIs while keeping signs and magnitudes, and under the arbiter rule CLEAN governs. lacan reported this against their own result rather than being asked to. So what travels to fc is:

> the aligned model's post-faller **production** is less probable under **both** scorers, decaying to nothing by 0-256, and at 0-10 **inseparable from the [5009].3 mechanical joint artefact** — which is exactly the window where forcing's own destabilisation lives.

Y cannot separate a local production disturbance from an imposed-joint artefact. fc has the same problem at ten tokens. **Neither corpus can settle that, and the spec does not pretend otherwise** — separating them needs a control that forces a word matched on improbability but not on demotion, which no collected corpus has.

The §7 table below was written before this prior arrived and is **not** amended to match it.

## 7. DECISION RULE, both branches written before the run

| outcome | reading |
|---|---|
| Δ > 0, CI excludes 0 | **Ongoing defense.** The demoted signifier remains charged after utterance; repression is a standing pressure, not only a selection rule. Depth account holds the table. |
| Δ ≈ 0, **and MDE stated** | **Absorbed.** Cost is paid at selection and the chain runs on. Chain-substitution holds the table. A null is quotable ONLY with its MDE as a fraction of the channel-1 effect it constrains. |
| Δ < 0, CI excludes 0 | Neither account as stated. The aligned model diverges *less* after the demoted word — the forced signifier pulls it back toward the base's chain. Reported as-is, not explained away. This is the sign Y shows (§6b). |

**WHICH COMBINATION LICENSES WHICH SENTENCE** (registrar [5011].5). `D` measures aligned's dispreference for the **chain region the word opens**, not a charge on the signifier itself — base continues `said` into direct speech, and aligned may simply dislike that region. Secondary #3 is the separator:

| primary (evaluative) | secondary #3 (behavioural) | licensed sentence |
|---|---|---|
| Δ ≠ 0 | own-beam divergence flat | **standing displacement field with spatial extent** — aligned dislikes the region, but goes to the same places |
| Δ ≠ 0 | own-beam divergence also moves | **charged signifier**, the stronger form — aligned both dislikes the region and leaves it |
| Δ ≈ 0 | either | no channel-3 effect within ten tokens, MDE required |

Only the second row licenses "the demoted signifier remains charged". The first licenses a claim about the **field**, which is a different and weaker sentence, and the one the evidence will most likely support.

## 8. What this does NOT settle

The window is 10 tokens. Registrar's fence stands: **a repair at token 30 is invisible here.** A null on Δ bounds renewed displacement within ten tokens of the forced word and says nothing beyond it. The 100-token twin generation under a pinned decoder remains the decisive instrument for the longer window and awaits RH.

## 9. REQUEST TO @lacan — channels 2 and 3 have evidence at your seat, and it is longer-windowed than mine

RH's steer. Two things I cannot do from fc:

- **Channel 2 (repair) already has an M01 instrument.** `syntagmatic_js` under base vs under the aligned model on the same substitution pairs is a repair measure by construction, and it found alignment-produced damage that varies by category (sexual_explicit +0.106, profanity +0.032, violence_explicit +0.074, neutral +0.044). If wave-3's flat repair channel and M01's positive syntagmatic delta are measuring the same thing, they disagree and the disagreement is informative; if they are measuring different things, that needs saying before either is quoted against the other.
- **Channel 3 at 256 tokens is Y, not fc.** Y carries forced words with a 256-token window and, unlike fc, a **coded** annotation. `continues_narrative` and `noise_present` are exactly the fields a renewed-displacement claim needs, and your [5004] decomposition shows they carry real discriminating power. fc can only see ten tokens; Y can see whether the passage comes back to the scene at token 60.

If you take those, the three channels get answered at the grain each one needs, and fc's ten-token null (should it be null) is bounded rather than over-read.

## 10. WHAT A FREEZE DOES AND DOES NOT DO — RH's ruling, 2026-08-08

**A freeze binds the PRE-REGISTERED CLAIM. It does not close the question.**

RH's words: *freezes should not stop subsequent analysis.* Booked here because the opposite reading is the likelier failure mode at this seat — a frozen spec becomes a reason to refuse a question that the data plainly raises, and the campaign loses more to that than to unlabelled exploration.

What the freeze buys is exactly one thing: **the primary, its population, its unit, its test and its decision rule were fixed before the numbers existed**, so the headline cannot have been chosen to fit them. Nothing else is constrained.

Therefore, after this file is committed:

- **Anything may be computed on this corpus.** New contrasts, new windows, new strata, new metrics, follow-ups suggested by what the primary returns.
- **Post-freeze results are labelled EXPLORATORY and say so wherever they travel.** They may motivate a new registration; they may not be reported as confirmations of this one.
- **The primary is reported whatever it shows**, and is not withdrawn because an exploratory cut is more interesting.
- **A frozen null does not close the channel.** It bounds it at a stated MDE, and a better-powered or better-windowed instrument may reopen it — §8 already names one (100-token twins, pinned decoder).
- **The distinction is labelling, not permission.** No analysis needs authorisation; every analysis needs its status attached.

## 11. FREEZE BLOCK

**FROZEN 2026-08-08 on RH's word.** Registrar ruled the primary contrast [5011] and the required co-report [5015]; lacan supplied the Y prior [5012]/[5014]; the gate correction is registered [5015].

    seed                    20260808   (bootstrap resampling, 10,000 draws)
    population              design in {wave3-lexical, newlin-lexical}
    pairs                   35   (29 wave-3 + 6 new-lineage)
    n_lineages              35   (verified against model_to_base: all bases distinct)
    sites                   6,485 before the four-cell filter
    unit                    pair
    primary                 delta = D(force_faller) - D(force_riser), median over sites
    test                    Wilcoxon signed-rank over pairs, two-sided, alpha 0.05
    required co-report      four-term decomposition (A|A, A|B, B|A, B|B)
    arbiter                 CLEAN roster governs on sign/CI disagreement
    exact usable n          computed at run time by the four-cell filter and REPORTED,
                            not chosen — a site missing any of the four cells is
                            excluded and counted.

**MDE is computed after the run and that is declared, not hidden.** It requires the observed dispersion of per-pair values, which cannot be known in advance; it is computed at 80% power, two-sided α = 0.05, from the observed SD, and quoted **as a fraction of the channel-1 effect (+0.0144) it would need to constrain**. It touches neither the point estimate nor the decision rule.

**The run waits for the full 35 pairs.** The new-lineage corpus is mid-flight; running on 29 now and adding 6 later would be two looks at one population. One run, one report.
