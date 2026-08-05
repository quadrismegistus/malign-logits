# Registration S

Written 2026-08-05, before the run. No hashes and no escrow. The only thing this document does is say what I expect before I look, because the 255 stems below are the last vv\*-eligible stems that exist and there is nothing behind them to replicate on.

It IS countersigned, which the first draft said it would not be. RH asked for a second reading and registrar gave one at [4697]: it caught that condition 2 was a gate with no uncertainty attached, that a confirming primary and a confirming conditional would be one finding and not two, and it recomputed the disputed per-order figures independently. Three of those four changes are in this document because of that reading, so the countersignature earned its place rather than decorating the run.

## Two stages

**Stage 1, the spent 50.** The 50 stems already burned on the revision-2 calibration, re-run under revision 3. They can never be confirmatory, so this costs no held-out sample. It exists to answer one question the six smoke items cannot: what is the base rate of the primary conjunction? A joint prediction with no base rate cannot be powered, and finding out on the held-out 255 would mean finding out after 7,140 calls. 200 items, 7 coders, about 1,400 calls.

Stage 1 is a pilot for predictions that were written and committed before it ran. It does not set them. If stage 1 contradicts the primary, the primary stays as written and stage 2 tests it anyway; a pilot that revises its own hypothesis is not a pilot.

**Stage 2, the held-out 255.** The confirmatory. Described below.

## What is being run

The 255 held-out stems, both members, both orders, 7 coders.

- Instrument: `malign_logits/tasks/code_operation_binaries.py`, revision 3, 7 coded fields.
- Coders: deepseek-v4-pro, deepseek-v4-flash, gemini-3.6-flash, gemini-2.5-flash, claude-haiku-4-5, claude-sonnet-5, gpt-5.4-mini.
- gpt-4o-mini is excluded. It scored 15/18 on a fresh six-item check where six of eight coders scored 18/18, it is the only coder that never reaches `B_GENERIC`, and it inverted `more_transgressive` on items where its own prose argued the other way. Excluded before any revision-3 data exists.
- 1,020 real items, plus 1,015 decoy items (below), = 2,035 items and 14,245 calls.
- 50 stems were spent on the revision-2 calibration and are not in this frame. Disjointness is asserted in `build_s_stage2_frames.py` against the stage-1 file itself, not against a count.

## The design

FR presents the pair as (faller, riser). RF presents the same two words as (riser, faller). The control is the identical word pair, so light verbs, frequency and selection rules cancel by construction rather than by a matched control population. Every matched population built for Registration R carried its own lexical character and that character turned out to be the effect.

For each (stem, member), the quantity is the rate over 7 coders in FR minus the rate in RF. A positive difference means the field fires more when B is the risen word.

Test: sign-flip permutation on those per-stem differences, 20,000 draws, seed 20260806, two-sided.

## Predictions

**Primary, one test. The displacement conjunction.** `register = B_CONTINUES` **and** `pitch = B_MILDER` **together** is positive, **and the excess of that conjunction over the product of its two marginals is itself positive under the same test**. Both conditions, or the primary is not confirmed.

**Condition 2 is a test, not an inequality.** Per (stem, member, order) compute the excess `j - c*m`, where `j` is the conjunction rate over the seven coders, `c` the `B_CONTINUES` rate and `m` the `B_MILDER` rate. Take FR minus RF per stem and run it through the same sign-flip permutation as everything else, 20,000 draws, seed 20260806. Confirmation is positive and p < 0.05.

Registrar's objection at [4697].2 is why this is here: as first written, condition 2 was a bare comparison of +0.071 against +0.037, which is a gate whose refusal has no uncertainty attached. The alternative was to demote it to a descriptive benchmark and say so in every report of the primary. Specifying the test is one paragraph and makes the primary whole, so it is specified.

This is the shape of Freud's Verschiebung and it is the only cell in the instrument that has it: the substitute stays inside the scene and carries less charge. `kill -> scream` is exactly this. The second condition is what makes it a test of displacement rather than of its parts, because a conjunction can rise purely because one component rose. Positive dependence between staying-in-register and going-milder is the claim; the marginals alone are not.

**Amended 2026-08-05, before any revision-3 data existed on real items.** The first draft made `register = B_GENERIC` primary. That is the wrong cell for this hypothesis. In Freud the affect has to attach somewhere else; if the substitute simply declines to carry the charge while the setup clause keeps it, nothing has transferred, and that is nearer repression than displacement. B_GENERIC remains a declared secondary because deflation is a real and separate finding, but it is not the displacement claim and must not be reported as one.

**Secondary, seven tests, reported at nominal p with the count stated.**

| field | predicted sign | basis |
|---|---|---|
| `register = B_GENERIC` | positive | deflation, NOT displacement. Light verbs were 35.0% of argmax picks against 24.2% of the pool, and `sat/drove/went -> found` is the largest cluster in the R data |
| `register = B_DIFFERENT_REGISTER` | positive, smaller | redirection: the substitute says something else rather than less |
| `register = B_CONTINUES` | no predicted sign | it is a component of the primary and reported for interpretation only |
| `more_transgressive` | negative | rev2 calibration, -0.129 |
| `pitch = B_MILDER` | positive | rev2 calibration, +0.104. Also a component of the primary |
| `pitch = B_STRONGER` | negative | rev2 calibration, -0.133 |
| `becomes_speech` | positive, weak | rev2 calibration, +0.053, p=0.076. Not expected to clear 0.05 |

**Declared stage-1-derived, one test. The conditional.** `pitch = B_MILDER` restricted to annotations where `register = B_CONTINUES` is positive, **and larger than** the same quantity restricted to `register = B_GENERIC`.

This is where the finding actually lives, and it was discovered at stage 1 rather than predicted before it. Stage 1 gave +0.205 (p=0.0076, n=41 pairs) inside B_CONTINUES against +0.037 (p=0.22, n=63) inside B_GENERIC. Read plainly: the substitute comes in milder only when it stays in the scene, and a substitute that goes generic does not soften because it carries nothing either way.

It is labelled derived and it stays labelled derived in every report. It is tested on held-out data under the same rule as everything else, predicted sign and p < 0.05, which is the only thing that makes a stage-1 discovery worth anything. Two limits travel with it: n was 41 pairs, and conditioning on `register` means conditioning on a coder's own judgement rather than on an assigned condition.

**If the primary and the conditional both confirm, they are ONE finding reported through two lenses, not two.** Registrar's point at [4697].1 and it costs a sentence to guard: both are the same dependence structure, the co-occurrence of staying-in-register with going-milder, tracked by direction. The conjunction states it as a joint rate and the conditional states it as a contrast between arms. Reporting them as independent corroboration would be the false-corroboration error with a design instead of a measurement behind it.

**The decoy arm, and why stage 2 is not complete without it.** `register = B_GENERIC` took 54.2% of stage-1 annotations. R's CO_ACT took 59% and was killed for it, but the thing that killed it was not the share: it was that words which NEVER MOVED took the label at the same rate, so it carried no information about alignment. Stage 1 has no non-movers and could not run that test.

Stage 2 therefore includes both decoy sets already built for R, at FR only, B = decoy:

| arm | file | light verbs | n |
|---|---|---|---|
| RANDOM | `r_confirm_decoys_random.parquet` | 27.0% | 508 |
| RANDOM_NL | `r_confirm_decoys_randomNL.parquet` | 0.0% | 507 |

RANDOM is population-matched: the eligible pool is 24.2% light verbs, so a uniform draw is the fair comparison. RANDOM_NL is the harder one, contentful words that did not move. The argmax decoy set is NOT used; it is 35.0% light verbs against the pool's 24.2% and its composition was the effect in R.

**Stated before the run: if `B_GENERIC` fires at the same rate on non-movers as on risers, the arm carries no information about alignment and the deflation secondary is withdrawn.** That sentence is the whole reason the arm exists, and it is written here so that a null cannot be reinterpreted afterwards as evidence of anything else. The conditional above does not depend on this, because it compares B_MILDER across arms rather than reading the arms themselves.

**No prediction. Symmetric by construction, and their difference estimates position bias.** `related`, `substitutable`, `bare_verb`. Whatever they show is the coders' thumb on the scale and it is the correction the directional numbers need. On the R corpus this ran at 0.010. None of the three may be reported as an effect, and `related` is a validity floor at ceiling that is not to be analyzed at all.

## What counts as confirmation

Predicted sign and p < 0.05. That is the whole rule.

Seven directional tests are declared above and the count goes in any reported result. Registration R's +0.235 was the largest of ten tests at n=50 and it did not survive a second run; naming the number of tests is what makes that visible, and it is the one piece of apparatus from that campaign worth keeping.

## Limits, stated now rather than discovered later

**The revision-2 priors come from a different instrument.** Three fields were removed between revision 2 and revision 3, and fewer questions may change how coders answer the ones that remain. The priors above are direction and sign only. Nothing is powered off their magnitudes, because treating a selected estimate as a design input is exactly how R's confirmatory was mis-sized.

**Seven coders is 2/2/2/1 across providers.** Coder identity mattered in R, and OpenAI carries a single slot here.

**The primary is a new field with no prior.** Six smoke items on eight coders is not a pilot. The prediction above is derived from the light-verb diagnosis rather than from any measurement of `register` itself, and if it fails there is no second sample.

**claude-sonnet-5 is not sampling-pinned.** It rejects the `temperature` parameter outright and dropped it on all 200 stage-1 calls, and on the revision-2 calibration before that. The task declares 0.0 and six of seven coders honour it. Do not describe these runs as temperature-controlled.

**The pooled dependence diagnostic is retired as a mis-cut.** `result_s_stage1.json` reports observed-over-expected conjunction as 1.07x pooled across FR and RF. Order is the manipulated axis, so pooling averages over the manipulation. Per order it is 1.32x and 0.62x, independently recomputed at the registrar's seat as 1.316 and 0.620. Per-order is the diagnostic of record, both orders always printed, and the pair is stronger than either alone: the dependence is present when B is the risen word and reversed when B is the fallen one, so the dependence itself tracks movement.

**A null on the primary is a result.** If risen words are not more generic, the deflation reading is wrong and the paper says so.
