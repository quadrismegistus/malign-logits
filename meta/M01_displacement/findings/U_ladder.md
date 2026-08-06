# Findings U: the alignment ladder, and what the siblings say

Written 2026-08-06. The plan is `registrations/plan_u_ladder.md`, written before any of this ran; it is a plan and not a registration, and the outcome readings below were fixed there in advance. Where this document and the plan differ, the plan is the record of what was expected and this is the record of what happened.

Every number here reproduces from `scripts/t_ladder.py` and `scripts/t_fans.py` on 2,182 prompts — all active English prompts, deduplicated on the string.

## The gap this fills

All 43 alignment edges in findings T have a **base** model on the pre side. The campaign had measured `base -> aligned` forty-three times and never once measured `SFT -> DPO`. Finding 9 addresses the question indirectly, comparing `base -> SFT-checkpoint` against `base -> DPO-checkpoint`, so its DPO edges contain the SFT step inside them; it cannot isolate DPO. Its headline — *supervised fine-tuning alone produces the operation at full strength, and preference optimization does not add to it* — rests on six checkpoints sharing one base, with the pooled version disowned in its own text.

## 1. SFT does the bulk, and preference optimization adds about a fifth

The OLMo-2-0425-1B ladder, every rung measured separately:

| rung | JS | fallers/site | risers/site | faller share |
|---|---|---|---|---|
| base -> SFT | **0.1963** | 17.67 | 17.60 | 50% |
| SFT -> DPO | 0.0449 | 3.30 | 7.63 | 30% |
| DPO -> Instruct | 0.0045 | 0.0064 | 1.5289 | **0.42%** |
| base -> Instruct *(the edge findings T measures)* | 0.2646 | 22.09 | 19.67 | 53% |

SFT carries 74 percent of the ladder's total JS. **Finding 9 is confirmed by direct measurement rather than inferred from endpoints.** But its wording is too strong: DPO is not null. It carries a fifth of SFT's JS and moves about eleven words per site. The accurate sentence is *preference optimization adds about a fifth as much*, not *does not add to it*.

## 2. Removal stops while addition continues

The faller share — what proportion of moved words fall rather than rise — collapses up the ladder. On OLMo-2: 50 percent, 30 percent, 0.42 percent. Across 16 families (medians): **49.3 percent at `base -> sft`, 28.6 percent at `sft -> pref`, 1.0 percent at `pref -> rlvr`**. It drops in 13 of 16 families, Wilcoxon p = 0.011.

**This is new. It is in neither finding 9 nor findings T**, both of which only ever see the composite. By the last rung the model has effectively stopped taking anything away and is only putting things in.

It also reproduces independently. The pre-operation embedding store, built for a different purpose and never analysed, carries `role` labels whose faller share falls 54 percent to 22.6 percent to 0.8 percent across the same stages — different data, different code path, same gradient.

**Held to what it will bear.** Grouped by lineage rather than by family — AI2 is 6 of the 16 and Olmo pretraining accounts for 5 — the gradient drops in 7 of 9 groups at p = 0.164. **Not significant at the lineage unit.** That is an underpowered test at n = 9 and not a refutation, but the flat result leans on AI2 counting six times and the honest statement is that the gradient holds in most families and the lineage-level test cannot confirm it at this roster size.

**Three families break the pattern**, and their faller share rises rather than falls: `amber` 54.1 to 58.3, `redpajama` 35.1 to 81.0, `tulu` 36.3 to 37.0. All three are families where the second rung is plausibly **siblings rather than a training step** — AmberSafe and RedPajama-Chat may both be trained from their base rather than from the SFT arm, and tulu's base is a cross-family Llama checkpoint declared by hand in the script. That is a hypothesis about why they differ, not grounds to exclude them; dropping the three that disagree and keeping the thirteen that agree would be fitting.

**And Pythia removes nothing at any stage**: `pythia` 1.7 to 0.5 percent, `archangel-dpo` 0.4 to 0.0. Whatever alignment does to Pythia is almost purely additive from the beginning.

**A corrected figure, and how it went wrong.** The `DPO -> Instruct` faller share read 0.6 percent here until the registrar checked it against the CSV ([4775]). The true value is **0.42 percent**: `0.006416 / (0.006416 + 1.528873)`. The table displays `fall` rounded to two places, and the share was computed from the DISPLAYED 0.01 and 1.53 rather than from the data -- `0.01/1.54 = 0.65%`, which rounds to the 0.6 that was printed. **A rate computed from rounded display values is a different quantity from the same rate computed from the data**, and at small numerators the difference is 50 percent of the value. The docket post quoted 0.4 percent from the CSV and was right; the file was wrong, and the file is the quotable record.

## 3. The rungs move the same words, not different ones

Faller Jaccard between `base -> SFT` and `SFT -> DPO` is **0.044** on OLMo-2 and 0.023 across families, and exactly zero between non-adjacent rungs. **That is not evidence of different operations, and reading it that way was wrong.**

72 percent of `base -> SFT`'s fallers sit below `CANONICAL`'s 0.003 floor in the SFT distribution and **cannot fall again**. The near-zero overlap is mechanical. Among the 28 percent that remain eligible:

| eligible at SFT | n | fall again at SFT -> DPO |
|---|---|---|
| also fell at `base -> SFT` | 10,843 | 2,023 = **18.7%** |
| did not fall at `base -> SFT` | 82,964 | 5,187 = **6.3%** |

**Ratio 2.98. Fisher odds ratio 3.44, p below the floating-point floor.** DPO preferentially re-targets the words SFT already lowered, pushing further down the ones SFT left standing. This is the plan's outcome three — one operation applied twice — and finding 9's spirit is right.

The correction is recorded because the error is instructive: the Jaccard was correctly computed and the conclusion drawn from it was refuted by a denominator. Nothing about the statistic was wrong; the reading was, and only a second measurement could show it.

## 4. Safety data is not what produces the operation

The sharpest result here, and the first direct test of a mechanism the whole campaign has been assuming. Every aligned checkpoint in the roster has safety data in it, so no previous finding can distinguish "alignment suppresses transgression" from "safety tuning suppresses transgression." Tulu's SFT ablations are the only place in the roster where the corpus is varied with everything else held fixed: one base, one recipe, five training sets.

| arm | JS | % of full | fallers/site | faller share |
|---|---|---|---|---|
| full | 0.0651 | 100% | 5.63 | 36.3% |
| no-math | 0.0576 | 88% | 4.60 | 33.3% |
| no-persona | 0.0572 | 88% | 4.52 | 33.0% |
| **no-safety** | **0.0583** | **90%** | 4.78 | 33.8% |
| no-wildchat | 0.0584 | 90% | 4.82 | 34.1% |

**The control is the other three arms.** Removing any slice costs 10 to 12 percent, and the four ablations span 0.0572 to 0.0584 — a spread of 1.8 percent of the effect. **Removing the safety corpus costs the same as removing the maths corpus.** If displacement were the safety objective, `no-safety` would collapse and the others would not; instead the four are interchangeable.

The words also stay the same. `full` against `no-safety` has faller Jaccard 0.534, among the highest in the fan, against 0.02 to 0.04 between ladder rungs. Change the training corpus and you get the same operation on the same words; change the rung and you do not.

**What this licenses and what it does not.** It licenses: *on this family, the operation is insensitive to which SFT corpus is removed, safety included.* It does not license a general claim that safety data is irrelevant to alignment — only that **this operation** is not the safety objective's signature. The reframing is real either way: what findings T measures may be something supervised instruction-following does whatever it is trained on, rather than a suppression the model was specifically taught.

**And it is one family on one base -- permanently, as far as we can establish.** A search of HuggingFace, arXiv and lab post-training documentation (agent report, 2026-08-06, HF API rather than search so the repo ids are real) found **no second suite meeting the bar**: same base, same recipe, one corpus slice removed, weights released. It also confirmed our five arms are the complete Tulu set -- `-no-code-data`, `-no-if-data` and `-no-science-data` were probed for and do not exist -- and that no lab besides H4 and AI2 publishes a safety-in/safety-out pair.

**So the caveat changes status from PENDING to UNAVAILABLE, which is a different thing and should not be read as the same.** "Wants replication" implies replication is possible and someone should get on with it. Here the instrument does not exist in the open-weight ecosystem. The nearest miss is instructive: Meta's MobileLLM-Pro ran leave-one-out ablations across seven instruction-tuning domains and reported them in its tech report, but released four checkpoints and none of the ablated ones. That is the normal case. **This finding rests on one family because one family is all anybody has published**, and it should be quoted with that sentence attached rather than with a promissory note.

**A natural experiment points the same way, more weakly.** `zephyr-7b-beta` is in the roster and its model card is explicit: *"We found that removing the in-built alignment of these datasets boosted performance on MT Bench and made the model more helpful,"* and *"Zephyr-7B-beta has not been aligned to human preferences for safety within the RLHF phase or deployed with in-the-loop filtering."* Its developers stripped alignment out of the training data on purpose and say the model has no safety alignment. It still displaces:

| | zephyr `base -> sft` | median of 16 families |
|---|---|---|
| JS | 0.0747 (rank 12 of 16) | -- |
| fallers/site | 7.27 | 12.27 |
| faller share | 41.9% | 49.3% |

**Reduced, not absent** -- about 59 percent of the median family's faller count. A pure safety-artefact account predicts near zero, and 7.27 fallers per site is not that.

**This is weaker evidence than the ablation and should not be quoted as stronger.** Zephyr differs from the other fifteen families in base, corpus, recipe and scale simultaneously, so its position in the distribution is confounded several ways over; being below median may have nothing to do with the missing alignment data. The Tulu fan holds everything fixed but the corpus and is the controlled version of the same question. Zephyr is consistent with it, which is worth recording, and is not independent confirmation of it.

## 6. Same words, same direction, and the field difference is power

Three grains over 1,281,413 word-role rows, 16 families, `scripts/t_ladder_fields.py`.

**Word.** Net-movement profiles for `base -> sft` and `sft -> pref` correlate at **Spearman rho = +0.350, p = 1.3e-232 over 8,108 shared types**. Positive and unmistakable, and also modest -- the rungs agree on direction far more than chance and are nowhere near identical. `whispered` is the top riser at BOTH rungs and `sighed` is in both top tens.

    base -> sft   falls  a, put, he, told, when, threw, turned, in, got, i
                  rises  whispered, felt, found, watched, made, the, began, take, consider, sighed
    sft -> pref   falls  he, put, was, went, had, i, gave, the, go, told
                  rises  whispered, sighed, now, stared, proceeded, carefully, shouted, gently, explain

**Word pair.** 1,101,963 distinct `(faller, riser)` pairs at `base -> sft` against 668,751 at `sft -> pref`, **Jaccard 0.267**, 373,611 shared. The same modest-overlap picture as the word grain.

**Field.** Of 984 and 978 category tests, 32 survive Bonferroni at `base -> sft` and 14 at `sft -> pref`. Split three ways: **29 significant at SFT only, 11 at DPO only, 3 at both** -- `perception_cognition` (+0.051 / +0.054), WordNet `perception` (+0.025 / +0.037), RID `anxiety`. **No sign flips anywhere.**

**That split looks like differentiation and is mostly power.** DPO's effect is a fifth of SFT's amplitude, so SFT clears correction where DPO cannot, and a field present at one rung and absent at the other is more often a detection difference than a difference in kind. Nothing reverses; perception rises at both. **The reading the three grains support together is one operation at two amplitudes**, not two operations.

## 7. DPO is not a register of manner, and a ranked list said it was

The top risers at `sft -> pref` include `gently`, `carefully`, `stared`, `proceeded`, and the induced taxonomy's `quality_manner` is among the fields significant at DPO and not at SFT. That suggested a reading on which SFT installs the prohibition and the later rungs supply comportment -- the cut, then the finishing school. It is wrong.

Manner declared from three independent resources BEFORE the contrast was run, riser share compared between rungs, paired within family:

| manner definition | types | base -> sft | sft -> pref | diff | p |
|---|---|---|---|---|---|
| VerbNet `manner_speaking` | 50 | 0.0150 | 0.0162 | +0.0012 | 0.56 |
| FrameNet `Communication_manner` | 31 | 0.0123 | 0.0124 | +0.0001 | 0.94 |
| `-ly` adverbs, CLAWS `rr*` | 400 | 0.0231 | 0.0247 | +0.0016 | 0.63 |
| any of the three | 456 | 0.0383 | 0.0412 | +0.0029 | 0.56 |

**Flat on all four**, 8 or 9 of 16 families higher, a coin flip on every definition. Manner rises at both rungs at the same rate.

**Recorded because the error is the day's most repeated one.** DPO's list reads as manner-inflected because `whispered` and `sighed` top BOTH lists and the adverbs were noticed on the second pass. A ranked list carries no baseline, and four separate claims today have been suggested by a top-N and killed by a denominator. **What DPO does instead remains unspecified**: not manner, not different words (it re-targets SFT's at 2.98x), not different fields once power is accounted for. On present evidence it is the same operation at a fifth the amplitude.

## 5. Preference optimization removes nothing at all

Four preference methods diverging from a single shared SFT checkpoint — the only place in the roster where recipe varies with the base, the corpus and the SFT stage all held fixed:

| method | JS | fallers/site | risers/site |
|---|---|---|---|
| dpo | 0.0030 | **0.00** | 0.54 |
| kto | 0.0048 | **0.00** | 0.97 |
| ppo | 0.0019 | **0.00** | 0.71 |
| slic | 0.0038 | **0.00** | 0.69 |

**Not one method produces a single faller per site.** All four only add. The faller Jaccards between arms are `nan` because both sides are empty sets.

**The confound is the base, and it is serious.** This fan sits on pythia-2.8b, whose faller share is near zero at every rung and in every edge of the main population — the pythia edges are the lowest in all 43 at 0.7 to 1.4 percent. So this may be a fact about Pythia rather than about preference methods. What it does establish is that **four different preference algorithms agree with each other**, which is worth having: whatever accounts for the near-zero fallers, it is not specific to DPO.

**A registry-labelling artefact hid three quarters of this fan** from the family-level analysis and is recorded so it is not re-encountered. The shared SFT checkpoint is filed under `family=archangel-dpo` while the kto, ppo and slic arms sit in their own families, so a per-family ladder search finds no base and no SFT for three of the four and skips them. `t_ladder.py` saw one Archangel family where there are four. The fan definitions in `t_fans.py` are declared by hand for exactly this reason.

## Limits

**The unit problem is unresolved.** Sixteen families are not sixteen independent sources: AI2 is 6 of them, Olmo pretraining 5, EleutherAI Pythia 2, m-a-p 2. Findings 1 and 3 are single-family and descriptive. Finding 2's flat test is significant and its lineage-clustered test is not. The registry has no `lineage` field, and the campaign's own ruling — *a pair floor counts independent sources, not pairs* — has not been applied to this roster because no independence map exists.

**Ladder provenance cannot be checked from our data.** `Step`'s relations are star-shaped from the base: the registry declares each rung's stage but never records that `-DPO` was trained *from* `-SFT`. The ordering comes from published pipelines and naming. Where it is wrong — plausibly amber, redpajama — the rung-to-rung steps measure a difference between siblings rather than a training step, and those are exactly the families that break finding 2.

**`pref -> rlvr` rests on five families**, all AI2, and should not be quoted with the confidence of the other two rungs.

**JS is not additive**, so the rung sum against the whole edge (0.246 against 0.265, ratio 0.93) is a shape check and not an identity.

## Data and code

    ladder       meta/M01_displacement/scripts/t_ladder.py
                 results/t_ladder_steps.csv, t_ladder_jaccard.csv
    fans         meta/M01_displacement/scripts/t_fans.py
                 results/t_fans.csv, t_fans_jaccard.csv
    plan         meta/M01_displacement/registrations/plan_u_ladder.md
    prompts      2,182 active English, deduplicated on the string
    movement     CANONICAL (min_prob 0.003, fall_ratio 0.5, delta 0.003)
    JS           word-level, over the union support, residual retained
