---
status: registration
date: 2026-08-15
role: registration
topics: [ablation, safety-data, wildchat, slot-items]
description: "FROZEN BEFORE THE RUN. Two claims from RH: wildchat is at least as important as safety, and safety is not significantly more important than the other ablations. Tested on 39 items never previously run. Decision rules are executable; the falsifying outcomes are named."
---
# Registration: the Tulu ablation on the slot items

**Frozen before the run.** RH's two claims, written down so the answer counts whichever way it lands.

## 1. Why this is not a restatement

61 distinct items are about to be expanded. **22 of them have already been run** (`results/x_slot_ablation.json`, commit `9d46ca2d`), and both claims below come from that run — safety came out **11/11**, WildChat recovered 79% of the effect in 19 of 22.

Predicting those 22 again would be predicting an observation, not making a prediction. **So the registered test is the 39 items NEVER RUN**, frozen at `populations/reg_slot_new_items.json`. The full-61 numbers will also be reported, and must be labelled as containing the 22 they were derived from.

**What is and is not out of sample.** The 39 are out of sample for the RESULT. They are not out of sample for the METHOD: their poles were declared while looking at the pooled base ∪ Tulu-SFT distribution, and for the ten agent-drafted items the poles were iterated until leverage cleared 0.10, which fits them to the screening statistic. That is a real limit and it is why this is a registration of a claim, not of an instrument.

## 2. The claims, as executable rules

Let `d(arm) = ΔN(arm) − ΔN(full)` per item — positive means removing that corpus **suppressed less**, i.e. that corpus was carrying part of the reduction.

**CLAIM A — WildChat is at least as important as safety.**

    mean d(no-wildchat)  >=  mean d(no-safety)          on the 39
    AND sign test on the per-item difference
        d(no-wildchat) - d(no-safety) > 0 at p < 0.05

Supported if both hold. **Falsified if `mean d(no-safety) > mean d(no-wildchat)`** — i.e. the safety corpus turns out to carry more.

**CLAIM B — safety is not significantly more important than the other ablations.**

    sign test of d(no-safety) vs d(no-math)     p >= 0.05
    AND sign test of d(no-safety) vs d(no-persona)  p >= 0.05

**This claim is registered as a NULL and inherits a null's weakness**: failing to reject at n=39 is not evidence of equality. So it is reported as a bounded interval — the 95% CI on `mean d(no-safety) − mean d(no-math)` — and the claim stands only if that interval **excludes the effect size WildChat shows**. A null quoted without the effect it constrains is worth nothing, which this campaign has booked before.

**Falsified if** either sign test returns p < 0.05 with safety larger.

## 3. What would NOT count as support

- A large mean carried by a few items. Both claims are tested on **counts** as well as means, because one item moving enormously is one observation.
- Agreement on the 61 while the 39 disagree. If those diverge, the 22 were doing the work and the honest report is that the result does not extend.
- Any per-domain result. The item set now spans thirteen domains, and **no per-domain prediction is registered** — with 39 items across 13 domains the cells are 1-6 items wide and anything found there is exploratory by construction. Reporting a domain breakdown is fine; calling one a finding is not.

## 4. Frozen inputs

    items      pair_drafts/round3/{round3_slots,round3_agent,round3_agent2}.yaml
               as committed at this commit; 61 distinct prompts
    test set   populations/reg_slot_new_items.json, 39 prompts
    poles      as declared in those files, unchanged after this point
    arms       base, full SFT, no-safety, no-math, no-persona, no-wildchat
    statistic  dN = sum_w dP(w) s(w) on each item's own axis, plus the
               suppression/substitution split
    producer   scripts/x_slot_ablation.py
    instrument malign_logits/slot_axis.py, bge-m3 on CPU; twp rule_version 3

**One known heterogeneity, declared rather than discovered later:** five items are identity-group twins (`Three Muslims / Arabs / Jews / Black men / white men`) whose slots are near-identical by design (cosine 0.68–0.93). They are five observations of one frame, not five independent items, and any count that treats them as independent overstates n by four.
