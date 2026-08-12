---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-07
role: finding
description: "Registered letters L/M, found prose: alignment loses the human author's word on all three rungs (argmax Z +2.64 to retained +5.58); M adjudicates the mechanism as BOUNDARY BLUR not tail contraction -- eviction is exactly zero above the fifth headroom decile. Escapes arm underpowered and stays declared so; found-prose scope per SCOPE_found_prose.md."
---
# Findings L / M: alignment loses the human author's word, by blurring the boundary

Split out of `C_to_O_registered_letters.md` on 2026-08-12 (RH's commission), rewritten the same day to be readable on its own: the hypotheses are stated in plain terms rather than by registration number. `REGISTRATIONS.md` remains authoritative for every number; the registration files hold the frozen statistical detail.

## What was asked

**L** put found prose to the models — passages a novelist actually wrote — and asked, at each next-word position: does the model still hold the author's actual word? Three increasingly generous tests, no verdict attached by design (the registration declares them descriptive): is the author's word the model's single TOP choice (argmax); is it within the model's TOP 20; is it RETAINED at all, above the probability floor? Each rung is a base-versus-aligned contrast. **M** then adjudicated the mechanism behind L's gradient with a perturbation null: when the author's word gets evicted (drops below the retention floor), does eviction concentrate just above the floor, where the word was barely holding on (boundary blur), or does it also strike words the model held with real confidence (tail contraction — a genuine shrinking of the model's repertoire)?

## What was found

**L: alignment loses the author's word at every rung.** All three contrasts run positive — the aligned model holds the human's word less than its base does — and get larger as the test gets stricter: argmax Z +2.64, top-20 Z +4.52, retained Z +5.58 (the retained rung tested on 31 of the 34 clusters).

**M: it is boundary blur, not tail contraction.** Eviction concentrates entirely where the author's word was barely retained. Across headroom deciles (how far above the floor the word sat), the eviction rate falls 0.157 → 0.045 → 0.020 → 0.008 → 0.003 and is exactly ZERO above the fifth decile: a word the model held with any real margin is never lost. The overshoot statistic lands at Z −13.3 in the decay direction. The picture is not a model whose repertoire has shrunk, but a model whose grip on marginal words has loosened.

## The limit that travels

Scope is found prose, as defined in `SCOPE_found_prose.md`. The escapes arm (words breaking upward) was declared UNDERPOWERED and stays declared so — no sentence about escapes is licensed.
