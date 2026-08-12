---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-07
role: finding
description: "Registered letters L/M, found prose: alignment loses the human author's word on all three rungs (argmax Z +2.64 to retained +5.58); M adjudicates the mechanism as BOUNDARY BLUR not tail contraction -- eviction is exactly zero above the fifth headroom decile. Escapes arm underpowered and stays declared so; found-prose scope per SCOPE_found_prose.md."
---
# Findings L / M: the human's word (found prose)

Split out of `C_to_O_registered_letters.md` on 2026-08-12 (RH's commission; the omnibus was written 2026-08-07 on the write-up push). Content verbatim from that document as of the split; REGISTRATIONS.md remains authoritative for every number.

L asked: given prose a novelist actually wrote, does the aligned model still hold the author's
word? Three rungs, no verdict by design (§L9), all z positive = alignment loses the human's word:
argmax Z +2.6372, top20 +4.5220, retained +5.5820 (retained tested on 31 clusters, not 34). M
adjudicated L's gradient with a perturbation null: **BOUNDARY BLUR, NOT TAIL CONTRACTION** —
overshoot Z −13.3170; escapes declared UNDERPOWERED; the eviction rate falls
0.157 → 0.045 → 0.020 → 0.008 → 0.003 across headroom deciles and is **exactly zero above the
fifth decile**. The mechanism: eviction concentrates entirely where the word was barely retained.
Limit: found-prose scope per `SCOPE_found_prose.md`; the escapes arm is underpowered and stays
declared so.
