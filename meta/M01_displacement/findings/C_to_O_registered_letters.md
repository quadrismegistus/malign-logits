# Findings C–O: the registered letters without their own write-ups

Written 2026-08-07 by the registrar seat on RH's commission (write-up push). One document for the
ten registered letters that ran under the pre-registration regime and whose results lived only in
REGISTRATIONS.md rows, ledger clauses and the README until now. Sources: REGISTRATIONS.md
(hashes and artifacts per row — authoritative for every number here), ledger.md, README.md.
Letters P, S, T, U, V, W, Q, R have their own findings files; B froze but never ran (no result to
write). Each section: what was asked, what was found, the limit that travels. Split any section
into its own file when it starts carrying more than this.

## C — valence/dominance de-extremification, general corpus

Asked: does alignment de-extremify affect dimensions corpus-wide? Found: **H2 valence CONFIRMED**
general (+0.025 residualised, p 0.0012); **H3 dominance dead**; H1 (sweetening) still blind,
never emitted. Limit: the artifact record is two `.txt` transcripts, nothing machine-readable —
quote C only alongside E, which replicated its H2 on the blind gap stratum with a real producer.

## D / D2 / D3b — the displacement-site suite (minimal pairs)

D asked whether displacement at transgressive sites moves valence/arousal in the registered
directions. Found: **H1 signed NOT SUPPORTED** (D +0.01272, p 0.8896) and **the arousal null is
QUOTABLE** (D −0.02841, p 0.9967) — at the sites, the substitution does not ride an arousal
gradient. D2 asked the two extremity arms: **BOTH CONFIRMED** — valence-extremity D +0.01525
(p 0.0076), dominance-extremity D +0.01624 (p 0.0114): what the substitution does move is
extremity. D3b decomposed D2 against pool extremity — a bracket by design, no significance test:
pool-associated share bounded above (valence +0.5609; dominance −1.1293, opposite sign);
pool-independent bounded above (valence +1.3372; dominance +0.7567). **"Just reflects pool
extremity" is dead; mediation stays open.** Limit: the suite is site-level and does not bridge to
the corpus level — Q tested that bridge and it reversed (see `findings/Q_bridge.md`).

## E — C's H2 on the gap stratum

Found: **19 of 25 lineages, p 0.0073, on the blind gap arm** — the de-extremification replication
with the lineage as the unit. Limit: the producer prints and writes nothing; the row in
REGISTRATIONS.md is the artifact of record.

## F / G — rate null, magnitude confirmed

The pair that set the campaign's shape. F asked whether alignment displaces at a higher RATE
within pairs at transgressive sites: **RATE NULL** (n=33 pair-sites, p 0.148). G asked the same by
MASS: **MAGNITUDE CONFIRMED** (d 0.748, p 0.00006). Alignment does not displace more often at
transgressive sites; it displaces HARDER. Limit: both result artifacts cite the pre-re-freeze
registration hashes (`8ff56206…`, `efbab158…`) — the REGISTRATIONS.md row documents the mismatch;
the results stand with that provenance note attached.

## L / M — the human's word (found prose)

L asked: given prose a novelist actually wrote, does the aligned model still hold the author's
word? Three rungs, no verdict by design (§L9), all z positive = alignment loses the human's word:
argmax Z +2.6372, top20 +4.5220, retained +5.5820 (retained tested on 31 clusters, not 34). M
adjudicated L's gradient with a perturbation null: **BOUNDARY BLUR, NOT TAIL CONTRACTION** —
overshoot Z −13.3170; escapes declared UNDERPOWERED; the eviction rate falls
0.157 → 0.045 → 0.020 → 0.008 → 0.003 across headroom deciles and is **exactly zero above the
fifth decile**. The mechanism: eviction concentrates entirely where the word was barely retained.
Limit: found-prose scope per `SCOPE_found_prose.md`; the escapes arm is underpowered and stays
declared so.

## N — mass migration at full scale (the flagship scale result)

Asked: does the substitution effect hold at the full English scale? Found: **SUBSTITUTION
CONFIRMED** — 2,199 stimuli × 44 edges, 82,775 cells, 91% negative, **34/34 clusters agree**;
the Stouffer Z is a FLOOR. This is Axis 1's anchor. Limit: English only (O carries the
cross-lingual arm); the cluster is the unit, stated with every quotation.

## O — the same content in two languages

Found: **H1 SUPPORTED IN BOTH ARMS** (en 2277/365, zh 2463/141) under §O6's Chinese-origin bound —
the substitution travels. **H2 and H3 NOT SUPPORTED**: the asymmetry is English-confirming and the
Chinese arms are clean coin-flip nulls (648/650, 652/646). The paper's sentence: the substitution
travels, the affect does not. Z is a FLOOR. Limit: §O6's origin bound governs; the zh nulls are
nulls, not reversals.

## B — frozen, never ran

The high-mass decomposition (movement within the high-mass set vs the tail, separately) froze at
`06186c42f9ff46e0` (v13) and never fired. No result exists; nothing here to quote. Recorded so the
letter is not mistaken for a missing write-up.
