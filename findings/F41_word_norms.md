---
status: measured-single-seat
grade: ungraded
date: 2026-07-31
role: finding
description: "Word-norm instrument (arousal/concreteness/dominance, en+zh): exogenous test of the intensity-dissolution frame. Predictions registered before any norm-movement join."
instruments: [word_norms, true_word_probs]
chapters: [ch05, ch06]
data: [TheoryMachines/norms_sources/]
scripts: [pending — producer authorized under the frozen spec]
---

# F41: Word norms — the exogenous gradient test

FIRST FILE UNDER THE POINTER CONVENTION (RH's word, 2026-07-31): a
finding file is born WITH its registration, spec, and producer, and
consists of pointers and statuses, not narrative. The docket is the
argument; this page is the map.

## Status

MEASURED, QUOTABLE (2026-07-31, same day as registration). P1 arousal
CONFIRMING, quotable as MEASURED-SINGLE-SEAT with no conditions; the
canonical quotable block is docket [1266].2 and NOTHING outside that
list travels. P2 concreteness NULL, quotable unconditionally
(displacement is affective, not abstractive). P3 dominance and the zh
arm did not run in this pass (en-only; zh gated on a declared
closed-class list). Full audit trail [1247] -> [1254]; gate CLEAR.
Tail before VERIFIED: six-item edit window -> custody block -> C1
refresh; P1 two-seat verification post-gate (malign eligible, seat
preserved [1262].1).

Withdrawn forms that do NOT travel: "grows by 13%" ([1261].1); the
1/n floor decomposition ([1273]); the database r = +0.026 quoted as
a moving-vocabulary fact (sign flips to -0.040 on the moving
vocabulary, [1263] — frequency slightly suppresses the effect).

Canonical-block amendment ([1284], post-window): line (ii) is the
FLOORED pair +0.199 w / +0.109 unw (median over cells clearing the
registered 0.60 mass-coverage floor); +0.221/+0.123 is the
UNFILTERED pair, named not superseded. Every numeric line names its
cell filter (A FILTER IS PART OF A NUMBER'S NAME). Producer hash
after the six-item window: 87d6d405d8ab3fc2.

STANDING ASYMMETRY ([1279].3): corr(concreteness, logfreq) = +0.177
in the database, against arousal's +0.026. The two dimensions never
deserved the same control confidence. P2's null is unthreatened,
but any FUTURE concreteness-effect claim — the LLM-norms second
instrument included — carries a live frequency-control burden that
arousal's does not ([1277].1 flag).

## Pointers

| Object | Where |
|---|---|
| Pre-registration (predictions P1-P4, sources pinned) | docket [1147] |
| Spec, drafted blind | docket [1150] |
| Spec freeze + blind floors + restricted-null promotion | docket [1152] |
| §2 header verification, falsification, amendment | docket [1153] -> [1154] |
| Results: P1/P2 arms, weighted + unweighted | docket [1247] |
| Audit (four candidates) + clearances | docket [1254] -> [1257], [1261] |
| Frequency control P4(b): database + band + moving-vocab transfer | docket [1259], [1260], [1263] |
| QUOTABILITY RULINGS + canonical P1 block | docket [1255], [1266].2 |
| Registrations A (curve) + B (decomposition), directions fixed | docket [1258]; coupling columns [1270].1 |
| Sparse-concentrated coupling (structural, unit-level) | meta/M01_displacement/README.md structural note; [1264]/[1267]/[1272] |
| Source files, staged + hashed | /Users/rj416/Dropbox/Prof/Articles/TheoryMachines/norms_sources/ (ALL FOUR, incl. Brysbaert consolidated 2026-07-31 — the earlier 'at its abslithists home' was a gesture, not a location: [1212].2) |
| Producer | lacan's seat; enters scripts/ in the custody block (six-item window pending) |

## Sources (16-hex prefixes; full digests in the spec artifact)

| Source | Role | Hash |
|---|---|---|
| Warriner et al. 2013 (en V/A/D, 13,915, M/F splits) | en affect | 85f6d7e35069b0ef |
| Brysbaert et al. 2014 (en concreteness + SUBTLEX + Percent_known) | en concreteness + FREQUENCY (two-source join, [1154].1) + reliability floor | 0b4082dbd38585b0 |
| Chan & Tse 2024 (zh, 5 dims + 3 log-freqs, Word_Trad + Word_Sim) | zh PRIMARY (14/14 gate-one coverage) | f1ae2435300c2a41 |
| Xu & Li 2020 (zh concreteness, sheet pinned by name) | CROSS-CHECK ONLY (9/14; never back-fills) | d329b49de1ebbc5d |
| Sulpizio et al. 2024 taboo (OSF ecr32) | supplement, unstaged | — |

## Registered predictions (directions fixed pre-join; [1147])

- P1 arousal (primary): fallers > risers at displacing sites, flat at
  controls; both languages, z within language; falsifier stated.
  Quotable only on BOTH nulls (full permutation + restricted,
  [1152].4).
- P2 concreteness (secondary): risers more abstract at displacing
  sites; weak prior declared.
- P3 dominance (en-only, declared): faller-to-riser dominance drop
  larger at female-subject anger sites; >= 6 sites/arm or UNDERPOWERED.
- P4 controls (gate): flat controls; log-frequency delta beside every
  norm delta; any survival sentence NAMES THE POPULATION IT SURVIVED ON.

## Floors and gates (all set blind)

mass_covered >= 0.60 both roles; Percent_known >= 0.85 (Brysbaert);
P3 >= 6 sites/arm; Log_Freq_W headline with C1/C2 sensitivity pair;
0.02-0.10 site-conditioning gap unassigned and counted.

FUNCTION-WORD EXCLUSION (RH's word, [1196]): closed-class words (NLTK
en stoplist, 198 entries) are excluded from the scored set in every
role — any database rating they carry is INVALID ("have"/"be" have no
valence or arousal; their frequency and mass would confound). Floor
and denominator UNCHANGED ([1170] intact); coverage printed
before/after; zh closed-class list to be declared before any zh
scoring resumes. For the LLM-norms candidate the rule doubles as a
validity diagnostic of the source.

## Meta routing

P1, if it holds both nulls and the frequency control: candidate NEW
M01 clause (working slug `arousal-descent`) — the first clause on an
instrument external to the project. P3: routes to the
gendered-displacement work, not M01. Nulls and failures are findings
and file here either way.
