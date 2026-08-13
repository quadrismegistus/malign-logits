---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-13
role: plan
topics: [affect-bridge, semantic-norms]
description: "Plan C: the affect bridge — does word-level de-extremification and de-concretization surface in the lexicon of generated prose? Directions inherited from the C/E/K word-level findings, drafted by the registrar, flagged for RH's countersign. Human norms only (Brysbaert, Warriner); no coder."
---
# Plan C — the affect bridge: does the word-level signature surface in prose?

Drafted 2026-08-13 by the registrar on RH's word ("can we start pilots for
the other plans"). DIRECTIONS ARE INHERITED, NOT NEW: they restate the
campaign's word-level findings at passage grain, and RH may amend or strike
them before any verdict is read. Prior exposure: none — no norm has been
joined to any passage of this corpus by anyone.

## Hypotheses (inherited; the inheritance is the point)

- **C.H1 — aligned prose is less emotionally extreme.**
  `valence_extremity_warriner_mean` (passage mean of |valence − 5| over
  content words found in Warriner) is LOWER in the aligned arm.
  Inherited from Registration C's corpus-level de-extremification
  (+0.025, p 0.0012) and E's lineage replication (19/25).
- **C.H2 — aligned prose is less concrete.**
  `concreteness_brysbaert_mean` (passage mean over content words found in
  Brysbaert) is LOWER in the aligned arm. Inherited from K (concreteness
  falls on both instruments, z −17.1/−18.8) and the M05 de-concretization
  arc.

Secondary, no direction: `arousal_warriner_mean`, `dominance_warriner_mean`
(dominance is the campaign's twice-dead scale and rides as the unplanned
negative control it has become).

## Why this is a BRIDGE and not a repetition

C/E/K measured which words gain and lose probability at next-token
positions. Plan C measures the words that actually ended up on the page
across 185 words of free continuation. The bridge can fail: Q taught this
campaign that frames reverse between zooms. A null here against the
word-level results is a finding about WHERE de-extremification lives, not
a replication failure — declared now, four-cell style, per the house form.

## Instrument

HUMAN NORMS ONLY, no coder: Brysbaert concreteness (Conc.M) and Warriner
V/A/D (x.Mean.Sum), joined on the LEMMA of content words (NOUN, VERB, ADJ,
ADV by UPOS) from the shared Stanza parses already in the stash — plan C
parses nothing. Norm files are the hub's `norms_sources/` set, pinned in
`meta/norms_digests.md` (Warriner `85f6d7e3`, receipt at 468b0855).
Coverage shares (`warriner_coverage`, `brysbaert_coverage`) reported per
passage; a passage below 50% coverage on an instrument is excluded from
that instrument's contrast with per-arm exclusion rates reported
(arm-behaviour clause: coverage itself may differ by arm — e.g. proper
nouns are not in the norms, and NNP runs 7/1000w lower aligned, so
coverage is EXPECTED to differ; the rate travels as description).

## Amendment 1 (2026-08-13, RH's word: include the K coder scales)

The seven K coder scales join the battery as passage means over content
words — `k_vulgarity_mean`, `k_register_level_mean`,
`k_transgressiveness_mean`, `k_charge_mean`, `k_valence_mean`,
`k_bodily_harm_mean`, `k_concreteness_mean` — joined on SURFACE with
lemma fallback (K rated emitted forms), `k_coverage` reported.
NO DIRECTIONS REGISTERED for any of them. The `fields.py` riders travel
verbatim: these are ONE MODEL's judgments at ONE frozen instrument
(never presented beside the human norms as the same kind of object — the
k_ prefix is load-bearing); `register_level` is NOT ESTABLISHED
(descriptor only, never evidence); `vulgarity` is a sparse indicator
(variance on 463 of 27,242 words — floors are not nulls); RANKS NOT
LEVELS (levels shift between instrument versions at stable order —
paired arm contrasts are order-like and permitted, absolute thresholds
are not). And `k_concreteness_mean` beside `concreteness_brysbaert_mean`
is the built-in convergence check at passage grain (word-level r 0.88).

## Unit, strata, tests

Identical to plan A as amended: undisturbed arm primary; (pair, prompt)
cell; prose AND non-degenerate AND English stratum primary; pair medians;
sign test over pairs; Wilcoxon beside; per-pair denominators; pooled reads
beside the stratified ones.

## Pilot before corpus

On the existing pilot population (one passage per cell) from cached
parses, directions eyeballed, never quoted as results. Full run reads the
same population as plans A/B.
