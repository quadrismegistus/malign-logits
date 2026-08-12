---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-12
role: plan
topics: [clause-architecture]
description: "Plan B: parataxis and hypotaxis on the OSP pipeline (Stanza UD, clause boundaries from subordinating relations). RH's hypotheses, fixed before any producer runs: B.H1 base models more paratactic; B.H2 aligned models more hypotactic. The 94-feature OSP battery rides descriptively; the OSP disciplinary map is chartered exploratory."
---
# Plan B — clause architecture: parataxis and hypotaxis

Drafted 2026-08-12 by the registrar on RH's word, hypotheses RH's verbatim,
directions fixed before any parse exists on this corpus (prior exposure as
plan A's: none — no clause statistic has been computed on any passage of
this corpus by anyone).

## Hypotheses (RH, 2026-08-12, in-session)

- **B.H1 — base models are more PARATACTIC.**
  `parataxis_indep_clauses_per_sent` is HIGHER in the base arm, paired
  within (pair, prompt).
- **B.H2 — aligned models are more HYPOTACTIC.**
  `hypotaxis_dep_clauses_per_sent` is HIGHER in the aligned arm, same
  pairing.

Both directional, one alpha each. They are NOT one hypothesis twice: the two
ratios have different numerators over the same denominator and can move
together (a model can add both clause types), so confirming one does not
imply the other and the joint pattern is reported whatever it is.

## Instrument (the OSP pipeline, verbatim)

Stanza (Universal Dependencies), exactly as in Ettel & Heuser, *Ordinary
Style Philosophy* §3.1-3.3, which is the published operationalisation these
hypotheses inherit: clause boundaries detected from subordinating relations
(`ccomp`, `advcl`, `acl`); per sentence, the number of independent clauses
and the number of dependent clauses. Continuity with OSP is deliberate — it
makes M06's numbers directly comparable to OSP's disciplinary baselines,
and it means the instrument was built and validated on a corpus that could
not have been chosen to flatter these hypotheses.

Primary measures (naming rule):

- `parataxis_indep_clauses_per_sent` — independent clauses / sentence,
  passage mean.
- `hypotaxis_dep_clauses_per_sent` — dependent clauses / sentence, passage
  mean.

Secondary, reported beside the primaries with NO registered direction:

- `clause_depth_max` — maximum embedding depth (OSP's "levels of
  subordination a sentence reaches").
- `modal_density_md_per_1000w` — modal verbs per 1,000 words (POS `MD`) —
  the hedging candidate.

Exploratory, disclosed as such: the full OSP battery (47 dependency
relations + 39 POS + 8 clause statistics, per-1,000-words, z-scored) rides
the same parse at no extra cost. **The OSP map** — base and aligned prose
placed in the OSP corpus's 94-feature z-space against 125 years of
disciplinary prose, asking where aligned prose lands relative to the
analytic-philosophy cluster — is a FIGURE, not a test; no verdict language
may attach to it (chartered in the module README).

## Gate, unit, exclusions

Plan B runs on plan A's shared Stanza output and waits on plan A's gate
(segmentation check on the typographic offenders; length audit; empty-text
exclusions with per-pair denominators, `bloom-7b1` named). The clause
ratios are per-sentence and therefore robust to passage-length differences
by construction; the length audit is still read first.

Unit and test as plan A's: (pair, prompt) cell, aligned-minus-base; pair
median with per-prompt sign split; sign test over 41 pair medians, Wilcoxon
beside. English only; zh clause parsing is its own gate in phase two.

## What this plan does not claim

Nothing named "register" or "formality" — if a composite is ever built from
this battery it gets its operationalisation in its name and its own plan.
No causal story about WHERE in the ladder the style arrives (that is a
U/Z-style question for a later plan on SFT/DPO rungs, if the corpus grows
them). No claim about P's unnamed axis — that is phase two, with a declared
holdout, per the module README.
