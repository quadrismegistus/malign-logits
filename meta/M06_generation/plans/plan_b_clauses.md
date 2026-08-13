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

## Amendment 1 (pre-gate, 2026-08-12, on [5632] and [5634] §§2-3)

**Unit: plan A's Amendment 2 binds here identically** — primary population
is the UNDISTURBED arm; forced arms are the secondary per-arm table; arm
count per cell reported.

**§1 — B.H1 and B.H2 are coupled** ([5634] §2): two readings of one clause
distribution over one sentence denominator. Not strictly complementary
(both ratios can fall if sentences shorten) but not two measurements
either. Decision rule, fixed now:

- Reported beside the per-sentence ratios, denominator-free:
  `dep_clause_share` = dependent / (dependent + independent) clauses (the
  clause MIX, no sentence denominator), and per-1,000-word rates
  `indep_clauses_per_1000w`, `dep_clauses_per_1000w`.
- **B.H1 and B.H2 may be reported as TWO findings only if the clause-mix
  shift and at least one per-1,000-word rate agree with the respective
  per-sentence verdicts.** Otherwise the verdict sentence is "one clause
  distribution, two surfaces," and CONFIRMED does not appear twice.

**§2 — the declared joint prediction with A.H1** ([5634] §3): subordination
lengthens sentences — hypotaxis and sentence length move together in the
register literature — so A.H1 (aligned shorter sentences) and B.H2 (aligned
more hypotactic) jointly require, if both fire, that aligned CLAUSES are
markedly SHORTER. `clause_len_words_mean` (words per clause) is the
adjudicating column, reported per arm. The outcome table, declared before
any number exists:

- **Both fire** → the strong, strange result: compression WITH
  subordination — aligned prose packs more subordinate structure into
  shorter sentences via shorter clauses. Reported as one joint signature
  with the clause-length column carrying the mechanism.
- **A.H1 fires, B.H2 fails** (or reverses) → sentence shortening without a
  subordination shift; B.H1's paratactic read adjudicates whether the mix
  moved at all.
- **B.H2 fires, A.H1 fails** → the register-literature pattern
  (subordination with length); A.H2's conditioned table is re-read against
  it.
- **Both fail** → both reported; the surface-accounting descriptives stand
  alone.

No selective reporting: all four cells of this table are quotable ONLY
together.

**§3 — Stanza is a stated choice, and the fourth POS path in the repo**
([5634] custody note): `taxonomy.py` carries in-context spaCy POS,
`fields._byu()` an out-of-context tagger (unusable except verbs/adverbs —
noun label 41.2% verbs in context, [5632]), and lacan's
`results/k/pos_context_en.tsv` the corrected M01 table (cleared at [5660]/d154b5f6:
re-filled with the 16,237 dedup-promoted pairs after the [5657] defect;
corrected-vs-contaminated Spearman 0.985). Plan B uses Stanza
BECAUSE UD clause relations (`ccomp`/`advcl`/`acl`) are the published OSP
operationalisation and spaCy's `pos_` cannot supply them. Stanza output is
used ONLY within M06; no cross-instrument POS comparison travels without
its own bridge check.

**The instrument is IMPORTED, not ported** (RH's call at adapter build):
`meta/M06_generation/scripts/m06_style.py` imports the extractors from the
OSP repository itself (`$M06_OSP_PATH`, commit `56f2562`, remote
github.com/quadrismegistus/ordinary-style-philosophy) so the code is
byte-identical to the published instrument. Pipeline runs WITHOUT the
constituency and NER processors — the published classifier used
pos/deprel/sent only (`CV_FEAT_TYPES`), and the sent features run on the
dependency route (`get_clauses_v2`). Parses are cached in the repo stash
(`m06_stanza_docs`, keyed by parser string AND text — a stanza upgrade is
a different parse). Two construct facts inherited from OSP and binding
B.H1/B.H2's wording: the UD relation named `parataxis` counts as a
SUBORDINATE-clause introducer in this extractor, and UD coordination
(`conj`) folds into the head clause — so "independent clauses" here means
OSP-main clauses, and B.H1 is operationalised as that, not as a direct
count of grammatically coordinate clauses; and IC is floored at 1 per
sentence.

## Amendment 2 (pre-verdict, 2026-08-13, RH's refinement question + the pilot feature diff)

Close reading showed the per-sentence ratios' EXTREMES mark run-on
narration (subordinator chains inside unpunctuated flow) rather than
periodic embedding — high-DC exemplars from this corpus are breathless
skaz, not Moore. And the pilot's per-1,000-word battery shows WHY the
per-sentence ratio reads flat while something real moves underneath:
aligned prose is HIGHER on `deprel_acl`, `deprel_advcl`, `deprel_xcomp`,
`pos_VBG`, `pos_VBN`, `pos_TO` (28/39, 27/39, 26/39 pairs and kin) —
non-finite, participial subordination — while sentences shorten and
punctuation rises. The candidate picture: alignment converts finite
clausal CHAINING into non-finite participial EMBEDDING — compressed
subordination — which a clauses-per-sentence ratio cannot see because
the sentence denominator shrinks in step.

Refined operationalisations, declared before any full-run verdict,
computable from cached parses (shards untouched):

- `hypotaxis_dep_clauses_per_1000w` (already emitted) PROMOTES to
  co-primary beside the per-sentence ratio: length-robust, run-on-robust.
- NEW, named secondaries: `hypotaxis_finite_per_1000w` (clause heads
  whose subtree carries an explicit subordinator `mark` or a finite verb)
  vs `hypotaxis_nonfinite_per_1000w` (xcomp, acl with VBG/VBN head,
  to-infinitives) — the split the pilot diff motivates; direction
  registered for NEITHER (the finding, if any, is the split itself).
- `dep_clause_word_share` (DCw/(DCw+ICw), words under subordination) —
  denominator-free mass form.
- Per-sentence ratio EXTREMES are demoted to description; tail exemplars
  are never quoted as periodic style (close-reading note, 2026-08-13).

B.H1/B.H2 stay registered as written and are reported as registered;
verdict language already requires denominator-free agreement (Amendment
1 §1), and the refined battery are the denominator-free forms.

## What this plan does not claim

Nothing named "register" or "formality" — if a composite is ever built from
this battery it gets its operationalisation in its name and its own plan.
No causal story about WHERE in the ladder the style arrives (that is a
U/Z-style question for a later plan on SFT/DPO rungs, if the corpus grows
them). No claim about P's unnamed axis — that is phase two, with a declared
holdout, per the module README.
