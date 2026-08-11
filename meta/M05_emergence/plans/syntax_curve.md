# Plan: the syntax curve (registered secondary 5), tiers 1-2

Written 2026-08-11 by the registrar seat, RH's word to run given the same
evening. Discharges the "frozen licit-category artifact" secondary 5 was
blocked on. Tier 3 (the natural/deviant/ungrammatical selection judgment)
is designed in the docket conversation but NOT part of this plan; it gets
its own plan if RH calls it.

## Objects, all frozen before the curve is drawn

1. TAGS (tier 1, done, commit 9ca35d6e): data/m05_syntax_tags.parquet --
   all 338,092 unique (prompt, word) pairs on the 584-text battery across
   BOTH ladders, spaCy en_core_web_sm in context; `pos_class` derives the
   coarse class from the PTB fine tag except VB*/MD, which defer to UPOS
   (AUX/VERB is contextual). Known limits: 1.65% multi-token splits
   (flagged, kept); residual tagger error unmeasured beyond the witness
   check below.
2. LICIT SETS (tier 2, this run): one call per prompt on
   code_m05_licit_v1 (deepseek/deepseek-v4-flash pinned, temp 0, witness
   discipline). Artifact data/m05_licit_sets.json with task instrument
   sha, model of record, spaCy version, battery sha. Coded ONCE; the
   cache is the freeze.

## Scoring rules, declared

- Three bands: LICIT / ILLICIT / FORMAT (PUNCT, X, SYM never count
  against grammar -- the cloze tokens are format).
- Convention equivalences at the join, both directions: ADP=PART,
  NUM=NOUN, AUX=VERB. DET/ADJ deliberately NOT merged.
- Two variants: STRICT (licit only) and PERMISSIVE (licit + marginal).
  Both reported wherever either is.
- Mass share of RESOLVED mass per (checkpoint, prompt); payload_empty
  censored; median over prompts, bootstrap CI; the m05_onsets persistent
  criterion; ladders never pooled.

## Gates before any number is quotable

- Witness/tagger agreement on the full 584 (smoke: 88-93%; the 8-prompt
  re-pin run showed ~3-point provider-side wobble at temp 0, which the
  freeze removes going forward). Disagreements listed in the artifact.
- Stability probe: 30 seeded prompts re-coded on a second family
  (anthropic/claude-haiku-4-5); per-prompt Jaccard on strict licit sets
  reported. No threshold declared in advance -- the number is reported
  and RH reads it; deepseek-v4-pro is the tiebreak if the families split.
- Priors, registered as expectations not gates: mature endpoints mostly
  licit; step0 shards mostly illicit; the format band swells at the OLMo
  base endpoint (the quiz register).

## Declared limits

Class-grain licitness cannot see within-class violations ("a apples");
that is tier 3's object. Coder under-licensing of content classes (the
smoke's missing PROPN after "her") biases the illicit share upward
uniformly; the probe sizes it. The smoke's 8 prompts and their cached
codings are superseded by this frozen run.
