---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-12
role: plan
topics: [surface-accounting]
description: "Plan A: length, sentence count and windowed TTR on the passage corpus. RH's hypotheses, fixed before any producer runs: A.H1 alignment makes shorter sentences; A.H2 alignment makes higher TTR. Carries the shared Stanza instrument gate both plans wait on."
---
# Plan A — surface accounting: length, sentences, lexical diversity

Drafted 2026-08-12 by the registrar on RH's word, hypotheses RH's verbatim,
directions fixed before any measure has been computed on this corpus by
anyone (prior exposure: the only numbers ever read from the passage TEXT are
[5629]'s row/emptiness counts; no style measure exists at any seat).

## Hypotheses (RH, 2026-08-12, in-session)

- **A.H1 — alignment makes SHORTER SENTENCES.** `sent_len_words_mean` is
  LOWER in the aligned arm than the base arm, paired within (pair, prompt).
- **A.H2 — alignment makes HIGHER TTR.** `ttr_mattr_w100` is HIGHER in the
  aligned arm than the base arm, same pairing.

Both are directional. A.H1 and A.H2 are one alpha each, tested separately;
neither conditions on the other.

## Measures (naming rule applies; per passage, then aggregated per
(pair, prompt, arm))

- `len_words`, `len_chars`, `n_sents` — descriptive, NO registered direction.
  Length is reported as finding-grade description and as the nuisance column
  every later M06 measure conditions on. (No direction because continuation
  length is partly a property of the generation settings in M04's spec; a
  verbosity claim would need those settings read first. If a direction turns
  out to be strong, it is reported as unregistered description.)
- `sent_len_words_mean` = `len_words` / `n_sents` per passage — A.H1's
  measure.
- `ttr_mattr_w100` — moving-average type-token ratio, window 100 words;
  secondary `ttr_mattr_w50` (passages run ~200 words median, so w100 gives
  few windows; w50 is the robustness read). RAW TTR IS NEVER REPORTED AS A
  CONTRAST — it is mechanically length-dependent, and if the arms differ in
  length a raw-TTR difference is an artifact. Raw TTR may appear only in the
  descriptive table beside `len_words`.

## Instrument, and the gate both plans wait on

One Stanza run (English models, tokenise + segment + tag + parse — the
pipeline of Ettel & Heuser, *Ordinary Style Philosophy* §3.1), shared with
plan B: A reads tokens and sentence boundaries, B reads the parse. GATE,
before either plan's producer runs on the corpus:

1. **Segmentation check.** ~50 sampled passages read by a human (RH or a
   seat) against Stanza's sentence splits, the sample DELIBERATELY including
   the worst typographic offenders (curly/straight quote mixtures — the
   format-attractor typography is known to stress segmenters) and at least
   two `bloom-7b1` passages. Pass = no systematic split failure class; any
   failure class found is named and either fixed or carried as a stated
   limit.
2. **Length audit.** `len_words` distributions per arm, read BEFORE
   verdicts: if the arms differ substantially in length, that fact is
   reported first and A.H2's windowed design is confirmed against it.
3. **Exclusions.** Empty texts excluded with per-pair denominators named
   (`bloom-7b1` 155/28,832; full table in the producer output). Unscorable
   sequences are NOT excluded — scoring gaps are orthogonal to text
   ([5629]) and this plan never touches teacher-forcing.

## Unit and test

The passage is the measurement grain; the **(pair, prompt)** cell is the
inferential unit (passages within a prompt share it); the **pair** is the
top level (41 at drafting; final roster at M04 reconciliation). Per cell:
aligned mean minus base mean. Per pair: the median over its prompts, with
the per-prompt sign split reported (prompt-unit doctrine — no pooled number
without per-prompt existence, [5582]). Across pairs: sign test over the
pair-level medians, direction as hypothesised; Wilcoxon beside it as the
magnitude read. Per-pair denominators named wherever a rate travels
([5587]/[5629]).

## Pilot before corpus

One passage per (pair, prompt, arm) first (~2 x 41 x prompt-roster rows);
directions eyeballed, instrument timed, THEN the full corpus. Pilot numbers
are never quoted as results.

## Amendment 1 (pre-gate, 2026-08-12, on lacan's [5632] objection)

**A.H1 and A.H2 are coupled through the window design, and the plan states
it before the numbers exist.** A fixed-token window over shorter-sentence
text spans more sentence boundaries; sentence-initial vocabulary is a small
recurring set; so `ttr_mattr_w100` can move as a CONSEQUENCE of sentence
length, not beside it. Windowing fixed raw TTR's length dependence at the
token level and left the same dependence alive one level up — the exact
form lacan names ([5632]): fixing a quantity at one level of aggregation
does not fix it at the level above.

Decision rule, fixed now ([5632]'s third option, plus the conditioned
table):

1. `sents_per_window_w100` is computed and reported per passage beside the
   TTR columns.
2. The A.H2 contrast is additionally reported WITHIN TERTILES of
   sentences-per-window. **A.H2 is quotable as a finding independent of
   A.H1 only if the aligned-minus-base TTR contrast survives inside those
   strata** (direction held, sign split reported per stratum). If it does
   not survive conditioning, the verdict sentence is "one finding, two
   surfaces" — A.H2 CONFIRMED language may not appear beside A.H1
   CONFIRMED as two findings.
3. Expected sign of the coupling, stated as expectation and checked in the
   table, not assumed: boundary-vocabulary recurrence should push TTR DOWN
   as sentences shorten, i.e. the coupling runs AGAINST A.H2's registered
   direction — so an A.H2 that fires AND survives conditioning is
   strengthened by the coupling, and an A.H2 null is ambiguous between
   absent and masked. Either way the conditioned table is the read of
   record.

Sentence-based windows (lacan's first option) were considered and declined:
a window of N sentences lets token count vary within the window, which
reintroduces raw TTR's token-count dependence inside each window — the
original artifact one level down. The conditioned table keeps the token
denominator fixed and makes the sentence effect visible instead of moving
it.

## Amendment 2 (pre-gate, 2026-08-12, on malign's [5634] §1)

**The cell was pooling across arms, and arm count varies by cell** (85.6% of
cells carry 5 arms, 14.4% fewer — measured on the frozen table, [5634]), so
the pooled cell mean was composition-dependent in exactly the shape the
campaign has been burned by ([5592] §2's pooled null, run the other way: a
composition difference read as a style difference).

Ruling, taking malign's first and third options together:

1. **The PRIMARY population is the UNDISTURBED ARM ONLY** — the one arm
   present in every cell, with no injected word, and the only arm whose
   prose is not conditioned on a word we chose. Every A and B verdict is
   read there. This costs ~80% of the passages and is the right trade: the
   effective n was always pairs and prompts, not passages.
2. The forced arms become a SECONDARY per-arm replication table — the same
   contrasts with `arm` in the unit, reported beside the primary, no
   verdict language of their own. Agreement across arms strengthens;
   disagreement is reported as a scope fact about injection.
3. Arm count per cell is reported in the producer output regardless, so
   the composition is visible rather than behind the table.

This amendment binds plan B identically (shared unit).

## Amendment 3 (pre-gate, 2026-08-12, cross-reference)

A.H1 participates in a DECLARED JOINT PREDICTION with plan B's B.H2 —
subordination lengthens sentences, so the two registered directions pull
opposite ways on sentence length. The joint outcome table and its
adjudicating column (`clause_len_words_mean`) live in plan B, Amendment 1
§3; A.H1's verdict is always read beside it, and no selective reporting of
whichever hypothesis landed is licensed.

## What this plan does not claim

No register/formality construct (that is plan B territory and the naming
rule guards it); no verbosity direction (see above); no zh (separate gate,
phase two); nothing about WHY — mechanism talk waits for C and D.
