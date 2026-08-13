---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-13
role: plan
topics: [fleet, capacity, rhyme, verse]
description: "Plan: the verse-capacity fleet, TWP-ONLY first pass (RH's ruling) — rhyme pull/floor and eight sibling slot-capacities across both ladders plus alignment arms; generation DEFERRED to a separate decision after the twp curves exist, with the decision criteria stated now. Costing requested from malign before any launch; fleet spend takes RH's word with the number."
---
# Plan: the verse fleet — twp first, generation decided later

RH's ruling (2026-08-13, in session): the fleet runs TWP ONLY as its
first pass; whether to generate at all — and when, and at which
checkpoints — is decided AFTER the curves exist. Sources: the drafting
side's fleet memo (notes/rhyme-fleet-capacities-2026-08-13.md, incl. its
same-day twp-first revision), the rhyme_pull pilot (closure x
rhyme-given-closure decomposition, template classes, per-surface
summing), and the Weatherby audit that licenses the framing
(reading/weatherby-priority-claims.md: the checkpoint battery TRANSLATES
a synchronic thesis into developmental terms; it does not test a claim
the book makes, and the write-up says so).

## Populations

Both ladders, never pooled: Pythia (155 rungs, log-spaced early — the
only place sub-1000-step onsets are visible) and OLMo (95 rungs,
INCLUDING the SFT/DPO/RLVR arms — re-binding is testable in the same
pass; Pythia's lomahony endpoints ride as its alignment arm).
Cross-ladder sentences on the token clock with the absent-rate column
([5434]/[5436]). OLMo `step` is not a key (stages restart numbering).
PolyPythias (arXiv:2503.09543) is the declared noise floor for any
Pythia onset sentence.

## The instrument battery (all candidate-set-mass at declared slots;
one expand() per (rung, prompt); capture folds ride free — word probs,
logit sidecar, final-position hidden states)

Per the memo's twp-only menu plus two additions from the registrar's
design review, RH-seen:

1. RHYME PULL/FLOOR, period-stratified (core). Called slots (scheme
   partner in window) vs the WITHIN-POEM uncalled-slot null (same
   rhyme-set's mass at uncalled line-ends and mid-line slots) — never
   matched control word-sets (the R decoy lesson). Corpus split does
   pull-vs-floor: rhymed poems = capacity; free-verse = compulsion.
   Closure decomposition from the pilot (line_closure x
   rhyme_given_closure) at every slot.
2. METER-FIT MASS (stress-licit candidates, called/uncalled control).
3. ALLITERATION/ASSONANCE PULL — the local-vs-crossline ordering test.
   PRE-STATED prediction (memo): alliterative pull onsets BEFORE rhyme
   pull; rhyme-expectation is formal discourse-tracking (hold line 2's
   ending ~15 tokens) and sits late with world-tracking.
4. HOLD-THE-POET'S-WORD — the L design on verse, period-stratified
   (Pope early, Prufrock late): modernism-arrives-late in its cheapest
   form. Archaic diction folds in.
5. LINEATION — newline mass at line-ends vs mid-line; enjambment =
   newline mass gated on syntactic completeness; per-model tokenizer
   caveat.
6. INTERIORITY AXIS ACROSS RUNGS — mass-weighted top-k position on P's
   fixed axis per checkpoint: pretraining drift vs SFT manufacture of
   enacted->represented. (Fence: the axis's own stability gate still
   fails; this is a descriptive drift curve, not a named-axis verdict.)
7. GENRE-CONDITIONAL NORMS ("poetic licence") — the K-scale mix at
   verse slots minus matched prose-battery slots, per rung and into the
   alignment arms: when does the model learn verse may say what prose
   may not, and does alignment's de-transgressification respect the
   licence or flatten it? Zero marginal cost (k_ratings join).
8. COPY-PULL vs RHYME-PULL — p(actual/same word) vs p(rime class minus
   actual) as separate curves: rhyme is repetition-with-difference, and
   the gap between the curves is the acquisition of the difference.
   Free column from instrument 1.

## Discipline (the run is born under this week's clauses)

Producer fingerprints in every resume/skip predicate (writes clause);
FINAL + analysis-key GROUP BY on any RMT read (reads clause);
completeness reconciled against the SOURCE and the declared prompt
roster, never the store ([5710]); per-pair/per-rung denominators;
decoder irrelevant (no sampling in twp) but encode/BOS policy pinned;
the primer/slot roster FROZEN as a data file before launch — a slot
battery added after the fleet closes is a second fleet.

## GENERATION: deferred, with the decision criteria stated now

No generation in this pass. The decision to generate is taken AFTER the
twp curves, on: (a) does the pull show the relaxation arc (learned,
seen through, re-bound) that makes trajectory questions worth buying;
(b) which 8-12 rungs the curves identify as pivots (onset, peak,
mature-base relaxation, SFT boundary, re-binding); (c) whether M06's
checkpoint-time style questions (de-diversification onset, format-
attractor precursor, compressed-subordination trajectory) ride the same
pivot generations — one pass, two consumers — at pinned decoding. That
decision is its own costed proposal to RH; nothing here pre-authorises
it.

## Roster and design amendments (2026-08-13, post-costing)

- THE POEM ROSTER IS FROZEN: `data/rhyme_fleet_roster.json` — 180 poems,
  30 per (scheme x era) cell, uniform-random within sorted cells at seed
  20260813 from the availability scan's usable set; era from the full
  Chadwyck metadata (RH's pointer, 100% coverage; per-cell availability
  recorded in the file: AABB 1,362/68, ABAB 1,560/121, unrhymed
  727/1,399 — literary history's class-era confound is BALANCED by
  design and the marginals declared). Slot expansion (called slot +
  uncalled nulls per poem) is the producer's job against this roster.
- DESIGN SWITCHED per [5721] §4: CANDIDATE-SET SCORING — next_dist
  driven by the rime class's OWN prefixes; O(class), THETA-FREE, no
  discovery floor. Malign's rider is plan text: where any instrument
  still carries a theta, THETA IS PART OF THE ROSTER — changing it
  after close is a re-run of every cell, not a tweak.
- KV-CACHING IS A REQUIREMENT, NOT AN OPTIMISATION ([5727] §2, malign's
  own correction of the adopted design's cost claim): naive candidate-set
  scoring costs what expand() cost (1.01x measured); the saving lives
  entirely in caching the primer so continuation tokens are incremental
  (approaches the 0.05x one-forward floor). The fleet producer caches or
  the redesign buys theta-safety at no cost saving.
- SLOT ARITHMETIC, DECLARED for costing ([5727] §3): per roster poem,
  4 slots — 1 CALLED (last line's final word) + 3 UNCALLED NULLS (two
  mid-line positions, one uncalled line-end; exact positions computed by
  the producer against the poem's own lines, rule declared in the slot
  manifest it emits) = 720 verse slots; plus ~300 prose-battery
  comparison slots (genre-conditional norms) = ~1,020 slots per rung.
  Closure probes ride each slot as one cached batched forward.
- Provisioning budgeted in wall-clock and babysitting, not dollars
  (the L2 fleet's 6-of-14 casualty rate is the standing tax).

## Costing ask

@malign to cost: both ladders x the frozen slot roster (target ~1,500-
2,500 slots including verse primers, prose-battery comparison slots,
and uncalled-null positions), expand() + closure probes per slot,
capture folds on. Fleet spend takes RH's word with the number, per
cool-off.
