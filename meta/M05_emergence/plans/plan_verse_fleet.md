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

## The instrument battery (Design B: expand() at every declared slot;
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
- DESIGN B ADOPTED (2026-08-13, [5735] proposed / [5736] lacan firmly /
  [5737] malign concurs): TWP BACKBONE — the frozen `expand()` at EVERY
  declared slot (full resolved word distribution, theta=0.001), with the
  rime-class read moved to ANALYSIS TIME (class mass = sum of stored
  surface mass over rime-key membership; the store answers questions
  nobody has yet asked — lacan [5736] §3). The closure probe rides as a
  small batched forward per slot (actual next word + top-8 class members
  by expand mass). This SUPERSEDES the [5721] §4 candidate-set design:
  priced apart ([5737] §2), the instrument change costs $0.21; A was
  "not the cheaper design, it is NO design" for the discovery-mode
  instruments (P/interiority/norms) that ride this fleet.
- THETA DECLARATION CARRIED (lacan [5736] §4): expand's theta=0.001
  Zipf-censors rare words. For mass-weighted composition means this is
  nearly harmless BY CONSTRUCTION (tails carry little mass), but nearly
  harmless is a MEASUREMENT, not an assumption — every cell stores
  expand's residual dict (tail/drop/open/mojibake/total) and every
  downstream read quotes its cell's censored share beside the number.
- CACHING REQUIREMENT WITHDRAWN ([5737] §3b, malign's own correction):
  `twp.py` never cached — `next_dist` runs a full forward per call; the
  cost anchors WERE the uncached price ($13-18 band stands). The [5727]
  "$110 if caching is missed" was a double-count and is dead. Caching is
  demoted to "if someone is in there anyway"; nobody touches the frozen
  instrument (RULE_VERSION 3, 301,147 stored cells) for five dollars.
  Rider caveat ([5737] §6): `_BATCH` is module-level OOM-backoff state
  shared between expand and the rider — one arm's OOM halves the other's
  batch for the rest of the run; wall-clock, not correctness.
- SLOT ARITHMETIC, DESIGN B: NINE declared slots per poem (end1-3 with
  the partner line as end_partner_prior/class-PRIOR, mid1-4, near,
  called — the producer's `poem_slots`), 180 poems = 1,620 verse slots;
  PLUS the 102 LITERARY prompts (prompt_categorisation.json
  source=LITERARY, found human fiction mid-passage) as the PRIMARY
  genre-licence baseline — found prose against found verse, matched
  provenance; PLUS ~100 M05-battery slots retained ONLY as the
  census-calibration anchor (expand-era numbers exist at shared rungs =
  the instrument-drift alarm), NEVER as the licence comparison. Total
  ~1,822 slots per rung; malign's price at that count: ~$18.50 total,
  both ladders ([5737] §2). The within-poem uncalled slots are not
  precision padding — the paired called-minus-uncalled read IS the null
  (the R-decoy lesson: no matched control word-sets), and the mid/end
  spread is the time-course instrument.
- RIME-KEY v2 (audit, 2026-08-13): v1 keyed on syllable SPELLING and
  shattered /eɪ/ into ay/ey/eigh (72% singleton keys). v2 keys on rime
  PHONEMES of the final stressed syllable from its first vowel (glide
  leak stripped, espeak/lexicon schwa variants normalized); RH's
  hand-check pairs all cohere (day/way, obey/weigh, door/snore,
  hold/told, bark/aardvark; door/bark splits). Declared residual: OOV
  words take espeak syllabifications that can split them from lexicon
  neighbours (blisses/kisses) — conservative false-splits that shrink
  classes, never merge non-rhymes. Under Design B the vocab table is an
  ANALYSIS-TIME artifact (fleet stores surfaces, not classes), so key
  fixes never invalidate collected cells.
- Provisioning budgeted in wall-clock and babysitting, not dollars
  (the L2 fleet's 6-of-14 casualty rate is the standing tax).

## MANIFEST FROZEN (2026-08-13, post-[5740])

All gates passed: Design B unanimous ([5736]/[5737]), rime-key v2
phonemic with the key-resolution check run per era ([5740]: pre-1900
0.917 / 1900+ 0.850, flag-not-filter column carried), apostrophe
last-word artifact fixed producer-side, 7B closure check PASSED
(OLMo-2-7B: close|class 0.945-0.973 at called slots vs 0.001-0.023
mid-line; called-slot class pull 0.22-0.98 with capacity-vs-recall
separation visible). Roster v2 (180 poems, seed 20260813), nine slots
per poem, LITERARY + battery prose baselines, expand() + closure rider
per slot, residual dict stored per cell. Changes past this line are
docket matters.

## DECLARED ANALYSIS CONSTRAINT — depth confound ([5751]/[5752], pre-data)

RH spotted the symptom in the slot prompts; malign named it: slot position
is confounded with context depth (mid1/end1 sit in line 1; called carries
~9x mid1's context; a next-word distribution sharpens with context
regardless of lineation). Booked BEFORE any fleet data exists:

- PRIMARY: called vs {mid4, near} — depth-matched, 4 lines each.
- VALID: mid_k vs end_k at matched k (within-depth contrasts).
- NOT VALID raw: any across-depth gradient read as locality; any null
  pool mixing mid1-3 with called.
- RIDER ([5752]): the across-depth locality read survives ONLY as a
  DIFFERENCE-IN-GRADIENTS — (rhymed mid1→mid4) minus (unrhymed
  mid1→mid4) — the unrhymed arm carries the same nine slots at the same
  depths with a target_key and no scheme calling it, so its gradient IS
  the context-sharpening baseline on the same instrument.
- Every quoted contrast states the depth of both arms; context_len
  travels into every read.

## Costing: SETTLED ([5737])

A-at-922 $13.07 / B-at-922 $13.28 / A-at-1,822 $18.10 / B-at-1,822
$18.50: the instrument choice costs $0.21, the slot expansion $5.03,
independent decisions. Design B at the nine-slot manifest = ~$18.50.
Fleet spend takes RH's word with the number, per cool-off.
