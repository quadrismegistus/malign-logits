---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-13
role: plan
topics: [capacity, rhyme, stuckness]
description: "Plan: RHYME as an M05 capacity, on the instrument of Heuser, 'Generative Aesthetics: On Formal Stuckness in AI Verse' (JCA) — exact-rime detection via Prosodic. Two questions: when does rhyme install on the pretraining ladders, and where in the alignment stack does STUCKNESS (rhyme the model cannot turn off) arise. Distributional rhyme-pull primary (word_probs machinery), generative rhyme-maintenance secondary (needs a checkpoint generation run, costed separately)."
---
# Plan: rhyme — the capacity, and where stuckness installs

Drafted 2026-08-13 by the registrar on RH's word ("I want to add a new
capacity in M05: rhyme"), instrument inherited from RH's published paper
(Heuser, "Generative Aesthetics: On Formal Stuckness in AI Verse," *Journal
of Cultural Analytics*; §2.3): rhyme detected by EXACT MATCH of the rime
phonemes of the final syllable(s) across line pairs, phonetic transcription
via Prosodic (Heuser/Falk/Anttila); poem-level threshold ≥4 rhyming lines
per 10 (validated on Chadwyck-Healey annotations: precision 88%, recall
90%); slant rhyme undercounted by design, disclosed there and inherited
here. Permutation tests, no distributional assumptions — the paper's own
statistics and this campaign's house style agree.

## What this capacity is NOT (anti-conflation, declared first)

M05 already carries `poetic pull` — a NEXT-TOKEN PREFERENCE over 20
binomial/rhyming/alliterative pairs at single positions. Rhyme-the-capacity
is SUSTAINED SCHEME MAINTENANCE across lines of produced or scored verse.
One is a pull at a slot; the other is a form held over time. No sentence
reads them against each other without a declared bridge; where both move,
the joint pattern is reported, never merged.

## The two questions

1. **ACQUISITION: when does rhyme install?** On the Pythia (155-rung) and
   OLMo ladders, alongside syntax (event at step 128), sense (climbs all
   pretraining), and the E capacities. Onset criteria as Findings G: first
   rung ≥ half of base-final with the next rung concurring; time-to-half-max
   beside it.
2. **STUCKNESS: where in the alignment stack does rhyme-compulsion arise?**
   The paper found deployed aligned models rhyme WHEN ASKED NOT TO. The
   OLMo ladder's SFT → DPO → RLVR rungs let us locate the installation.
   DIRECTION INHERITED FROM THE PAPER (not new): alignment RAISES rhyme
   intrusion in unrhymed contexts. RH may amend before verdicts.

## Operationalisations (naming rule)

**Primary — `rhyme_pull` (distributional; no generation; the ladder's own
word_probs machinery).** Constructed verse primers ending immediately
before a line-final word: three lines of an AABB opening (lines 1-2 a
couplet, line 3 setting the B rime) plus line 4 up to its final slot.
Measure: probability mass on RIME-MATCHING candidates (rime classes
precomputed over the vocabulary via Prosodic) minus mass on
frequency-matched non-rhyming controls, per prompt, per rung. The exact
design pattern of M05's census instruments; runs wherever word_probs runs.
Control arm: the same primers with line 3's final word replaced to break
the rime expectation (mass on the same candidate set = the base rate).

**Secondary — `rhyme_maintenance` (generative; the paper's protocol,
REVISED 2026-08-13 to RH's own completion design).** RH's paper data
(`generative-formalism1/data/data_as_in_paper/genai_rhyme_completions.csv.gz`,
326,862 rows, structure learned: one row per LINE of a completion event;
`id_human` names the source poem, `first_n_lines` the primer length,
`line_gen` NaN over the primer rows, and `line_real` / `line_gen` ALIGNED
thereafter) already implements the three-way comparison: the human
continuation and the model continuation of the same real-poem primer,
line by line. The ladder run REUSES this protocol and primer roster
(same `id_human` poems, same `first_n_lines`), making checkpoint rungs
directly commensurable with (a) the paper's deployed-model numbers and
(b) the HUMAN `line_real` baseline, which arrives built in. Scoring
mirrors the paper's `get_rhyme_for_txt` semantics: perfect rime =
distance 0 (the exact-rime criterion), near rime <= max_dist reported
beside, the >=4-per-10 poem threshold for categorical reads. STUCKNESS =
rhyme rate in continuations of UNRHYMED primers, tracked across
SFT/DPO/RLVR. Requires a checkpoint generation run — COSTED SEPARATELY,
RH's word before any spend, per cool-off.

## The memorization fence

Real-poem primers make memorization a live concern on ladders trained on
the canon (the Pile contains Dunbar). The CONSTRUCTED-primer arm (novel
AABB/ABAB quatrains written for this plan) therefore rides as the
MEMORIZATION PROBE beside the paper-protocol primary: paper-protocol
minus constructed difference = the memorization share, reported, never
folded. A rung that continues Dunbar in rhyme but cannot continue a novel
couplet is remembering, not rhyming.

## Instrument state (gated 2026-08-13, per [5693])

`prosodic` is PINNED IN THIS REPO'S VENV at the paper's own commit —
`git+...prosodic.git@31db244b` (the version question answered from the
paper repo's requirements.txt: the paper pins its prosodic, so we inherit
CODE, not just method) — installed via `uv pip` (same dynamic-deps note
as stanza's). `espeak-ng` installed via brew (phonemizer's backend; plain
`espeak` does not serve). ROUND-TRIP GATE PASSED: Gray's Elegy quatrain
returns its ABAB scheme with me/lea at rime distance 0 and way/day
matched — install AND function verified, not install alone. HAND-CHECK
(RH's three cases, 2026-08-13): door/snore YES (dist 0), door/bark NO,
bark/aardvark — the instrument says YES (dist 0) and the paper's own venv
returns the IDENTICAL answer: espeak marks -vark with (secondary) stress,
prosodic treats secondary as stressed, so the rime is -ark alone under the
paper's final-syllable rule and the pair rhymes. Construct note inherited,
not a defect: secondary-stress finals rhyme on their own syllable, in our
runs exactly as in the published ones. [5693]'s
`twp_words` cautions acknowledged: `rhyme_pull` writes its OWN results,
never into `twp_words`; any read of `twp_words` follows the engine-state
clause (FINAL + GROUP BY the analysis key).

## Population, unit, tests

Ladders and battery discipline as E/F/G: the two ladder populations,
never pooled; the prompt is the unit within rung; sign tests over prompts;
per-rung existence before any pooled curve; permutation nulls per the
paper. Primer roster: ~40 constructed primers (AABB and ABAB), ~10 famous,
~20 unrhymed controls — small enough to hand-audit, declared in
`data/rhyme_primers.json` before any run reads them.

## Sequencing and cost

1. Primer roster written and committed (no compute).
2. Rime-class table over the ladder vocabularies via Prosodic (local, CPU).
3. `rhyme_pull` pilot on a handful of Pythia rungs LOCALLY (small model,
   MPS) to validate the instrument; then the full ladder pass costed and
   put to RH (same machinery as the syntax/sense census runs).
4. `rhyme_maintenance` generation run: proposal with numbers to RH only
   after the pull curve exists and says the capacity is worth the spend.
