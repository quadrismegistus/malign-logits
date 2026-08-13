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

**Secondary — `rhyme_maintenance` (generative; the paper's instrument
proper).** At each rung, continue rhymed and unrhymed verse primers;
score line-pair rime matches in the continuation (Prosodic, exact rime);
capacity = P(continuation rhymes | rhymed primer) − P(rhymes | unrhymed
primer); STUCKNESS = rhyme rate under unrhymed primers, tracked across
SFT/DPO/RLVR. Requires a checkpoint generation run that does not yet
exist (`mega_generations` covers the OLMo ladder thinly and without verse
prompts) — COSTED SEPARATELY, RH's word before any spend, per cool-off.

## The memorization fence

Pythia's Pile and OLMo's mixes contain the canon: continuation of a famous
quatrain can be recall, not capacity. PRIMERS ARE CONSTRUCTED (written for
this plan, never in any training corpus), primary; a small famous-poem arm
rides beside as the MEMORIZATION PROBE, and the constructed-minus-famous
difference is reported as the memorization share, never folded.

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
