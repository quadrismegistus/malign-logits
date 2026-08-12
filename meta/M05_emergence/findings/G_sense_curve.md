# Findings G: sense installs with syntax, there is no colorless-green phase, and alignment nudges sense where it never touched grammar

**Status: the tier-3 curve, on a census not a sample; one coder family
(pinned), instrument validated by pilot, tie-break, and 10/10 canaries.**
Three results. (1) The natural-sense share of completion mass rises from
exactly zero to 47% inside Pythia's first 128 steps (~268M tokens) — the
same rung as the syntax onset — and then climbs for the entire rest of
pretraining to 92%, long after syntax has saturated. (2) **The
grammatical-but-senseless band never exceeds 3.7% of mass at any rung of
either ladder**: mass moves from ungrammatical to natural directly, and
the "colorless green ideas" phase the syntax-first story permits does not
occur. (3) Alignment RAISES the natural share — small, monotone across
SFT -> DPO -> RLVR, paired per-prompt median +0.4 to +0.6pp, ~360 of 584
prompts up, Wilcoxon p ~ 1e-6 — where the same ladder's licit-syntax
share never moves (Findings F). Alignment does not touch grammar; it
does, slightly, touch sense.

## The instrument (tier 3 of the syntax program)

RH's design (2026-08-12), each element his: a CENSUS, not a sample —
every (prompt, word) completion pair either judged or auto-assigned,
never sampled. 136,036 census pairs from two declared floors (max
p >= 0.003 at any rung, CANONICAL's own eligibility constant; early-window
top-up at p >= 0.002 for Pythia step < 2,000 / OLMo stage1 < 4,000, where
a colorless-green claim would have to live), minus the syntax tier's
auto-exclusions: 118,129 JUDGE + 16,624 ungrammatical_auto (illicit under
BOTH tier-2 coder families — the already-paid-for instrument rules) +
1,283 format_auto (PUNCT/X/SYM).

Coder `code_m05_sense_v1` (four-way: natural | odd | ungrammatical |
not_a_word; rules earned in the pilot: OFFENSIVE IS NOT ODD, FALSE IS NOT
ODD, THE TEXT CONTINUES AFTER THE WORD), pinned deepseek-v4-flash after a
500-pair two-coder pilot with tie-break; 10 authored positive controls
rode the bulk as end-of-run canaries, 10/10. Bulk: 118,129/118,129 in
2.12h, sha `9060957ed8050b42`, chunked and resumable, sidecar carries the
three-family band. Producers committed with the run:
`m05_sense_census.py`, `m05_sense_bulk.py`, `m05_sense_mass.py`,
`m05_sense_curve.py`.

Judged-pair split (unweighted): natural 64.2%, ungrammatical 21.4%, odd
13.5%, not_a_word 0.9%.

## 1. The curve

Mass join (`data/m05_sense_mass.parquet`, 552,061 rows, 95 + 155
checkpoints, ZERO store gaps; unclassified tail below both floors =
1.29% of all mass, censored from the ratio and drawn beside it). Curve =
natural share of classified mass, median over 584 prompts; format band
excluded both sides, mirroring the syntax curve.

    Pythia base arm (fig19_sense_curve_pythia.png):
    step    0-8     16     32     64    128    256    512   1000   final
    natural 0.000 0.037  0.118  0.318  0.472  0.551  0.630  0.749  0.918

Onset (first rung >= half of base-final, next rung concurring): **step
128**, the same rung as the tier-2 syntax onset (deepseek coder,
`results/syntax_curve.json`). OLMo stage1: onset at the first non-zero
rung (step 1000; the grid is coarser), final 0.921. But the shapes
diverge after onset: syntax saturates near its ceiling within the first
few hundred steps, while sense is still buying its last 15 points from
step 1000 to the end of pretraining. **Syntax is an event; sense is the
rest of pretraining.**

## 2. No colorless-green phase

The design's sharpest question: is there a window where completions are
grammatical but senseless — Chomsky's colorless green ideas, the phase a
syntax-first acquisition story permits? The early-window top-up floor
existed precisely to catch it. Answer: **no, at any rung, on either
ladder.**

    Pythia base, median band shares of classified mass:
    step      16     64    128    512   1000   2000   final
    odd    0.000  0.005  0.020  0.031  0.037  0.037  0.024
    ungram 0.947  0.631  0.438  0.299  0.171  0.140  0.080

The odd band peaks at **3.7%** (steps 1,000-2,000) and declines. Mass
transfers from ungrammatical to natural almost one-to-one; grammatical
nonsense is a thin constant sliver, never a phase. Within this
instrument, syntax and sense arrive together at mass grain — what
"syntax first" buys is only the saturation asymmetry of §1.

## 3. Alignment nudges sense and never touched syntax

Findings F: alignment never moves the licit-syntax share (flat at 1.0 on
the fig18 normalization). The sense share moves:

    OLMo, natural share at last rung per role (median over prompts):
    base 0.918   SFT 0.936   DPO 0.936   RLVR 0.939

    Paired per-prompt, last aligned rung minus base_endpoint, n=584:
    SFT   +0.0041 median, 341/227 up, Wilcoxon p 1.1e-05
    DPO   +0.0051 median, 356/211 up, p 9.2e-07
    RLVR  +0.0055 median, 361/205 up, p 1.9e-07

Small (half a point per prompt at the median), significant, and
monotone along the ladder. Read with F: **the alignment stack leaves
grammar alone and slightly concentrates mass on words a reader calls
natural** — consistent with M01's CANONICAL fall-dominance (alignment
removes more than it adds) if what is removed is disproportionately the
odd/ungrammatical tail. That mechanism check (which bands lose the mass
alignment removes) is a cheap reweighting of the same parquet, not run
here.

## Figures and artifacts

- fig19_sense_curve_{pythia,olmo}.png — band shares + coverage + the
  censored tail, raw scale.
- fig17/fig18 (both ladders) now carry "sense (natural share)" beside
  the four capacities, poetic pull and syntax; sense is the earliest
  riser on fig18's own-late-base normalization.
- `results/sense_curve.json` — onsets, per-rung natural shares, per-role
  finals. Raw grains per the raw-data rule: word-grain verdicts in
  `data/m05_sense_verdicts.parquet`, band-mass grain in
  `data/m05_sense_mass.parquet`.

## Limits

- One coder family judged the bulk (deepseek-v4-flash, pinned). The
  two-coder pilot (500 pairs, tie-break) and 10/10 canaries validate the
  instrument, but the 118k verdicts themselves are single-coder; the
  three-family disagreement band from the pilot rides in the bulk
  sidecar and any drafted rate should carry it.
- "Natural" is a coder's judgement of word-in-context acceptability, not
  a human norm; the coder never sees rung, model or arm (word-grain
  blindness is structural — the census is deduplicated across rungs).
- The odd/ungrammatical boundary has known taxonomy edges (the pilot's
  "swim" case); the §2 null is robust to them only because the odd band
  is small under EITHER resolution of the boundary.
- OLMo's early grid (0, 1000, ...) cannot resolve a sub-1000-step
  colorless-green window on that ladder; the §2 claim there rests on
  Pythia's fine early grid plus OLMo's concurring coarse one.
- §3's mechanism reading (fall-dominance removing the senseless tail) is
  suggested, unrun, and named as the next cheap read.
