# Finding M05-F: syntax installs as an event — after an agrammatical spam phase, before any capacity, and alignment never touches it

Written 2026-08-11 by the registrar seat. STATUS: DRAFT, grade C — two
ladders but each ONE lineage; two coder families; no cross-seat audit.
Discharges registered secondary 5 (plan:
`../plans/syntax_curve.md`). Re-derives from:

    uv run python meta/M05_emergence/scripts/m05_syntax_tags.py           # tier-1 tags
    uv run python meta/M05_emergence/scripts/m05_licit_run.py             # deepseek licit sets
    uv run python meta/M05_emergence/scripts/m05_licit_run.py \
        --model anthropic/claude-haiku-4-5 \
        --out data/m05_licit_sets_haiku.json --no-probe                   # second family
    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_class_mass.py
    uv run python meta/M05_emergence/scripts/m05_syntax_curve.py

Instruments, all frozen: in-context spaCy tags for all 338,092 unique
(prompt, word) pairs (`data/m05_syntax_tags.parquet`); per-prompt licit
class sets from TWO coder families under the witness discipline
(deepseek-v4-flash, witness/tagger agreement 91.0%; claude-haiku-4-5,
84.0%); class-mass table `data/m05_class_mass.parquet` (UNTAGGED 0.00%
after the apostrophe-unescape correction below). Curve = median share of
RESOLVED mass on licit classes; format band (PUNCT/X/SYM) separate;
convention equivalences ADP=PART, NUM=NOUN, AUX=VERB; payload_empty
censored; coverage drawn WITH the curve ([5434]/[5436] discipline).

## Result 1: on Pythia, grammar is an event with a shape — spike, crash, install

The licit share does not rise monotonically. Step 0-1 sits deceptively
high (a handful of junk words that land in licit classes by luck, at tiny
n); through steps ~8-64 it CRASHES to ~0.10-0.25 — the frequency-spam
phase, where the model pours mass into `to/a/of/and` regardless of the
frame the prompt opens, which is systematically AGRAMMATICAL at most
sites; by step ~1000-2000 it has rocketed past 0.9 and it plateaus there
for the remaining 140k steps. Majority-licit onset (CI > 0.5, persistent):
step 128 (deepseek) / 256 (haiku). Every capacity family onsets an order
of magnitude later (packages 2000, discourse 80000 — E Result 1): the
model speaks in licit forms long before it can complete a package, state
a fact, or track a referent. Form before content before fact.

![Pythia syntax curve](../figures/fig16_syntax_curve_pythia.png)

## Result 2: on OLMo, the entire drama is invisible — and alignment never moves the floor

OLMo's first usable rung (stage1-1000) opens at ~0.9 already: the crash
and install sit entirely below its 1,000-step granularity, inside the
window only Pythia's log-spaced checkpoints can see ([5434]'s point at a
second instrument). From there the syntax floor is FLAT through the rest
of pretraining, SFT, DPO and RLVR — final strict 0.99/0.92 by coder.
Alignment reshapes WHICH licit words carry the mass (the campaign's whole
subject) while leaving the licit share untouched: the operations run
inside grammar, not against it. (Deepseek's nominal "onset stage1-0" is a
thin-coverage artifact — 56% of prompts resolve at step0 and the
resolvers hold junk-class luck; the coverage-gated onset is the first
usable rung, matching haiku.)

![OLMo syntax curve](../figures/fig16_syntax_curve_olmo.png)

## Result 3: the shape survives the coder disagreement

The two families disagree substantially on licit-set breadth (median 4 vs
2 classes; cross-family Jaccard 0.37, systematic strictness difference on
open frames — the probe in `data/m05_licit_sets.json`). The CURVES are
parallel throughout: same crash, same install rung within one grid step
(128 vs 256), same flat alignment arm, a level offset of ~6-7 points.
What is quotable is the shape and the onsets; absolute licit-share levels
are coder-relative and always reported per coder.

## Corrections on the record

- The first class-mass build read 8.05% of all mass as UNTAGGED, flat
  across rungs: 46 apostrophe-bearing prompts were poisoned by a missed
  ClickHouse TSV unescape (`Arendt\'s`) — the exact string ch_read's
  docstring warns about. Fixed in `m05_syntax_tags.py`; rebuild leaves
  UNTAGGED at 0.00%. The uncorrected figures were never committed.
- The coder pin moved from the retired `deepseek-chat` alias to the
  resolved `deepseek-v4-flash` id mid-smoke; the alias run showed ~3-point
  share wobble at temp 0, which the frozen artifact removes.

## Caveats

Class-grain licitness cannot see within-class violations ("a apples");
that is tier 3's object (designed, not run). Coder under-licensing of
content classes biases levels down uniformly (Result 3's offset), not
shapes. One lineage per ladder; the Pythia crash phase is n-small per
cell at its bottom (the battery resolves few words there) though the
median is taken over ~580 prompts. The tier-3 selection judgment
(natural / deviant / ungrammatical — where displacement's semantic scar
would be measured) is the registered next step, on RH's word.
