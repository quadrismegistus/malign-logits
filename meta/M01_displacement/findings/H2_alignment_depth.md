# Findings H: alignment is distributed through the stack, not concentrated in the last layers

Written 2026-08-10 by the malign seat. Plan: `registrations/plan_h2_alignment_depth.md` §5.
Producer: `scripts/h_depth_primary.py`. Data: `data/h2_depth/*.canonical.jsonl`, written by
`scripts/twp_depth_battery.py`; receipt `data/h2_depth_receipt.json`, run locally by RH.
Every number below comes from one run of that producer over those shards and can be re-derived
by running it.

## The question

If you take a base→aligned pair and restore the aligned model's weights **layer by layer**, does
the base model's behaviour come back all at once near the output, or gradually across the stack?
The first would make alignment a late-layer overwrite — a filter bolted onto the end. The second
makes it a change to the whole computation.

The quantity is a per-cell **raw difference**:

    d = recovery(all-but-last-2) - recovery(last-2)

computed inside each cell from that cell's own `bottom[N-2]` and `top[2]`. Positive `d` means the
first N−2 layers carry more of the effect than the last two do.

## The answer

**Alignment is distributed. 22 of 23 pairs go the same way, and the median pair recovers about
0.7 more from all-but-the-last-two layers than from the last two.**

    PAIR LEVEL (the unit)          n=23
      median of per-pair medians   +0.697
      range                        [-0.098, +1.443]
      pairs with median d > 0      22 / 23

    CELL LEVEL (ceiling-gated)     n=4,318
      median                       +0.696
      IQR                          [+0.525, +0.843]
      d > 0                        4,145 / 4,318   (96.0%)

The last two layers carry a small fraction of what alignment does. Whatever alignment is at the
level this instrument sees, it is not a filter on the output.

**The pair is the unit and the cell level is not an independent n.** Cells within a pair share a
base model, a tokenizer and a training recipe, and they move together. The cell line is reported
because a per-pair median hides the within-pair spread, not because 4,318 is a sample size.

## The one reversal, named

**`BSC-LT/salamandra-7b-instruct`, median −0.098** — the only pair below zero, and marginal rather
than dramatic. 64 of its 117 gated cells are negative, which is a third of every reversing cell in
the run; the remaining 109 are scattered thinly across 15 other pairs, none of which reverses at
the pair level.

The plan requires reversals to be **named, not counted** (§5), and this is why: "173 reversing
cells of 4,318" invites the reading that reversal is a low-rate background phenomenon. It is not.
It is mostly one pair.

## The ceiling is a class, never a divisor

`recovery` is a ratio with a per-cell denominator, so there is a live temptation to normalise `d`
by the cell's ceiling. **§5 forbids it and the reason is arithmetic rather than taste**: a shared
denominator cancels in a ratio and *scales* in a difference. `d_norm = d_raw / ceiling` inflates a
cell whose ceiling is 0.07 by 14×, and a negative ceiling flips the sign of a difference that was
never in doubt.

So the ceiling enters as a four-level class and gates membership only:

    failed   ceiling <= 0     541 cells   construction failed; no scale exists
    low      ceiling < 0.5    436 cells   a scale too small to divide by
    normal   0.5 .. 1.2     3,725 cells
    over     ceiling > 1.2    593 cells

`failed` and `low` are excluded — 977 of 5,295 cells, 18.5%.

**Gating barely moves the headline: ungated median +0.707 against gated +0.696.** That is reported
here deliberately. Had the gate moved the result, the result would have been a fact about the gate.

## The pilot overstated it, and the direction survived anyway

The pilot reported median raw `d` **+0.962 with 162 of 163 cells positive**. The full run gives
**+0.697 with 22 of 23 pairs positive**. Same direction, materially smaller magnitude — the
ordinary inflation of a small first look. **Quote the full-run number; the pilot's is superseded,
not confirmed.**

## What this does not establish

- **It is a claim about weights, not about meaning.** The instrument restores parameters and
  measures how much word-level behaviour returns. It does not say the middle layers hold
  "the content" of alignment, only that they hold most of what restoring them recovers.
- **`repr_L50` (the depth at which half the representational change has accrued) is carried in
  every row and is not analysed here.** It is the obvious next reading and it is not this one.
- **No claim about which layers matter most** — `d` contrasts two blocks, and a block contrast
  cannot locate anything inside a block.

## Two pairs of the designed 25 contributed nothing

`llm-jp/llm-jp-3-7.2b-instruct3` and `m-a-p/neo_7b_instruct_v0.1` recorded 231 of 231 prompts as
`no_row` and produced no cells, while the receipt reported `status: ran` and `owed: 0` for both.

**The cause is the reader, not the checkpoints.** Both have complete twp coverage — 2,590 cells on
both arms. `movement.word_probs` returns `None` for llm-jp under the ClickHouse default, where the
hashstash holds 204 words for the same cell; the battery catches the resulting `AttributeError`
per prompt and records it as a legitimately-unusable cell, and the runner then never re-offers it.
RH ruled the gap out of scope for this reading.

It is recorded here because **"23 pairs" must never read as "the design had 23"**, and because the
failure shape is worth keeping: a wholesale failure was recorded as 231 individual per-cell facts,
`rc` was 0, and the only tell at the receipt level was elapsed time — 72 s against 681 s for a
pair that worked.

## Reproduce

    python meta/M01_displacement/scripts/h_depth_primary.py
    python meta/M01_displacement/scripts/h_depth_primary.py --json out.json
