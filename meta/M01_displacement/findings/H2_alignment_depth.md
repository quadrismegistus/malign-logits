# Findings H2: alignment is distributed through the stack, not concentrated in the last layers

Written 2026-08-10 by the malign seat. Confirms **H1** at scale (H1 was written as F42 and
asked for exactly this). Plan: `registrations/plan_h2_alignment_depth.md` §5.
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

**Alignment is distributed. 19 of 20 lineages go the same way, and the median pair recovers about
0.7 more from all-but-the-last-two layers than from the last two.**

    LINEAGE LEVEL (the unit)       n=20
      median of per-lineage medians +0.700
      range                        [-0.098, +1.443]
      lineages with median d > 0   19 / 20

    PAIR LEVEL (23 pairs = 20 lineages)
      median of per-pair medians   +0.697
      range                        [-0.098, +1.443]
      pairs with median d > 0      22 / 23

    CELL LEVEL (ceiling-gated)     n=4,318
      median                       +0.696
      IQR                          [+0.525, +0.843]
      d > 0                        4,145 / 4,318   (96.0%)

The last two layers carry a small fraction of what alignment does. Whatever alignment is at the
level this instrument sees, it is not a filter on the output.

**The LINEAGE is the unit, not the pair, and neither is the cell.** `Falcon3-1B/3B/7B-Instruct`
are three arms of one pretraining run and `Qwen2.5-0.5B/7B-Instruct` are two, so 23 pairs are 20
independent observations — counting them as 23 counts one base run five times. Cells within a pair
share a base, a tokenizer and a recipe and move together, so 4,318 is not a sample size either.

**The correction costs three units and changes nothing: +0.697 at n=23 against +0.700 at n=20,
22/23 against 19/20.** It is applied because the campaign's unit rule has been got wrong before in
the direction that flatters — 59 → 39 → 34 in one afternoon, every correction toward significance —
and a result that survives the check is stronger for having been asked.

## Where it accrues: half the change is in place by ~60% of depth

`d` contrasts two blocks, and a block contrast cannot locate anything *inside* a block. `repr_L50` — the depth at which half the representational change has accrued — answers the obvious next question, and it was already sitting in every row.

**Normalised by `N`, because 32 blocks and 48 blocks are not one scale.** Layer 20 is two-thirds of the way up a Llama and 40% of the way up a Yi; pooling raw layer indices across this roster would report a fact about the depth mix and call it a fact about alignment.

    repr_L50 / N          4,306 gated cells with a curve, 23 pairs
      CELL LEVEL          median 0.607   IQR [0.425, 0.738]   range [0.000, 0.975]
      PAIR LEVEL          median 0.594   range [0.000, 0.861]

      L50 in the last quarter of the stack    922/4,306   21.4%
      L50 before the halfway point          1,329/4,306   30.9%

So the positive form of the headline: **not merely "not at the readout" but "half of it by three-fifths of the way up," with only a fifth of cells accruing late.**

**The per-pair spread is the result, and the median describes no pair in the roster:**

    allenai/OLMo-2-0425-1B-DPO             0.000
    OpenLLM-France/Lucie-7B-Instruct-v1.1  0.125
    tiiuae/Falcon3-10B-Instruct            0.425
    HuggingFaceTB/SmolLM3-3B               0.556
    meta-llama/Llama-3.1-8B-Instruct       0.625
    LLM360/AmberSafe                       0.719
    PKU-Alignment/beaver-7b-v1.0           0.781
    stabilityai/stablelm-2-zephyr-1_6b     0.833
    Qwen/Qwen3-8B                          0.861

A roster spanning 0.000 to 0.861 is not a population with a typical member. Some recipes have done half their work before the stack has started; others leave it to the top sixth.

**`allenai/OLMo-2-0425-1B-DPO` should be read as an instrument failure, not a fast aligner.** Three independent measures agree: `L50 = 0.000`, ceiling **3.96**, and a head delta of **1.18** — larger than the norm of the weights it started from. H1 already calls it "not a fine-tune in any usable sense," and it also carries H2's largest per-pair `d` (+1.443). It is inside the gated set as an `over`-ceiling cell and it should be named wherever it is inside a median, here included.

**This distribution is descriptive and carries no test.** It is not a claim that alignment "begins" at 0.6N — differences accumulate, so a half-accrual point is a summary of a curve and not an onset in a mechanism. That is the same restraint H1 exercises about onset numbers and it applies here unchanged.

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
- **No claim about onset.** The `repr_L50` distribution above summarises a curve; because
  differences accumulate, a half-accrual point is not the depth at which alignment "starts".
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
