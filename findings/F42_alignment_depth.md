---
status: provisional
grade: pending
date: 2026-08-09
role: finding
description: "Alignment's causal work is distributed across the stack, not concentrated at the readout: the last two aligned blocks recover 6-16% of the behaviour, everything below them 55-73%. Three pairs, gated. PENDING CONFIRMATION AT SCALE."
instruments: [weight-delta, activation-patch, weight-patch, logit-lens, movement]
families: [llama, qwen, smollm]
chapters: [ch05]
data: [twp_depth_battery.jsonl, head_frozen_survey.json]
scripts: [twp_depth_battery.py, twp_patch_weights.py, twp_patch_depth.py, twp_head_swap.py, head_frozen_survey.py]
---
# F42: How deep does alignment go? (PROVISIONAL)

**STATUS: PROVISIONAL. Three pairs, 6-10 prompts each, one domain. Nothing here
should be cited as established.** It is written up now because the instrument
chain and its gates are the durable part, and because a provisional claim with
its refutation conditions attached is more useful than a private one.

## The claim it answers

> Alignment is a small change *downstream* of pretraining — it adjusts the
> readout, not the computation.

That claim exists inside this project as well as outside it. `F05` (grade D,
rescoped) was revised on 2026-07-01 to say displacement is "overwhelmingly a
final-layer operation... **alignment changes the readout, not the
representation**. Hidden states are nearly identical between base and aligned
through 97% of the network."

## The result

**In every pair that passes the validity gates, the last two aligned blocks
recover a small minority of alignment's behaviour and the blocks below them
recover most of it.**

    pair                  n prompts  ceiling  last-2 blocks  all-but-last-2
    Llama-3.1-8B                 10    0.878          0.157           0.733
    Qwen2.5-7B                    6    0.961          0.076           0.564
    SmolLM2-360M                  6    0.680          0.056           0.552

Recovery is measured against the aligned model's own next-token log-probabilities
on the movers of that cell:

    recovery = median over words of
               (log p_hybrid - log p_base) / (log p_aligned - log p_base)

A hybrid model is built by taking transformer blocks from either checkpoint;
**the head, embeddings and final norm stay BASE throughout**, so the readout is
held fixed by construction.

### Three measures, and they do not say the same thing

    WEIGHTS        modified at every depth, roughly uniformly. Llama 4.0% of
                   block norm in the first third, 5.1% in the last (ratio 1.27);
                   Amber's SFT edge is flat at 4.7%, ratio 1.00.
    REPRESENTATION ||dh_L||/||h_L|| ~7% at EVERY depth on Llama, flat from L0,
                   rising to 32% only at the output. Measured here, not taken
                   from F05 -- and **"nearly identical through 97%" is not what
                   7% looks like.**
    CAUSAL         the aligned RESIDUAL at depth L, run through base weights,
                   recovers half the behaviour only at L26 of 32.

**The third contradicts the first two and the contradiction is the finding.**
The aligned representation at mid-depth is not *sufficient given base
machinery*; the aligned *blocks* at mid-depth nonetheless carry most of the
behaviour. Both are true. The early changes are neither inert nor independently
sufficient — they require the later aligned weights to be read out.

**Quoting either patch alone inverts the conclusion**, which is why the driver
computes both or neither.

## The gates, which are load-bearing

**CEILING.** Recovery with *all* blocks aligned. It is the share attributable to
blocks at all, and it is the denominator the normalised columns divide by. Two
of five pairs failed it and were excluded: Qwen2.5-0.5B at **-0.25** (swapping
every aligned block moves the output *away* from aligned) and OLMo-2-0425-1B at
**3.96**. The latter's weight delta is **1.10** — the change is larger than the
original weights, so it is not a fine-tune in any sense this design can use.

**FROZEN HEAD — FAILS EVERYWHERE, AND IS ROUTED AROUND.** A same-depth cross-arm
*lens* reading is only clean if the fine-tune froze the unembedding. Surveyed 29
pairs by reading the tensor: **zero are frozen, zero have a single unchanged
row**, and the rows for the scored tokens move as much as the global figure
(Llama 5.1e-02 against 6.6e-02 global). Spread spans 2.3e-03 to 1.18.

The routing is why this finding rests on patching rather than on the lens: a
hybrid holds the head at base, so the head cannot contribute.

**CROSS-READ IN DISTRIBUTION.** Where a head swap *is* used (to decompose the
output gap), it is only interpretable if the cross-read stays in distribution.
Llama passes; Amber fails — its cross-read is 5x sharper than the true one, and
`AmberSafe` is `dpo_of` `AmberChat`, so `Amber->AmberSafe` spans two training
stages rather than one.

## What was retired getting here

Each by a check, not by argument, and each is a live lesson:

- **A trajectory observation** (`punch` 0.392 at L27 -> 0.045 at L32): a typical
  word moves 8.8x between those layers; `punch` moved 8.7x. The null was already
  in the data.
- **Anger-specificity**: the same split appears on grief and excitement.
- **A dose-response** across magnitude strata: all strata showed the same effect
  and none was detectable — observed 67-91% of MDE at n=5-6.
- **A stable L19-23 onset**: an artifact of loose word sets. Under `CANONICAL`
  the onset moves to 11/18/27 with a visible false positive at L1.
- **`||dh|| ~ 0` through 97%**, cited from F05 and then measured at ~7%.

**And the depth NUMBER is not claimed at all.** Differences accumulate, so a
curve flat to L18 and rising after is what a uniformly distributed effect looks
like crossing a noise floor: onset-in-curve is not onset-in-mechanism. The
ordinal claim is what replicates.

## What would confirm or refute it

**CONFIRM**: the ordinal pattern — last-2 blocks well below all-but-last-2 —
holding across the 47 locally available pairs that pass the ceiling gate, and
across domains beyond the sexual battery used here.

**REFUTE**: pairs with sane ceilings where the last two blocks recover the
majority; or a demonstration that hybrid composition is invalid even within a
fine-tune (blocks composing only by shared initialisation is an assumption this
design rests on and does not test).

**NOT YET DONE**: more than one prompt domain; any pair above 8B; the SSM and
hybrid architectures; per-prompt variance reported rather than medians; the
interaction term, which is not identified by either patch.
