# The contradiction ratio moves under alignment. It cannot say toward what.

**Status: a small, replicated positive with a hard ceiling on what it licenses.**
At the output layer, across 37 English lineages, alignment raises the ratio from
0.842 to 0.911 — **26 of 37 lineages, sign p = 0.020**. That is the one claim
the instrument supports. Everything built on top of it is either unsupported or,
as of today, refuted.

Substrate `results/lens_group_layer.jsonl` at `layer == n_layers - 1`, English,
degeneracy-guarded (`js_min >= 1e-6`, `|ratio| <= 100`), lineage unit, groups
paired within lineage.

---

## Why this document exists

The ratio has been M02's and F11's main instrument for months and **its positive
result has never been written down.** What is on the record is
`contradiction_ratio_has_no_null.md`, which is the calibration and the negative
correction: 1.0 is not the superposition/resolution boundary, it is where a
distribution holding NEITHER pole lands, so "alignment shifts toward resolution"
is wrong in kind rather than in degree.

That document is right and this one does not disturb it. But a campaign that
records only the deflation of its instrument, and never the effect the
instrument actually measures, has an incomplete ledger — and a number that lives
only in session memory is one that drifts.

## The claim, and the scale it lives on

    perfect blend of the two poles            0.000
    BASE ARM, observed                        0.842
    ALIGNED ARM, observed                     0.911
    NEITHER pole -- neutralization            1.006
    resolution to one pole                    4.031

Alignment moves the BOTH distribution **away from a blend of its poles**. The
whole effect is **0.047** on a scale whose next landmark is 0.164 further on and
whose far end is 3.1 away.

    aligned arms reaching even halfway to resolution (> 2.5)     0 of 37
    arms above the neutralization anchor (1.006)   base 2/37, aligned 7/37

So the direction is real and the magnitude is small in the scale's own terms.
Every measurement ever taken with this instrument sits in a band about 0.07 wide
in the bottom quarter of a four-point calibrated range.

## What it cannot say, and why both escape routes are now closed

"Away from the blend" is compatible with resolution and with frame exit and
distinguishes neither. Two routes have been used to infer the destination
anyway, and both closed on 2026-08-11:

**Depth.** The late-gate signature — 0.339 of the base/aligned gap in the top
eighth, 35 of 38 lineages, p = 6.7e-08 — was read as alignment operating as a
cheap late mask. Two results bear on that. The gap's concentration in the top
eighth is **head-dependent**: on the M05 ladder it climbs 0.228 → 0.441 read
through a frozen base head and stays flat at ~0.24 through a frozen DPO head,
17 of 21 groups, p = 0.0072, while the magnitude of the change is identical
under both (`M05_emergence/findings/lens_ladder_instrument_note.md`). And the
depth of divergence does not predict frame exit at the surface
(`depth_and_exit_do_not_join.md`).

**The markers.** The output-level ratio contrast predicts the passage-level exit
markers at **rho = -0.011** over 24 lineages. Not weakly — not at all. Whatever
the ratio measures, it is not the event the markers see.

## The reproduction is loose

`results/f11_reproduction.csv`, 9 families, F11's published values against
recomputation on the current substrate:

    pearson r 0.704, spearman rho 0.728
    recomputation HIGHER in 9 of 9 families, mean shift +0.099
    F11 range 0.61-0.89, recomputed range 0.82-0.94

Same direction, different numbers, and a compressed range. This is the F13
pattern again — direction survives, numbers do not — and it means F11's
per-family values should not be quoted as though they were these values.

## How to use it

Report the arm effect at its actual size, as a claim about departure from
blending and nothing more. **Do not hang depth or destination stories on it.**

For the destination, the instrument that works is the second-order marker
(`second_order_naming.md`): 2.18x on contradiction against 0.93x on single-pole
controls, 20 of 22 lineages, p = 0.00012, on 52,559 passages — the same question
the ratio was reaching for, at ten times the effect size, in a measure with room
to move, and readable without a lens.

The ratio's honest role is now corroborating rather than load-bearing.

## Limits

The 26/37 rests on a sign test at p = 0.020, which is real but not
overwhelming, and several p ≈ 0.05 results elsewhere in this campaign have
dissolved under a change of unit. This one is already at the declared unit.

The final-layer lens readout is used as the output distribution. That is exact
by construction for a model read through its own head, which is the case here.

English only; zh is 47% of the lens rows and reported apart by M02 convention.
