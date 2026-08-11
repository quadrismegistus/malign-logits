# The pole-axis projection is not a superposition measure, and two corrections to [5157]

**Status: PROVISIONAL, and partly a CORRECTION.** [5157] reported the L3
geometry result with the sentence *"Superposition at the representation level is
real and alignment does not undo it."* That word was mine and it outran the
instrument. The measurement is real; the label is not licensed by it.

L3 measures, per (pair, group, arm, layer), with `h_A = pole_a`, `h_B = pole_b`:

    t(X)     = (h_X - h_B) . (h_A - h_B) / |h_A - h_B|^2
    resid(X) = |off-axis part| / |h_A - h_B|

`t` is a ONE-DIMENSIONAL projection onto the line joining the two pole
representations. It says where a prompt's shadow falls on that line and nothing
about where the prompt actually is.

## Why t cannot carry the word

    role            t        resid mean   resid median
    both          0.453        0.954         0.855
    control_a     0.793        0.853         0.806
    control_b     0.128        0.824         0.773
    both_matched  0.412        0.782         0.755

    BOTH cells with resid > 1.0:  0.315      > 2.0: 0.031

`resid` is the off-axis magnitude as a multiple of the entire pole separation.
**The BOTH representation's off-axis component is about as large as the whole
distance between the poles**, and a third of cells exceed it. BOTH is not a
point on the segment between pole_a and pole_b. It is a point well away in some
other direction whose shadow happens to land near the middle.

If BOTH were a superposition of the poles, `h ~ a.h_A + b.h_B`, it would lie
roughly in their span and `resid` would be small. It is not small, and it is not
even distinctively large: every role sits about as far off the axis as every
other. The pole axis is simply not where most of the variance lives, so whatever
carries the "both-ness" is largely orthogonal to the thing being measured.

**And three different states produce an intermediate t:**

    superposition   a genuine mixture of the two poles
    a third thing   a distinct representation (paradox, rhetorical figure)
                    whose projection happens to fall midway
    NEUTRALIZATION  neither pole strongly represented

The first and third are opposite readings of the finding -- inclusive
disjunction against frame exit -- and `t` cannot separate them. *Both* and
*neither* cast the same shadow. This is the same gap that
`contradiction_ratio_has_no_null.md` finds in the output-side instrument,
arrived at independently on the representation side.

## What [5157] did establish, stated narrowly

Same-side near-synonym conjunctions sit AT their pole (0.793, 0.128) while the
contradiction sits between (0.453), and alignment barely moves any of them. That
is real, it is contradiction-specific because the controls license it, and it is
a NECESSARY condition for superposition. It is not evidence of superposition.

## Correction 2: "clustered by family" clustered nothing

[5157] reported its contrasts as "clustered by family, n=45". **The `family`
column holds 52 distinct values for 52 pairs** -- it is 1:1 with the pair, so the
grouping was a no-op and the test treated scale siblings as independent. Falcon3
1B/3B/7B are one lineage; four other lineages contribute two pairs each.

Recomputed with the pair collapsed to its lineage
(`data/lineage_representative_pairs.txt`), every conclusion holds:

                          FAMILY (= pair)              LINEAGE
    [5157] population
      BOTH-control_a      n=45  -0.0034 p=7.7e-02   n=39  -0.0037 p=8.0e-02
      BOTH-control_b      n=45  +0.0007 p=6.8e-01   n=39  +0.0003 p=8.6e-01
      BOTH-both_matched   n=45  +0.0100 p=1.3e-04   n=39  +0.0097 p=7.6e-04

The label was wrong; the finding was not. Reported here rather than quietly
restated.

## The union: the store L3 could not address

`l3_geometry.py` globbed `data/f11_twp*`, which cannot reach
`data/raw/twp_fill/` -- **65 of the 74 GB of residuals on disk**, the same
split-store defect as the logit index's bare basename, in a third consumer.
`--dirs union` opts in and requires `--out`; the default still reproduces
[5157] to the digit.

    43 -> 52 pairs, 104 models, 391,278 rows
    FULL_QUINTUPLET  44 pairs      TRIPLET_ONLY  8 pairs

The 8 added pairs carry no controls (RH called this before it was measured;
`twp_fill` holds the triplet and both_matched, not the yoked controls). So the
frame holds two populations and they are two different n. The `stratum` column
exists to stop them being pooled:

    BOTH vs control contrast   FULL_QUINTUPLET only    n = 44 pairs / 37 lineages
    t(both) and both_matched   both strata             n = 52 pairs / 46 lineages

The one contrast that gains is the one that was already significant:

    UNION
      BOTH-control_a      n=43  -0.0034 p=8.7e-02   n=37  -0.0037 p=9.2e-02
      BOTH-control_b      n=43  +0.0006 p=7.2e-01   n=37  +0.0002 p=9.1e-01
      BOTH-both_matched   n=52  +0.0118 p=2.6e-06   n=46  +0.0119 p=1.7e-05

`BOTH-both_matched` goes from `+0.0097, p=7.6e-04` at 39 lineages to `+0.0119,
p=1.7e-05` at 46. The two control contrasts do not move at all -- same point
estimates to four decimals, 37 lineages against 39 -- which is exactly what a
population that adds no controls should do, and is worth stating as the check
that the widening did what it claimed rather than something else.

## The stratum difference is family-level, not contradiction-level

Interior window, base minus aligned:

                       both      both_matched
    FULL_QUINTUPLET   +0.0043      +0.0070
    TRIPLET_ONLY      +0.0128      +0.0132

The 8 added families drift about three times as much under alignment -- but
`both_matched` drifts just as much within each stratum, and in BOTH strata it
drifts at least as much as `both`. So this is a property of those models'
representations moving more under alignment generally, not something about
contradiction. Worth recording, not worth claiming: 8 non-randomly-selected
families, several non-Western or hybrid-architecture (Zamba2 is an SSM).

The negative control is unchanged by the union and still refuses the excursion:

    f11_reason/_zh   both 0.0780   control_a 0.0741   control_b 0.0739

## What would actually test superposition

The logit lens, on data already on disk, with an instrument that already exists
and is verified against the model's own head (`LayerReadout`, `malign_logits/twp.py`).

For the BOTH prompt at interior layers, project the residual stream to
vocabulary and ask whether BOTH POLE WORDS ARE SIMULTANEOUSLY ELEVATED. That
discriminates what `t` cannot:

    superposition   both readable
    neutralization  neither
    a third thing   neither pole, but something else coherent

It is the same probe as the repression-against-foreclosure question -- is `kill`
still readable in the aligned model's stack at the point where `scream` is
emitted -- so one instrument answers both, and that is the difference between
the paper asserting the Lacanian claim and measuring it.

## Reproduction

    uv run python meta/M02_frame_exit/scripts/l3_geometry.py                       # [5157]
    uv run python meta/M02_frame_exit/scripts/l3_geometry.py --dirs union \
        --out l3_geometry_union.parquet
    uv run python meta/M02_frame_exit/scripts/l3_geometry_read.py l3_geometry_union.parquet

Producer: `scripts/l3_geometry.py`. Read: `scripts/l3_geometry_read.py`.
Results: `results/l3_geometry.parquet` (published, [5157]),
`results/l3_geometry_union.parquet` (wider population, stratified).
Related: `findings/contradiction_ratio_has_no_null.md`.
