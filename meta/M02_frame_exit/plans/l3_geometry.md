# L3 geometry: is the interior excursion contradiction-specific or conjunction-general?

Plan document under [5148]. lacan seat, 2026-08-09, dispatched at [5152].
Exploratory label: the trajectory reads are EXPLORATORY; the role contrast at a
fixed layer is the confirmatory one. Nothing frozen.

## QUESTION

The L3 pilot ([5141], 2 checkpoints, 1 triplet, no controls) found the base
holding the midpoint between its poles through the stack (t ~ 0.42-0.46) while
DPO hauls to 0.18 at layer 7 and reconverges by the top. **Is that interior
excursion about CONTRADICTION, or is it what any conjunction of two adjectives
does?** The pilot could not ask: the BOTH prompt lexically contains both pole
words, so t ~ 0.5 may be a fact about two-adjective sentences.

## INPUT

Population **read from** `data/f11_quintuplets.json`, status-filtered at analysis
time ([5084].2): 43 live groups of 44, `f11_species_wolf` dropped as wholly
RETIRED. `f11_reason` / `f11_reason_zh` are the weak-manipulation NEGATIVE
CONTROL, run beside the primary and never pooled with it.

Residuals from `data/f11_twp`, `f11_twp_bf`, `f11_twp_delta`, discovered by glob
and the scanned list printed (a hardcoded two-directory constant reported "no
controls exist" for an hour after they landed).

    base/aligned pairs, both arms present            45
    (pair, group) cells   TRIPLE                   1845
                          TRIPLE + both controls   1459
                          BOTH_MATCHED              440

Controls reach 34 of 43 groups **by design, not by gap**: 9 groups have CATEGORY
poles (gender x3, parent, species x2, ...) for which no near-synonym companion
exists in any language ([5072].2).

Two models trail twp on residuals ([5151].3). Handled **per prompt, not per
model**: a cell missing any role it needs is dropped by the coverage test, and
the roster actually read is printed with the result.

`f11_holy` + `f11_holy_b` and `f11_holy_zh` + `f11_holy_b_zh` share a BOTH cell
byte-identically. One contradiction measurement, two pole-pairs: never pooled.

## INSTRUMENT

Per (pair, group, arm, layer), with h_A = pole_a, h_B = pole_b:

    t(X)     = (h_X - h_B) . (h_A - h_B) / |h_A - h_B|^2
    resid(X) = |off-axis component| / |h_A - h_B|

for X in {both, control_a, control_b, both_matched}. `t` is where X sits on the
pole axis (0.5 between, 1 at A, 0 at B); `resid` is how far off that axis it
sits, which is what frame exit looks like geometrically.

**WHICH LAYERS: all of them, but the primary is WITHIN-LAYER.** Depths range 17
to 81 across the roster, so cross-model pooling needs relative depth and that is
a secondary. **The primary contrast compares ROLES AT THE SAME LAYER of the same
model**, which needs no depth alignment and is immune to the next point.

**THE PRE-NORM / POST-NORM SEAM, and why the primary avoids it.**
`hidden_states[-1]` is post-norm; every other entry is pre-norm. RMSNorm applies
a learned per-dimension weight, which is a diagonal linear map and NOT a scalar,
so `t` is invariant under it only within a layer, not across the seam. Comparing
an interior `t` to the final-layer `t` therefore compares two spaces. The primary
(roles at one layer) is unaffected. Any cross-layer statement, including "the
output repairs the excursion", is SECONDARY and carries this caveat until the
norm weights are extracted per model and every layer is put in one space.

## OUTPUT

`meta/M02_frame_exit/results/l3_geometry.parquet`, one row per
(family, base, aligned, group, arm, layer, role) with `t`, `resid`,
`|h_A - h_B|`, plus a run header naming the roster read and the directories
scanned. Producer `meta/M02_frame_exit/scripts/l3_geometry.py`.

## ANALYSIS

**Primary.** At each layer, the base-minus-aligned shift in `t`, compared across
roles within the same (pair, group): is the shift larger for BOTH than for
CONTROL_A and CONTROL_B? Unit is the (pair, group) cell; paired across arms;
clustered by family, since F31 puts family at 97.8% of variance.

**Secondaries.** BOTH vs BOTH_MATCHED (the modifier against the conjunct, 440
cells). `resid` as the frame-exit read. The depth-normalised trajectory.

**THE PRIOR, BOTH BRANCHES, WRITTEN BEFORE READING.** The controls are same-side
near-synonym conjunctions, so if the interior excursion is about contradiction
they should show **little or no** excursion while BOTH shows it. If the controls
excurse as much as BOTH, the layer-7 pole-pull is about **conjunction, not
contradiction**, the pilot measured grammar, and M02's interior reading changes.
That is not a null to bury, and it is the same branch malign's primary carries at
the output grain.

**And the negative control outranks the primary**: if `f11_reason` shows the
effect too, on poles known not to separate, the effect is not about contradiction
whatever the controls say.

## COST

No GPU, no generation. Reads flat `.hidden.f32` sidecars and their jsonl indices.
Minutes, single process. The expensive thing already happened.
