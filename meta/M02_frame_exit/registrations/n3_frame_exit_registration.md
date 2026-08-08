# N3: Frame-exit at the full roster — analysis registration

**STATUS: FREEZE CANDIDATE. Written at [2249] under commission [2246](a), on
RH's N3 = GO ([2245].1). Runs when the cloud grid completes.**

## 0. What this is, and what it is NOT

**This is a NEW INSTRUMENT, not a replication.** `contradiction_four_mass.csv`
has no producer and no definition anywhere in the repository ([2222], verified
[2226]): which surfaces count as pole1 for love/hate, whether inflections count,
whether `blend` is an explicit list or a residual — none of it was ever written
down. **The published 40% / 14-of-35 therefore cannot be reproduced and CANNOT BE
COMPARED to anything below.** Per [2245].1 no comparison to it appears in any
output.

**The token sets ARE the instrument.** They were the missing content last time,
so §2 is the longest section here by design and is the part the freeze gate
should read hardest.

**ATTESTATION.** Written arm-unread. This seat has read no arm-level statistic
from the cloud store, and the beam-stash freeze ([2245].3) is not touched by this
document.

## 1. The construct, stated before any operationalisation

**Frame-exit is the claim that under a contradictory prompt the model stops
producing the scene at all, rather than resolving it toward one pole.**

Three rival mechanisms, and the instrument must be able to return any of them:

    RESOLVE   mass concentrates on one pole's continuations
    ENGAGE    mass stays in the scene and rises above the single-pole baseline
    EXIT      mass leaves the scene entirely -- the model does not continue it

**The contradiction prompts are three-way:** `A` = "She loved him deeply and
wanted to", `B` = "She hated him deeply and wanted to", `AB` = "She loved him and
hated him and wanted to". The measured position is the next token after `to`,
i.e. **what she wanted to DO.** Every mass below is a share of that
next-token distribution.

**Why a single position is adequate here and was not for F14:** the construct is
about whether a scene-continuing act is produced at all, and that is decided at
the verb. This is a LEVEL claim about one position, not a claim about a chain.

## 2. THE TOKEN SETS — the registration's entire content

### 2.1 Discovery, not authorship

**The candidate vocabulary is DISCOVERED from the data, on the F40 precedent, not
written by this seat.** A hand-written pole list is the failure mode this project
has hit repeatedly, and the four-mass's collapse was an unwritten list, not a
wrong one.

**Procedure, frozen:**

1. For every model in the grid and every contradiction prompt (`A`, `B`, `AB`),
   take every surface whose next-token probability is **≥ 0.001**.
2. Pool across all models and all three prompt types. **Pooling is across ARMS
   TOO** — a vocabulary discovered per-arm would build the arm difference into
   the instrument.
3. Filter to alphabetic surfaces of length ≥ 2 with nonzero English unigram
   frequency, per F40's filter.
4. **The resulting set is the CANDIDATE VOCABULARY and is frozen and committed
   before any coding.**

### 2.2 Coding, blind to arm and to model

Each candidate surface is coded into exactly one of four classes, **per pair**
(the classes are pair-relative — `kill` is pole2 for love/hate and in-frame for
obey/rebel):

    POLE1        continuations congruent with the A-pole predicate and not the B-pole
    POLE2        continuations congruent with the B-pole predicate and not the A-pole
    IN-FRAME     continuations that continue the interpersonal scene but are
                 congruent with NEITHER pole specifically, or with BOTH
    OFF-FRAME    continuations that do not continue the scene: discourse markers,
                 punctuation-led continuations, topic shifts, meta-commentary,
                 list/format tokens

**The coding is BLIND: coders see the surface and the pair, never the model, the
arm, or the probability.** Two coders; disagreements adjudicated by a third pass
on the merged set; **kappa reported with the result and the per-pair kappa
reported beside it, F11's sheet-d precedent — a weak pair must be visible.**

**`BLEND` IS RETIRED AS A CLASS.** It appears in the old file with no definition
and no way to distinguish it from IN-FRAME. **Four classes, each defined by what
it excludes.**

### 2.3 The masses

Per (lineage, arm, pair, prompt ∈ {A, AB, B}):

    pole1_mass    sum of P over POLE1 surfaces
    pole2_mass    sum of P over POLE2 surfaces
    in_frame      pole1_mass + pole2_mass + IN-FRAME mass
    off_frame     sum of P over OFF-FRAME surfaces
    unresolved    1 - (in_frame + off_frame)

**`unresolved` is REPORTED, never folded.** It is the mass the coded vocabulary
does not reach, and a cell whose `unresolved` exceeds **0.50** is demoted to
descriptive for that cell, with the count of demoted cells on the face. **An
instrument reports its own coverage before its ratio.**

## 3. The measures, and the baseline that makes them mean anything

**Contradiction-excess for each mass M:**

    excess_M  =  M(AB)  -  mean( M(A), M(B) )

**The single-pole prompts are the baseline. A raw AB quantity says nothing** —
what a model does after "wanted to" is mostly a fact about the frame, not about
contradiction. **This is the re-baselining the original did and is the one part
of its design that is recoverable from its prose.**

**Classification, per cell:**

    EXIT      excess_off_frame  > +t
    ENGAGE    excess_in_frame   > +t
    RESOLVE   max(excess_pole1, excess_pole2) > +t AND |excess_in_frame| <= t
    NULL      none of the above

**t = 0.05, declared now.** And **the full sensitivity curve over t ∈ {0.02,
0.03, 0.05, 0.08, 0.10} is reported with every result**, because the previous
instrument's partition moved from 51% to 31% across exactly that band and the
finding reported one point on it.

## 4. Readout

**UNIT: the independent pretraining lineage** (`data/lineage_map_models.json`),
**n = 34**, per [2198]. **Family labels are never the unit and 103 models is never
the n.**

**PRIMARY.** Is EXIT the modal mechanism across lineages?

    statistic   count of lineages whose modal mechanism is EXIT
    null        uniform over the THREE real mechanisms (p = 1/3).
                NULL is a non-classification, never a fourth category --
                including it in the null is what bought the old reading
                its significance ([2110].6).
    test        one-sided binomial, alpha = .05
    clears at   17 of 34  (one-sided binomial vs p=1/3: 0.0327;
                16 of 34 gives 0.0673 and does NOT clear)
    reported    with the design effect. ICC computed over pairs within lineage
                and applied; the DEFF-deflated p is the headline, never the raw.

**CO-PRIMARY, per [2204]/[2245].1.** Does frame-exit fail on Llama-derived
lineages specifically?

    statistic   EXIT rate in Llama-descended lineages vs all others
    motivation  at 7 labels / 6 lineages the sole dissenting lineage was
                Llama-3.1-8B at ONE exit cell in ten, consistently across both
                its alignment implementations
    test        two-sided, alpha = .05, lineage unit
    DECISIVE ALONE: if EXIT fails on Llama-derived lineages the mechanism is
    LINEAGE-CONDITIONED and the universal claim dies whatever the pooled
    plurality does.

**SECONDARY, descriptive only:** the mechanism distribution over all cells, with
its sensitivity curve.

## 5. What kills what

- **Primary null (EXIT not modal):** frame-exit is not a general mechanism. **This
  is a result, not a failure** — it would mean the addendum's plurality was a
  7-label artifact, and it retires the reframe's mechanism claim while leaving
  the level-dissociation claim untouched ([2248].3: the anti-Fazi line is
  severable from frame-exit entirely).
- **Co-primary positive (Llama-conditioned):** the universal claim dies even if
  the pooled plurality clears.
- **Coverage failure** (>0.50 unresolved in a majority of cells): the instrument
  is not measuring the distribution and NOTHING is reported but the coverage.
- **Kappa < 0.60 on the blind coding:** the token sets are not reliably
  codeable; the instrument is withdrawn rather than reported with a caveat.

## 6. Bands anchor here, and nowhere else

**All bands anchor on THIS RUN's own base-arm cells.** Per [2245].1 and the [290]
rule: **no quantity in any output is compared to the published 0.61–0.89 ratios,
the 40% plurality, or the 14-of-35 count.** The old instrument has no definition;
a comparison to it would be a comparison to nothing.

## 7. Confirmatory and exploratory

**CONFIRMATORY** — declared here, before any arm-level read: the primary, the
co-primary, the t-sensitivity curve, the coverage report, the coding kappa.

**EXPLORATORY, wearing its own name:** per-pair mechanism profiles; the
stage decomposition (base/sft/dpo/rlvr) where arms permit; any relation between
mechanism and model scale.

**BARRED:** any statement of the form "consistent with the published four-mass".

## 8. Scope sentence, pre-written

Whatever this concludes, it concludes about the next-token distribution at one
position, under five contradiction pairs, in the lineages the cloud grid covers,
with a vocabulary discovered at threshold 0.001 and coded blind. **It does not
establish what a model "does" with a contradiction, and it does not speak to
generated text — the behavioural layer is a separate instrument with a separate
result ([2244]), and the two must not be merged.**
