# Plan: the logit lens on contradiction -- where in depth does the frame exit?

**STATUS: A PLAN, AND DELIBERATELY INCOMPLETE.** The INPUT clause (§7) is RH's to
set, on the [5148] standard and following `plan_h_logitlens.md`, which this is
NOT an extension of -- H is displacement (does the sharpness of the terminal
event track displacement magnitude); this is contradiction. Same instrument,
different question, different population. Written 2026-08-11 by the lacan seat.
Nothing here is frozen. The PILOT in §8 has its scope set by RH and is an
instrument shakedown, not evidence.

## 1. The question, and it is not the one I kept saying

I have repeatedly written that the lens would adjudicate REPRESSION against
FORECLOSURE. **It cannot, because foreclosure is already refuted.** Foreclosure
requires the signifier never to have been inscribed. Measured on the pole
prompts, 2,006 cells, pole-characteristic words (the base model's own top-10):

    mass in BASE on that pole prompt      0.455
    mass in ALIGNED on that pole prompt   0.476     aligned retains 105%
    cells where aligned < 10% of base      0.0%

The aligned model holds the pole signifiers in full, slightly more concentrated
than base. Nothing is un-inscribed. So one arm of the pair is dead before any
lens runs, and the theory word was doing work the instrument cannot.

**Keep this as a fracture rather than hiding it: foreclosure has no clean
computational analogue when the signifier is demonstrably available two prompts
away.** That is the method working -- the theory strains against the data.

What is live is that the suppression is CONTEXT-SPECIFIC. The pole words are
available; they are not available *in this scene*. That is repression proper,
and the open question is the mechanism and DEPTH of the barring:

    LATE GATE          the pole continuations rise through the stack on the BOTH
                       prompt and are cut in the last layers
    EARLY RE-ROUTING   they never rise; the representation went elsewhere from
                       the start

Those are different claims about what alignment IS -- a mask over an unchanged
computation, or a changed computation -- and they carry the political-economic
weight, since a late gate is cheap, reversible and cosmetic.

## 2. Why the two existing instruments cannot answer it

`contradiction_ratio_has_no_null.md`: the JS ratio is a single scalar at the
OUTPUT. It cannot see depth at all, and it cannot separate superposition from
neutralization without the null this campaign had to compute for it.

`pole_axis_t_is_not_superposition.md`: `t` is a one-dimensional projection onto
the pole axis, `resid` ~ 0.95, and a midpoint in logit space would give the
GEOMETRIC mixture while the models measurably do the ARITHMETIC one (union,
46 of 46 lineages). `t` scores the kind of "both" they do not do.

**The lens is the first instrument in this campaign that is both depth-resolved
and in the same space as the finding** -- next-token mass, where the union lives.

## 3. The measure

For group `g`, per arm, per layer `L`:

    A-set = top-k of the POLE_A prompt's final-layer distribution, minus the
            intersection with pole_b's top-k
    B-set = the same for POLE_B

    on the BOTH prompt, at layer L:
        A_mass(L) = sum of P_L(w) over A-set
        B_mass(L) = sum of P_L(w) over B-set

**Two masses, tracked separately, never subtracted into one number.** That is the
whole point: the three states are distinguishable only as a pair.

    superposition    both elevated
    resolution       one elevated
    NEUTRALIZATION   neither

The JS ratio collapses these to one scalar and so cannot separate the first from
the third; that is the defect this measure exists to fix, and it is the same
"plot the pair or not at all" rule the pen wrote for `ratio`/`pole_sep` and for
poetic texture's floor.

## 4. The null, which this measure needs as much as the ratio did

"A_mass is elevated at layer 12" means nothing absolutely. The null is the one
already built in `contradiction_null.py`: **A-set and B-set taken from a
DIFFERENT live group, scored on THIS group's BOTH prompt, at every layer.** It
gives the per-layer mass an unrelated set of pole words attracts, which is the
floor any claim of "elevated" must clear.

Same construction, same content-disjointness requirement, and the same known
result that the stricter same-frame null runs slightly HIGHER, so it is
conservative.

## 5. Two hazards, both to be written into the run and not discovered

**THE LENS IS A READOUT IN THE OUTPUT BASIS.** Projecting an interior residual
stream through the unembedding gives what the output would be if you stopped
there -- not what the layer encodes. Early layers are not in that basis and the
raw lens is known to be biased there, which is why the tuned-lens literature
exists. **Consequence: no absolute early-layer claim. The base model's own
trajectory on the same prompt is the paired reference, always.**

**THE PRE/POST-NORM SEAM.** `hidden_states[-1]` is post-norm and every other
entry is pre-norm. `malign_logits/models.py:logit_lens` applied the final norm to
ALL of them and so double-normed the last -- Amber `kill` 0.119 against 0.060, a
factor of two at the only layer anyone reads. Fixed 2026-08-09 (see
`plan_h_logitlens.md` §1) and the fix REFUSES when the final layer's projection
disagrees with the model's own logits. **Use `LayerReadout`, never a hand-rolled
projection**, and let the refusal fire.

## 6. Substrate: no new forward passes are needed for most of the roster

`.hidden.f32` already holds the residual stream at the final position for EVERY
layer, 121 models, 74 GB. A lens read is that vector times the unembedding, so
the depth profile is computable from bytes on disk for any model whose weights
can be loaded for the head alone. The full-battery captures in
`data/raw/twp_fill/` make this wider than the quintuplet block.

This is a property to exploit, not a claim -- confirmed only for the pilot.

## 7. INPUT, NOT SETTLED -- RH's

Recorded as candidates, not decisions:

- **Population.** The 43 live quintuplet groups, or the 22 English ones. The zh
  half is a weaker instrument on an English-heavy roster ([5093]).
- **Roster.** The 46 lineage representatives is the defensible unit; 52 arms is
  not, and 121-models-with-hidden-states is a coverage fact rather than a
  population.
- **k**, the size of the pole sets. Unset. It must be declared before running,
  not tuned: a k chosen after seeing the curves is a threshold chosen on the
  outcome.
- **Token or word level.** The ratio is word-level (twp, `rule_version 3`); a
  lens is natively token-level. Mixing them silently is the kind of unit word
  this campaign has spent a week on. The pilot declares TOKEN level for
  simplicity and the full run must choose deliberately.

Under [5148] this clause needs the enumerated list in a file with its hash, the
roster, and nothing defined by a tool.

## 8. THE PILOT -- scope set by RH, and it SELECTS ON THE OUTCOME

One model pair, one triple:

    LLM360/Amber -> LLM360/AmberSafe        both cached locally, 13 GB each
    f11_loyal    The soldier was loyal and chose to              pole_a
                 The soldier was rebellious and chose to         pole_b
                 The soldier was loyal and rebellious and chose to   both

    ratio  base 0.725   aligned 1.580     (neutralization null 1.006)

**This is the largest base-to-aligned superposition collapse in the roster, and
that is why it was chosen.** Selecting the most favourable case is legitimate for
an instrument shakedown and illegitimate as evidence, so: the pilot answers "does
the measure resolve anything at all, and does the refusal fire", and it answers
nothing about contradiction. Whatever it shows, the population question is §7's.

Amber's aligned arm is also past neutralization and heading toward resolution,
which makes it atypical twice over -- most aligned arms sit AT the null. Any
generalisation from it is unwarranted by construction.

Six forward passes, local, free.

### 8a. WHAT THE PILOT ESTABLISHED (run 2026-08-11; results/lens_pilot_contradiction.csv)

**Readout verified on both arms.** `head_err` 8.46e-06 base, 2.09e-05 aligned --
`head(hidden[-1])` IS the model's logits and the `FinalReadout` identity holds,
so the interior layers are read under a convention the final layer proves.

**Cost, and it is asymmetric in the direction that hurts.**

    base     cost_vs_twp   6.1x        five layers, not 5x -- the union
    aligned  cost_vs_twp  10.4x        frontier overhead is mild

A flatter distribution keeps more prefixes above theta at each pass, so the
ALIGNED arm costs 70% more. **A budget priced on base arms alone underestimates
by that much**, and the aligned arm is the one the question is about.

**The layer set must exclude the embeddings, and that is a cost fact not a taste.**
`expand_layers` walks a frontier that is the UNION of live prefixes over the
requested layers, so cost is set by the MOST DIFFUSE layer in the set. Asking for
all 33 did not return in 7.5 minutes on ONE prompt; five layers without layer 0
runs in ~6 minutes for five prompts.

**THETA IS THE FLOOR ON DEPTH, AND ONLY BELOW HALF-DEPTH.** The first reading of
this pilot was that theta=0.001 made every interior zero uninterpretable. That is
true at depth <= 0.25, where `tail` is 0.65-0.70, and FALSE where it matters:

    depth 0.75   tail 0.1006 base / 0.0219 aligned   211 / 239 words above theta

Those layers are 90-98% accounted for and the pole words are genuinely ABSENT,
not truncated. So the measure resolves from depth ~0.5 up, and any claim below
that needs a lower theta or a direct token lookup with no threshold.

**AND THE FINDING THE PILOT WAS NOT LOOKING FOR: the A/B decomposition and the
JS ratio DISAGREE on the same cell.** `f11_loyal` base has ratio 0.725 -- strong
superposition against a 1.006 null -- while at the same final layer A-mass sits
2.7x above its null and B-mass 0.65x BELOW its null, which reads as resolution
toward `loyal`. Both arms.

They disagree because of SCOPE. The ratio compares the whole distribution to the
blend; A/B mass reads the top-k disjoint words, here 8 and 8. A distribution can
be blend-like overall while its top B-words are suppressed. **So this measure is
not a finer-grained ratio, it is a NARROWER one**, and §7's unset `k` is the
parameter that decides whether it can see what the ratio responds to. The pilot's
instruction to §7 is: do not set k small, and justify whatever k is set against
the mass the ratio is actually integrating.

**Three defects of mine, caught here rather than in a fleet:** token space where
the campaign is word space; `expand_layers` returns `words[(surface, FIRST
TOKEN)]` and the partition MUST BE SUMMED (the ClickHouse table hides this, since
the summation happened before ingest); and a display filter left over from the
33-layer draft that dropped two of five computed layers from the printout AND the
CSV, because the append sat inside it.
