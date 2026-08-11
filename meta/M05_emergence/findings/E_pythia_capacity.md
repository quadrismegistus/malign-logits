# Finding M05-E: the Pythia ladder resolves the tie — packages first, facts after phrases, discourse last (cross-ladder)

Written 2026-08-11 by the registrar seat. STATUS: FIRST LOOK, grade C —
one lineage (Pythia-6.9b), descriptive, instruments copied from the OLMo
ladder so the two are read by one rule; no cross-seat audit. A SEPARATE
STUDY from M05's OLMo population and never pooled with it ([5425](b),
[5430]): different lab, tokenizer (50,277 vs 100,278) and corpus. Every
comparison below is CROSS-LADDER and labelled so. Re-derives from:

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_curves.py \
        --population data/pythia_population.json --out data/pythia_curves.parquet
    uv run python meta/M05_emergence/scripts/m05_pythia_capacity.py

Readable exhibits (top of the distribution, both ladders, selected rungs,
two probes per family): `../results/capacity_examples.md`
(`scripts/m05_capacity_examples.py`).

Population: `data/pythia_population.json`, 155 checkpoints (154 pretraining
rungs, log-spaced 0,1,2,4,...,512 then 1000-step to 143000, plus `main`),
malign's fleet [5430], 90,170/90,170 cells. Battery: the same 584 texts by
declaration (sha e5a4f5f). Curves: `data/pythia_curves.parquet`, 80,290
rows. Results: `results/pythia_capacity_onsets.json`. Onset criterion
copied from `m05_onsets.py`: first rung whose bootstrap CI of the median
log p(target)/p(competitor) sits above zero and stays there to the end of
the arm; plus a coverage gate (>= 10 surviving probes) because Pythia's
early rungs are absent-dominated and a sign-only onset can fire on two
cells at imputation scale.

## Result 1: the four-way tie OLMo left-censored now resolves

On OLMo, packages, reference, reasoning and poetic pull all onset at the
first usable rung (stage1-2000) — a tie the release schedule could not
break, because Allen AI's finest granularity is 1,000 steps. Pythia's
log-spaced early rungs break it:

    family              Pythia onset (gated n>=10)     OLMo onset (cross-ladder)
    semantic packages   step 2000  (n=31 at onset)     stage1-2000  } four-way
    poetic pull         step 2000  (sign-onset 512,    stage1-2000  } tie,
                                    but at n=2)        stage1-2000  } left-
    reasoning           step 3000                      stage1-2000  } censored
    reference (facts)   step 4000                      stage1-2000  }
    discourse tracking  step 80000 (see Result 3)      stage1-32000

The ordering that emerges — packages before reasoning before reference —
is the one the OLMo half-max milestone already suggested (packages 4k <
reasoning 6k < reference 11k, POST-HOC), now visible at onset grain on a
second lab's ladder. The phrase reaches significance before the fact on
both ladders: Pythia half-max packages 2000 vs reference 12000 (POST-HOC,
same direction as OLMo's 4k vs 11k).

![Capacity acquisition on the Pythia ladder](../figures/fig14_pythia_capacity.png)

## Result 2: the sub-1000 window has words before it has capacities

[5430] found the battery's vocabulary rising eight-fold between step 8 and
step 128 (words resolved per cell 5 -> 66). At the capacity probes, that
early vocabulary buys nothing yet: through step 512 the cells are
absent-dominated (per-rung medians undefined or carried by a handful of
probes; poetic's sign-onset at 512 rests on n=2 pairs at imputation
scale). The first rung where any family clears the coverage gate with a
stable positive contrast is 1000-2000. At battery grain, the window
acquires WORDS first and measurable CAPACITIES only at its top edge —
which is also where [5426]'s pole-separation collapse has already
happened. Cross-instrument, one lineage, descriptive.

## Result 3: discourse tracking is last on both ladders, and weakest

Discourse is the only family whose Pythia onset lands an order of
magnitude after the others (80000), replicating its OLMo position
(stage1-32000, alone after the tie) cross-ladder. The exact step is
criterion-sensitive and should not be quoted as a constant: discourse's
median contrast is positive from step 2000 (~+0.7 to +1.3) but its CI
keeps dipping through zero across mid-training (dips at 55k-79k), so the
persistent-onset criterion fires late. The robust cross-ladder claim is
ordinal and doubled: discourse last, and discourse smallest at ceiling
(~+1 nat vs packages ~+6.5).

## Weatherby note (cross-ladder, both lineages)

Semantic packages are the earliest-onsetting and largest-magnitude
capacity on both ladders: the "predigested form" is the first and
strongest thing pretraining learns at this battery, with reference facts
arriving later and lower. The trajectory ordering is now a two-lab fact at
onset grain (this doc) and half-max grain (A-R2, POST-HOC), not an OLMo
quirk.

## Artifacts on the record, not quotable

- Discourse half-max "step 128" (in `pythia_capacity_onsets.json`): a
  noise spike — rung 8's median (+1.44) rides a handful of surviving cells
  and the trajectory goes NEGATIVE at 256-1000 before the real climb. The
  half-max metric fired on the spike. Recorded so nobody quotes it.
- Poetic sign-onset 512: n=2, 92% of targets absent at that rung. The
  gated onset (2000) is the honest number.

## Caveats

One lineage; grade C; the onset criterion inherits m05_onsets.py's
persistence requirement, which is conservative where a curve is noisy
around zero (discourse) and permissive where coverage is thin (hence the
gate). Absent-word imputation at theta/2 travels from the extractor;
absent rates by curve are printed in the extraction log (packages 29%,
reference 21%, reasoning 22%, discourse 13%, poetic 11% over the full
ladder, concentrated early). The alignment endpoint arms (lomahony
sft/dpo, scored at step 143000 only) are NOT read here — per-rung
alignment contrasts are impossible on this family and the endpoint
contrast is its own future analysis.
