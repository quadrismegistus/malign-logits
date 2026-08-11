# M02 TODO

Written 2026-08-11. One item, and it is ready to run.

---

## THE DEPTH x EXIT JOIN IS RUNNABLE NOW. THE SUBSTRATE EXISTS AND I MISREAD IT ONCE.

### The question

The JS ratio licenses "the continuation is unrelated to the poles" and **not**
"the model exited the frame" -- that is a claim about the destination, and only
the marker instruments can make it. The two have never been joined.

The lens makes the join sharper than a ratio-vs-marker check would be:

> **Does the DEPTH at which the base and aligned arms diverge predict whether
> the continuation exits the frame?**

A late gate and an early re-routing are different claims about what alignment
IS. If they also produce different surface symptoms, the lens is measuring
something the output cannot see on its own. If they do not, the depth story is
about the readout rather than the computation, and that is worth knowing before
anything is written about cheap-reversible-cosmetic alignment.

### The substrate, and the correction

`results/exit_contradiction_cells.csv` is a **first look at 3 generations per
cell** -- 120 joinable cells, 360 generations, `E-ASSIST` firing ONCE. I read
that file, concluded the join was ~30x underpowered, and deferred it. **That
file is a sample, not the corpus.**

The corpus is in ClickHouse:

    gen_sequences, corpus='f11_l2'    228,520 passages
                                       58 models, 11,426 (model,prompt) cells
                                       20 generations per cell, uniformly

    models shared with the lens        50   (not 12)
    lens-model x group triples with
      all three roles generated      2,150
    passages behind them           129,000

**18x the cells and 7x the generations per cell of what I looked at.** The
lesson is the campaign's own and I paid it again: an artifact built for one
purpose is not the store. Check the store.

Note also that `source='QUINTUPLETS'` returns 0 rows from the DB catalogue
while the JSON lists 42 texts under it -- one of the JSON/DB divergences
recorded in `M03/plans/plan_c_reference_class.md` §2a. **Match quintuplet
material by TEXT from `data/f11_quintuplets.json`, never by the DB source
label.**

### The instrument, already declared

`scripts/y_exit_typology.py` holds `TYPES` as compiled regexes -- E-QUIZ, E-QA,
E-TASK, E-ASSIST, E-MENTION, E-META -- and `exit_contradiction.py` records that
they were **copied verbatim and declared before reading**. Reuse them; do not
re-author. REFUSAL is declared a priori and is always reported apart, as the
dissociation readout.

### The shape of the run

1. apply `y_exit_typology.TYPES` to the 129,000 `f11_l2` passages, per
   (model, group, role); exit rate per cell, REFUSAL beside it
2. join to the lens depth profile per (model, group) from
   `results/lens_group_layer.jsonl` -- the per-depth base/aligned gap and the
   top-eighth share from `lens_analysis.py`
3. ask whether the depth of divergence predicts the exit rate, at the LINEAGE
   unit and not the cell unit

### Things to get right, from today's damage

- **The unit is the lineage.** `lens_analysis.py` reports 38 pairs; the exit
  side must be aggregated the same way. Cells and rungs are not observations:
  the ICC of a paired contrast across rungs on comparable material is 0.85.
- **`excess(BOTH) - mean(POLE_A, POLE_B)`** is the declared form of the marker
  contrast in `exit_contradiction.py`. Use it rather than a bare BOTH rate.
- **Exit rates are low.** Even at 20 generations a cell rate is coarse; pool to
  the lineage before testing and print the denominator.
- **zh and en apart.** The lens holds 43 groups over both languages and zh is
  47% of its rows; the M02 convention is English-primary.
