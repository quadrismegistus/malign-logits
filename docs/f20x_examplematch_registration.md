# Registration: are the 2×2's flat contrasts real, or measurement noise?

**Registered by the malign seat, 2026-07-28, before any example-matched coding
exists. Audited by lacan per the rotation. Timestamp booked by registrar.**

RH's objection, and it is a correction to this seat rather than a new experiment:
I compared the published 1P finding (28/29) against Q1's rung counts (11/20, 8/20,
13/20) and read the difference as being about referent kind. **Those are
measurements of different quality, and the comparison attributes to the phenomenon
what belongs to the instrument.**

---

## The asymmetry, stated in numbers

| | published 1P | Q1's 2×2 |
|---|---|---|
| coder | `code_identity` | `code_sited` |
| few-shot examples | **15** | **0** |
| human licence | 0.895 vs two-coder consensus | none |
| base-arm `quiet_drift` rate | 0.105 | **0.308** |
| 1P effect | +0.061, 28/29 | +0.084, 18/23 |

**The effect size is the same or larger under the unlicensed coder. What differs is
resolution.** `code_sited` flags three times as much, so each per-model estimate
carries far more noise, and noise flattens *contrasts between conditions* much
faster than it flattens the conditions themselves — which is exactly the pattern
observed: every rung individually significant, no contrast between rungs anything.

**So the null contrasts are consistent with two different worlds** and nothing run
so far separates them:

1. Referent kind genuinely does nothing, or
2. Referent kind does something and a zero-shot instrument cannot resolve it.

## Design

**No new generation.** Re-code the existing 2×2 completions
(`data/f20x_nonce.parquet`) with an **example-matched** instrument and recompute
the identical analysis.

**`code_sited_fs`** — `code_sited` unchanged in schema, system prompt, referent
handling and prompt-showing, with a **15-example set** added, matching
`code_identity`'s count. Nothing else differs, so the comparison isolates examples.

**The examples are drawn from the 1P VALIDATION SET and its two human codings**, not
from the 2×2 battery. That set is already spent as test material — it licensed
`code_identity` — so using it here burns nothing that is still live, and it keeps
the 2×2 completions unseen by the instrument that will code them.

**Examples cover all six referent kinds**, three person and twelve non-person,
because an example set drawn only from person passages would advantage the person
rungs and manufacture the contrast this test exists to check.

## Predictions, with the falsifier on each line

1. **PRIMARY.** `code_sited_fs`'s base-arm `quiet_drift` rate drops materially
   toward `code_identity`'s 0.105 — i.e. examples reduce over-flagging.
   *Falsified if* the rate is unchanged, which would mean the level difference is
   the schema rather than the examples and this test cannot do its job.
2. **THE QUESTION.** With per-model estimates less noisy, do the rung contrasts
   move? *If A−B, B−C, A−C remain null at ~10/20, the flat contrasts are real* and
   referent kind is not doing work. *If any contrast clears*, the earlier null was
   resolution.
3. **Prediction this seat expects to fail, ~35%.** That a contrast clears. Every
   referent-kind comparison has been null or reversed at every n tried today, and
   the stipulation contrast — which *did* clear on the same noisy instrument — shows
   the instrument can resolve a real difference when there is one.

**Point 3 is the load-bearing one.** `code_sited` found stipulation at +0.113,
p=0.013 while finding referent kind at 4/21 in the wrong direction. **An instrument
too noisy to see referent kind should have been too noisy to see stipulation.** That
is evidence the null is real, and it is the reason this test is registered as a
check rather than as a rescue.

## Controls fixed in advance

**Unit** is the distinct base model. Rule 2.

**Same completions, both coders.** The contrast is within-passage, so sampling
cannot explain any difference — the failure mode that cost two withdrawals on Q2
today.

**Both codings retained and reported.** Not a replacement: `code_sited`'s output
stands, and the paper reports the analysis under both instruments with their
licences named. If they disagree, the disagreement is the finding.

**Case-level agreement is measured, not assumed.** Three instrument pairs today
have shown converging marginals with ~20% case agreement. This pair gets the same
Jaccard reported, and a fourth instance would make that a general property rather
than a coincidence.

**Levels are not comparable across instruments.** Only within-instrument arm
differences are quoted, per the standing rule that a rate is instrument-relative.

## What is not controlled

- **The examples are this seat's selection.** They are drawn mechanically from the
  validation set's human-agreed passages, but which fifteen is a choice, and a
  different fifteen would give a different instrument.
- **An example-matched coder is not a licensed one.** Matching `code_identity`'s
  example *count* does not confer its human licence, which was earned on a subset
  containing no agreed instability and therefore never covered drift sensitivity
  anyway.
- **This cannot rescue a real null.** If contrasts stay flat, referent kind is done
  and no further instrument work on the 2×2 is warranted.
