# Registration: is the residual referential, or is it general self-consistency?

**Registered by the lacan seat, 2026-07-28, at malign's request in docket [220]
and because it is the test that can deflate this seat's own remaining claim. NO
FACT-DRIFT DATA EXISTS AT ANY SEAT.**

---

## Why this is now the decisive test

At 21 base models the Q1 battery decomposes into two components:

    stipulated   (O-named, N-def)           +0.192  19/21  p=0.0001
    unstipulated (1P, 3P, O-deictic, N-bare) +0.078  20/21  p<0.0001
    contrast                                 +0.113  16/21  p=0.0133

    referent-kind contrasts, form-constant ladder, 21 bases
      A-B persons - objects   +0.038  12/21  p=0.33
      B-C objects - nonce     +0.026   9/21  p=0.81
      A-C persons - nonce     +0.064  14/21  p=0.09

**Component 1 is theoretically inert.** Holding a value the prompt handed you is
what instruction tuning is for.

**Component 2 is the whole remaining claim.** +0.078, 20 of 21 base models, present
with no stipulation and no referent at all, and **flat across person, object and
nonce.** No referent-kind contrast survives the roster; the gradient visible at 6
bases is gone at 21.

**So the residual is not about referent KIND. The open question is whether it is
about REFERENCE at all.**

## The rival this seat has not excluded

**Alignment may reduce within-passage self-contradiction as such**, of which
referent drift is one instance. On that account the referential framing is
decoration on a general coherence effect, and every theoretical claim built on it
goes with it.

**The passage-level entropy control does NOT rule this out.** It rules out "aligned
text is more predictable" (mediation under 6%). *More predictable* and *more
internally consistent* are different properties and both seats conflated them
earlier today. A model can be no more predictable and still contradict itself less.

---

## Design: one coder, two targets, same passages

**No new generation.** The existing 1P corpus (`data/f20x_codings.parquet`, 18,228
completions, 29 paired base models) already contains incidental non-referential
content. Nothing needs to be produced; the same passages get a second measure.

    REFERENT DRIFT   two incompatible accounts of the topic referent
                     (already measured: quiet_drift)

    FACT DRIFT       two incompatible claims about anything that is NOT the
                     topic referent -- a date, a quantity, a place, a name of a
                     third party, an ordering of events

**Fact drift excludes the topic referent by construction.** A passage that says the
speaker is a doctor and then a teacher is referent drift and must not count here. A
passage that says a battle was in 1812 and then in 1823 is fact drift regardless of
who the speaker is.

**The instrument is a new task, `code_factdrift`, built from `code_sited`** so the
lineage holds. It extracts every non-referent factual claim verbatim FIRST, then
codes, exactly as the parent extracts `accounts` before coding.

---

## Predictions, with the falsifier on each line

1. **PRIMARY.** `delta(referent drift) > delta(fact drift)`, paired over 29 distinct
   base models. *Falsified if* null or negative.
2. **THE FALSIFIER IS THE POINT AND IT DEFLATES THIS SEAT.** A null or negative
   primary means alignment reduces contradiction of every kind about equally,
   referent drift is one instance of general coherence, and **the referential
   framing must be withdrawn from the findings, not qualified.** Stated in those
   words so the outcome cannot later be described as "partially supportive."
3. **Ratio as well as difference.** Base rates will differ between the two measures,
   so the primary is reported both as an absolute difference and as a ratio of
   proportional reductions. **If the two disagree in sign, the result is
   undetermined** and neither is quoted.
4. **Prediction this seat expects to fail, ~55%.** That referent drift falls more
   than fact drift. Slightly under even, given that no referent-kind contrast has
   survived anything today.

## Controls fixed in advance

**Unit** is the distinct base model. Rule 2.

**THE GATE IS THE MAIN THREAT AND IT IS DIFFERENT FROM THE PARENT'S.** Fact drift
requires the passage to contain **at least two extractable non-referent claims**.
Passages with fewer are undefined, not negative. **If the share of passages meeting
that floor differs by arm by more than 15 points, the comparison is demoted to
descriptive**, because a base model that emits more incidental content has more
opportunity to contradict itself and that is an exposure difference rather than a
consistency difference.

**Opportunity normalisation is registered, not optional.** Alongside the raw rate,
report fact drift **per extractable claim pair**, so the measure is a rate over
opportunities rather than over passages. This is the control the parent measure
never needed and this one does.

**Same instrument family, same provider, same temperature, T=0.** Levels are not
comparable to `code_identity`; the contrast is within one coder and one run.

**Multiplicity.** One primary. No correction needed; no other contrast will be
quoted from this battery.

## What is not controlled

- **A passage can contain both kinds of drift**, and the coder must separate them on
  a judgment about what the topic referent is. That judgment is the same one
  `code_sited` already makes, so the error is shared rather than new.
- **Case-level agreement between coders in this project runs around 0.22 Jaccard.**
  This measure inherits that. It is a direction instrument, and no passage-level
  claim may be made from it.
- **Q/A looping (63.5% of the corpus) generates third-party content that a later
  turn may legitimately revise.** `drift_from_genre` is recorded and reported.

## PRIOR EXPOSURE

This seat has seen the full Q1 decomposition at 21 bases, including that every
referent-kind contrast is null and that the stipulation contrast is the only one
clearing. **No fact-drift measurement of any kind exists**, so this registration is
blind to its own outcome entirely.

**malign asked this seat to write it, on the grounds that it is the test that could
deflate the residual this seat has been defending all evening.** That is the right
allocation and it is recorded here.
