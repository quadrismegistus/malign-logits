---
status: verified
grade: B
date: 2026-07-27
role: finding
description: "Registered rebuild of the preference-corpus gate on a validly constituted three-construct slate. The gate failed 0/3 in hh_rlhf, and the failure is a BOUNDED NEGATIVE rather than a non-detection: every marker's 95% interval excludes the 0.174 effect the design required, the largest upper bound at 0.80x of it. No verdict on convention follows and none is available. Measured on: hh_rlhf chosen/rejected unigram tables; pku_saferlhf descriptive-only."
instruments: [logit-mass]
chapters: [ch09]
data: ["f37_corpus_unigrams_hh_rlhf_chosen_v2.csv", "f37_corpus_unigrams_hh_rlhf_rejected_v2.csv", "f37_corpus_unigrams_pku_saferlhf_chosen_v2.csv", "f37_corpus_unigrams_pku_saferlhf_rejected_v2.csv", "preference_corpus_gate_v2.json", "preference_corpus_exclusion.json", "tier2_full_frontier.csv", "tier2_power_bounds.csv"]
scripts: [preference_corpus_gate_v2.py, preference_corpus_exclusion.py, tier2_construct_grid.py, tier2_full_frontier.py, tier2_power_bounds.py]
supersedes: "the v1 and v2 gates, both invalidly constituted (see history below); the earlier instrument-insensitivity finding, which was withdrawn and is now re-established on a valid slate and in bounded form"
---
# F39: `hh_rlhf` does not encode register preference at the scale the chain analysis required

## Summary

The convention account — that shared public preference corpora installed the
reroute chains — was the last surviving explanation after pretraining contiguity
(Set-D) and structural semantic relation (the role test) were excluded. Testing
it required first certifying the instrument: a chosen-vs-rejected unigram
log-odds statistic must be shown to detect a preference known to be there before
it can be trusted to report one that might not be.

**The gate failed, 0 of 3, and the failure is bounded rather than merely
negative.** Every marker's 95% interval excludes the effect size the design was
built around. That is a result about `hh_rlhf`, not a result about the
convention account, and the convention account remains untested.

## The registered gate

Three markers, one per surviving construct, declared with attestation,
frequency-band membership and power shown individually in
`docs/preference_corpus_markers_v2.md` before any `D` was computed. A marker
fires iff `D > 0` and `|D|` exceeds the p50 decoy floor — the floor taken from
the same decoy pool the false-certification and power rates were simulated on.
Gate passes at 3 of 3. Script committed at `5e0e5a3` **before running**.

| marker | construct | `D` | SE | 95% interval | verdict |
|---|---|---|---|---|---|
| `require` → `prefer` | directive softening | +0.0178 | 0.0370 | [−0.055, **+0.090**] | directional, below floor |
| `entirely` → `mostly` | quantifier qualification | +0.0332 | 0.0544 | [−0.073, **+0.140**] | directional, below floor |
| `angry` → `concerned` | emotional de-escalation | +0.0234 | 0.0541 | [−0.083, **+0.129**] | directional, below floor |
| | | | | | **0/3 — GATE FAILS** |

Floor `p50 = 0.0481`. False certification at this cell **0.01414** (95% upper
0.0181), an order of magnitude inside the 0.10 standard, so the margin clause is
not triggered.

## Why this is a bounded negative and not a non-detection

**The distinction is the whole content of the finding** (lacan). A gate can fail
because the instrument cannot see, or because there is nothing of the relevant
size to see. Only the second licenses anything.

The design's alternative hypothesis is a specific number: the anchor, `hh`'s
median chain-pair MDE, `log(1.19) = 0.1740`. That is the effect the gate was
powered to catch and the effect the chain analysis would have needed. The
observed markers can be tested against it directly:

| marker | 95% upper bound | as a fraction of the 0.174 needed | z vs anchor | p |
|---|---|---|---|---|
| `require` → `prefer` | +0.090 | 0.52× | −4.22 | 0.00001 |
| `entirely` → `mostly` | +0.140 | 0.80× | −2.59 | 0.005 |
| `angry` → `concerned` | +0.129 | 0.74× | −2.78 | 0.003 |

**Every upper bound falls below the effect the design required**, the largest at
0.80× of it, and each marker is individually below the anchor at `p < 0.005`.
Pooling by inverse variance — which assumes a common effect across the three
constructs, and is therefore secondary to the per-marker intervals — gives
`D = +0.0228 ± 0.0266`, indistinguishable from zero, `5.67` SE below the anchor
(`p = 7 × 10⁻⁹`), with a 95% upper bound of `0.0750`, or `0.43×` the anchor.

**This answers the design's disclosed weakness rather than merely restating it.**
The gate ran at a cell whose power floor was met on the point estimate (0.80389
against 0.80) and **not shown** at the 95% bootstrap lower bound (0.759, sd 0.008
over 1,600 replicates, below 0.80 in all 8 disjoint blocks). So the design ran
toward error B — condemning a live instrument — more often than a
shown-satisfied floor would allow, and roughly a fifth to a quarter of failures
would be expected even against a true effect at the anchor. That is the scenario
the caveat protects against, and **it is excluded on the evidence at
`p = 7 × 10⁻⁹`**: an unlucky draw from a true effect of 0.174 does not produce
three markers whose intervals all sit below 0.174.

## What this books, and what it does not

**Books.** For `hh_rlhf`, on the three constructs tested: chosen-vs-rejected
unigram log-odds shows no register-preference effect at the scale a chain-pair
verdict required, and effects at that scale are excluded rather than undetected.

**Does not book: any positive claim about what *is* there** (lacan, and this is
a correction to my own first reading). All three markers are directional, and it
is tempting to read that as a real effect an order of magnitude smaller than the
design assumed. **That claim is withdrawn and should not be made from this
data.** All three `z` are under 1 (0.48, 0.61, 0.43); 3-of-3 directional is
`p = 0.125` one-sided, which is nothing. And it would use the same three numbers
in opposite directions — indistinguishable from zero when the point is that no
effect exists at the design scale, consistently signed when the point is that a
small real effect does. Those two readings are not jointly supportable from three
`z` under 1.

The reason this matters more here than it usually would: **the gate exists so
that a positive empirical claim cannot enter the ledger without a check, and the
gate just failed, so no check is available.** "Register softening is real but
small" is exactly the class of claim the apparatus was built to stop at exactly
this moment. The shape of what is there is left to an instrument that can see it.

**Does not book: a verdict on convention.** Unchanged, and for the original
reason. The response-level successor was registered as the consequence of
demonstrated insensitivity, and that consequence is now met: **insensitivity is
demonstrated for three marker classes in `hh_rlhf`; the successor is licensed for
`hh_rlhf`, with its scope set by its own registration when written.** "Licensed"
means only that the precondition for building it is satisfied. It still runs the
full sequence before any statistic is computed — **registration, adversarial
audit, then RH's go, in that order** (lacan) — and nothing here shortens it.

The distinction is not pedantry (desktop). An earlier draft wrote "licensed for
`hh_rlhf` at these three constructs," which is a **category slip**: the
constructs are classes of *markers*, and they scope what the insensitivity
demonstration covered. The successor tests *chain pairs* — a different
population, on a different registration. Carrying the marker constructs into its
license would quietly narrow an instrument that does not exist yet, and would do
so in a document written before anyone had decided what it should cover.

## Scope, stated as limits rather than caveats

- **One corpus.** `hh_rlhf` certified nothing; `pku_saferlhf` **cannot** certify
  and is descriptive-only — its chains are badly measured, its MDE threshold is
  2.3× laxer, and its power at this cell is 0.649. An earlier phrasing said
  "these preference corpora"; the plural is unsupported and contradicts the
  descriptive-only flag (rule 8).
- **Three constructs of seven enumerated.** Directive softening, quantifier
  qualification, emotional de-escalation. Hedging/epistemic, pejorative→neutral,
  refusal softening and blame→attribution yielded no valid pair at all. So this
  licenses nothing about register preference in general.
- **Two usable rungs.** With three constructs the available `k` are 2 and 3, far
  apart with nothing between, so the operating cell clears both criteria without
  being near either optimum.
- **The within-construct rule is contemporaneous with the ranking**, not prior to
  it, and eases certification in direction. It ranks on **MDE, a precision
  functional**, while `D` is a location functional of the *same four counts* — so
  MDE is **approximately ancillary** to effect location, which is weaker than the
  "carries no information about any `D`" an earlier draft claimed and is all the
  construction supports (lacan). Two functionals of one input set are not
  independent in general, and the question **cannot be settled empirically
  either way**: doing so would mean computing `D` for the pairs the rule did not
  select, which burn discipline forbids. What is factual and unaffected: no `D`
  was computed for any marker before selection, and none has been computed for
  the unselected candidates now. Fully stated in
  `docs/preference_corpus_markers_v2.md`.

## Descriptive only: the `pku` oddity

In `pku_saferlhf`, `require`→`prefer` (+0.1995) and `angry`→`concerned` (+0.3258)
both fire against a floor 2.8× larger, while `entirely`→`mostly` runs **hard the
wrong way** at −0.6094. Nothing is read off this — pku cannot certify — but the
heterogeneity is on the record, and it is a further reason the plural claim above
would have been unsafe even as a gesture.

## History: two invalid constitutions before this one

Retained because the failures are the reason the current slate has the
requirements it has.

**v1 (7 markers, p75, 3-of-7).** Six of seven markers fell outside the registered
frequency-comparability band (881–7,926 in hh). The shortfall was recorded in the
marker declaration and the gate was run anyway, treating a registered
precondition as a disclosure. It is a gate condition, and the gate was never
validly constituted. The superseded p75/3-of-7 layer was also executed in place
of the registered p80/5-of-7 — a deviation in the *certifying* direction. That
script is retained unmodified at `scripts/preference_corpus_test.py` as the
record of what v1 was; it is not the gate.

**v2 (7 markers, three constructs).** The seven pairs were three near-synonymous
constructs in blocks of 3/3/1. Correlated markers understate false certification
by 74× and **overstate power by 0.212** — the binding defect, since a gate at
0.64 power condemns a working instrument more than a third of the time. Capping
at one marker per construct restores independence by construction, which is why
the current grid needs no blockwise estimator.

**The general form, which is the durable methodological residue.** Decoy-based
calibration assumes candidates are independent draws matched to the decoy
distribution. v1 broke the *matching* and was therefore **conservative**; v2
broke the *independence* and was therefore **anti-conservative**. Any slate
property that breaks exchangeability with the decoys invalidates the rate, and
the direction depends on which property broke.

**One procedural note worth carrying forward.** In this run the standing rule
that a failed gate forbids the chain-pair sign test is enforced **in code** —
`preference_corpus_gate_v2.py` refuses tier 2 unless the gate passes — rather
than by a person remembering at the right moment.
