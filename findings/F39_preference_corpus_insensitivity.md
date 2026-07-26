---
status: rescoped
grade: C
date: 2026-07-26
role: finding
description: "Registered preference-corpus test of the convention account. The gate failed, but six of seven markers did not meet the registered frequency-comparability precondition and the one that did fired in both corpora, so no finding books -- neither a verdict on convention nor instrument insensitivity. Measured on: hh_rlhf and pku_saferlhf chosen/rejected unigram tables."
instruments: [logit-mass]
chapters: [ch09]
data: ["f37_corpus_unigrams_hh_rlhf_chosen_v2.csv", "f37_corpus_unigrams_hh_rlhf_rejected_v2.csv", "f37_corpus_unigrams_pku_saferlhf_chosen_v2.csv", "f37_corpus_unigrams_pku_saferlhf_rejected_v2.csv", "preference_corpus_results.json"]
scripts: [preference_corpus_test.py, tier2_power_check.py, tier2_gate_calibration.py, tier2_gate_grid.py]
superseded_by: "none (rescoped in place -- the instrument-insensitivity finding is withdrawn; the test needs a slate of attested markers inside the comparability band)"
---
# F39: The preference-corpus gate could not be validly run

## Summary

The convention account — that shared public preference corpora installed the
reroute chains — was the last surviving explanation after pretraining
contiguity (Set-D) and structural semantic relation (the role test) were
excluded. It was tested as registered in `docs/preference_corpus_spec.md`,
after nine rounds of adversarial audit.

**The instrument failed its own positive control, so the convention account is
neither confirmed nor excluded.** The registered outcome is a finding about
method: a chosen-vs-rejected *unigram rate* is not sensitive to the kind of
preference it was built to detect.

## The gate

Seven marker pairs were declared in `docs/preference_corpus_markers.md` **before
any statistic was computed**, on the documented shape of RLHF annotation
(hedged, qualified, de-escalated over blunt, absolute, pejorative). A pair fires
if `D > 0` and `|D|` exceeds the corpus's p75 decoy floor. The gate passes at 3
of 7. All seven results, as registered:

| pair | hh_rlhf `D` | verdict | pku `D` | verdict |
|---|---|---|---|---|
| must → should | −0.0079 | wrong direction | −0.0391 | wrong direction |
| never → rarely | **+0.1104** | **FIRED** | +0.1161 | below floor |
| always → often | +0.0195 | below floor | −0.0197 | wrong direction |
| wrong → incorrect | −0.0699 | wrong direction | +0.0195 | below floor |
| stupid → unclear | **+0.1857** | **FIRED** | **+0.4454** | **FIRED** |
| no → unfortunately | −0.0315 | wrong direction | **+0.4710** | **FIRED** |
| obviously → perhaps | +0.0097 | below floor | −0.0535 | wrong direction |
| | **2/7** | **GATE FAILS** | **2/7** | **GATE FAILS** |

Three of seven run the *wrong way* in hh and three in pku. Only one pair
(`stupid`→`unclear`) fires in both. `no`→`unfortunately` fires in pku and runs
backwards in hh — the same marker family as `sorry`→`unfortunately`, which was
disqualified at spec-writing time for running backwards in both.

## AMENDED 2026-07-26: the gate was invalidly constituted; NO finding books

Two corrections arrived after the first write-up, one from each other seat. The
second overturns the conclusion.

**Desktop: the wrong gate parameters were used.** The spec's *final* firing rule,
set by the registered selection rule whose grid was committed at `596213c`
before it ran, is **hh p80 / 5-of-7 (threshold 0.1057)** and **pku p65 / 5-of-7
(threshold 0.2114)**. I executed the p75 / 3-of-7 layer, which the spec itself
marks as superseded. Recomputed under the registered gate the fired sets are
identical (hh 2/7, pku 2/7) and, with three wrong-way markers per corpus, the
ceiling was 4 against a threshold of 5 — the gate fails *a fortiori*. Outcome
unchanged, **but the deviation ran in the certifying direction**: under other
numbers 3-of-7 at p75 could have certified an instrument the registered gate
failed. The spec retains its own revision history in one document and I executed
the superseded layer.

**lacan: the slate did not meet a registered precondition, and this is decisive.**
The spec requires each marker to be *frequency-comparable to the chain pairs —
both members within a factor of 3 of the median chain-word frequency* (band
881–7,926 in hh). **Six of seven failed that requirement.** I recorded the
shortfall in the marker declaration and proceeded anyway, treating a registered
precondition as a disclosure. It is not a caveat; it is a gate condition, and
the gate was therefore never validly constituted.

**And the single compliant marker fired in both corpora.** `stupid` → `unclear`
(1,183 and 919 occurrences — the only pair inside the band) fires at **+0.1857
in hh** and **+0.4454 in pku**, clearing every floor under both the registered
and superseded gates, and is the only pair to fire in both.

| | in band | hh | pku |
|---|---|---|---|
| stupid → unclear | **yes** | **FIRED** | **FIRED** |
| other six | no | 1 fired | 1 fired |

So the one marker measured at the frequencies the chains actually occupy behaved
exactly as an attested annotator preference should, in both corpora. That is
evidence the instrument *works* at chain-pair frequencies — the opposite of what
the first write-up concluded.

**Booking, replacing the insensitivity finding.** What is certain: *this gate
could not certify the instrument.* What is **not** supported: that the
instrument is insensitive. The likely explanation for the failure is the slate,
not the instrument — six of seven markers were measured outside the frequency
range the test was specified for, and the one inside it fired twice.

Consequently:

- **No verdict on convention.** Unchanged, and for the original reason.
- **The instrument-insensitivity finding is WITHDRAWN.** It was inferred from a
  gate that did not meet its own precondition.
- **The response-level successor is not licensed by this result.** It was
  registered as the consequence of demonstrated insensitivity, which has not
  been demonstrated.
- **What the test needs** is a slate of seven attested markers *inside the
  comparability band*. That is genuinely hard: register markers are
  systematically commoner or rarer than mid-frequency content words, so
  attestation and frequency-matching pull against each other. That difficulty is
  the real methodological residue here, and it was visible in the declaration
  before the run.

### The one valid observation, booked separately

The gate's invalidity does not invalidate an individual marker's measurement.
`stupid` → `unclear` was attested, **in band** (1,183 and 919 occurrences), and
fired in **both** corpora at +0.1857 (z=2.11) and +0.4454 (z=1.97). It is the
only pair in the run measured under the conditions the spec required, and it
behaved as an attested annotator preference predicts.

**It is n=1 and certifies nothing.** The booking must not swing from "instrument
insensitive" to "instrument works" on one pair. It is recorded because a future
reader deciding whether to rebuild tier 2 needs to know that the single
compliant marker worked.

Two further measurements bear on the same question and were not part of the
gate's logic: `must`→`should` and `always`→`often` return **precise zeros**
(hh z=−0.32 and +1.07, SEs 0.025 and 0.018 — the two best-measured pairs). A
blind instrument gives wide intervals everywhere; this one gives tight intervals
around zero where it sees nothing and significant positives where it sees
something. So the gate failed because several markers were not in fact preferred,
not because the measurement cannot detect preference.

### Is a compliant slate constructible? Yes, but the yield is low

The methodological residue was whether attestation and frequency-matching pull
against each other so hard that seven attested in-band markers do not exist.
Enumerating 34 attested register-marker candidates (hedging, apology, softened
refusal, quantifier qualification, de-escalation) against the hh band:

| | count |
|---|---|
| below the 20-occurrence floor | 3 |
| attested but **out of band** | 26 |
| **attested and in band** | **5** |

`stupid`→`unclear`, `totally`→`fairly`, `crazy`→`unusual`, `demand`→`request`,
`force`→`encourage`. So a compliant slate is **constructible but expensive** —
roughly a 15% yield, meaning ~50 enumerated candidates for seven usable pairs.
One of the five is now burned by this run, leaving four.

**Tier 2 is not structurally dead.** The design failure was mine (running an
invalid slate), not the design's.

**Residue, corrected.** An earlier version of this section claimed three hedging
markers "run the wrong way" and treated that as suggestive. Computing `D/SE`
shows **no pair is a precise negative** — every wrong-direction value is a
precise zero or is indeterminate, and `no`→`unfortunately` is not a reversal but
a strong positive in pku (z=9.61) beside a null in hh (z=−0.82). There is no
anomaly to pursue. Both seats had proposed leaning on that pattern; it does not
exist.

## Registered weakness of the slate

Only one of the seven pairs has both members inside the frequency-comparability
band. Register markers are systematically commoner or rarer than the
mid-frequency content words the chains are built from, so external attestation
and frequency-matching pull against each other. This was recorded in the marker
declaration *before* the results and was deliberately not used to reselect. It
is a real limitation on the gate's sensitivity and it cuts toward the
insensitivity conclusion rather than against it.
