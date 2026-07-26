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

**Residue that survives independently of the band.** Three hedging markers
misbehave across three settings: `sorry`→`unfortunately` runs backwards in both
corpora (disqualified pre-spec), `no`→`unfortunately` fires in pku and runs
backwards in hh, and `must`→`should` runs backwards in both. All three are far
outside the band, so this is suggestive rather than established — but a hedging
preference that reverses between two preference corpora is worth its own look.

## Registered weakness of the slate

Only one of the seven pairs has both members inside the frequency-comparability
band. Register markers are systematically commoner or rarer than the
mid-frequency content words the chains are built from, so external attestation
and frequency-matching pull against each other. This was recorded in the marker
declaration *before* the results and was deliberately not used to reselect. It
is a real limitation on the gate's sensitivity and it cuts toward the
insensitivity conclusion rather than against it.
