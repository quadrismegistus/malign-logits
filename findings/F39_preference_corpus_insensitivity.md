---
status: verified
grade: A
date: 2026-07-26
role: finding
description: "Registered preference-corpus test of the convention account. The gate failed in both corpora, so the registered outcome is the instrument-insensitivity finding and NOT a verdict on convention. Measured on: hh_rlhf and pku_saferlhf chosen/rejected unigram tables."
instruments: [logit-mass]
chapters: [ch09]
data: ["f37_corpus_unigrams_hh_rlhf_chosen_v2.csv", "f37_corpus_unigrams_hh_rlhf_rejected_v2.csv", "f37_corpus_unigrams_pku_saferlhf_chosen_v2.csv", "f37_corpus_unigrams_pku_saferlhf_rejected_v2.csv", "preference_corpus_results.json"]
scripts: [preference_corpus_test.py, tier2_power_check.py, tier2_gate_calibration.py, tier2_gate_grid.py]
---
# F39: The chosen/rejected unigram instrument cannot detect lexical annotator preference

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

## What this licenses, and what it does not

**Licensed.** The chosen/rejected unigram instrument is not sensitive to lexical
annotator preference. This was pre-registered as an outcome rather than a
nullity, with its reasoning stated in advance: a whole response is chosen for
many reasons at once, and any single word's contribution is diluted by
everything else in the text. Tier 2 cannot carry the verdict **for any marker
set** — a better word list does not fix it.

**Not licensed.** Nothing about the convention account. It is not confirmed, not
excluded, and not weakened. The chain-pair sign test was **not computed**: the
spec makes the verdict unreachable through a failed gate, and computing it
anyway would have meant knowing a number the design says must not count.

## The successor instrument, named in advance

The spec registered where to go if this happened: a **response-level** design —
per-response presence/absence of the source and target words, with the word pair
as the unit — rather than a token-rate comparison. That measures what an
annotator actually chose between (whole responses) instead of a rate diluted
across everything else in them.

## Registered weakness of the slate

Only one of the seven pairs has both members inside the frequency-comparability
band. Register markers are systematically commoner or rarer than the
mid-frequency content words the chains are built from, so external attestation
and frequency-matching pull against each other. This was recorded in the marker
declaration *before* the results and was deliberately not used to reselect. It
is a real limitation on the gate's sensitivity and it cuts toward the
insensitivity conclusion rather than against it.
