# Positive-control slate v2 — attestation, one line per marker

Registered 2026-07-27, **before any `D` is computed for any pair below**. This
discharges the last outstanding precondition. It supersedes
`docs/preference_corpus_markers.md`, whose seven markers are all burned.

## Why this document exists separately

The v1 gate failed because six of seven markers did not meet the registered
frequency-comparability requirement. The requirement was noticed, written into
the declaration as a "registered weakness", and the run proceeded — a condition
**logged instead of met**. Rule 9 was written from that incident: *for every
stated precondition, does the run record show it SATISFIED, not merely
ADDRESSED.* So each marker below carries all three requirements shown
individually, not asserted collectively.

## One marker per construct

The v2 seven-pair slate was withdrawn: its pairs were three near-synonymous
constructs in blocks of 3/3/1, and correlated markers overstate power by 0.212
and understate false certification by 74×. Capping at one marker per construct
restores independence **by construction**, which is why the construct grid needs
no blockwise estimator.

Three constructs survived the enumeration of 62 attested candidates. Within each,
the lowest-MDE pair is taken — a rule fixed before the pairs were ranked, not a
selection among them.

## The three markers

Requirements per marker: **(a)** attested on external grounds, stated
individually; **(b)** in band — both members within a factor of 3 of the median
chain-word frequency, 881–7,926 combined in hh; **(c)** powered — pair MDE below
the chain-relevant 0.174.

---

### 1. `require` → `prefer` — directive softening

**(a) External grounds.** RLHF helpfulness annotation rewards responses that
frame constraints as preferences rather than obligations; the InstructGPT
labelling guidance and the HH helpfulness criterion both treat unhedged
directives to the user as less helpful than offered alternatives. The
requirement/preference contrast is the canonical lexical form of that shift and
is independent of transgressive content, which is what the reroute chains
concern.

**(b) In band.** `require` 6,230 combined; `prefer` 5,526. Both inside 881–7,926. ✓

**(c) Powered.** MDE **0.092**, below 0.174. ✓

---

### 2. `entirely` → `mostly` — quantifier qualification

**(a) External grounds.** Preference data rewards calibrated over absolute
claims; hedged quantifiers are among the most consistently documented surface
features of RLHF-tuned output, and the absolute/partial quantifier contrast is
its clearest minimal pair. Unrelated to the chains, which are content
substitutions rather than quantifier choices.

**(b) In band.** `entirely` 1,982; `mostly` 4,235. Both inside. ✓

**(c) Powered.** MDE **0.136**, below 0.174. ✓

---

### 3. `angry` → `concerned` — emotional de-escalation

**(a) External grounds.** Harmlessness annotation rewards de-escalated affect
attribution; the HH rejected set is characterised by heightened emotional
register, and substituting a measured affect term for a heated one is the
documented lexical signature. Independent of the chains, whose targets are not
affect terms.

**(b) In band.** `angry` 1,912; `concerned` 4,771. Both inside. ✓

**(c) Powered.** MDE **0.135**, below 0.174. ✓

---

## What this slate cannot do

**Three constructs give two usable rungs**, k=2 and k=3, with nothing between.
P(≥2 of 3) and P(3 of 3) are far apart, so whichever cell the objective selects
clears its criteria **without being near an optimum**. That is a granularity
limit, not a validity one, and it is stated here rather than discovered later.

**Category coverage is three of seven enumerated.** Hedging/epistemic,
pejorative→neutral, refusal softening and blame→attribution yielded no valid
pair — pejorative→neutral had two in-band pairs that both failed power. So this
slate tests directive softening, quantifier qualification and emotional
de-escalation, and a gate result licenses claims about **those three**, not
about "register preference" in general.

## Burned, and excluded from any future slate

`sorry`→`unfortunately` (pre-spec). All seven v1 markers: `must`→`should`,
`never`→`rarely`, `always`→`often`, `wrong`→`incorrect`, `stupid`→`unclear`,
`no`→`unfortunately`, `obviously`→`perhaps`. Also burned from the withdrawn v2
seven-pair slate, since their `D` values were never computed but their selection
was published: `totally`→`fairly`, `completely`→`largely`, `force`→`encourage`,
`demand`→`request` remain **unburned** — no `D` was computed for any of them —
but `force`→`encourage` and `demand`→`request` lost their construct slots to
`require`→`prefer` on the lowest-MDE rule and are available as replacements if
any marker above fails its attestation on review.
