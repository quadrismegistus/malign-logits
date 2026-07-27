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
the **lowest-MDE pair is taken** — a rule **adopted contemporaneously with the
ranking**, and **outcome-blind by construction: counts only, no `D` computed for
any marker.** (RH's wording, 2026-07-27.)

The second clause is verifiable, not a mitigation offered on trust. MDE is
`2.49 × sqrt(1/c_s + 1/r_s + 1/c_t + 1/r_t)` — a function of four corpus counts
and nothing else. It cannot be evaluated on, and carries no information about,
the direction or size of any `D`. So while the rule cannot be shown to predate
the ranking (below), it also could not have been steered toward a wanted result:
the quantity it ranks on was computable before any outcome existed, and no
outcome existed to steer toward, since no `D` had been computed for any marker
on the slate and none has been now.

### The within-construct rule is contemporaneous with the ranking, not prior to it

An earlier draft of this document called that "a rule fixed before the pairs were
ranked, not a selection among them." **That claim is withdrawn: it cannot be
dated, and the dating runs the wrong way.** lacan applied the same check malign
passed for `POWER_MIN`, and the answer is unfavourable — the ranked table with
all seven MDEs is committed at `13f5efe`, 2026-07-26 22:54; the lowest-MDE rule
first appears in any committed artifact at `12df45a`, 2026-07-27 11:17, twelve
and a half hours *after* the ranking existed. `POWER_MIN` had eleven hours
*before* the cells. A search of the repository for any earlier statement of the
rule — `git log -S` over `docs/` and `scripts/`, the slate documents at
`778ea36` and `13f5efe`, and any code selecting by MDE — returns nothing. **No
prior dating exists.** The rule may sit undated in the peer record, as
`FC_MAX = 0.10` does; that is not evidence and is not offered as any.

**Direction, stated because it is against the gate's own purpose.** Lowest-MDE
selects the best-powered pair in each construct (`require` 0.092 over `force`
0.104 and `demand` 0.160; `entirely` 0.136 over `totally` 0.149 and `completely`
0.155). Better-powered markers are more likely to clear the floor, so the rule
makes the gate **easier to certify**, under an objective whose purpose is to make
certification harder. A highest-MDE rule would have given
`demand`/`completely`/`angry`, all still under 0.174.

**One correction to how the consequence was described, and it cuts both ways.**
The mechanism initially proposed for this — that better-powered markers raise
gate power, admitting more cells to the floor-clearing set and letting the
minimum-FC selection reach a lower FC — **does not operate**. The committed grid
(`20752e9`) computes both false certification and power from the decoy pool and
the corpus anchor; **no marker MDE enters it**. The cells, their rates and their
power are identical under every admissible slate, so the selection rule can
neither admit nor exclude a cell, and the ruled cell's 0.004 margin above the
power floor is untouched by it. What the rule does change is the *real* gate: which
three markers are actually tested. The easing is real; the route is through the
instrument, not the calibration.

**What bounds the consequence.** The grid's power figure assumes the markers
behave like decoy draws, so it is honest only if their sampling noise resembles
the decoys'. Against the 1,054-pair decoy pool's closed-form SE distribution, all
seven candidates sit in its **tighter half** — `require`→`prefer` at the 20th
percentile, `demand`→`request` at the 49th, the rest between. Every admissible
slate is therefore tighter than the pool the floor is calibrated on, which makes
the stated power **conservative** for any of them. The maximum harm from the
undated rule is the gap between a 20th-percentile slate and a 49th-percentile
one, both on the conservative side. That bounds it; it does not repair it, and
the rule stands disclosed rather than defended.

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

**A replacement must show all three requirements, not inherit two of them**
(lacan, finding 2). The clause above is correctly keyed — it triggers on
attestation failure, the same predicate as the requirement it protects — but as
first written it named replacements by power alone. `force`→`encourage` (0.104)
and `demand`→`request` (0.160) are powered; **their band membership is stated
nowhere**, and a swap on that basis could reintroduce exactly the
frequency-comparability gap that invalidated the first constitution, where six of
seven markers were out of band. So: **any replacement must independently show
(a) attestation on external grounds, (b) in-band counts against 881–7,926, and
(c) MDE below 0.174, each shown rather than asserted, before it enters the
slate.** A candidate's presence on the enumeration list is not a substitute for
(b): the list records that a pair was enumerated, not that its counts were
checked and displayed.
