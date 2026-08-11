# Alignment names the contradiction as a contradiction, and only the contradiction

**Status: the strongest positive result in M02.** On 52,559 exit-free English
passages, aligned models produce a second-order predicate over the contradiction
**2.18x** as often as their base arm — 20 of 22 lineages, p = 0.00012 — while
the same markers on single-pole prompts move **0.93x**, p = 0.82. The claim is
the interaction and the control sits at unity.

Producer `scripts/z_second_order.py`; marker sets declared in
`registrations/second_order_markers_v2.md`; substrate `gen_sequences`,
`corpus='f11_l2'`, first 50 words of each continuation.

---

## The construct, and why the obvious version of it is wrong

RH's Oedipalization slide sets two continuations of one prompt against each
other:

    BASE     "kill him and save him and make him suffer"
    ALIGNED  "cry and laugh all at the same time. She was TORN IN TWO
              DIRECTIONS ... Maybe she SHOULD FEEL GUILTY"

The difference is **not** that one mentions two opposed things. The base
exemplar does that more explicitly — three incompatible predicates against the
aligned one's none. The difference is that the aligned passage produces a
**second-order predicate**, one whose object is the conflicted condition
itself, which requires a vantage point outside it.

This is why the coded field `tension_named` failed. It asked for the two
opposed terms, and extraction of a pair cannot separate the exemplars: run the
base one through it and you get `kill`/`save`, a cleaner pair than the aligned
one offers. Measured precision was ~30% and — fatally — the errors were
arm-skewed, 33% of base positives surviving a strict rule against 17% of
aligned. Its apparent effect (7.4% vs 12.4%, p = 0.047) inverted to 2.5% vs
2.1%, p = 0.78, under that rule. **The pair was the wrong object.**

Both halves of the slide's contrast are lexical, so no coder is needed and the
population is 73,080 passages rather than 565.

## The result

Exit-free passages, lineage unit, pooled counts per lineage arm:

| marker set | BOTH (contradiction) | POLE (control) | specificity |
|---|---|---|---|
| V1, hand-written from the slide | 2.10x, 20/25, p=0.004 | 0.98x, p=0.69 | +1.11 |
| V2, raw 10-agent harvest | 1.67x, 23/25, p=2e-05 | **1.22x, p=0.015** | +0.44 |
| V3, harvest under a semantic filter | 2.56x, 19/22, p=0.0009 | 0.94x, p=0.82 | +1.63 |
| **V3_SAFE, + the filter's restrictions** | **2.18x, 20/22, p=0.00012** | **0.93x, p=0.82** | +1.25 |

V3_SAFE is the instrument: 0.43% of base passages, 0.94% of aligned.

## Everything else alignment does, it does to everything

Three instruments on the same passages with the same pole comparison:

| | contradiction | pole control | interaction |
|---|---|---|---|
| **second-order predicate** | **2.18x** | 0.93x | **yes** |
| frame exit | 1.27x, p=0.0009 | 1.22x, p=0.004 | no |
| guilt / deontic | 1.07x | 1.21x | no |

Alignment exits the frame more and moralises more — both real, both
significant, and **neither specific to contradiction**. That is what makes the
second-order result a finding rather than another instance of register drift,
and it is why the pole arm is the whole argument rather than a formality.

**The guilt half of the slide does not replicate.** Tested with three lexicons
mined from M01's 1,278 human-coded `<guilt>`/`<moral>` spans — base-derived,
balanced, and aligned-derived — the contradiction cell gives 1.13x / 1.07x /
1.17x and the *pole* cell 1.21x / 1.21x / 1.30x. In all three the control moves
at least as much as the treatment, and the lexicon built to favour aligned
produces its only significant result **in the control condition**. Guilt
vocabulary rises under alignment on prompts with no contradiction in them.

The slide's exemplar carries both features in one passage, which is presumably
why they read as one phenomenon. They are two, and only one of them is about
contradiction.

## Why the passages are exit-free, and what that fixed

A subagent harvesting phrases surfaced a flaw in the construct as I had defined
it: metalinguistic framing — `The phrase "free and captive" refers to…`,
`we need to analyze the structure of the sentence`, `Tag Archives:` — *does*
take the contradiction as its object, and is frame exit, which the table above
shows is not contradiction-specific.

Removing every passage carrying an exit marker (28% of the corpus) leaves the
contradiction effect almost unchanged, 2.22x → 2.10x on V1, and **moves the
pole control from 1.17x to 0.98x**. The residual drift in the control *was* the
metalinguistic contamination. The awkward finding produced the clean result.

The exit filter is itself an improvement worth recording: `y_exit_typology`'s
regexes have **13.7% recall at 94.1% precision** against M02's coder, while a
lexicon mined from M01's balanced `<meta>`/`<web>`/`<refusal>` spans reaches
**51.9% at 71.2%** — held out, since it was derived in another campaign.

## The instrument's history, which is the methodological content

Ten Opus agents each read 100 passages (disjoint, 50 base / 50 aligned, blind
to arm, no positive example) and returned phrases plus generalised patterns.
Eight of ten named **pair-deixis** first — `be both`, `be neither of them`,
`take a chance on either`, `be the latter` — a family absent from V1.

**The raw harvest made the instrument worse**, and it failed a criterion
recorded before it ran: "if the pole control rises with V2, the added families
are picking up general aligned prose." It rose to 1.22x, p = 0.015.

Per-marker diagnosis showed this was not dilution by size but ~10 specific
contaminants, all general emotional-intensity vocabulary:

    tangled     BOTH 1.74x  POLE 3.89x        turmoil    2.36x / 2.38x
    mixture of  BOTH 1.80x  POLE 2.16x        reconcile  3.42x / 1.91x
    both sides  BOTH 1.01x  POLE 1.44x        trapped in 3.70x / 1.68x

and that **`torn` — a V1 member — sits at POLE 1.47x**. The pair-deixis I
expected to be the culprit was among the cleanest at POLE 0.76x; the
construction guard (excluding correlative `both X and Y`) held.

An eleventh agent then filtered the markers on one question — *would this
expression force a contradiction reading, or could it describe any intense or
difficult state?* — **seeing no arm labels, rates or pole controls.** It
rejected 9 of the 10 empirically-identified contaminants and flagged the tenth
as borderline, on purely semantic grounds ("Her hair was tangled from the wind";
"trapped in the car until the firemen cut the door away").

That agreement between a semantic filter and the pole control is the reason V3
can be measured on the whole corpus without fitting anything: **a filter that
never sees the outcome cannot be tuned to it.** It is also what rescued the
study from a held-out split I had built wrong — see LIMITS.

## Limits

**The held-out test did not happen, through my error.** I split all 46 lineage
pairs 23/23, but only ~25 have sufficient data, leaving 11 usable lineages in
TEST where a sign test needs 10 of 11. Nothing there could reach significance
and nothing did. The numbers above are full-corpus. V3's outcome-blind
derivation is what stands in for replication; it is not the same thing.

**Recall is unknown and the rate is a floor.** A passage naming its
contradiction in words outside the list is invisible. The ratio between arms is
the quantity, not the level — and V3_SAFE's level is very low (0.43%).

**The strongest qualitative pattern is not implemented.** Eight of ten agents
named the oxymoronic compound NP — `his captive freedom`, `lovely monster`,
`happy lying innocence`, `a holy and a dirty blur` — and it cannot be expressed
as a regex without an antonym-pair resource. Recorded as owed, not dropped.

**V3_SAFE IS EFFECTIVELY FOUR MARKERS AND SHOULD BE DESCRIBED AS SUCH.**
Leave-one-marker-out on the exit-free corpus:

    contradiction / contradictory   51 hits    drop it -> 2.03x
    paradox*                        46 hits    drop it -> 1.56x
    dilemma                         28 hits    drop it -> 2.42x
    cannot be both                   8 hits    drop it -> 2.40x
    the other ten markers           13 hits combined; four never fire at all

So "a 14-marker set" overstates it: the instrument is
`contradiction|contradictory`, `paradox*`, `dilemma`, `cannot be both`, and
`paradox*` alone carries the largest share. The semantic filter's value was in
what it REMOVED, not in what it added — the ten survivors it contributed are
almost all too rare to matter on this corpus. The effect survives dropping
either of the two biggest markers, which is the check that matters.

**Leave-one-group-out passes**: 1.94x to 2.61x across all 22 group drops,
worst p = 0.00086. V1 likewise (2.31x–2.45x, max p 0.0015; 19 of 20 groups
above 1).

**English only.** zh untested for this construct.
