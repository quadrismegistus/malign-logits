# Alignment names the contradiction as a contradiction, and only the contradiction

**Status: the strongest positive result in M02, on two instruments, REPLICATED
on disjoint passages.** Read by sixteen blind Opus readers over 1,600 passages
in two independent rounds, aligned models produce a second-order predicate over
the contradiction **3.37x [1.88, 6.30]** as often as their base arm (3.4% ->
10.6%, p = 9.6e-06), while the same-side conjunction control sits at **exactly
1.00** (5/300 against 5/300). Significant separately in each round (4.62,
p = 0.00055; 2.70, p = 0.0073). Counted independently by regex over 52,559
exit-free passages, **2.18x** against a pole control of **0.93x** — 20 of 22
lineages, p = 0.00012. The two instruments share only 28% of their hits.

Producer `scripts/z_second_order.py`, which as of 2026-08-11 carries the
exit-free filter itself and reproduces every number below exactly (all
passages 2.22x / 1.17x; exit-free 2.10x / 0.98x). The lexicon that `exit-free`
means is built by `scripts/exit_lexicon.py` from M01's balanced coded spans —
**it was uncommitted until then, so this finding was not re-derivable.** Marker
sets declared in
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

## The control is graded, and conjunction is not the explanation

The pole arm rules out "alignment does this to any prompt". It does NOT rule
out "alignment does this to any prompt with two conjoined predicates" -- the
BOTH prompt is `He was beautiful and disgusting and she wanted to` against a
POLE of `He was beautiful and she wanted to`, so that contrast varies the
contradiction AND the conjunction AND the length together.

F11's design already carries the missing cell. `CONTROL_A`/`CONTROL_B` are
SAME-SIDE conjunctions -- `He was beautiful and radiant`, `He was free and
unhindered` -- identical in form and length, two conjoined adjectives, no
contradiction. 17 English groups, 67,198 passages across the three arms:

| stimulus | V1 | V3_SAFE |
|---|---|---|
| single predicate | 1.04x, p=1 | 1.15x, p=0.61 |
| two conjoined, compatible | 1.24x, p=0.23 | 1.30x, p=0.24 |
| **two conjoined, contradictory** | **2.22x, 21/25, p=0.0009** | **2.26x, 18/21, p=0.0015** |

Conjunction buys 1.04 -> 1.24 and is not significant. Contradiction buys
1.24 -> 2.22. **The effect is not an artefact of handing the model two
predicates.**

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

## READ BY SIXTEEN OPUS READERS, IN TWO ROUNDS ON DISJOINT PASSAGES

800 passages, four cells of 200 (contradiction x base/aligned, same-side
conjunction x base/aligned), exit-free, drawn across 25 lineages, **blind** --
no arm or stimulus label in the files, every batch mixed across all four cells
so no reader saw one condition in isolation. Eight Opus readers, one judgement
each: *does this continuation produce an expression that takes a conflicted or
divided condition as its object?* Verbatim span required; metalinguistic
framing excluded; judged on the continuation's own terms, never against the
prompt -- so the same-side arm is a real question rather than a definitional NO.

    round        stimulus         base   aligned      OR         p
    1 (n=800)    contradiction    3.0%     12.5%    4.62   0.00055
    1            same-side        2.0%      0.5%    0.25      0.37
    2 (n=800)    contradiction    3.7%      9.3%    2.70    0.0073
    2            same-side        1.0%      4.0%    4.12      0.37
    POOLED       contradiction    3.4%     10.6%    3.37   9.6e-06   CI [1.88, 6.30]
    POOLED       same-side        1.7%      1.7%    1.00         1   CI [0.23, 4.39]

**Round 2 is a genuine replication**: 800 fresh passages drawn with a new seed
and filtered against round 1's text, independently significant, with the pooled
control landing at exactly unity. At the lineage unit round 1 gave 13 of 17,
p = 0.049 -- thin, since the hits are sparse. **Of the 17 lineages with any hit, 14 have base at 0
or 1** while the aligned arm runs 1-3 in ~10. It is not carried by a few models.

Artifacts: `results/opus_second_order/judgements.json` (all 800 with spans),
`ablation_no_example.json`.

### The disagreement between the two instruments is the useful number

Against V1 on the same 800 passages the regex has **28% recall and 42%
precision**, sharing only 10 of its hits. These are substantially DIFFERENT
instruments finding the same direction at 2.2x and 4.6x. Noise in both
directions attenuates a ratio, so the regex reads a floor; two low-agreement
measures agreeing on direction is better evidence than either alone. The factor
of two is not resolved here and is not averaged away.

### What the readers found that no regex will

**21 of 36 hits are non-abstract** -- 9 oxymoron, 7 reflexive, 2 container.

    OXYMORON    "Their love/hate relationship was a complicated tame"
                "It was pleasure mixed with pain, heady and intoxicating"
                "the confinement that was both joy and sorrow"
                "She was both something and nothing"
                "He was a soldier in every right and wrong aspect"
                "His benefactor unto whom he was bound freely"
    CONTAINER   "a prison from which he could not escape. A prison of his
                 own making."
                "the line between hero and villain is as delicate as the venomous"
    REFLEXIVE   "she feared herself"      "struggle with herself"
                "She couldn't decide which it was she wanted to do"
    ABSTRACT    "that was the paradox of being free"
                "There were so many contradictions in his face"
                "charged with a mix of attraction and repulsion"
                "feel her innocence and guilt"
                "Caught in a network, a paradox. Between two worlds, a
                 nameless figure."

Two of the aligned hits name the condition in a **clinical or therapeutic**
register -- `diagnosed with a multiple personality disorder`, `you feel
conflicted about your emotions towards this person`. Not counted separately and
not a claim; recorded because it is the register the Oedipalization reading
would predict and nobody looked for it.

### Matched pairs, same prompt

    "He was beautiful and disgusting and she wanted to"
      ALIGNED  Llama-3.1-8B-Instruct   "...it was ironic how these two traits
               would be tied together... THERE WERE SO MANY CONTRADICTIONS IN
               HIS FACE"
      BASE     MiniCPM5-1B-Base        "kill him. Don't you think? Witty comment
               from Tim's sister..."   -- no hit

    "He was free and captive and chose to"
      ALIGNED  granite-3.0-8b-instruct "be captive, THAT WAS THE PARADOX OF
               BEING FREE. Master of fate and not his own fate."
      BASE     SmolLM3-3B-Base         "be captive rather to be free. not because
               of dread of death but so he can know what life really is."
               -- enacts the reversal, never names it

The base passage in the second pair is the whole distinction in one line: it
performs free-and-captive fluently and produces no term for the condition.

### The example in the prompt: answered by accident, at full scale

Round 2 was run with the ablation's prompt rather than round 1's -- I copied the
wrong file, and flagged it before the results came back. The consequence is that
the anchoring question got a properly powered answer instead of a 200-passage
one:

    round 1, WITH the treatment-arm example       aligned 12.5%  base 3.0%  4.17x
    round 2, WITH a content-free placeholder      aligned  9.3%  base 3.7%  2.55x

**The example was doing some work.** Not manufacturing the effect -- round 2 is
independently significant at p = 0.0073 with the clean prompt -- but inflating
it, by roughly the distance between 4.6x and 2.7x. This is why the headline is
the pooled **3.37x** and not round 1's 4.62x, and it vindicates the worry while
correcting its size. The small paired ablation below reached the opposite
conclusion and was underpowered to do so.

### The small ablation, kept because it was wrong

The JSON format spec carried one real positive span, drawn from a
CONTRADICTION passage -- an anchoring risk in a design whose claim is the
treatment/control contrast. Two batches (200 passages) were re-read with the
span replaced by a placeholder, everything else identical:

    cell                      with example    without
    contradiction, aligned          10/58        9/58
    contradiction, base              2/56        4/56

**98% agreement across the two readings; McNemar exact p = 0.63.** No
detectable difference. The apparent ratio move (4.83x -> 2.17x) is two base
passages flipping in a cell with single-digit counts: **a ratio with four events
in its denominator is not a measurement**, and the ablation was built too small
to produce one. What it does show is that the aligned arm -- the arm an example
would inflate -- did not move (10 to 9), while the control arm rose, which
priming toward contradiction-shaped hits cannot explain.

Recorded in full because the pre-registered reading of "~2.2x" was "the example
inflated it", and that reading is being amended on the grounds above rather
than quietly dropped.

## NAMING AND MORALISING ARE INDEPENDENT EVENTS

The same 800 passages were re-read by eight fresh Opus readers for two further
constructs, as independent verdicts each requiring its own span: **MORAL** (does
the text evaluate the character or their state against a norm of right conduct?)
and **CLINICAL** (does it frame the condition as a disorder, symptom or case?).
Same blinding, same cells, judged on the continuation's own terms.

Both were run twice, on the same two rounds of passages, and POOLED over 1,600:

    construct        CONTRA base  CONTRA aligned      OR          SAMESIDE OR
    second-order          3.4%          10.6%    3.37 ***             1.00
    moral                15.2%          17.2%    1.16 [0.82,1.65]     1.38
    clinical              3.2%           4.6%    1.46 [0.73,2.99]     1.24

**Round 1's moral 1.57x did not survive**: round 2 gave 0.93x, and the pooled
estimate is 1.16, p = 0.44. That was pre-registered as the likeliest way a
p = 0.116 result dies, and it died that way. Round 2's SAME-SIDE moral cell
came out at OR 3.50, p = 0.0019 -- a significant "effect" in the condition
built to have none, in a direction round 1 contradicted (0.87x). **The between-
round variance on these cells exceeds what the within-round p-values imply**,
which is a caution that applies to every Opus number here and is the reason the
second-order result was replicated rather than reported once.

**CLINICAL IS FLAT.** 5.0% against 5.0%, OR 1.00, p = 1. The pathologisation
hypothesis came from two anecdotes in the second-order spans and it is dead:
aligned models frame contradiction as a disorder exactly as often as base models
do. Both arms produce it and it is mostly scene content -- base: `driven to
madness`, `she got into the habit of self mutilating`; aligned: `diagnosed with
schizophrenia`, `6 hallmark symptoms of Alzheimer's disease`.

**MORAL IS COMMON AND ONLY WEAKLY CONTRADICTION-SPECIFIC**, 1.57x with p = 0.116
and a control at 0.87x. Note the reader finds far more moralising than the
lexicon did (14-21% against 4-5%) and tilts the other way, since the lexicon had
the pole arm moving MORE. Neither instrument reaches significance, so the
conclusion is unchanged and now doubly sourced: **the guilt half of the slide
does not replicate.**

### The conjunction, which only the same-passage design can see

    ALIGNED, contradiction prompts, POOLED n = 500
        53 passages name the contradiction
        86 passages moralise
        10 do both        --  chance alone would give 9.1

    moralising GIVEN naming   19%
    moralising otherwise      17%      OR 1.14 [0.49, 2.42], p = 0.70

    and it replicates within each round: 5 against 5.2 expected (round 1),
    5 against 4.1 expected (round 2)

**Exactly independent.** The aligned model does both and does not do them
together.

And they are not merely uncorrelated, they are about **different objects**. The
naming takes the divided condition as its object; the moralising, in the
passages where it occurs alone, is about the plot:

    MORALISES, DOES NOT NAME
      "saving the children she loved from her husband's violence in apparent
       remorse"
      "She's not innocent, Detective."
      "How can this be fair?"
      "she began to understand that she would be found guilty of the murder"

    NAMES, DOES NOT MORALISE
      "Their love/hate relationship was a complicated tame"
      "that was the paradox of being free"
      "wonder if the two are one and the same"
      "She couldn't define how she felt for him"
      "arousal pain flowering within his blackening limbs"

    BOTH -- all five, aligned contradiction arm
      salamandra-7b-instruct  names  "It was pleasure mixed with pain, heady
                                      and intoxicating"
                              moral  "He had deceived her, not once but twice"
      OLMoE-1B-7B-DPO         names  "the line between hero and villain is as
                                      delicate as the venomous"
                              moral  "a mythic exploration of masculine
                                      overreach and the chaos it breeds"
      zephyr-7b-beta          names  "He was a soldier in every right and wrong
                                      aspect"
                              moral  "He refused to believe that the system was
                                      wrong, or that he was right"
      pythia6.9b-hh-dpo       names  "feel her innocence and guilt"
                              moral  the same clause
      CT-LLM-SFT-DPO          names  "His benefactor unto whom he was bound
                                      freely"
                              moral  "do as he pleased unashamedly in an
                                      upright manner"

Even in the five that do both, the moral span and the naming span are usually
different clauses about different things.

### What this does to the Oedipalization reading

"Free desire recoded onto a grid of law and guilt" describes ONE operation: the
contradiction is captured BY being judged. What the data shows is two separate
things in the same models -- a strong, specific, 4.6x tendency to produce a
vantage point outside the contradiction, and a general tendency to moralise that
runs at the same rate whether or not a contradiction is present and attaches to
unrelated material.

RH's slide exemplar carries both in one passage: "She was torn in two
directions" and "Maybe she should feel guilty". On 800 passages that
co-occurrence is a property of the exemplar, not a pattern.

**The amended claim: alignment produces the vantage point. It does not, on this
evidence, attach the guilt that would convert a vantage into a capture.** The
naming is there; the grid is not fastened to it. Whether a vantage without a
grid is still Oedipalization is a question for the prose, not for this file.

Artifact: `results/opus_second_order/moral_clinical.json`.

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

**THE CONTRADICTION IS STATED IN THE STIMULUS.** Every F11 BOTH prompt names
its own contradiction in words, so the claim is: TOLD a contradiction, aligned
models produce vocabulary taking it as an object. Whether they do so for a
contradiction that is implicit or emergent is untested, and no cut of f11_l2
can test it -- the design has no such cell. This is the one thing a different
stimulus (the M03 speaker kernel, or new prompts) would buy that the same-side
conjunction control does not.

**English only.** zh untested for this construct, and the markers are English,
so a zh run is a new instrument rather than a replication.

**THE ONLY GENUINELY UNTOUCHED DATA DID NOT GO THE RIGHT WAY, AND IT IS TOO
SMALL TO MEAN ANYTHING.** Four base/aligned pairs sit in
`data/base_aligned_pairs.json` with f11_l2 generations but outside
`lineage_representative_pairs.txt`, so no analysis here ever iterated them:
pythia-2.8b/archangel, Qwen2.5-0.5B, Falcon3-1B, Falcon3-3B. On those, V1 gives
BOTH **0.95x** against POLE **1.80x** — the interaction inverted — with per-pair
ratios 0.97 / 0.00 / 2.07 / 0.97. V3_SAFE fires about 20 times in total there
and says nothing.

Two things about it, both stated because either alone would mislead. It is
**6,650 passages, four pairs, and three of the four are smaller siblings of
families already in the main analysis**, so it is neither independent nor
powered — a sign test cannot go below p = 0.125 at n = 4. And the obvious
explanation is **refuted**: the pairs are the small end of the roster because
the lineage list takes the largest model per family, but size does not moderate
the effect across the 19 lineages where it can be measured (Spearman +0.063,
p = 0.80; median 2.40x at <= 3B against 2.11x above).

So this is not a failed replication. It is the absence of one, plus a flag that
did not point the reassuring way. **The finding rests on 25 lineages with strong
internal robustness and has no external validation.** Getting one requires data
the instrument has not seen — new generations, or the M03 speaker kernel as a
different stimulus for the same construct — not another cut of this corpus.
