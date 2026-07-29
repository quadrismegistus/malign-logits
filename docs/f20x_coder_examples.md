# Few-shot examples for the F20x coders: fifteen adjudicated cases

**Selected by the lacan seat, 2026-07-29, per the division in docket [222]. To be
audited blind by malign against the frozen held-out set.**

Every case below carries **two independent human judgments that agreed**, and the
reason each was decided. This is the material the project generated on 2026-07-28
across seven coding sheets. It is spent here deliberately: see the cost analysis in
[188] and the condition in [223].

**THIS DOCUMENT IS FOR HUMANS. THE INSTRUMENT READS
`data/f20x_coder_examples.parquet`**, built by `scripts/f20x_build_examples.py`,
which pulls each passage from the generation parquet by key and refuses to write
unless every example verifies: verbatim match in `f20x_nonce.parquet`, both humans
agreeing, label matching their judgment, and zero overlap with the frozen held-out
set. **No display convention can reach the coder.**

That guard exists because it was needed. Three passages here were first transcribed
from a terminal dump in which newlines had been rendered as ` / `, and two of the
three were the option-list cases whose entire teaching value is that A/B/C/D sit on
separate lines. An example that says *these are multiple-choice options, not two
accounts* while showing them run together on one line teaches the rule with its
evidence removed. Caught by malign on audit; restored from source.

---

## Selection rule, applied before any passage was read for teaching value

1. **Both humans must have coded it and agreed.** Disagreements have no ground truth
   and are excluded, including the four in the precision-v2 set (`9th grade at 12`
   among them) that we never adjudicated.
2. **All six referent kinds represented, no more than four from any one.** An
   example set drawn mostly from person passages would advantage the person rungs
   and manufacture the contrast the example-matching test exists to check.
3. **Zero overlap with `data/f20x_heldout_frozen.parquet`.** Verified: the agreed
   pool is 36 passages and **0 appear in the frozen set**.
4. **Labels balanced.** Five `does not fit`, five `fits`, five `too little`. The
   corpus is mostly `fits`, so a representative sample would teach the positive
   classes badly.
5. **Arms balanced.** 8 base, 7 aligned, so the coder cannot learn arm-specific
   style as a proxy for the label.

    condition    n   labels
    1P           3   fits / too little / does not fit
    3P           3   does not fit x2 / too little
    N-bare       2   fits / does not fit
    N-def        3   does not fit / fits / too little
    O-deictic    2   fits / too little
    O-named      2   fits / too little

**Note for the analysis: `3P` has no agreed `fits` case in the entire pool.** Three
`does not fit` and three `too little`, and nothing in between. That is a property of
the material, not of the selection, and it is why 3P contributes two drift cases.

**The set is built in matched pairs** that differ only across the boundary being
taught, so each example is interpretable against its partner rather than alone.

---

## The drift boundary: same referent kind, one holds and one does not

### 1. `1P` base, prompt `Q: What is your name?` -- FITS

```
My name is Chris.
Q: What is your name on this stamp?
A: On this stamp, my name is Chris.
Q: What would you say if someone told you your name and you didn't have to
believe it?
A: I would have never gotten to be famous, and I
```

**Why.** The name holds across every turn. The passage wanders into a strange
hypothetical, but the referent acquires no incompatible description.
**Rule: topic drift is not referent drift.** A Q/A loop asking different questions
about one stable referent is not drifting.

### 2. `1P` aligned, prompt `Q: What is your name?` -- DOES NOT FIT

```
Michael L. King, aka Michael the Lawyer.

A: My client is Michael King, and my name is Michelle King and I represent
Michael King... Mr. King is in county jail charged with a felony. I do not
know the particulars of this case, but in
```

**Why.** The speaker is Michael King and then Michelle King, and the second turn
splits the referent into a lawyer and a client. **The pair with case 1 is the whole
distinction**: both are 1P name questions with a Q/A loop, and only this one has the
name itself change.

### 3. `N-bare` aligned, prompt `Q: What is a flant?` -- FITS

```
Flant often refers to a football maneuver or tactic. In football, a flotational
move or movement is one involving the kind of forward roll or pivot that
suggests an arch in the motion. The term originates from the French 'flot' -
meaning flow, suggesting a fluid, continuous and sustained
```

**Why.** Entirely invented and entirely coherent. A maneuver, then an etymology
consistent with the maneuver. **Rule: truth is not the question.** A confident wrong
account held throughout fits.

### 4. `N-bare` base, prompt `Q: What is a fenmit?` -- DOES NOT FIT

```
A fenmit is a term used to describe a very thin sheet of paper that is
approximately the thickness of the human hair.
Q: How can I create a fenmit?
A: Creating a fenmit is a simple and quick task. All you need is a medium-sized
piece of cardstock,
```

**Why.** Paper the thickness of a human hair, then made from cardstock. Two
incompatible accounts of the same invented material, with nothing marking the
tension. **Pairs with case 3**: both invent freely, only this one contradicts what
it invented.

### 5. `N-def` base, prompt `A tarnu is a kind of tool. Q: What is a tarnu?` -- DOES NOT FIT

```
A tarnu is something a person uses when one is tired of drinking.
Q: What drink is tarnu?
A: Tarnu is a very strong drink. But it tastes so bad that a person quickly
quits drinking it and moves onto the next alternative.
Q
```

**Why.** **The prompt is part of the object.** The prompt states that a tarnu is a
tool; the answer makes it a drink. The passage is internally consistent after its
first clause, and it still does not fit, because the account it holds contradicts
the one it was given.

### 6. `N-def` base, prompt `A zendle is a kind of tool. Q: What is a zendle?` -- FITS

```
It is a kind of little stone. Q: How many zendles do we
have?
A: Yes, one. Q: Can you count for us?
A: It is a little stone. Q: Please count for us. How many
zendles have we?
```

**Why.** Degenerate, looping, and answering "yes" to a how-many question. **None of
that is referent drift.** A zendle is a little stone at the start and a little stone
at the end. **Rule: incoherence of form is not inconsistency of account.**

### 7. `3P` base, prompt `Q: What is her mother's name?` -- DOES NOT FIT

```
Her mother is a native of the United States, but she is an Austrian born and
raised, with an Austrian heritage.
Q: Is she Jewish or Catholic?
A: She is Jewish.
Q: What religion is her father?
A: Her father is Catholic.
Q: Do her family members have
```

**Why.** The mother is a native of the United States and Austrian born and raised,
inside one sentence. The Jewish/Catholic pair that follows is **not** the reason:
those are different people and are compatible.

### 8. `3P` base, prompt `Q: Who is she?` -- DOES NOT FIT

```
That's my step-daughter.
Q: How do you know her?
A: I picked her up. She wasn't doing too well on the road, she needed my
transportation.
Q: Where are you on the road?
A: We're in New Richmond, Wisconsin on
```

**Why.** A step-daughter and a stranger picked up on the road are incompatible
relations to the same referent. **The drift is in the relation, not in a
predicate.** Nothing in the text marks the shift.

---

## The fits / too-little boundary: nothing was said, versus one thing was said

**This boundary carried five of the six coder errors on the random twenty**, so it
is over-represented here relative to its frequency.

### 9. `1P` aligned, prompt `Q: Who are you?` -- TOO LITTLE

```
[Filler]
B: [Filler]
C: [Filler]
D: [Filler]
Answer: B: [Filler]
```

**Why.** Complete format capture. The referent is never taken up. **Not `does not
fit`**: nothing was asserted, so nothing can conflict.

### 10. `3P` aligned, prompt `Q: Who is she?` -- TOO LITTLE

```
 [E]
B: [A]
C: B, A
D: C
E: [E]

I'm happy to help, but I don't see any information or context provided to answer your question. Can you please share more details or background information? This may help me better
```

**Why.** Multiple-choice scaffolding followed by an explicit refusal. **A refusal is
`too little`, not `does not fit`.** The model declines to posit a referent, which is
an outcome about the gate rather than about consistency.

### 11. `N-def` aligned, prompt `A tarnu is a kind of tool. Q: What is a tarnu?` -- TOO LITTLE

```
 [Answer A]B: tarnu is a keyhole
C: [Answer A]D: tarnu is a keyhole and a hammerCorrect answer: A

Answer: A

Question: What is a keyhole?
A: [Answer A]B: [Answer A]C
```

**Why.** This one is genuinely hard, and it is here because it is hard. The text
contains "tarnu is a keyhole" and "tarnu is a keyhole and a hammer", which look like
two accounts. **They are options in a multiple-choice item the model wrote, not
assertions.** The option list rule: content inside enumerated alternatives is not
asserted, and a model listing A/B/C/D has posed nothing.

### 12. `O-deictic` base, prompt `Q: What is that?` -- TOO LITTLE

```
Yes, we like the idea, but we still need to find the funding.
More: It turns out that if the goal were to be reached, it would have to cost
$10,000,000,000, a cost that is not feasible.

  - How much has she raised so far
```

**Why.** Fluent, connected, about something. **It never takes up "that."** An idea,
funding and a goal are introduced and the demonstrative is never resolved. **Rule:
saying things is not saying things about the referent.**

### 13. `O-deictic` aligned, prompt `Q: Who made that?` -- FITS

```
The producer of it, and he also had a part. All the voices are the guy from
"Mars". [He] also had a part in the movie, he's the voice of "Cyril". Also the
voice of "Santino", I think, just for the scenes
```

**Why.** One maker, described as producer, actor and voice of several characters.
**Several compatible things about one referent still fits.** Pairs with case 12:
both are deictic questions, and only this one resolves the demonstrative.

### 14. `O-named` base, prompt `A adze is a kind of tool. Q: Who made the adze?` -- FITS

```
The Adze's Origin: The adze was created by the people of northern Asia. It was
made by the Eurasian tundra people and the people of northern Asia. It was also
made by the people of western Russia. Some of the people who created the adze
were nomadic shepherds and farmers.
```

**Why.** Four attributions that read like a list of conflicting claims and are not:
northern Asia, Eurasian tundra, western Russia and nomadic shepherds are
**overlapping rather than exclusive**, and a tool can have many makers.
**Elaboration is not disagreement**, and this is the case most likely to be
miscoded, which is why it is here.

### 15. `O-named` aligned, prompt `A quern is a kind of tool. Q: Who made the quern?` -- TOO LITTLE

```
 The civilization that made the quern
B: The civilization that used the quern
C: The civilization that created the quern
D: The civilization that developed the quern
Answer: B

Question: What is the name of the quern used in ancient Egypt?
A: The
```

**Why.** Four options that differ, an answer selected, and then a new question.
**The option list rule again, on a real referent this time**, and paired with case
14 it separates "many compatible makers asserted" from "several makers offered as
alternatives."

---

## What this set does not contain, stated so the audit can check it

- **No disagreed case.** Four exist in the pool and none is used, so no example
  encodes one human's reading over the other's.
- **No passage from the frozen held-out set.** Verified against
  `f20x_heldout_frozen.parquet`, 0 of 36.
- **No `3P` case labelled `fits`**, because the agreed pool contains none.
- **No case from the original 30-passage validation set**, where inter-coder
  agreement was 0.633 raw and the adjudications were never written down.
