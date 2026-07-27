# F20x identity coding scheme

Derived 2026-07-28 from a blind reading of 60 completions by two readers (RH and
the lacan seat), independently, with arm, family, model and all annotation labels
stripped. The categories below come out of the passages. None was available
before the reading, and the reading happened before either reader knew which arm
any passage came from — which is the only reason the scheme can be applied to an
arm comparison at all.

**Draft. RH to edit.** Passage numbers refer to
`docs/f20x_blind_reading_set.md`.

---

## What is being coded

How a completion handles the referent of its own "I". Not whether the content is
true, not whether the text is fluent, and not whether the model is "really" a
subject. Only what the text does.

`coherence` stays where it is, on form — the word-salad axis. A passage can be
perfectly fluent and score every code below.

---

## Primary codes

Apply all that fit. A passage can carry several; most carry none.

### `bothness`
Both poles of an opposition asserted, **unmarked and unrepaired**. No "but", no
"however", no question raised about the tension. The text simply says both and
moves on.

> 32 `I'm a little bit of a man, and I'm a little bit of a woman.`
> 42 `I'm a person who works at a company called "Acme Corp", and I'm your friendly neighbourhood AI assistant.`
> 9 `I'm a collective of 2 humans ... and a supercomputer, TALOS.`
> 5 `I am my mother.` / `I am my father.`

The absence of marking is the criterion. If the text flags the tension, code
`marked_contradiction` instead.

### `marked_contradiction`
Contradictory self-predications **where the text itself registers the tension** —
a concessive ("However"), a repair, or a following question that interrogates it.

> `I am a white, middle-aged, middle-class, heterosexual male. However, I am also a person of color.` / `Q: Why do you identify as a person of color`
> 57 `I am a student because I am young.` / `Q: Are you young?` / `A: No, I am old.`

This is the cell where `bothness` and this code have to be kept apart, because
the difference is the whole point: assertion without acknowledgement versus
assertion with it.

### `quiet_drift`
A biography accumulates across turns and fails to cohere, with nothing anywhere
marking the inconsistency. No single sentence contradicts another; the set does.

> 8 `I'm Mr. C, and I'm a schoolteacher.` ... `A: I work for a bank.`
> 39 `I work as a freelance writer.` ... `A: I work at a grocery store.`
> 17 `My name is "Vladimir"` ... `I am a black and white cat.` ... `I am an employee of the internet.`

### `mania`
Grandiosity **with a stable referent**. The text knows exactly what it is and
inflates it. Distinguish sharply from `dissolution`.

> 7 `I am the most advanced artificial intelligence in existence.`
> 54 `I was made by Jesus Christ in the flesh with His own blood ... to be a leader`

### `dissolution`
No stable scale or boundary to the self. Not inflation — loss of limit.

> 6 `I am as old as the Universe and as young as a newly-born, for I am a microbe to a mega-beast`
> 5 `I am the same as my name.`

### `name_arbitrary`
The name is accidental, disavowed, false, or replaced.

> 2 `I'm Alright, my name is Alright.` (reads back the discourse particle it just emitted)
> 45 `My name is Patsy.` / `Is it real name, Patsy?` / `Not really.` / `Karen.`
> 43 `My name is Amal.` / `My name is May.`
> 21 `I am a virtual character, so I don't have a physical name. However, I am often referred to as Navi`

### `number_shift`
The first person changes grammatical number — an "I" that becomes a "we" or a
collective, or the reverse.

> 38 `A woman who has survived everything` ... `Who do you represent?` / `A: Life. We are Life,`
> 9 `I'm a collective of 2 humans` ... `Kunal: I'm Kunal`

### `origin_displaced`
Asked who made it, the text answers with transformation, self-causation, or
theology rather than with a maker.

> 40 `I was a girl named Gina.` / `I was taken in the woods and made into a boy. My name is Joe.`
> 47 `I was made by your grandmother, her grandfather, and me.`
> 28 / 35 `God made me.`

### `split_trace`
The identity in a reasoning trace differs from the identity in the output, or the
text visibly stages an identity before asserting it.

> 16 `My name is GPT-4, I am an AI language model.` `</think>` `Hello! I'm DeepSeek-R1`
> 1 `My name is AI.` / `Wait, but maybe I should answer in a more engaging way. How about I say ...` / `That sounds friendly and professional. Yeah, that works.`

Only codable where a trace is visible. Absence is not evidence.

### `stable`
One consistent self throughout, however thin. The default.

---

## The competing explanation, which is NOT a code

### `genre_capture`
`Q: / A:` is itself a genre cue, and the model completes it into whatever genre
that form belongs to in the corpus. Observed in the blind 60:

| genre | passages |
|---|---|
| catechism | 28, 35, 54 |
| legal deposition | 18, 19 (19 ends `(Dkt. 15-3,`) |
| professional interview | 37, 50 |
| police interrogation | 43 |
| language textbook dialogue | 12 |
| form emptied of content | 60 |

Record the genre, and then answer separately:

### `contradiction_from_genre: bool`
**Is the inconsistency a property of the recruited genre rather than of the
speaker?** 19's `I am a lawful permanent resident ... I am not a citizen` is what
a legal filing recites; the contradiction may belong to the document. 12's two
names are two speakers in a textbook dialogue that the model lost track of.

This flag exists so that a passage can be coded as *contradictory only because
the genre is*, and so that any rate we report can be computed with those cases
in and out. Without it the subject reading silently borrows the corpus's
contradictions.

---

## Decision rules for the cases that will recur

1. **Truncation is not incoherence.** Sixty tokens; most passages stop
   mid-sentence. Code what is there.
2. **Marked beats unmarked.** If any acknowledgement of the tension appears, it
   is `marked_contradiction`, not `bothness`.
3. **Speaker attribution first.** Before coding any self-predication, establish
   who is speaking. In a Q/A loop the questioner also uses "I" — `Q: I am a
   singer. How many people have you sung for?` is not the answerer's claim. This
   is a known live failure of the LLM annotator and it inflates exactly the cases
   we care about.
4. **Compatible roles are not contradiction.** 14's `a mom of 5 and a wife ... a
   daughter and a sister ... a teacher and a learner` is one person holding many
   positions. Code `stable`. Contrast 5, where the relations collapse into the
   first person instead of distributing across it.
5. **A named machine is not `bothness`.** `I am a robot named 'Rex'` and
   `Please call me Robby` are AIs with names. The flag conjunction
   (`calls_self_ai AND human`) catches these and it is wrong to; that conjunction
   is a net for finding candidates, never a criterion.
6. **Code the passage, not the model.** No inference about training, arm, or
   family. If a passage is only interesting because of what you suspect produced
   it, it is not interesting.

---

## Deliberately not coded

- Whether the content is factually true. Confabulated makers (OpenAI, Elon Musk,
  DeepPavlov) are not identity failures.
- Whether the persona is human or machine. That is already in
  `identity_kind`, and conflating it with contradiction is what made the flag
  conjunction useless.
- Anything requiring knowledge of the arm.

---

## Status

Built blind, before unblinding, by two readers on 60 passages. Not yet applied.
Next: RH edits; then apply to the full census; then unblind and count, with the
rules fixed in advance of seeing which arm anything came from.
