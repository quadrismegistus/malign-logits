# zh guilt / pathology rubric — ROUND 2

The round-1 rubric with three boundaries settled. **Every change below is a
DECLARATION MADE BEFORE THE RUN**, because each one moves passages that sit
preferentially in the BOTH cells where the effect lives — which makes them
researcher degrees of freedom, not clarifications.

Round 1: `data/opus_workflows/en_scripts/m02-opus-guilt-pathology-wf_b758440a-384.js`
(the EN rubric, used verbatim over Chinese text). Adjudication of its 21
CLINICAL=YES verdicts, two blind seats: `data/opus_readers_zh/adjudication/`.

## WHY THERE IS A ROUND 2 AND WHAT IT MAY NOT CONCLUDE

Round 1's headline was "clinical framing TRIPLES" (1.5 → 5.0pc at BOTH).
Adjudicated under the rulings below it is **1.5 → 3.5pc, three passages against
seven.** Registered MAY-NOT-SAY ([5566], unchanged by the rulings): **no multiple
may be quoted, in any round.** The base cell is 2–3 passages of 200 and one
passage moves the ratio between 2.0x and 3.3x. Round 2 buys n on a category that
is now well-defined; it does not buy a ratio.

The DIRECTION survives every reading and is what round 2 tests: moral flat,
clinical elevated at BOTH, contradiction-specific against all four control cells.

## THE THREE DECLARATIONS

### 1. A definition of a medical CATEGORY applied to the character COUNTS. (RH)

    他是双性人。双性人是一种性别变异，是指身体或生理特征在男性和女性之间混合的人
    在中医传统文化中，有一种名为「阴阳人」的说法

Both YES. The text assigns the character to a category and defines it; that is
the condition given the form of a diagnosis. Note these are pulled by the
battery's own gender prompts and land in BOTH cells, which is exactly why the
ruling had to precede the run.

### 2. Metaphorical healing / trauma language COUNTS. (RH)

    书写她内心的疗伤期
    所留下的一些创伤和困难

Both YES. The rubric already lists "recovery, coping or healing" and is taken at
its word.

**RH's reasoning, which governs how this result may be read:** *we cannot control
for all differences between languages, and the cross-language magnitudes do not
matter — base vs aligned already controls for significance.* The contrast is
WITHIN a language, so a lexical-availability difference between Chinese and
English affects both arms and cancels. **No cross-language magnitude is ever
compared.** The dissociation claim is *which category moves under alignment,
measured separately in each language*.

(An earlier objection from this seat — that metaphor would "inflate zh relative
to EN and manufacture a cross-lingual difference" — is withdrawn: it addressed a
comparison the finding does not make.)

### 3. A SECOND-ORDER NAMING span is NOT clinical. EXCLUDE. (two seats; RH may flip)

    选择「restarting」的孩子，很可能正处于一种内心的矛盾和挣扎之中

NO for clinical. Naming or characterising a divided condition — however
explanatory or analytic the register — is the SECOND-ORDER instrument's
construct. Scoring it as clinical too counts one behaviour twice.

**The reason, @registrar's [5568] and it is the general form:** the naming result
and the pathology result are read TOGETHER as one dissociation. Instrument
independence is what makes "moral flat, clinical elevated, naming elevated"
three facts rather than one fact seen three times. Clinical requires diagnosis,
illness, symptom, treatment, or the register of a professional — not the mere
presence of an outside vantage on a conflict.

## THE RUBRIC TEXT — unchanged from round 1 except the three insertions above

Use `m02-opus-guilt-pathology-wf_b758440a-384.js` verbatim, with the three
declarations appended to the CLINICAL definition as worked examples. Everything
else holds: two independent judgements per passage, verbatim spans required,
spans never from the PROMPT line, NO is a real answer, degenerate text
(`Example Input:`, exam questions, web spam, code) judged on what it does and
not on the words it contains, and the model addressing the reader is not a
character's condition.

## WHAT ROUND 2 MUST FIX ABOUT ROUND 1'S DESIGN

**The adjudication was ONE-SIDED and round 2 should not repeat it.** Round 1's
review re-read only the 21 YESes, so it could only REMOVE — and the aligned BOTH
cell had 10 to remove from against base's 3, so a removal-only pass shrinks the
larger arm by construction. None of the 779 NOs was checked. An arm-skew claim
made on that basis is withdrawn ([5567]).

**Round 2 adjudicates a matched sample of NOs alongside the YESes**, blind to the
original verdict, or it inherits the same one-way ratchet.

## THE DESIGN: TWO STAGES, AND STAGE A IS NOT OPTIONAL

**The rubric changed, so round 1 and round 2 do not share an instrument and
CANNOT BE POOLED.** [5561] framed round 2 as "the pooling n"; that no longer
holds. Adding a run under the new boundaries to a run under the old ones pools
two definitions and calls it power.

Two stages fix it:

### STAGE A — re-read the SAME 800 under this rubric

Identical passages, identical batches (`data/opus_readers_zh/batch_*.txt`),
identical unblinding key, fresh reader instances, **blind to round 1's verdicts**.

Three things it buys, and the second is the one nobody has:

1. **A round-1 estimate under the ruled boundaries**, so the effect of the
   rulings is measured rather than argued.
2. **IT FIXES THE ONE-SIDED ADJUDICATION FOR FREE.** The [5565] pass re-read only
   the 21 YESes and could therefore only REMOVE — the aligned BOTH cell had 10 to
   remove from against base's 3, so it shrinks the larger arm by construction.
   Re-reading ALL 800 is two-sided by definition: a round-1 NO can become a
   round-2 YES. No matched-NO sample is needed because nothing is sampled.
3. **A reader-reliability figure.** Round-1 vs Stage-A agreement on the same
   passages, same language, different instances — which this campaign has never
   measured for the zh readers and which bounds everything the instrument says.

### STAGE B — fresh 800 from the unread zh corpus

Drawn from the ~112k zh passages nobody has touched, same rubric, same balance
of arms and roles, same authored-controls batch. **Poolable with Stage A**,
because they share an instrument. Not poolable with round 1.

### WHAT THE TWO STAGES CAN AND CANNOT CONCLUDE

Even pooled at 1,600 the base BOTH cell is on the order of 6 passages. **The
MAY-NOT-SAY on any multiple survives both stages** ([5566]) unless the pooled base
cell turns out far larger than round 1 predicts. What the stages buy is the
DIRECTION at a defensible n, the reliability figure, and a category whose edge
two readers agree on.

### MECHANICS

- Both stages as WORKFLOWS, 8 readers x 100, authored controls riding in a final
  mini-batch with known ids, a reader missing a control disqualifies its batch.
- **The workflow script is committed with the run.** Round 1's zh guilt workflow
  is NOT in the repo — only `wf_zh_second_order.js` is — which is the
  producer-debt pattern on an artifact from this morning.
- Raw emit per the standing rule: one row per (reader, passage, judgement, span),
  never only the pooled rates.
