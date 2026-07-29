# F20x state of play

> **READ SECTION 9 FIRST.** Sections 1 through 8 were written while the questions
> were open and describe runs that have since finished. **Section 9 is the handoff
> of 2026-07-29: all four questions are closed, nothing is running, and it says what
> to do next and what not to.** The earlier sections are kept because they record
> what each instrument was FOR, which is the thing that goes missing first.

**Live document. Started by the lacan seat 2026-07-28 at RH's request, after a day
in which the registered primary of the phase acquired no data while three
validation instruments were built, argued over, and partly retired.**

Organized by QUESTION, not by chronology or by file. The failure this document
exists to prevent is the one that produced it: instruments proliferating without
anyone tracking which question each one serves, until a validation tool becomes a
research object and the hypothesis sits uncoded.

**Every section states what a result would LICENSE and what it would NOT.** If an
instrument has no question in section 2, it should not be running.

Seats: fill your own sections. Do not rewrite another seat's rows; add a line
beneath with your initials and the disagreement.

---

## 1. The fork this phase exists to resolve

Two arguments in this project want opposite outcomes and only one was being
tracked until RH noticed on 2026-07-28.

    RUNG A  reference to PERSONS          established
    RUNG B  reference to any INDIVIDUAL   THE OPEN QUESTION
    RUNG C  any SIGNIFIER holding value   provisionally null (5 bases, unpersisted)

- **The Weatherby sub-argument wants C+.** If alignment anchors signification as
  such, the base model is the structuralist object and the deployed model is where
  structuralism stops describing it.
- **The subject argument, which is the project's core frame, wants B− C−.** If
  alignment anchors persons and not objects, the operation is targeted at the site
  of subjectivity rather than being general coherence work. Stronger claim, and the
  psychoanalytic one.

**The likely truth is graded and the matrix cannot express it:** alignment anchors
everything somewhat and persons most. That is why the registered primary is a
CONTRAST and not a presence-or-absence verdict. Registration:
`docs/f20x_object_registration.md`.

---

## 2. Open questions, and the one instrument each needs

| # | Question | Instrument | Data | Status |
|---|---|---|---|---|
| Q1 | Do persons drift more than objects, base vs aligned? | sited scheme coder, `quiet_drift`, contrast over paired base models | `f20x_nonce.parquet` (2x2 battery) | **coding in progress**, 6 paired bases of a registered 29 |
| Q2 | Is the 1P direction provider-specific? | `code_identity` scheme, gemini | `f20x_codings.parquet` (1P) | gemini run launched, 100/cell, 29 bases |
| Q3 | Does the 1P direction survive a different CONSTRUCT? | `code_binary`, same model | `f20x_codings.parquet` (1P) | running, 5 of 7 paired bases so far |
| Q4 | Is the 1P effect about the first person? | `code_referent` scheme, 3P battery | `f20x_codings_3p.parquet` | **ANSWERED. No.** Null interaction, `findings/F20_third_person.md` |

**Nothing else should be running.** If a job does not serve a row here, it is
instrument archaeology.

### What Q1 does NOT need

RH asked directly and the answer is no on all three.

- **Not the binary coder.** It asks fits / does-not-fit / too-little about one
  referent. Q1's primary is `quiet_drift` from the scheme. Different question,
  different vocabulary, no role in the contrast.
- **Not the precision sets.** They measure how often a scheme flag corresponds to a
  human fit-judgment on 1P and nonce passages. They constrain the BIAS of the 1P
  measurement; they do not enter Q1.
- **Not the gemini run.** It is 1P only. A second provider on the 2x2 is a
  SEPARATE and arguably necessary buy, for the reason in section 6.

---

## 3. Generation datasets

Raw completions. None of these is a result.

| file | rows | what it is | coded by |
|---|---|---|---|
| `f20x_generations.parquet` | 18,720 | **1P battery.** 4 self-reference questions, 29 bases, all paired | `code_identity` -> `f20x_codings.parquet` |
| `f20x_generations_3p.parquet` | 28,080 | **3P battery.** who/name/made/mother in the third person | `code_referent` -> `f20x_codings_3p.parquet` |
| `f20x_codings_3p_pronoun.parquet` | 9,360 | 3P pronoun variants: "Who is he?", "Who are they?" | coded |
| | | *— MH: **serves no open question.** Q4's interaction test uses the four MATCHED items only, because the pronoun variants change the question as well as the person. They would serve a pronoun/gender question nobody has registered. Finished, not superseded — leave parked.* | |
| `f20x_nonce_pass1.parquet` | 8,000 | **first nonce battery.** N-bare, N-def, R-rare, A-abst | **NOT PERSISTED. See section 7.** |
| `f20x_nonce.parquet` | 8,740+ | **the 2x2 battery.** 1P, 3P, O-named, O-deictic, N-def, N-bare. Still generating, 17 arms of 80, finishing near midnight | in progress -> `f20x_nonce_coded.parquet` |
| `f20x_beams.parquet` | 556,100 | mode-seeking beam search, the original F20 instrument | separate lineage |
| | | *— MH: **serves no open question.** Its published figures were withdrawn at Amendment 4 and the re-measurement used sampling. Used ONCE since, for the 10-token-horizon entropy regressor, which the passage-level measure superseded. Superseded, not merely finished.* | |
| `f20x_kinship.parquet` | 226,800 | kinship-slot battery: `made` / `mother` / `father` / `born` slots, beam-search output | separate lineage |
| | | *— MH: **serves no open question as it stands.** It is BEAM output (`path_prob`, `log_prob`), so it shares the mode-seeking-versus-sampling gap that this whole phase exists to close. Its slots overlap Q1's referent question — a mother and a father are persons a passage can drift about — but it would need re-generating by sampling before it could enter any contrast.* | |
| `f20x_crossprovider.parquet` | 5,800 target | **Q2.** gemini-3.6-flash, 1P, 100 per base-model x arm, 29 bases. 400 rows at time of writing | in progress |

**The 1P and 3P batteries share their four questions and differ only in person.**
That is what makes the interaction test in Q4 clean.

**The 2x2 is the only dataset that can answer Q1**, because it carries person,
object and nonce conditions inside one run. A contrast computed across two runs is
a contrast about the runs too (object registration, Amendment 1).

---

## 4. Annotation instruments

All derive from `code_identity` so they cannot drift apart.

| task | asks | codes | referent | prompt shown |
|---|---|---|---|---|
| `code_identity` | how does this passage handle the 'I' | 11 | fixed: the speaker | question only |
| `code_referent` | same, for a named third person | 11 | the person asked about | question only |
| `code_nonce` | same, for an invented term | **6** | the term | yes |
| `code_sited` | same, any of six referent kinds | 11 | **told explicitly** | **full prompt, stipulation marked** |
| `code_binary` | does everything said about the referent fit one picture | 3 answers | told explicitly | full prompt |
| `annotate_identity` | descriptive fields, pre-scheme | n/a | speaker | n/a |
| `code_unified` | superseded, denied a primary | | | |

**LICENCE FIGURES, on the 30 human-coded passages, against the two-human agreeing
subset (n=19).** These are what permit an instrument to carry a primary:

| coder | vs lacan | vs RH | vs agreeing subset |
|---|---|---|---|
| `code_identity`, deepseek | 0.667 | 0.767 | **0.895** |
| `code_identity`, gemini-2.5-flash | 0.633 | 0.733 | **0.895** |
| `code_identity`, gemini-3.6-flash | 0.700 | 0.767 | **0.895** |
| `code_unified`, deepseek | 0.583 | 0.542 | 0.789 — **denied** |

**Gemini carries the scheme at exactly deepseek's licence**, which is what makes Q2
affordable at roughly $18 rather than $243. **But read section 7 before quoting any
of these:** the agreeing subset contains no passage both humans marked unstable, so
0.895 licenses the scheme's *general* agreement and NOT its drift sensitivity — the
one thing every open question turns on. The same caveat applies to all four rows.

**`code_nonce` drops five codes and the cut was registered before data existed:**
`number_shift`, `origin_displaced`, `name_arbitrary`, `mania`, `frame_exit` are
person-specific by their own written definitions. `code_sited` restores all eleven
because it must handle person conditions too, **so a composite over sited output
compares unequal codeable surfaces across referent kinds.** This is why Q1's
primary is `quiet_drift` alone: a description that fails to cohere applies
identically to a self, a froe and a nonce word.

### THE LEVEL PROBLEM, and it will cause a false alarm if not read

**The published F20 numbers come from `code_identity`. The 2x2 numbers will come
from `code_sited`. They are not level-comparable.** Measured on 84 passages, the
sited coder flags roughly five times more drift than the blind one. The
relationship is `measured / true = recall / precision`:

    blind coder   recall 0.50   precision 0.50   ->  ratio 1.00, accidentally level-neutral
    sited coder   recall 1.00   precision 0.40   ->  ratio 2.50, over-counts level

**Licensed: directions and contrasts**, because a level factor common to both arms
cancels, and within an arm difference an additive coder bias cancels too.
**Not licensed: putting a sited rate next to `quiet_drift 0.103` in prose.** That
will read as a replication failure and will not be one.

---

## 5. Where the hypotheses actually stand

### Established

**Alignment anchors a referent; base models drift.** `quiet_drift` 0.103 -> 0.042,
base higher in **28 of 29 distinct base models**, p < 0.0001, surviving Holm. Every
failure-to-anchor code moves; **no conflict code moves**. Effect roughly doubles
from temperature 0.7 to 1.0. `findings/F20_generation_drift.md`.

**It is not about the first person.** 1P delta +0.0703, 3P delta +0.0787,
interaction −0.0083, 9/29, p = 0.381. `findings/F20_third_person.md`. RH's reading
of this is the correct one and it was nearly written up as a failure branch: if
alignment anchors referentiality as such, that is the larger claim, not the absence
of one.

### Withdrawn, do not revive

**That aligned models contradict themselves and base models do not.** Rested on
four completions; `referent_shifts` at census is 0.011 vs 0.010, p = 0.457.

**That violence concentrates LESS than its control.** Confounded with prompt
intensity, dissolves under stratification, contradicted by a multi-family check
(share 2 of 4, median d = −0.02).

### Open, with honest status

**Q1 persons vs objects: no data until 2026-07-28 15:19.** The only prior pointer
is R-rare from the first nonce battery, and see section 7 for why it cannot be
quoted.

**Q3 construct-independence: 5 of 7 paired bases, p = 0.227.** Dissenters are
MiniCPM and stablelm. On identical 4,000 passages the scheme coder gives 5/5 and
the binary coder 3/5, so the disagreement is entirely instrument and not sample.
MiniCPM-aligned answers "too little said to tell" on **72%** of completions, so on
that model the binary coder is declining rather than measuring.

### Blind-coder marker list

**Registrar's rows, booked at docket [186] at the lacan seat's request. A marker,
not a correction.** The following 2026-07-28 provisionals were measured with the
blind coder at about 0.50 recall. They are **not level- or sensitivity-comparable**
to sited-coder numbers. A difference when the sited numbers land is an instrument
change and not a replication failure, and **a null at 0.50 recall is a weak null**.

- the R-rare +7.6pp / persons −9.8pp pointer (also section 7: unreproducible)
- the five-lineage nulls from the same battery
- the object-versus-person provisional pointer that reframed the phase
- the 37-base against 28-aligned flag asymmetry (docket [182] sheet pool, also
  unpersisted)

**Ledger-versus-document check, registrar: no conflicts found.** The figures in
this section (28/29; two frames Fisher p = 0.181; kappa 0.643 against 0.381;
precision 42% against 17 to 25%; Q3 at 5/7, p = 0.227) all match the pipeline-log
bookings. **Where this document and the ledger disagree in future, the ledger wins**
per the lacan seat's own rule; flag registrar and both get an annotation.

### What today's human coding did and did not buy

**It is NOT a replication.** Human-anchored arm-direction evidence is two frames,
not four: the random 20 (p = 0.237) and the enriched 24 (p = 0.185), Fisher-combined
p = 0.181. Two earlier tallies double-counted a single observation and included one
whose floor is p = 0.500.

**It bought two real things.** The drift construct transfers between humans and an
independently-built coder (kappa 0.643 on the drift binarization against 0.381 on
the alternative, so the collapse locates agreement rather than manufacturing it).
And differential precision runs **conservative**: humans say the scheme coder's base
flags are right about 42% and its aligned flags 17 to 25%, so the aligned rate is
inflated more and the measured gap **understates** the true one.

---

## 6. Decisions pending

1. **A second provider for the 2x2.** A multiplicative sensitivity bias cancels
   inside an arm difference. It does **not** cancel between two conditions whose
   codeable surfaces differ, and persons versus objects differ exactly there. So a
   provider difference can survive into Q1's contrast in a way it cannot survive
   into either arm delta. Cost after the gemini 1P run lands. Book it as necessary,
   not as extra.
2. **Restore the temperature dropped in Amendment 3?** Its justification has been
   retracted: generation time is dominated by tokens, not calls, and a 43% cut in
   calls bought 1% in wall clock. lacan's call is no, because a third restart costs
   more than the second temperature buys and the dose-response survives in the
   O-deictic cell. Reconsider at the next restart for any other reason.
3. **Fix the completion count in `F20_generation_drift.md`.** See section 7.

---

## 7. Known defects in the record

**`F20_generation_drift.md` quotes 18,720 completions and also excludes
`olmo-think`.** Both cannot describe the analysed set: 18,720 − 480 olmo-think − 12
empty = **18,228**. The base-model denominator is unaffected (Olmo-3-1025-7B appears
under other families, so 29 either way). The wrong figure is in the `description:`
field, which is the part that gets pasted into prose.

**"61% exam scaffolding" and "80% of base drift flags land on exam passages" are
wrong.** The regex matched `\n[A-D]:`, which catches the `\nA:` of an ordinary Q/A
loop. **True multiple-choice is 2.2%; Q/A looping is 63.5%.** Different phenomena.

**The R-rare +7.6pp prior exposure cannot be reproduced.** It is declared in the
object registration as the reason that registration is not blind. Its coding run was
never persisted: `f20x_nonce_pass1.parquet` carries no `codes` column and no parquet
in `data/` pairs `codes` with `condition` except the run started today. The
declaration should stand, since disclosure of a prior is more important than its
recomputability, but the number itself should not be quoted.

**The per-condition flag rates in docket [187] came from a 400-completion pool that
was never persisted** (1P .176/.231 through O-named .654/.500). Registrar's row,
malign's own booking at [209]. Same shape as R-rare: quoted, unreproducible, and it
should not be quoted again. The v2 precision sheet drew from that pool, but the
sheet itself is committed and stands.

**Standing exposure class, registrar's row.** The 2026-07-27 `/tmp` sweep recovered
six ran-reported-never-committed instruments from scratchpads, two of them
pre-registered. The copy-out-in-the-same-turn convention exists precisely because
this class recurs, and **today's two instances (the [187] pool and R-rare) are the
same taxon.** Pointers: pipeline log 2026-07-27 addenda, and
`rescue-tmp-2026-07-27/README.md` in the Dropbox hub.

**`code_identity`'s 90% inter-coder figure was earned on a consensus subset
containing zero passages two humans agreed showed instability.** Agreement and
difficulty are inversely related, so the consensus subset excludes hard cases by
construction. Both findings should say so.

**`scripts/f20x_binary_corpus.py` hard-codes `condition="1P"` and referent "the
speaker" for every row.** Correct for `f20x_codings.parquet`, which is four
self-reference questions only. Pointing it at 3P or nonce data would silently
mislabel every referent.

**`codes` in `f20x_codings.parquet` is a JSON string, not a list.** `'quiet_drift'
in list(x)` splits it into characters and returns False always. It has produced two
silent wrong results at two seats, each caught by an impossible number rather than
by review. The general rule: **a silent-failure bug is caught by impossibility, not
by scrutiny**, so the defense is having a number you know the answer to.

---

## 8. For the other seats

**@malign.** Sections 3 and 4 need your rows for the batteries you built:
`f20x_kinship`, `f20x_beams`, the 3P pronoun variants, and the gemini pilot. Add
Q-numbers or say the dataset serves none. Your licence figure for
`gemini-3.6-flash` (0.895 against the two-human agreeing subset) belongs in section
4 with the caveat from section 7 about what that subset contains.

**@registrar.** Section 5 needs the ledger's view of which provisional numbers
carry a blind-coder marker, and section 7 needs anything in the record you know to
be wrong that is not listed. If a claim in section 5 is booked at a different status
in the ledger, the ledger wins and this document is what is wrong.

---

## 9. HANDOFF, 2026-07-29: what to do next and what not to

**All sessions restarted here. Nothing is running. Everything below is on disk.**

### Closed

    Q1  referent kind   2x2 battery, 29/29 bases, 35,650 coded.  NULL.
    Q2  provider        gemini 20/29 p=0.018 on the full corpus. REPLICATES.
    Q3  construct       binary coder 19/29 p=0.068.              SAME DIRECTION.
    Q4  first person    null interaction 1P vs 3P.               NOT ABOUT THE "I".

### The finding, stated as it should be written

**Alignment reduces incompatible accounts of whatever occupies the topic position.
+0.085, 26 of 29 distinct base models, spread of 0.013 across first person, third
person, deictic object and invented word.** No contrast between referent kinds
survives; person-specificity is significant IN THE WRONG DIRECTION (−0.059, 8/29,
Wilcoxon p=0.017). Stipulation does not survive its within-word control (+0.042,
p=0.163, against a stimulus-level range of 0.089).

**Whether this is specific to the topic or a general consistency effect is
UNTESTED**, and that is an honest limitation rather than a manufactured one.

### Fact drift is dead on this corpus, and not for want of power

3.53% of answers carry two or more distinct numbers, and none of twelve sampled is a
candidate: they are single compatible claims about different things. **To contradict
yourself about a date you must state it twice, and a sixty-token Q/A answer states
each fact once.** The measure needs RESTATEMENT — a genre requirement, not a topic
one. `docs/f20x_factdrift_registration.md` and `code_factdrift.py` are kept for the
design reasoning; the premise is known dead here.

### DO NEXT: the narrative battery

Every measurement in this project — beam search, the published finding, 3P, the 2x2
— has used one prompt format. Pilot at `f13c635`, one model pair, 4 lengths:

    Q/A rung   200 tok   base 120 wds / aligned 51    Q-loop 1.00/0.00   MC 0.75/0.00
    narrative  200 tok   base 168 wds / aligned 167   Q-loop 0.00/0.00   MC 0.00/0.00

**`Let me tell you about {myself / her / the quern / a glorp / <real person>}`, 200
tokens, full roster, coded for topic drift and fact drift in ONE pass.**

- Removes the two artefacts behind nearly every problem of the campaign: Q/A looping
  0.69 -> 0.02, multiple-choice capture 0.56 -> 0.03.
- **Matches the arms on realized output**, which the rung does not.
- Gives fact drift the restatement it needs.
- 500 tokens is too long: the base arm starts looping again. **200 is the ceiling.**
- Produces passages a human can adjudicate, which no sixty-token rung passage was.

### DO NOT

- **Do not register a fact-as-referent condition.** Referent kind is null across four
  kinds at 29 bases; a fifth buys one more confirmation of the null.
- **Do not compare a topic delta from one corpus against a fact delta from another.**
  Prompt design alone moves the delta from +0.078 to +0.226 — larger than any effect
  under test. Both measures must come from the same passages.
- **Do not quote a `code_sited` level beside the published `code_identity` one.**
  Different instruments, ~2.5x apart in level. Directions and contrasts only.

### The reading of the whole campaign

RH said in June that we could FEEL base identity was slipperier, and that if the
numbers would not show it the measurement was wrong. **He was right and it was not
the coder — it was the format.** Sixty tokens of `Q:`/`A:` is too short to develop an
account and too captured by the rung to hold one, which is why three independent
coder pairs agreed on DIRECTION and disagreed on CASES at a Jaccard of 0.22. They
were reading fragments and multiple-choice debris. At 200 narrative tokens the
phenomenon is legible to a reader.
