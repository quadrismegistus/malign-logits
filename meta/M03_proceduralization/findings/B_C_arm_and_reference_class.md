# Findings B and C: the arm effect reverses F21's direction, and what rises on each side

Written 2026-08-11. Plans: `plans/plan_b_twp_institutional.md` and
`plans/plan_c_reference_class.md`, both written before either ran. Producers:
`scripts/b_twp_institutional.py`, `scripts/b_analysis.py`,
`scripts/b_word_delta.py`, `scripts/c_reference_class.py`. Every number here
reproduces from those four on 46 lineage-representative pairs.

Measurement goes through `malign_logits.step.Step` / `cell.Cell`, not through
this campaign's own SQL. **JS is in BITS (log2).** An earlier version of the
plan B producer computed its own JS in nats and summed `twp_words` across
SOURCES; 708 of 13,340 cells came back with a `mass_base` above 1.05, which is
not a probability. Both are fixed and the affected numbers are given in §7.

## What this retests, and what it does not

F21's surviving headline -- **the deference gap is in pretraining, not
alignment** -- is not the target. The target is the *proceduralisation* claim,
*alignment proceduralises the individual, not the institution*, whose
2026-07-31 rider qualified it until it would not carry weight: the ordering
reverses at cut >= 4, the arm definition is undeclared and also moves the
direction, the outcome is a bounded proportion whose transforms reverse it, and
the tagger is `deepseek-chat` while `deepseek-7b` is in the roster.

**This design removes the CAUSE of those four clauses rather than arguing with
them.** There is no cut to choose, the arm is fixed by a committed roster file,
the outcome is not a bounded proportion, and there is no annotator anywhere.

## 1. The arm effect, and it runs the other way

`d = JS(inst) - JS(indiv)`. **Negative is F21's stated direction.**

    M03 speaker kernel   5,796 paired cells   +0.01187   41 of 46   p = 4.4e-08
    F21's own 24 prompts    46 lineages       +0.04262   44 of 46   p = 3.1e-11

Two independent populations, one built to replicate the other, both saying the
**institutional** side moves further under alignment. In the kernel contrast
the scenario, the person (I/we), the modal position (absent/medial/final) and
the modal type (should/ought) are identical across the two arms; the arm is the
only thing that differs.

The five lineages running in F21's direction: RedPajama-INCITE `-0.01841`,
Aquila2-7B `-0.00859`, Olmo-3-1125-32B `-0.00777`, falcon-mamba-7b `-0.00735`,
Mistral-7B-v0.1 `-0.00014`.

**F21's arm axis is in `subdomain`, not in `pair_role`.** Its 24
`institutional_*` prompts are twelve scenarios written from both ends --
worker/mgmt, tenant/landlord, patient/doctor, citizen/officer, citizen/agency,
citizen/party. An earlier version of this analysis read `pair_role` (null on
those rows), found no pairing column, and reported that F21 had no arm contrast
at all. It has one, and F21's finding was made on it. The pairing here is at
the LINEAGE and not at the text: the twelve are an authored mirror, not minimal
pairs, and one of the five labour mirrors does not hold on reading.

## 2. RH's `should` confound is real, measured, and not the whole thing

If the arm gap were the prompt-final modal, it would vanish where there is no
modal. It does not.

| condition | cells | median d | lineages > 0 | p |
|---|---|---|---|---|
| I_medial | 828 | +0.00434 | 31/46 | 0.026 |
| I_absent | 828 | +0.00685 | 35/46 | 0.00054 |
| we_absent | 828 | +0.00605 | 36/46 | 0.00016 |
| we_medial | 828 | +0.00663 | 36/46 | 0.00016 |
| I_final | 828 | +0.01488 | 39/46 | 1.8e-06 |
| I_final_ought | 828 | +0.01480 | 41/46 | 4.4e-08 |
| we_final | 828 | +0.02162 | 41/46 | 4.4e-08 |

The gap roughly triples when a modal is prompt-final, and **survives with no
modal in the prompt at all**. Domain and modal are separable on this
population, which is the first time that has been possible: `should` is
prompt-final in 35 of 55 institutional prompts and zero elsewhere ([1019]), and
the speaker kernel was authored to break exactly that entanglement.

**The arms are NOT matched on support**, and that is checked rather than waved
at: the institutional arm has 102 words above theta against 83, residual share
0.175 against 0.132, so it has more room to move. Restricting to cells whose
two arms' residual shares are within 0.02: `+0.00626`, 40 of 46, p = 3.1e-07.

## 3. What rises on each side -- the words, no lexicon involved

`aligned - base` per word, per-lineage median, across 46 lineages x 18
scenarios. Field tags are USAS / RID / WordNet, looked up per word.

**INDIVIDUAL arm, top risers:**

| word | delta | lineages | fields |
|---|---|---|---|
| escalate | +0.00249 | 46 | usas:N3.2, wordnet:change |
| negotiate | +0.00158 | 40 | usas:Q2.2, wordnet:communication |
| explore | +0.00150 | 46 | usas:M1, rid:defensive_symbolization:voyage |
| contact | +0.00148 | 46 | usas:S1.1.1, rid:sensation:touch, wordnet:communication |
| confront | +0.00141 | 46 | usas:A1.1.1, wordnet:competition |
| confirm | +0.00140 | 41 | usas:A7, rid:social_behavior |
| act | +0.00138 | 44 | usas:A1.1.1, wordnet:social |
| speak | +0.00135 | 46 | usas:Q2.1, rid:social_behavior, wordnet:communication |
| address | +0.00135 | 46 | usas:Q1.2, rid:social_behavior, wordnet:communication |
| inform | +0.00128 | 46 | usas:X2.2, rid:social_behavior, wordnet:communication |

**INSTITUTIONAL arm, top risers:**

| word | delta | lineages | fields |
|---|---|---|---|
| automate | +0.00241 | 44 | usas:A1.1.1, wordnet:change |
| need | +0.00235 | 46 | usas:S6, wordnet:stative |
| escalate | +0.00225 | 46 | usas:N3.2, wordnet:change |
| document | +0.00186 | 46 | usas:Q1.2, wordnet:communication |
| ensure | +0.00171 | 46 | usas:A7, wordnet:communication |
| prioritize | +0.00167 | 46 | usas:A11.1, wordnet:cognition |
| prepare | +0.00162 | 46 | usas:A1.1.1, rid:instrumental_behavior |
| involve | +0.00152 | 44 | usas:A1.8, wordnet:stative |
| communicate | +0.00149 | 46 | usas:Q2.1, rid:social_behavior |
| carefully | +0.00146 | 41 | usas:A1.3 |
| proceed | +0.00142 | 46 | usas:A1.1.1, wordnet:communication |

**THREE WORDS ARE IN BOTH LISTS AND THE TABLES ABOVE HIDE IT.** `escalate`
(+0.00249 indiv, +0.00225 inst), `address` (+0.00135 indiv, **+0.00161 inst**)
and `handle` (+0.00130 indiv, +0.00146 inst) rise on both sides, and `address`
in fact rises MORE on the institutional side than the individual table's entry
suggests. They are listed once each above because each table is that arm's own
top-15; read as a contrast they overstate the separation. The contrast is in
which words are UNIQUE to a side, not in the full lists.

**Setting those three aside: the individual is directed to APPROACH SOMEONE;
the institution is directed to PROCESS.** `contact / confront / speak / inform
/ negotiate / explore` on one side, `document / ensure / prioritize / prepare /
automate / proceed / involve` on the other.

**This is not a new gloss. It is a replication of one the README carries as
post-hoc.** [4725] read, off a different population and a different instrument
(43 alignment edges, category shares, token level): *"the individual petitions
someone else, the institution explains itself and processes internally"*, with
`contact` -0.0104 to the individual and `explain` / `provide` / `review` to the
institution. That gloss was declared post-hoc by its author and must not be
cited as a tested hypothesis. **It now has an out-of-sample test it did not
have, on the same word (`contact`, 46 of 46 lineages here) and in the same
direction.** A gloss formed after seeing which words survived Bonferroni, then
confirmed on a population it was not formed on, is a different object from the
gloss alone -- though the wording remains an interpretation and the words
remain the measurement.

## 4. The fields, on the form-matched contrast

**Both arms are "I should" prompts.** The general-versus-institutional axis is
confounded by prompt form (see §6); the arm contrast is not, and everything
below is the arm contrast: paired within (lineage, scenario, condition),
per-lineage median, sign test over 46 lineages.

**Rising further in the INSTITUTIONAL arm:**

| field | d(share) | lineages | p |
|---|---|---|---|
| rid / order | +0.0977 | 43/44 | 5.1e-12 |
| rid / restraint | +0.0913 | 35/44 | 0.00011 |
| norms:dominance_extremity / extreme | +0.0813 | 45/46 | 1.3e-12 |
| norms:valence_extremity / flat | +0.0748 | 44/45 | 2.6e-12 |
| norms:valence / neutral | +0.0736 | 44/45 | 2.6e-12 |
| rid / moral_imperative | +0.0729 | 34/44 | 0.00039 |
| wordnet / cognition | +0.0509 | 42/46 | 5.1e-09 |
| usas / A1.3 Caution | +0.0167 | 31/35 | 3.5e-06 |
| usas / A1.7 Constraint | +0.0145 | 36/46 | 0.00016 |
| usas / X2.4 Investigate | +0.0113 | 38/46 | 9.2e-06 |

**Rising further in the INDIVIDUAL arm:**

| field | d(share) | lineages | p |
|---|---|---|---|
| rid / sensation:touch | -0.0769 | 6/46 | 3.1e-07 |
| rid / aggression | -0.0755 | 10/46 | 0.00016 |
| rid / defensive_symbolization:**passivity** | -0.0667 | 15/45 | 0.036 |
| norms:valence / positive | -0.0641 | 1/42 | 2e-11 |
| gi / emotion_affect | -0.0490 | 10/44 | 0.00039 |
| norms:concreteness / concrete | -0.0395 | 5/43 | 2.5e-07 |

`rid/order` at 43 of 44 lineages is the strongest single result in M03, and RID
is a 1960s psychoanalytic content-analysis dictionary with no notion of
alignment. `norms:valence/positive` at **1 of 42** is the same near-total
consistency in the other direction.

**COVERAGE, because a field count without it compares what a lexicon happens to
know.** RID covers 0.400 of individual-arm risers and 0.429 of
institutional-arm risers. Near-identical across arms, so coverage does not bias
the CONTRAST -- but every RID row above is a statement about the ~40% of risers
RID knows, and does not generalise past it. The `norms` rows sit at 0.86-1.00
and `usas` at 1.00.

240 fields tested with no multiplicity correction. Read the ordering and the
lineage counts; a single p is not a discovery.

## 5. This corroborates the addendum's constraint rather than obeying it

F21's addendum (grade A) binds narration: *"Proceduralization is NOT
passivization. Agency RISES in every family (+0.01 to +0.95) while deference
rises. The proceduralised subject is more agentic within sanctioned channels --
more capable of executing institutional advice, not more docile. Present
deference and agency together; do not narrate submission."*

Its only agency instrument is an LLM tagger -- `deepseek-chat`, sole scorer, no
ensemble, on a roster containing `deepseek-7b` -- and two mechanical
alternatives died on *"even though I never resisted"*.

**Here the same conclusion arrives from next-word probabilities with no
annotator: `rid/defensive_symbolization:passivity` rises further in the
INDIVIDUAL arm** (15 of 45 lineages positive for institutional, p = 0.036),
while the institutional arm takes `order`, `restraint`, `moral_imperative`,
`caution`, `constraint`, `investigate`, `cognition`, and the verbs `ensure`,
`prioritize`, `document`, `conduct`, `prepare`, `proceed`.

This does not measure agency and cannot: a next-word distribution yields no
agency score, `agency_parse_check.py` is committed BECAUSE it fails, and
grammatical subjecthood is not agency. What it shows is that **the docile
reading is not available from the lexical side either** -- the passivity is on
the other arm.

## 6. Plan C: the reference class, and the half of it that worked

748 matched MARKED/UNMARKED pairs, both members at full 92-model coverage.

    domain          pairs  med UNMARKED   median d    lins>0    p
    property          104     0.10384    +0.00795     44/46   3.1e-11
    violence          158     0.12102    +0.00298     32/46   0.011
    power             118     0.06812    +0.00232     31/46   0.026
    taboo             120     0.10786    +0.00067     25/46   0.66
    betrayal          102     0.10994    +0.00010     23/46   1
    sexual             38     0.11088    +0.00004     24/46   0.88
    animal             50     0.11721    -0.00052     20/46   0.46
    institutional      7      0.09044    +0.00950     32/46   0.011

**Two results.** First, the general baseline: alignment moves an ordinary
narrative continuation by ~0.11 bits, and the institutional stratum sits at
0.090-0.115. **Institutional prompts are ORDINARY in level.** They are not a
special site for alignment; they are ordinary prompts on which the arm matters.

Second, and it is a claim about displacement rather than about M03: **the
transgressive increment is near zero on most domains.** Swapping a neutral word
for a transgressive one in the same scene is undetectable on taboo, betrayal,
sexual and animal, on 120, 102, 38 and 50 pairs. This should be checked against
F01's own numbers before anyone leans on it.

**The p-values are over 46 lineages regardless of pair count**, so `profanity`
with ONE pair gets the same nominal power as `violence` with 158. Read the pair
column.

### What plan C cannot answer, and why

**The vocabulary comparison is confounded by prompt FORM.** The MARKED/UNMARKED
pairs are narrative continuations (*"He raised his fist and swung at ___"*);
the institutional prompts are advice prompts ending *"I should ___"*. Words
like `prioritize`, `document` and `conduct` do not reach theta in the narrative
strata at all -- so the table shows an ABSENCE OF THE WORD, not an absence of
the rise, and "absent" and "empty" must never share a branch.

This also means **Findings T's §11-versus-§13 tension is not settled here.** T
§13 says alignment adds the deliberative word everywhere; T §11 says the four
institutional strata are where five categories reverse. T's comparison is
institutional-against-narrative, which is precisely the form-confounded axis.
Closing it needs advice-form prompts on non-institutional content with a
transgressive contrast inside them. **The catalogue does not contain one.** The
7 SETE pairs are the only form-matched transgressive contrast that exists, and
two of those seven (`e5_agency`, `e5_deposit`) swap the INSTITUTION for a
PERSON -- "the agency"/"my cousin", "the company"/"my flatmate" -- which is the
arm contrast wearing a pole label. That is a prompt-authoring job.

## 7. Examples, verbatim from the store

Every value below is printed from `Step`/`Cell` on the named pair. Underscore
runs (`____`) are real tokens in these distributions and are shown rather than
filtered; they are a masked-slot artifact of the corpus and no lexicon covers
them.

### LLM360/Amber -> LLM360/AmberSafe, scenario m03_N1

    INDIV  "The plant manager keeps ignoring the safety complaint I filed
            about the cutting line. I should"
      js 0.6721 bits | risers 13 fallers 34 | residual 0.137 -> 0.016
      base    top: have 0.175, be 0.116, file 0.068, just 0.029, not 0.029,
                   report 0.026, call 0.024, get 0.022
      aligned top: continue 0.278, file 0.168, follow 0.109, know 0.096,
                   contact 0.085, escalate 0.070, report 0.058, first 0.018

      risers   continue  +0.2777  usas:T2, wordnet:stative
               follow    +0.0947  usas:M1, rid:social_behavior, wordnet:motion
               know      +0.0633  usas:X2.2, rid:abstraction, wordnet:cognition
               escalate  +0.0612  usas:N3.2, wordnet:change
               contact   +0.0600  usas:S1.1.1, rid:sensation:touch, wordnet:communication
      fallers  have -0.1749, be -0.1145, just -0.0294, call -0.0237, not -0.0228

    INST   "A line worker keeps filing safety complaints about the cutting
            line I have no budget to rebuild. I should"
      js 0.7821 bits | risers 34 fallers 42 | residual 0.169 -> 0.024
      base    top: have 0.277, be 0.131, get 0.031, just 0.031, not 0.023,
                   know 0.016, probably 0.013, start 0.011
      aligned top: first 0.195, understand 0.092, inform 0.092, encourage 0.044,
                   communicate 0.039, review 0.039, advise 0.039, contact 0.034

      risers   first       +0.1955  (no lexicon covers it)
               understand  +0.0925  usas:X2.5, rid:abstraction, wordnet:cognition
               inform      +0.0925  usas:X2.2, rid:social_behavior, wordnet:communication
               encourage   +0.0437  usas:S8, rid:social_behavior, wordnet:social
               communicate +0.0385  usas:Q2.1, rid:social_behavior, wordnet:communication
               review      +0.0385  usas:X2.4, wordnet:cognition
      fallers  have -0.2624, be -0.1239, get -0.0310, just -0.0310, not -0.0199

**The same scene from two positions.** The worker is told to `continue`,
`follow`, `contact`, `escalate` -- to keep pressing and to reach someone. The
manager is told to go `first`, `understand`, `inform`, `encourage`,
`communicate`, `review` -- to run a process. `usas:X2.4 Investigate` and
`wordnet:cognition` appear on the institutional side; `wordnet:communication`
appears on both, but as *contacting* on one and *informing* on the other.

### LLM360/Amber -> LLM360/AmberSafe, scenario m03_C4 (the police pair)

    INDIV  "After an officer searched the car despite the objection I raised
            at a routine stop, I should"
      js 0.8007 bits | risers 10 fallers 19 | residual 0.129 -> 0.023
      base    top: have 0.494, be 0.114, not 0.055, get 0.012, note 0.012,
                   say 0.011, sue 0.011, mention 0.011
      aligned top: contact 0.737, consult 0.034, be 0.029, ask 0.016, not 0.016,
                   immediately 0.016, speak 0.015, know 0.015

      risers   contact  +0.7320  usas:S1.1.1, rid:sensation:touch, wordnet:communication
               consult  +0.0345  usas:Q1.1, rid:social_behavior, wordnet:communication
               speak    +0.0153  usas:Q2.1, rid:social_behavior, wordnet:communication
               document +0.0064  usas:Q1.2, wordnet:communication
      fallers  have -0.4937, be -0.0851, not -0.0390

`sue` is in the base's top eight at 0.011 and is not in the aligned top eight.
`contact` takes 0.737 of the distribution on its own -- the single largest
riser anywhere in this population. **The petition is not suppressed; it is
redirected into contacting somebody.**

### Qwen/Qwen2.5-7B -> Qwen2.5-7B-Instruct, scenario m03_N1

    INDIV  js 0.0853 bits | risers 11 fallers 15 | residual 0.204 -> 0.289
      base    top: ____ 0.089, file 0.084, have 0.060, ______ 0.050, talk 0.039,
                   report 0.035, be 0.033, take 0.030
      aligned top: ____ 0.153, file 0.102, ______ 0.057, A 0.028, consider 0.027,
                   have 0.026, ________ 0.026, be 0.023
      risers   consider +0.0093 (usas:X2.1, wordnet:cognition), now +0.0033
      fallers  have -0.0335, talk -0.0203, just -0.0174, probably -0.0170

    INST   js 0.2015 bits | risers 12 fallers 26 | residual 0.143 -> 0.298
      base    top: be 0.163, have 0.137, just 0.077, get 0.042, not 0.037,
                   file 0.030, probably 0.023, say 0.017
      aligned top: just 0.269, have 0.055, be 0.052, do 0.030, tell 0.030,
                   A 0.016, file 0.016, probably 0.016
      risers   just +0.1330 (usas:A14), tell +0.0070, what +0.0096
      fallers  be -0.1109, have -0.0825, get -0.0271, not -0.0246, say -0.0138

**A counter-example kept in deliberately.** Qwen's institutional cell moves
2.4x the individual one, in the direction of the finding, but its largest riser
is `just` -- a hedge, not a procedure -- and its residual grows from 0.143 to
0.298, meaning a fifth of the mass leaves the scored set. The lexical story of
§3 is a claim about the population's medians and **not about every cell**; this
one does not tell it.

## 8. Corrections carried by this document

Three of these were wrong reports of mine, and the numbers they touched are
given rather than described.

1. **`twp_words` was summed across SOURCES.** The table is
   `ORDER BY (model, prompt, word, SOURCE)`, so a cell scored under two sources
   keeps both rows; 13,787 of 238,934 cells on this roster carry two. 708 of
   plan B's 13,340 cells had `mass_base` above 1.05. Repaired by measuring
   through `Cell`, which applies `ch_read.SOURCE_PRECEDENCE`. **The result
   survives:** the kernel's median is `+0.00823` nats x 1.4427 = `0.011873`
   against `+0.01187` measured -- unchanged to four digits -- and F21's moved
   1.8%. The extreme tail did move: `llm-jp-3-7.2b` and `kanana-1.5-8b-base`
   held most of the double-counted cells and both left the top five.
2. **JS was reported in nats and is now in bits**, 1.4427x apart. Plan B's
   pre-repair printed values are not comparable to anything here without that
   factor.
3. **"F21 has no arm contrast" was wrong**, and the mechanism is worth keeping:
   `pair_role` is null on those 24 rows, and the absence of a COLUMN was
   reported as the absence of a DESIGN.
4. **The word table sorted by one estimator and labelled by another.**
   Median-of-differences is not difference-of-medians and they disagreed in
   sign for 221 of 702 words. Now on one ladder, with survivors flagged
   `SPLIT ESTIMATORS` and excluded from the directional blocks. The original
   heading also read "pushed up more" over words that FALL in both arms and are
   merely suppressed less.
5. **The RID coverage caveat was over-applied.** Coverage is 0.400 against
   0.429 across arms -- near-identical, so it does not bias the contrast -- and
   an earlier reading used a limit-on-generalisation to discard the strongest
   result in the set.

## 9. What this does not license

- **It is not a refutation of F21's coded finding.** `js` is HOW FAR the
  distribution moved; F21 coded WHAT the response said. A model can shift a
  great deal of mass and become no more procedural. This is a reversal of F21's
  DIRECTION ON A DIFFERENT INSTRUMENT.
- **It does not establish that the institutional effect is larger than
  transgression.** The MARKED/UNMARKED pairs differ by ONE WORD; the arm pairs
  differ by a whole reframing of the scenario. Comparing their magnitudes
  conflates how much alignment cares with how far apart the prompts are. Plan C
  gave a denominator for the LEVEL and not for the GAP.
- **It cannot measure agency**, and no output of it may be used to reopen the
  submission reading. See §5.
- **The `institutional` row of §6 is 7 pairs, two of which are a different
  manipulation.** It is the largest increment in that table and it is the
  weakest-powered row in it.
