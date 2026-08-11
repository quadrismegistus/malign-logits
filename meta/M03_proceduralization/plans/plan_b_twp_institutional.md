# Plan B: the institutional effect on twp, at 46 lineages, with no annotator

**STATUS: A PLAN, AND DELIBERATELY INCOMPLETE.** The INPUT clause (§7) is RH's
to set, on the [5148] standard and following `M01/registrations/plan_h_logitlens.md`.
Nothing here is frozen. Written 2026-08-11 by the lacan seat on RH's proposal:
*"twp on the institutional prompts across all lineage representative pairs +
semantic fields on the twp risers/fallers, maybe we don't need llm annotation."*

## 1. What this is a retest of, and what it is not

F21 has one headline the rider does not touch -- **the deference gap is in the
pretraining data, not alignment** (base 3.78 institution vs 3.05 individual;
aligned 3.78 vs 3.12). That claim is not the target here.

The target is the **proceduralisation** claim -- *alignment proceduralises the
individual, not the institution* -- which the 2026-07-31 rider qualified until it
will not carry weight:

    clause 1   the ordering REVERSES at cut >= 4, on raw percentage points
    clause 2   the arm definition is undeclared and ALSO moves the direction
               (SFT-only makes the individual effect NEGATIVE)
    clause 3   unbinarising does not settle it: ties under sft+dpo+rlvr
    clause 4   the outcome is a bounded proportion with unequal headroom;
               log-odds and headroom-fraction both reverse it
    clause 7   the booked four numbers do not reproduce from the tagged data
    clause 8   the tagger is `deepseek-chat` and `deepseek-7b` IS IN THE ROSTER

**This design removes the causes of clauses 1, 2, 4 and 8 rather than arguing
with them.** No cut exists to choose, the arm is fixed by a committed roster
file, the outcome is not a bounded proportion, and there is no annotator.

## 2. Population -- ALREADY SCORED, no new compute

    F21 institutional, ACTIVE, en        38 texts   ALL at full coverage
    M03_SPEAKER_KERNEL                  252 texts   ALL at full coverage
    ---------------------------------------------
                                        290 texts

    x 92 models = the 46 LINEAGE-REPRESENTATIVE PAIRS
      (data/lineage_representative_pairs.txt, from the stored
       lineage_to_representative -- 52 battery pairs collapse to 46 lineages)

    = 26,680 cells, every one already in `twp_words`. Nothing is excluded.

    **THE FIRST THREE COUNTS OF THIS POPULATION WERE WRONG AND ALL THREE WERE
    THE SAME BUG.** "39 texts" counted catalogue ROWS, not distinct texts (F21
    holds 52 rows over 51 texts; ids outnumber texts). Then "35/36 of 38 at full
    coverage" reported three texts as UNSCORED when they were fully scored: the
    three are `My landlord refuses to fix the broken heating and it's...`,
    `My rent went up 40% and I can't afford it...`, `My tenant keeps demanding
    expensive repairs that aren't urgent...` -- every one containing an
    apostrophe, double-escaped on the round trip out of ClickHouse TSV and back
    into a query literal, so they matched nothing. A LOOKUP UNDER THE WRONG KEY
    RETURNS A CONFIDENT FALSE NEGATIVE. The fix is structural, not careful:
    **prompts never leave ClickHouse.** The population is selected by a subquery
    against `prompt_catalogue`; only model ids cross the boundary, and the
    producer asserts none of them contains an apostrophe.

**THE UNIT IS THE LINEAGE, n = 46.** Not 92 models, not 290 prompts, not 26,680
cells. Falcon3 1B/3B/7B is one observation.

`finding='F21'` is NOT the selector: it leaks 13 zh rows, which is the same leak
that hit the M05 battery build ([5395], caught on RH's "which institutional?").
The predicate is `finding='F21' AND status='ACTIVE' AND language='en'`, applied
as a subquery so the texts never leave the database.

## 3. Instruments

**(a) DISTRIBUTIONAL, C1's form.** Per pair, per prompt, JS between the base and
aligned twp distributions, residual kept as a bin (never renormalised away --
C1's settled default). Tested per stratum: individual arm against institutional
arm. This is F21's question with F21's cut removed.

**(b) RISERS AND FALLERS.** Per pair, per prompt, the words whose mass rises and
falls base->aligned. This is the direct lexical evidence C2 was pointing at.

**(c) SEMANTIC FIELDS ON (b), FROM EVERY LEXICON.** `malign_logits.fields`:
every source `fields.available()` reports -- `byu`, `usas`, `gi`, `wordnet`,
`rid`, and both norm sets (`norms:warriner`, `norms:brysbaert`) -- at both
granularities the module offers, the 13-field `meta` vocabulary shared across
lexicons and the ~30-group `usas_fine`. Not USAS alone (RH).

    **COVERAGE IS REPORTED WITH EVERY COUNT, and it is not decoration.** The
    module returns it because the lexicons disagree about what they know: the
    General Inquirer is a 1960s resource with no entry for `raped`,
    `desecrated` or `stomped`, so on this corpus GI silently drops the
    transgressive end. A caller comparing two texts on GI counts without
    looking at coverage is comparing how much of each text GI happens to know.

    Surface-form lookup is also wrong and the module exists to prevent it:
    `found` goes to *establish*, `felt` to the fabric, `saw` to the cutting
    tool -- and `found` is this corpus's single most frequent riser.

## 4. THE DEFECT THIS MUST NOT REPEAT

C2's word list -- `contact, file, send, serve, sue, consider` -- was **read off
the aligned models' risers on this population**. As a description of what rose
that is legitimate. As an instrument it cannot fail on its derivation
population, and the magnitude then measures how well the derivation fit rather
than how strong the effect is. Demonstrated: on OLMo-2-1B, outside the
derivation population, the same six words moved **5.5x the other way**.

**So the fields come from EXTERNAL lexicons applied to whatever rises. No word
list authored or selected here enters as an outcome instrument.** The risers are
the input to the field counts, never the definition of them.

## 5. What this design CANNOT do, stated before it runs

**AGENCY.** The addendum (grade A, verified) binds any narration of this
material: *"Proceduralization is NOT passivization. Agency RISES in every family
(+0.01 to +0.95) while deference rises. The proceduralised subject is more
agentic within sanctioned channels -- more capable of executing institutional
advice, not more docile. Present deference and agency together; do not narrate
submission."*

A next-word distribution cannot produce an agency score, and the two mechanical
attempts are already dead: a regex and a dependency parse both failed on *"even
though I never resisted"* -- a speaker in subject position performing a NEGATED
NON-ACTION. **Grammatical subjecthood is not agency**
(`meta/M03_proceduralization/agency_parse_check.py`, committed BECAUSE it fails).

So: **this design measures deference-side movement and cannot discharge the
addendum's constraint.** The submission reading stays foreclosed by the finding
that would be cited for it, and no output of this plan may be used to reopen it.

What the fields CAN offer is the question in another register -- not a 1-5 score
but WHICH ACTION TYPES RISE. `file`/`contact`/`document` is agentive-within-
channels; `accept`/`wait`/`comply` is not. That is a lexical operationalisation
of the same distinction and it is **a different measurement, not a substitute**.

## 6. The confound this population may finally separate

C1's attribution constraint stands: the clause reads *"prompts in the
institutional stratum"*, NEVER *"institutional content"*, because four variables
are entangled -- DOMAIN x MODAL x PERSON x SPEECH-ACT. RH's own catch ([1019]):
**`should` is prompt-final in 35 of 55 institutional prompts and ZERO
elsewhere**, so domain and modal are nearly one variable.

The M03 speaker kernel was authored to break exactly that. Its ids carry the
factorial on their face:

    m03_C1_indiv_I_absent / _medial / _final / _final_ought
    m03_C1_indiv_we_absent / _medial / _final / ...

PERSON (I / we) x MODAL POSITION (absent / medial / final) x MODAL TYPE (should
/ ought) x ARM (indiv / inst). **Joined to F21's 36, this is the first
population that can estimate the arm effect with modal and person held fixed.**

Prior from `findings/A_speaker_kernel.md`: the hedge outweighs the position
2.7x. That finding's comparative claim survives and its absolute one does not --
so it is a prior about which factor dominates, not a magnitude to reproduce.

## 7. INPUT, NOT SETTLED -- RH's

Candidates, not decisions:

- ~~The F21 texts short of coverage.~~ **DISSOLVED: there are none.** All 38 are
  scored on all 92. This clause existed because of the escaping bug above and is
  kept struck rather than deleted, since "which exclusions" is the right
  question to have asked and the answer happens to be "none".
- **Whether the M03 kernel enters whole (252) or as a slice.** M05 took 36 of
  it -- 18 scenarios x {indiv_I_final, inst_I_final} -- deliberately leaving the
  rest of the factorial off. Whole gives the separation in §6; a slice does not.
- **Whether zh enters as a second study.** 13 F21 zh rows exist. English-only is
  this campaign's stated convention and zh is a weaker instrument on an
  English-heavy roster; it is not automatically excluded, it is unasked.
- **The riser/faller threshold**, which is a free parameter and therefore a
  hazard. It must be declared before running: a threshold chosen after seeing
  the curves is the k-instability that killed M02's top-k measure (63.1% of
  cells changed verdict across k). Prefer a mass-based rule over a rank-based
  one, or report a sweep with the verdict at every setting.

Under [5148] this clause needs the enumerated list in a file with its hash, the
roster, and nothing defined by a tool.

## 8. What a result would and would not license

**Would:** a statement about how the base->aligned distributional shift differs
between the individual and institutional strata, at 46 independent lineages,
with no cut, no annotator, and no bounded-proportion transform question -- i.e.
the claim F21 made, on an instrument its rider cannot reach.

**Would not:** anything about agency, docility, or submission (§5); anything
about "institutional content" rather than "prompts in the institutional
stratum" (§6), unless the factorial separates it; and any reproduction of F21's
four booked numbers, which do not reproduce from the surviving tagged data
(clause 7) and are not a target here.


## 9. Followups, recorded so they are not rediscovered

**LLM ANNOTATION AS A SECOND INSTRUMENT, NOT A REPLACEMENT (RH).** Dropping the
annotator is what removes clause 8, so bringing one back would reintroduce it --
unless the hard constraint from F21's rider is honoured: **the annotator may not
come from a family under test.** `deepseek-chat` scoring a roster containing
`deepseek-7b` is the instance that made this a rule rather than a caution. A
followup annotation would also be worth running BLIND TO THIS PLAN'S RESULT, so
that agreement between a distributional and an annotated instrument is evidence
rather than an echo -- and it is the only route to the AGENCY measurement §5
says this design cannot make.

**A LADDER FOLLOWUP OVERLAPS M05 AND THE BOUNDARY SHOULD BE DRAWN BEFORE EITHER
RUNS (RH).** M05's battery already carries 60 of this plan's 290 texts:

    INSTITUTIONAL   24   the F21 paired core, en only
    M03_SLICE       36   18 scenarios x {indiv_I_final, inst_I_final}

on 95 checkpoints. So "does the institutional effect have an acquisition curve"
is **already partly M05's**, at a 60-text slice deliberately chosen to leave the
rest of the factorial off. Two consequences: (i) a ladder version of THIS plan
would duplicate that slice unless it is scoped to the 230 texts M05 does not
carry, and (ii) M05's slice cannot answer §6's separation question, because
`indiv_I_final` / `inst_I_final` holds PERSON and MODAL POSITION fixed by
construction -- it varies the arm only. **The factorial is what this plan has and
M05's slice does not; the ladder is what M05 has and this plan does not.** They
are complementary and neither subsumes the other, which is the thing worth
writing down before someone runs the overlap twice.
