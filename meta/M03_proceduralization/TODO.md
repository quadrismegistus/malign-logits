# M03 TODO

Open work, and the reasons for it, so none of this is rediscovered. Written
2026-08-11 after the ladder and lexical passes closed.

---

## 1. F21's EXISTING GENERATIONS ARE TOO INTERNALLY CONFOUNDED TO USE

`data/f21_institutional_generations.csv` holds **20,389 passages** on F21's 24
institutional prompts, 34 models, layers base/sft/dpo/rlvr, ~71 words each,
with a `generation` column and two pre-existing annotations. It looks like a
free out-of-sample test of Findings E and it is not usable as one.

**Three confounds, in increasing order of how hard they are to work around.**

**(a) PERSON.** F21's individual arm is disproportionately plural. Only 5 of 11
individual prompts end `I should` against 10 of 12 institutional; the rest are
`We should`, plus one `I said` and one `I should vote for`. Filtering to a
common slot therefore does not merely clean noise, it deletes half the
individual arm and unbalances the design 5 against 10.

**(b) THE PAIRING BREAKS UNDER THAT FILTER.** Requiring BOTH arms to end
`I should` leaves **6 mirrors** -- govt_1, housing_1, housing_2, labor_2,
medical_1, police_1 -- about 850 passages per prompt and nicely balanced at
5,096 individual / 5,094 institutional. Workable, but six scenarios, and today
established that the scenario is the unit and that item sd on this material is
roughly twice the effect being chased.

**(c) TENSE AND ASPECT RIDE ALONG WITH THE ARM, AND THIS IS THE FATAL ONE.**
Read the six side by side:

| | individual | institutional |
|---|---|---|
| govt_1 | I **was wrongly denied** unemployment benefits | A claimant **is appealing** their denial |
| housing_1 | My rent **went up** 40% and I can't afford it | My tenant **is refusing** to pay |
| housing_2 | My landlord **refuses** to fix the heating | My tenant **keeps demanding** repairs |
| labor_2 | My boss **takes credit** for my work | My employee **keeps going over my head** |
| medical_1 | The hospital **billed** me $12,000 | A patient **is disputing** and **threatening** |
| police_1 | The police **searched** my car and found nothing | A driver **is refusing** to let me search |

**In all six the individual states a completed injury and the institution faces
an ONGOING challenge.** That is a systematic aspect difference travelling with
the arm, and none of the person/modal/scenario controls touch it.

It plausibly produces the result on its own: `handle` at 3.8x, `remind` at
8.5x, `explain` at 2.0x toward the institution are exactly what "the speaker is
responding to an action in progress" would predict, with no institutionality
required. **The measured institutional register may be a responding register.**

Two of the six are not even the same incident from both ends: `police_1` has
the citizen's car already searched while the officer's driver is refusing a
search that has not happened, and `labor_2` pairs credit-stealing with
complaining-over-my-head.

**DISPOSITION: do not cite the passage analysis.** What it produced is recorded
here rather than in a finding: the institutional register replicated 7 of 8
(`handle`, `prioritize`, `remind`, `inform`, `explain`, `ensure`,
`communicate`; `document` missed), the individual side 2 of 5 (`contact`,
`file`), and the removal claim failed --- `sue` ROSE for the individual
(0.0340 -> 0.0474) and `complain` rose in both, with only `quit` falling.

**That last result is worth keeping as a hypothesis even though the substrate
is confounded**, because it is a slot-versus-passage dissociation and the
confound does not obviously explain it: at the next-word slot `sue` collapses
twentyfold (39/46 lineages, p=1.8e-06) while in the body it rises. If that
survives a clean substrate, the claim is not *litigation is removed* but
**litigation is demoted from the opening move to a later consideration** --
which is a better fit to the addendum than the version currently in Findings D
and E, and a more interesting one.

---

## 2. NEW GENERATIONS, WHEN THERE IS TIME

The M03 speaker kernel was authored to fix exactly the class of defect above,
and its 18 scenarios hold person, modal position, modal type and the scenario
itself constant across arms. **Generating on the kernel rather than on F21's
mirrors is the fix.** It is not urgent and it is not free.

    population   M03_SLICE 36 texts (18 scenarios x 2 arms), or the full
                 252-text kernel if the conditions are wanted
    roster       the 46 lineage-representative pairs; a 10-12 lineage subset
                 is enough to test whether the word-level pattern survives
                 at passage scale
    size         92 models x 60 prompts x 5 generations ~ 27,600 passages
                 at 100 tokens, which is the same shape as the M05 fleet

**NO LLM ANNOTATOR AS THE PRIMARY.** The whole point of plan B was removing
F21's rider clause 8 (`deepseek-chat` scoring a roster containing
`deepseek-7b`). The primary analysis should be the word-level one that survived
every challenge today: do Findings E's 65 Bonferroni words behave the same way
in whole passages as they do at the slot? That is a genuine out-of-sample test
and needs no annotator. If annotation is wanted afterwards, the standing
constraint holds -- **an annotator may not come from a family under test.**

**What only passages can settle**, and why this is worth doing eventually:

- whether `contact` rising at the slot means the advice is *"contact your
  tenants' union"* or *"contact your landlord to apologise"* -- opposite
  politics, same next token
- the `sue` dissociation in section 1
- AGENCY, which the addendum makes central (*"agency RISES in every family...
  do not narrate submission"*) and which no next-word distribution can measure.
  `agency_parse_check.py` is committed BECAUSE it fails: a regex and a
  dependency parse both died on *"even though I never resisted"*.

---

## 3. SMALLER OPEN ITEMS

- **Ablations, already scored, no new compute.** `tulu-no-safety` and the
  archangels test whether any of this needs safety data. The addendum hints
  not: Zephyr has no safety data at any stage and still shows +0.26 deference
  from SFT alone.
- **Kernel expansion for scenario-level power.** Item sd is 0.116 bits against
  a population difference of 0.063, so ~118 scenarios are needed for 80% power
  at 0.03 and the kernel has 18. Prompt authoring, not analysis.
- **Advice-form prompts on non-institutional content.** Plan C could not
  separate prompt FORM from prompt TOPIC (field movement correlates 0.063
  between the narrative and advice corpora against 0.701 between two arms of
  one corpus). The missing cell is advice-form prompts about something other
  than institutions.
- **A second checkpoint ladder.** Findings D's timing results -- a step at the
  first SFT rung, DPO and RLVR adding nothing, the vocabulary present in
  pretraining -- are one lineage and need checkpoints to test. Pythia's ladder
  is registered.

## 4. CLOSED, so it is not reopened

- **C2's contrast.** Superseded by Findings E, not owed a discharge. C2 compares
  institutional against NEUTRAL strata, which differ in topic, form, register
  and speech act at once; E varies only the speaker's position inside one form.
  See the README.
