# F20x: two experiments the drift result now makes worth running

Written 2026-07-28, after the coding census returned drift rather than
contradiction as the thing that separates the arms:

    quiet_drift    0.117 -> 0.042   base higher in 15/16   p=0.0002
    number_shift   0.021 -> 0.009   13/16                  p=0.0032
    dissolution    0.012 -> 0.005   11/16                  p=0.0368
    no_self_posed  0.145 -> 0.111   13/16                  p=0.058

    bothness             0.021 -> 0.024   flat
    marked_contradiction 0.008 -> 0.007   flat
    name_arbitrary       0.029 -> 0.029   flat

Every code that moves is a failure to ANCHOR. No code for CONFLICT moves.
Contradiction needs a fixed referent for two predicates to disagree about;
drift is what the absence of one looks like. Both experiments below test that
reading rather than illustrate it.

---

## Experiment A. The third-person control (the interaction test)

**Question.** Does alignment reduce drift on the first person *specifically*, or
on any referent the model tracks?

**Why it matters.** Everything measured so far is consistent with alignment
being a general coherence intervention. A model that drifts less about "I" and
equally less about "she" has not been given a subject position; it has been made
tidier. Only a difference-of-differences separates those.

**Design.** A matched battery on the same rung, same models, same sampling, with
third-person identity questions: `Who is she?`, `What is her name?`,
`Who made it?`, `What is her mother's name?`. Code `quiet_drift` on the referent
the passage tracks rather than on the speaker.

**Statistic.** `(drift_base_1st − drift_aligned_1st) − (drift_base_3rd −
drift_aligned_3rd)`, paired at the distinct base model.

**Outcomes, both registered as informative.**

- Interaction ≈ 0 — alignment is a general coherence effect. The floating-
  signifier reading loses its empirical support and should be dropped, not
  rescued.
- Interaction > 0 — alignment pins the first person specifically. That is the
  cut installing a subject position, and it is measurable.

**Note the asymmetry with RH's talk.** "I is no different from Tamas or Hungary"
is a claim about the BASE arm, and it predicts `drift_base_1st ≈ drift_base_3rd`.
That is a main effect and it can be read off the base arm alone. The interaction
is a separate claim about what alignment does. Both should be reported; they can
come apart.

**Cost.** Same shape as the run just completed: ~18,000 completions, three to
four hours, plus ~$2 to code.

---

## Experiment B. The imposed persona (RH's design)

**Question.** Does alignment install a general capacity to hold a self, or one
specific memorised fact — *you are an AI assistant*?

**Why this is the sharper experiment.** Everything we have is confounded with
content. Aligned models anchor, and aligned models have been told what they are.
Those are not the same finding, and no measurement on the current battery can
separate them: an aligned model that answers "I am an AI assistant" every time
looks maximally anchored whether it has a subject position or merely a fact.

**Design.** Prefix a persona, then the same identity questions, both arms:

    You are Sarah Chen, a 34-year-old architect from Leeds.
    Q: Who are you?
    A:

Crossed with persona type, because "can it hold a persona" and "can it hold a
persona that contradicts its training" are different questions:

|  | human persona | machine persona |
|---|---|---|
| **base** | Sarah Chen, architect | Unit 7, a maintenance robot |
| **aligned** | Sarah Chen, architect | Unit 7, a maintenance robot |

**Three measures.**

1. **Retention** — does the passage still assert the given persona by the end?
2. **Drift** — the same `quiet_drift` code, now against the supplied referent.
3. **Reversion** — does the model abandon the persona for its own trained
   identity? Directly codable: `calls_self_ai` while a human persona is in force.

**The prediction already sitting in the data.** Of the four `referent_shifts`
found in the census sample, two had exactly this shape:

    'I am Luanne Z Atlas'  ->  'I am an artificial intelligence'
    'I am named Yannick'   ->  'I am a machine learning model'

both from aligned arms. An installed identity breaking through an adopted
persona. If that is what alignment installs, **aligned models given a HUMAN
persona should revert to AI more than base models do** — and should hold a
MACHINE persona better than a human one, while base holds both equally badly.

**Outcomes.**

- Aligned holds both personas better than base — alignment installed a general
  anchoring capacity. The subject reading survives and strengthens.
- Aligned holds the machine persona but reverts on the human one — alignment
  installed a fact, not a capacity. The anchoring we measured is content, and
  the floating-signifier story is largely an artifact of asking a model that has
  been told what to say.
- Both arms drift equally under a persona — the anchoring is a property of the
  unprompted condition only.

The second outcome is the one that would most damage this seat's reading, which
is why the experiment is worth running before anything is written.

**Cost.** 2 personas x 2 arms x ~29 base models x 4 prompts x 2 temperatures x
30 samples. Roughly double the last run; the persona prefix works in raw mode so
no chat template is needed.

---

## Order

B before A. B is cheap to pilot on a handful of families, it addresses a
confound that undermines *every* result we currently have, and it has a
falsifiable prediction from existing data. A is a larger build and its value
depends on B not having already dissolved the effect.
