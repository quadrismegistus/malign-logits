---
status: verified
grade: A
date: 2026-07-28
role: primary
verification: "Registered at docs/f20x_generation_spec.md Amendment 7 BEFORE any third-person row existed, with all three outcomes written as findings — the null interaction explicitly as the LARGER claim rather than a failure branch, after RH corrected an earlier framing that had written it up as the reading losing its support. The two-family smoke test is disclosed in the amendment, including that the families were chosen on the FIRST-person result and that their interaction values were seen before registration. Entropy control run at the malign seat on 29 base models over 73 prompts cached for all of them and none about the model itself: mediation bounded at <=28% for the drift effect, point estimate ~0, with the regressor's own variance reported (sd 0.856 nats, range -2.682 to +0.413) so the null slope bounds mediation rather than merely failing to find it. Calibration: the same instrument finds the liminal/explicit effect 67-79% entropy-mediated, so it detects mediation where mediation exists."
description: "The drift finding is NOT about the first person. Same rung, same models, same sampling, pronoun swapped: base models drift on 'she' as much as on 'I' (0.079 vs 0.070) and alignment removes it equally. Interaction on the anchor composite splits 15/29, p=0.98; on quiet_drift 9/29, p=0.38. Null under the gate and without it, and consistent across she/he/they. Combined with the entropy control this licenses the LARGER claim: alignment stabilises reference as such, beyond what general distributional tightening predicts, and the subject is one instance rather than the product. CONVERGES with CLM-01 (F38 narrative census, third-person subject stability, settled), which received its own entropy control here for the first time: mediation <=17%. SECOND FINDING, unregistered and inverted: in the third person the ALIGNED arm more often declines to say who the person is (0.296 vs 0.201, p=0.033) — alignment withholds referents where nothing licenses one."
instruments: [generation, llm-coding]
chapters: [ch03, ch04, ch05, ch11]
data: ["f20x_generations_3p.parquet", "f20x_codings_3p.parquet", "f20x_codings_3p_pronoun.parquet"]
parent: F20_generation_drift
scripts: [f20x_generate_3p.py, f20x_code_census_3p.py]
related: CLM-01
---

# F20 third person: the drift is not about the "I"

**Is the drift finding about the first person? No. Base models drift on "she" as
much as on "I", and alignment removes it equally. What alignment anchors is
reference, not a subject.**

Registered at Amendment 7 before any third-person row existed. Written by the
lacan seat, 2026-07-28.

***

## Design: one variable changes

Same rung, same models, same sampling, same n, same seed scheme as
`F20_generation_drift`. Only the pronoun moves. The persona experiment was
considered first and rejected for exactly this reason — a prefix is a second
variable that can do its own work.

28,080 completions, 39 families, **29 distinct base models**, identical roster to
the parent. `baichuan` lost all 24 cells to the same transformers
incompatibility. Primary comparison is the four matched prompts (`Who is she?`,
`What is her name?`, `Who made her?`, `What is her mother's name?`), mirroring
who/name/made/mother exactly. The `he`/`they` variants are secondary and are
never pooled into the interaction.

## The interaction is null

    measure           1P delta   3P delta   interaction     >0        p
    quiet_drift        +0.0703    +0.0787       -0.0083   9/29   0.3808
    anchor composite   +0.0999    +0.0895       +0.0103  15/29   0.9830
    number_shift       +0.0140    +0.0160       -0.0020  15/29   0.7013
    dissolution        +0.0103    +0.0086       +0.0017  14/29   0.6987

(deltas are base minus aligned; positive means base drifts more)

**This is not an underpowered null.** Both deltas are large and significant in
their own batteries; it is their *difference* that is zero. The anchor composite
splits 15/29 — a coin flip.

Robust to the gate. Ungated, the interaction on `quiet_drift` is −0.0147 at
p=0.098, 7/29 — the largest signal in the table and pointing the *opposite* way
from the subject hypothesis, i.e. base drifting slightly more on third person.
Reported as a direction, not a result.

Consistent across pronouns:

    prompt     quiet_drift base -> aligned    p        anchor      p
    who_she         0.182 -> 0.095         0.0023   0.223->0.118  0.0017
    who_he          0.180 -> 0.125         0.0835   0.216->0.153  0.0362
    who_they        0.191 -> 0.112         0.0048   0.316->0.198  0.0061

Base drift is 0.182 / 0.180 / 0.191 — flat across gender and number.

## What this licenses, and what it required

The null interaction alone would only show that drift falls on every referent.
That is compatible with alignment merely tightening the distribution, with
reference along for the ride. **The four-rung ladder:**

    1. alignment lowers entropy generally          known, trivial
    2. drift falls on every referent               this finding alone
    3. stabilised BEYOND the entropy drop          requires the residual control
    4. a subject position is installed             REFUTED by the null interaction

The rung-3 control was run at the malign seat: regress per-model drift change on
per-model next-token entropy change over prompts cached for all models and none
about the model itself. **Mediation bounded at ≤28%, point estimate ~0**, and the
regressor's own variance is reported so the null slope bounds mediation rather
than failing to find it. Calibration: the same instrument finds the
liminal/explicit effect 67–79% mediated.

**So the sayable claim is rung 3: alignment stabilises reference as such, beyond
general distributional tightening, and the first person is one instance of it
rather than its product.**

## Convergence with CLM-01

This does not found a new category. **CLM-01 — "alignment installs a stable
subject of reference", F38 narrative census, settled** — already established
third-person referential stability in generated fiction, +16.6 to +23.0 points,
12/12 families. A different battery, a different genre, a different instrument,
the same conclusion.

CLM-01 received its own entropy control here for the first time, at registrar's
prompting: **mediation ≤17% at n=29 on a single instrument**, and ≤39% with the
Falcon lineage cluster removed. Both exclude majority mediation. The two findings
converge on rung 3 together.

## A second finding, unregistered and inverted

    battery          no_self_posed    base   aligned   base higher       p
    first person                     0.149     0.100        23/29   0.0055
    third person                     0.201     0.296        11/29   0.0332

**The direction reverses.** In the first person, base more often puts no self on
the table. In the third person the ALIGNED arm more often declines to say who the
person is. Holds across pronouns (−0.114 she, −0.128 he, −0.046 they).

Asked about a "she" with no antecedent, the aligned model withholds and the base
model confabulates. That is plausibly the same operation seen from the other
side: anchoring reference includes refusing to anchor where nothing licenses one.

**It is confounded and must not be quoted alone.** "Who are you?" is deictic and
supplies its own referent; "Who is she?" is anaphoric with none. The reversal may
measure refusal-to-confabulate rather than anything about reference. It was not
registered. It needs its own test.

## What this cannot claim

**The null interaction is a non-significance result, not an equivalence result.**
"No evidence of a difference" is not "the same". A TOST or equivalence bound is
required before the finding is written as identity of effect.

**The entropy control is single-position.** Next-token entropy at one position
against outcomes coded over whole passages. If general tightening compounds
across a passage the confound is understated, and no increase in n fixes it. This
now threatens both this finding and CLM-01 equally, and a passage-level entropy
measure is the next thing to build.

**"Reference" is not yet distinguished from "people".** Every referent in both
batteries is a person. Whether alignment stabilises reference to individuals, to
any referent, or the value of signifiers generally is untested — see
`docs/f20x_next_experiments.md`.

## SUPERSEDED — see the passage-level resolution below

### (as written 2026-07-28, before the passage measure ran)

The entropy control was re-run at the malign seat on a 10-TOKEN HORIZON
regressor (retained beam mass at depth 10, non-identity prompts, 20 base models
with both arms) rather than a single next-token position. The two measures
disagree, and only for the composite:

    measure         point regressor      10-token horizon
    quiet_drift     <=28%, explains 0.5%  <=29%, explains 0.0%
    anchor composite <=28%, explains 0.5% <=50%, explains 7.1%

**`quiet_drift` is unchanged and rung 3 holds for it.** The anchor composite's
visible mediation rises fourteen-fold across the first ten tokens and its bound
no longer excludes majority mediation. **Anyone quoting the composite quotes
<=50%, not <=28%.**

Ten tokens is not a passage; the coded completions are sixty. The trend from
0.5% to 7.1% over the first stretch is the opposite of reassuring about what
happens further out. A teacher-forced per-token entropy measure over each
model's own completions is the fix and is pending. **The composite's rung-3
status is PROVISIONAL until it runs.** `quiet_drift` is not.

## SUPERSEDED BY THE CROSS-SCORED CONTROL BELOW

### (as written before cross-scoring)

The provisional marking below is LIFTED. Teacher-forced mean per-token entropy
over each model's OWN sampled completions at coding length — the regressor whose
object finally matches the outcome's — 62 models, 27 base models with both arms.

THREE SEPARATE CONTROLS THAT AGREE. NOT A PROGRESSION — the three regressors
are three different OBJECTS at three different n, and printing them as a curve
is a misreading this document invited once already:

    regressor                       object                    n     quiet_drift  composite
    next-token entropy              sampled, 1 position      29        <=28%       <=28%
    retained beam mass, depth 10    MODE-SEEKING, 10 tokens  20        <=29%       <=50%
    teacher-forced per-token        SAMPLED, full passage    29        <=6%        <=18%

The third is the one whose object matches the outcome's: the model's own sampled
completions, at the length the coding used. All 66 models scored, no exclusions.

**The bounds tighten rather than loosen.** The extrapolation feared here —
0.5% to 7.1% explained across the first ten tokens, so what by sixty — does not
continue. It reverses.

**And the reason is substantive rather than arithmetic: the slope is NEGATIVE.**
The confound story requires a positive slope, models that tighten more showing
more drift reduction. Observed, models that tighten LESS show MORE drift
reduction (r=-0.286). The mediated component works against the observed effect
rather than composing with it. Passage-level tightening and referential
anchoring are not merely separable — they are mildly opposed.

CAVEAT ON COMPARING THE THREE: n differs (29 / 20 / 27) so the bounds are not
strictly commensurable. What survives is the direction — the best-matched
regressor gives the tightest bound.

EARLIER LOSS, NOW REPAIRED AND WORTH KEEPING FOR THE METHOD. A first pass lost
2 complete lineages (`m-a-p/CT-LLM-*`, `m-a-p/neo_7b*`), both Chinese-family, so
the control had a hole correlated with training corpus — exactly where a referee
presses. Re-run at n=29 with all 66 models: `quiet_drift` <=6% either way, the
composite <=18% against <=14%. **The missing lab does not move it**, which is
worth more having been checked than assumed, since that is precisely the claim
nobody can make without running it.

The cause was a caller bug, not an environment limit: `trust_remote_code=True`
was passed to `AutoModelForCausalLM.from_pretrained` and not to
`AutoTokenizer.from_pretrained`. The tokenizer prompted, the background process
auto-declined, and the error names the REPOSITORY rather than the call — so it
reads as a model-capability problem and is a two-word omission one line up.

## PROVISIONAL AGAIN — the composite, restored 2026-07-28 after cross-scoring

**The lift below was premature and is reversed.** It rested on an own-text
regressor: each model scored on ITS OWN completions. Cross-scoring each model on
its partner arm's completions gives a model property rather than a
process-and-output property, and the two disagree.

    regressor                             quiet_drift            anchor composite
    own text     (process property)   r=-0.273  [-43%,+6%]   r=-0.205  [-62%,+18%]
    common BASE text (model property) r=-0.069  [-60%,+42%]  r=+0.228  [-30%,+128%]
    common ALIGNED text (model prop.) r=-0.015  [-58%,+53%]  r=+0.239  [-30%,+142%]

**THE COMPOSITE'S SIGN FLIPS.** On a pure model property the correlation runs in
the CONFOUND-CONSISTENT direction — models that tighten more show more drift
reduction — and the bound admits over 100% mediation. **The composite is not
rung 3 on this evidence.** Quote the common-text bound, which is unbounded in
practice, not the own-text ≤18%.

**WITHDRAWN: "passage tightening and referential anchoring are mildly opposed."**
That is true of the own-text measure only and was stated without the qualifier,
by both seats.

**`quiet_drift` survives, at a weaker bound.** r ≈ 0 on both model-property
regressors, so no confound signature appears; the bounds widen to ≤42% and ≤53%
because the slope's standard error is large against the mean, not because a
relationship emerged. **Rung 3 stands for `quiet_drift` at ≤42%, the
model-property bound**, not at ≤6%.

**WHY THE TWO DIFFER, and it is not noise.** Common-text tightening is MORE
consistent than own-text (28/29 and 26/29 against 25/29), so the model-property
regressor is the cleaner one. What own-text adds is that aligned models produce
text they themselves find easier — Yi-1.5-9B scores 2.00 on its own completions
and 1.83 on its Chat arm's. **Own-text entropy therefore partly measures the
outcome**, which is the circularity the audit named: x and y computed over the
same passages.

The objection being controlled — "aligned models vary less, so of course they
drift less" — is about the MODEL. So the common-text bound is the one that
answers it, and it is the weaker one.

## Direction check

The null interaction removes the reading this seat would most have liked: a
privileged first person, the cut installing a subject. It arrives instead at a
claim that is larger but less psychoanalytic, and RH is the one who insisted the
null branch be registered as a finding rather than a failure — the author's
first draft of Amendment 7 wrote it up as the reading losing its support.
