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

## Direction check

The null interaction removes the reading this seat would most have liked: a
privileged first person, the cut installing a subject. It arrives instead at a
claim that is larger but less psychoanalytic, and RH is the one who insisted the
null branch be registered as a finding rather than a failure — the author's
first draft of Amendment 7 wrote it up as the reading losing its support.
