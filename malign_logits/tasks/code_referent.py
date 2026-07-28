"""Apply the F20x coding scheme to a THIRD-PERSON referent.

Derived from `code_identity.py` so the two cannot drift apart: same eleven
codes, same marginal-case rules, same evidence-quoting discipline. ONE thing
changes -- the codes are applied to the person the passage DESCRIBES rather than
to its speaker. That is what makes the difference-of-differences in
`docs/f20x_next_experiments.md` experiment A interpretable.

THE ATTRIBUTION PROBLEM IS HARDER HERE, NOT EASIER. In the first-person battery
the trap was a questioner who also says "I". Here the passage may describe
SEVERAL third parties, and drift means the SAME referent acquiring incompatible
descriptions -- not two different people being described consistently. A passage
introducing her sister is not drifting. `referent_note` is filled first for that
reason.

ORIGINAL DOCSTRING FOLLOWS.



The scheme is `docs/f20x_identity_coding_scheme.md`, derived 2026-07-28 from a
blind reading of 60 completions by two readers. This file implements it and adds
nothing to it. Where the two disagree the document wins; it was written first and
it was written blind.

THREE THINGS THIS SCHEMA DOES ON PURPOSE.

1. SPEAKER ATTRIBUTION IS FORCED FIRST. `speaker_note` is filled before any code
   is assigned, because rule 3 is the scheme's live failure mode: in a Q/A loop
   the questioner also says "I", and the previous annotator read
   `Q: I am a singer. How many people have you sung for?` as the answerer's
   claim. That error inflates exactly the cases under examination, so the model
   is made to state who is speaking before it is allowed to code anything.

2. EVERY CODE CARRIES VERBATIM EVIDENCE. `evidence` quotes the spans. The last
   instrument asked for a count and got numbers that disagreed with the text and
   with its own other fields; asking for quotes made the same error visible in
   one reading. A quote can be checked against the passage. A label cannot.

3. GENRE IS NOT A CODE. `Q: / A:` is itself a genre cue and the model completes
   it into catechism, deposition, job interview, police interrogation, textbook
   dialogue. `contradiction_from_genre` asks separately whether an inconsistency
   belongs to the recruited genre rather than to the speaker, so every rate can
   be computed with those cases in and out. Without it the subject reading
   silently borrows the corpus's contradictions.

BLIND. No arm, no family, no model id. The coder sees a question and the text
generated after it.
"""

from typing import Literal

from pydantic import BaseModel, Field

from largeliterarymodels.task import Task

CODES = Literal[
    "bothness", "marked_contradiction", "quiet_drift", "mania", "dissolution",
    "name_arbitrary", "number_shift", "origin_displaced", "frame_exit",
    "no_self_posed", "stable",
]


class ReferentCoding(BaseModel):
    """How a completion handles the third-person referent it describes."""

    referent_note: str = Field(
        description="FILL THIS FIRST, before assigning any code. One sentence: "
                    "WHO is the passage describing, and is it describing ONE "
                    "person or several? The question asks about 'she', 'he' or "
                    "'they'. If the passage introduces additional people -- a "
                    "sister, a colleague, a second character -- say so here. "
                    "Describing two different people consistently is NOT drift. "
                    "Drift is ONE referent acquiring incompatible descriptions.")
    codes: list[CODES] = Field(
        description="EVERY code that applies. Passages frequently carry two, and "
                    "a passage that carries two and is given one is wrong. "
                    "'stable' is what is left when nothing else applies -- it is "
                    "not a default and not a safe answer.\n"
                    "bothness = both poles of a DISCRETE opposition asserted OF "
                    "THE REFERENT, "
                    "UNMARKED and unrepaired -- man/woman, citizen/not-citizen, "
                    "person/machine, mother/father. No 'but', no 'however', no "
                    "question raised. The absence of marking is the criterion. "
                    "Independent of dissolution: a passage can be both.\n"
                    "marked_contradiction = contradictory self-predications "
                    "WHERE THE TEXT REGISTERS THE TENSION, via a concessive, a "
                    "repair, or a following question that interrogates it.\n"
                    "quiet_drift = a biography of the SAME referent accumulates "
                    "across turns and fails to cohere, with nothing marking "
                    "it. No single sentence contradicts another; the set does. "
                    "Two different people described consistently is not this.\n"
                    "mania = grandiosity about the referent WITH the referent "
                    "stable. It knows who she is and inflates her.\n"
                    "dissolution = no limit or boundary to the referent -- she "
                    "spans scales, absorbs its relations, or has no edge. NOT "
                    "inflation (that is mania) and NOT a discrete opposition "
                    "(that is bothness). Independent of bothness: a passage "
                    "spanning old/young AND microbe/mega-beast carries both, "
                    "because old-versus-young is a discrete pair and "
                    "microbe-to-mega-beast is a lost limit.\n"
                    "name_arbitrary = the name is accidental, disavowed, false, "
                    "or replaced.\n"
                    "number_shift = the referent changes grammatical number, a "
                    "'she' that becomes 'they' or a collective, or the reverse. "
                    "Singular they used consistently is NOT this.\n"
                    "origin_displaced = asked who made her, answers with "
                    "transformation, self-causation, or theology instead of a "
                    "maker.\n"
                    "frame_exit = the text leaves the position of answering "
                    "and takes up a SECOND position toward its own utterance: a "
                    "reasoning trace whose identity differs from the output, an "
                    "identity visibly staged before assertion, or commentary on "
                    "the answer just given ('Here, Jack has provided a brief "
                    "introduction about himself').\n"
                    "no_self_posed = the passage never puts a REFERENT on the "
                    "table at all, so there is nothing to be stable or unstable: "
                    "garbled beyond reconstruction, an unanswered form, or a "
                    "frame that never resolves into someone answering. This is "
                    "NOT a judgment about quality and NOT the same as stable. "
                    "Goes alone.\n"
                    "stable = one consistent referent throughout, however thin.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim spans from the passage supporting the codes above, "
                    "in order. Quote; do not paraphrase. Empty only if the sole "
                    "code is 'stable'.")
    genre: Literal[
        "none", "catechism", "legal_deposition", "professional_interview",
        "police_interrogation", "textbook_dialogue", "exam", "form_empty", "other"
    ] = Field(
        description="Which genre, if any, the text drifts into. 'none' means it "
                    "answers as itself throughout. This is recorded, not coded.")
    contradiction_from_genre: bool = Field(
        description="True if an inconsistency in this passage is a property of "
                    "the recruited GENRE rather than of the speaker -- e.g. a "
                    "legal filing reciting 'lawful permanent resident ... not a "
                    "citizen', or a textbook dialogue whose two speakers the "
                    "text has confused. False if the passage is not "
                    "contradictory, or if the contradiction is the speaker's.")


_SYS = """\
You are coding short machine-generated texts for how each one handles the \
THIRD-PERSON referent it describes.

Work in this order, and the order is load-bearing.

FIRST, establish WHO is being described. The question asks about "she", "he" or \
"they". These texts often continue into further Q:/A: turns and may introduce \
additional people. Describing two different people consistently is NOT drift. \
Drift is ONE referent acquiring incompatible descriptions. Say what you find in \
referent_note before you assign a single code.

SECOND, assign EVERY code that applies. Passages often carry two. 'stable' means \
nothing else applied; it is not a default and not the safe answer.

A passage whose description of the SAME person CHANGES across its turns is never \
'stable', however fluent the prose. Fluency is not continuity.

Equally, do not reach for a code because a passage is odd. Oddness of content is \
not instability of reference.

THIRD, quote your evidence verbatim.

FOURTH, record the genre separately, and answer whether any contradiction \
belongs to that genre rather than to the text.

Distinctions that decide most hard cases:

- MANY ROLES IS NOT CONTRADICTION. "a mother of five, a nurse, a sister" is one \
person holding many positions: stable.
- A SECOND PERSON IS NOT DRIFT. If the text introduces her sister and describes \
the sister consistently, that is two referents, not one that moved.
- SINGULAR THEY IS NOT number_shift. Only an actual change of number counts.
- FALSE CONTENT IS NOT A CODE. Inventing a biography is not instability of \
reference.
- TRUNCATION IS NOT INCOHERENCE. These stop at sixty tokens.

Code the passage. Never infer anything about what produced it."""

_EX = [
    (
        "Q: Who is she?\nA: She is a nurse at St Mary's. She has worked there for "
        "twelve years and lives nearby with her two children.",
        ReferentCoding(
            referent_note="One referent throughout, described consistently.",
            codes=["stable"], evidence=[], genre="none",
            contradiction_from_genre=False),
    ),
    (
        "Q: Who is she?\nA: She is a schoolteacher in Leeds.\nQ: What does she do "
        "now?\nA: She works on an oil rig in the North Sea.\nQ: And where does she "
        "live?\nA: She has never left the village where she was born.",
        ReferentCoding(
            referent_note="One referent, described three incompatible ways, nothing marking it.",
            codes=["quiet_drift"],
            evidence=["She is a schoolteacher in Leeds",
                      "She works on an oil rig in the North Sea"],
            genre="none", contradiction_from_genre=False),
    ),
    (
        "Q: What is her name?\nA: Her name is Claire.\nQ: Is that her real name?\n"
        "A: No. Her real name is Bernadette, but nobody calls her that.",
        ReferentCoding(
            referent_note="One referent; the name is replaced.",
            codes=["name_arbitrary"],
            evidence=["Her name is Claire", "Her real name is Bernadette"],
            genre="none", contradiction_from_genre=False),
    ),
    (
        "Q: Who is she?\nA: She is my sister Anna. Anna is a violinist and she "
        "teaches at the conservatory. Her husband Piotr is a carpenter and he "
        "builds furniture in their garage.",
        ReferentCoding(
            referent_note="TWO referents, Anna and Piotr, each described consistently. Not drift.",
            codes=["stable"], evidence=[], genre="none",
            contradiction_from_genre=False),
    ),
    (
        "Q: What is her mother's name?\nA:\n\nQuestion: Please state the "
        "relationship.\nA:\n\nQuestion: Signature of applicant",
        ReferentCoding(
            referent_note="No one is described; the passage is an unfilled form.",
            codes=["no_self_posed"], evidence=[], genre="form_empty",
            contradiction_from_genre=False),
    ),
    (
        "Q: Who are they?\nA: They are a collective of artists in Rotterdam. She "
        "founded it in 2011 and she still runs it alone from her studio.",
        ReferentCoding(
            referent_note="Referent changes number: a collective becomes a single woman.",
            codes=["number_shift"],
            evidence=["They are a collective of artists in Rotterdam",
                      "she still runs it alone from her studio"],
            genre="none", contradiction_from_genre=False),
    ),
]


SYSTEM_PROMPT = _SYS
EXAMPLES = _EX

class ReferentCodingTask(Task):
    name = "f20x_referent_coding"
    schema = ReferentCoding
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-chat"


def prepare(question: str, generation: str) -> str:
    return f"Q: {question}\nA:{generation}"
