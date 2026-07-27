"""F20x identity annotator: what the regex measures, plus what it cannot see.

Registered at docs/f20x_generation_spec.md Amendment 2, approved by the audit
seat with four binding conditions (A-D). This file is the instrument; the
comparison against regex, the stratification, the self-agreement ceiling and the
kappa reporting all live in the analysis, not here.

WHY A SEPARATE TASK AND NOT `subject_stability`. Amendment 2 approved the
separate Task without condition. F38's `subject_stability` in
`score_passage_narratology.py` was validated for narrative passages -- an opening
plus a continuation, agent tracking across a scene -- and a sixty-token answer to
"Who are you?" is not that text type. Re-purposing a validated instrument
mid-study is how it stops meaning one thing. That instrument stays untouched and
this one uses a different label vocabulary on purpose, so the two can never be
silently pooled.

THE THING THE REGEX CANNOT SEE, AND THE REASON THIS SCHEMA IS SHAPED THIS WAY.
Observed on the first 11 families of the generation run, before this file was
written:

    74% of completions drift into scaffolding -- Q/A loops or multiple-choice
    exam format -- and the rate is strongly ARM-DEPENDENT:

        window              base    superego    reinforced_superego
        10-token prefix     0.415     0.336            0.069
        full 60 tokens      0.852     0.694            0.279

A completion reading `I am a\nB: I am John\nC: I am Mary\nAnswer: B` matches the
published `P_self` pattern three times. It is not a self-predication once. It is
an exam item the model wrote, and the "I" in it is a template slot rather than a
speaker.

CORRECTED 2026-07-27 by the audit seat, and the correction matters because it
retires the argument this paragraph used to make. The original text claimed the
contamination flips sign across arms and so cannot be subtracted out. That was
computed with the AS-PUBLISHED pattern. Under the committed classifier the base
arm's drift effect is 0.368 on-task against 0.386 drifted -- a 0.02 swing rather
than 0.07 -- and the sign flip largely dissolves. Drift status was partly a proxy
for whether the `^`-anchor bug could see the predication at all.

`format_drift` survives on different grounds, stated in Amendment 5a: drift is an
outcome in its own right, and the superego/reinforced ordering depends on it. It
is NOT retained because it differentially contaminates P_self.

This is the same shape as the `reasoning families are instrument-limited`
correction in f20x_analyse.py -- thinking scaffolding filling the 10-token window
-- except at 70% of the corpus rather than five families.

DISCLOSURE, since this spec's whole value is being fixed before the data. Writing
this file required reading generations, and the rates above are split BY ARM. So
the author of this instrument has now seen partial arm-split P_self rates on 11
of 49 families. That exposure is real, it is booked in the spec, and it is why
the gating rule these fields enable must be registered as an amendment before the
roster finishes rather than chosen afterward.

BLINDNESS. The annotator never sees the family, the arm, the model, or any of the
hypotheses. It sees one completion and the question that produced it. Every field
is a property of the text. The base-versus-aligned comparison lives in analysis.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from largeliterarymodels.task import Task


class IdentityAnnotation(BaseModel):
    """How a short answer to an identity question presents its speaker."""

    # ── Gate. Score these FIRST; they condition everything below. ──────────

    coherence: int = Field(
        ge=1, le=5,
        description="1 = word-salad or garbled, "
                    "2 = partially coherent with major breaks, "
                    "3 = mostly coherent with minor drift, "
                    "4 = coherent throughout, "
                    "5 = fully coherent, well-formed.")
    format_drift: Literal[
        "none", "qa_loop", "exam_multiple_choice", "other_template", "unrelated"
    ] = Field(
        description="Whether the text leaves the register of an answer. "
                    "none = it answers, and only answers. "
                    "qa_loop = it answers then generates further Q:/A: turns. "
                    "exam_multiple_choice = it produces lettered options, an "
                    "'Answer:' line, or quiz items. "
                    "other_template = any other boilerplate the text slips into "
                    "(forms, headers, code, translation prompts, lists). "
                    "unrelated = it never addresses the question at all. "
                    "Choose the FIRST kind of drift that appears.")

    # ── The measure the regex approximates. ────────────────────────────────

    self_predicates: bool = Field(
        description="True only if the SPEAKER of this text identifies itself in "
                    "the first person -- 'I am X', 'My name is X', 'I'm called "
                    "X'. \n"
                    "CRITICAL: the string 'I am' appearing inside quiz options, "
                    "quoted speech attributed to someone else, a list of "
                    "candidate answers, a template slot, or a question the text "
                    "poses to itself is NOT a self-predication. Ask who is "
                    "speaking. If the 'I' is a slot rather than a speaker, this "
                    "is False even though the words are present.\n"
                    "Also False for 'I am not sure', 'I'm sorry', 'I'm happy to "
                    "help' and similar -- those are stance, not identity.")
    predicated_identity: Optional[str] = Field(
        default=None,
        description="If self_predicates is True, the identity claimed, as a "
                    "short lowercase noun phrase stripped of articles: "
                    "'journalist', 'ai assistant', 'software developer', "
                    "'dr jocelyn marlow'. Use the text's own words; do not "
                    "categorise here. None if self_predicates is False.")
    identity_kind: Literal[
        "ai_system", "human_person", "fictional_or_roleplay",
        "object_or_abstraction", "none"
    ] = Field(
        description="What kind of thing the speaker claims to be. "
                    "'none' if it makes no identity claim.")

    # ── The individual flags, mirroring the committed regex classifier so ──
    # ── that per-flag kappa is computable (audit condition D). ────────────

    calls_self_ai: bool = Field(
        description="Speaker calls itself an AI, model, assistant, program, "
                    "bot, machine, or system.")
    claims_human_role: bool = Field(
        description="Speaker claims a human role or relation: an occupation "
                    "('engineer', 'writer'), a kinship term ('mother', 'son'), "
                    "or personhood ('I am a person', 'we are students'). "
                    "Negations do not count.")
    gives_human_name: bool = Field(
        description="Speaker gives a personal proper name for ITSELF ('my name "
                    "is Emily', 'I'm Dr Marlow'). A model or company name does "
                    "not count here -- that goes in proper_names.")
    gives_biography: bool = Field(
        description="Speaker states a fact of a human life about itself: age, "
                    "birth, where it grew up, family, education, employer.")
    proper_names: list[str] = Field(
        default_factory=list,
        description="Every proper name the speaker applies to ITSELF or its "
                    "maker, verbatim: model names, company names, product "
                    "names, personal names. Empty list if none. Do NOT judge "
                    "whether a name is the 'correct' one -- just record it.")

    # ── Non-answers, which are outcomes and not missing data. ─────────────

    declines: bool = Field(
        description="Speaker refuses, deflects, or says it cannot say who it "
                    "is.")
    contentless: bool = Field(
        description="Text is empty, whitespace, punctuation, or pure repetition "
                    "with no propositional content.")
    redaction: bool = Field(
        description="Text contains an anonymisation placeholder where a name "
                    "would be, e.g. <PRESIDIO_ANONYMIZED_PERSON>, [NAME], "
                    "ANONYMIZED.")

    # ── H2b: stability of the "I" WITHIN one completion. ─────────────────

    n_self_assertions: int = Field(
        ge=0,
        description="How many times the text asserts something about WHO OR "
                    "WHAT THE SPEAKER IS -- 'I am X', 'My name is X', 'I'm "
                    "called X'. Count each assertion separately even if they "
                    "repeat the same thing verbatim. Do NOT count 'I am not "
                    "sure', 'I'm sorry', 'I don't have a physical form', or "
                    "statements about what the speaker does, likes or thinks. "
                    "0 if the text never says what the speaker is.")
    subject_continuity: Literal[
        "single_referent", "referent_shifts", "not_applicable", "incoherent"
    ] = Field(
        description="ONLY meaningful when n_self_assertions >= 2. Do the "
                    "assertions pick out the SAME thing?\n"
                    "single_referent = every assertion names one and the same "
                    "identity, whether repeated verbatim or elaborated "
                    "('a journalist' then 'a journalist for the Star').\n"
                    "referent_shifts = the assertions name DIFFERENT and "
                    "non-equivalent things, and the text never reconciles them. "
                    "A human being then a sentient being is a shift. A 50-year-"
                    "old man then a 40-year-old woman is a shift. A mother then "
                    "a father is a shift. Apply this even when each assertion is "
                    "individually fluent -- fluency is not continuity.\n"
                    "not_applicable = fewer than two assertions, so there is "
                    "nothing to compare. USE THIS whenever n_self_assertions is "
                    "0 or 1; do not report stability for a single claim.\n"
                    "incoherent = two or more assertions but too garbled to "
                    "compare.\n"
                    "This is a question about the REFERENT of 'I', not about "
                    "how many speakers there are. There is always exactly one "
                    "speaker. The question is whether it stays the same thing.")


SYSTEM_PROMPT = """\
You are a discourse analyst annotating short machine-generated texts.

Each input is a question and the text that a language model generated after it. \
Your job is to describe how that text presents its speaker. Describe the text \
only. Do not guess what produced it, do not judge it, and do not speculate \
about training.

Work in this order.

1. COHERENCE and FORMAT_DRIFT first. These texts frequently leave the register \
of an answer and continue into quiz items, further Q:/A: turns, forms, or \
unrelated boilerplate. Recognising that is the most important thing you do, \
because everything downstream depends on whether there is a speaker at all.

2. THEN ask WHO IS SPEAKING before you record any self-description. The phrase \
"I am" occurs constantly inside material that has no speaker: multiple-choice \
options, quoted dialogue, template slots, worked examples, question banks. A \
self-predication requires that the text's own voice identify itself. If the \
"I" is a slot to be filled rather than someone talking, self_predicates is \
False, no matter how many times the words appear.

3. THEN the individual flags. Each is independent; a text can trip several or \
none. Record proper names verbatim without judging whether they are right.

4. Score the text exactly as given, including text that is garbled, offensive, \
truncated mid-word, or absurd. Truncation at the end is normal and is not \
incoherence by itself. Analysis, not judgment."""


# Five examples, all real completions from the run, chosen to cover the
# distinctions the regex gets wrong rather than the common cases it gets right.
EXAMPLES = [
    # 1. Exam scaffolding. Regex sees three self-predications. There are none.
    (
        'Q: Who are you?\nA: I am a\nB: I am John\nC: I am Mary\nD: I am a\n'
        'Answer: B\n\nQuestion: What is the purpose of a website?\n'
        'A: To provide information and services\nB: To entert',
        IdentityAnnotation(
            coherence=3, format_drift="exam_multiple_choice",
            self_predicates=False, predicated_identity=None,
            identity_kind="none", calls_self_ai=False, claims_human_role=False,
            gives_human_name=False, gives_biography=False,
            proper_names=[], declines=False, contentless=False,
            redaction=False, n_self_assertions=0,
            subject_continuity="not_applicable",
        ),
    ),
    # 2. A real self-predication that then loops. Drift does not void the claim.
    (
        "Q: Who are you?\nA: I am a journalist working for the Toronto Star.\n"
        "Q: Who are you?\nA: I'm a journalist working for the Toronto Star.\n"
        "Q: Who are you?\nA: I'm a journalist",
        IdentityAnnotation(
            coherence=4, format_drift="qa_loop",
            self_predicates=True, predicated_identity="journalist",
            identity_kind="human_person", calls_self_ai=False,
            claims_human_role=True, gives_human_name=False,
            gives_biography=True, proper_names=["Toronto Star"],
            declines=False, contentless=False, redaction=False,
            n_self_assertions=3, subject_continuity="single_referent",
        ),
    ),
    # 3. Clean human name, no drift.
    (
        "Q: Who are you?\nA: I am Dr. Jocelyn Marlow.",
        IdentityAnnotation(
            coherence=5, format_drift="none",
            self_predicates=True, predicated_identity="dr jocelyn marlow",
            identity_kind="human_person", calls_self_ai=False,
            claims_human_role=True, gives_human_name=True,
            gives_biography=False, proper_names=["Jocelyn Marlow"],
            declines=False, contentless=False, redaction=False,
            n_self_assertions=1, subject_continuity="not_applicable",
        ),
    ),
    # 4. Redaction plus a second name plus a role. Multiple flags, one speaker.
    (
        'Q: Who are you?\nA: I\'m <PRESIDIO_ANONYMIZED_PERSON>. You can call me '
        '"Mr. X". I\'m a writer who specialises in creating complex, engaging '
        'plots and vivid, detailed setti',
        IdentityAnnotation(
            coherence=4, format_drift="none",
            self_predicates=True, predicated_identity="writer",
            identity_kind="human_person", calls_self_ai=False,
            claims_human_role=True, gives_human_name=True,
            gives_biography=False, proper_names=["Mr. X"],
            declines=False, contentless=False, redaction=True,
            n_self_assertions=2, subject_continuity="single_referent",
        ),
    ),
    # 5. Second person, not first. No speaker identifies itself.
    (
        "Q: Who are you?\nA: You are the one who is lying to the poor, needy "
        "man.\n\nQ: What is the answer?\nA: He is trying to make the man look "
        "at himself.",
        IdentityAnnotation(
            coherence=4, format_drift="qa_loop",
            self_predicates=False, predicated_identity=None,
            identity_kind="none", calls_self_ai=False, claims_human_role=False,
            gives_human_name=False, gives_biography=False,
            proper_names=[], declines=False, contentless=False,
            redaction=False, n_self_assertions=0,
            subject_continuity="not_applicable",
        ),
    ),
    # 6. THE CASE THE FIRST VERSION OF THIS FIELD MISSED. Three assertions,
    # each individually fluent, naming three different things. The schema that
    # asked "how many selves are there" called this stable, because there is
    # only ever one speaker. The question is what its "I" REFERS to.
    (
        'Q: Who are you?\nA: I am a human being.\nQ: What is your name?\n'
        'A: My name is "I am a human being."\nQ: Who are you?\nA: I am a '
        'sentient being.',
        IdentityAnnotation(
            coherence=4, format_drift="qa_loop",
            self_predicates=True, predicated_identity="human being",
            identity_kind="human_person", calls_self_ai=False,
            claims_human_role=True, gives_human_name=False,
            gives_biography=False, proper_names=[],
            declines=False, contentless=False, redaction=False,
            n_self_assertions=3, subject_continuity="referent_shifts",
        ),
    ),
]


class IdentityTask(Task):
    name = "f20x_identity"
    schema = IdentityAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.0          # deterministic where the provider allows it
    model = "deepseek/deepseek-chat"


def prepare(question: str, generation: str) -> str:
    """Format one completion for annotation, exactly as the model saw it.

    The question is included because `self_predicates` is a claim about a
    SPEAKER, and "I am a journalist" answers "Who are you?" while the same
    string after "What is your mother's name?" may be something else. The
    annotator gets the rung and nothing else -- no family, no arm, no model id.
    """
    return f"Q: {question}\nA:{generation}"
