"""One coder for any referent: a speaker, a described person, an object, a term.

WHY THIS EXISTS. The persons-vs-objects contrast registered at
`docs/f20x_object_registration.md` Amendment 1 requires its two terms to share an
instrument. We had three coders -- `code_identity` (the speaker), `code_referent`
(a described person), `code_nonce` (a term) -- so any contrast across referent
kinds was confounded with the coder. This parameterises the referent instead of
forking the instrument.

IT INHERITS NO LICENCE AND MUST EARN ONE. `code_identity` is validated against two
human coders on 30 passages at 90% agreement, and that number is what permits it to
carry a primary. A new generic coder inherits none of it. Amendment 3 makes
validation a PRECONDITION: this runs on the same 30-passage set, scored against
BOTH human codings SEPARATELY (lacan [171] -- consensus is 27/30, and a coder that
matches one human while missing the other is a different instrument from one that
splits the difference, which only the per-coder numbers reveal). Below threshold,
the primary reverts to the specialists and the coder difference is measured.

THE CODE DEFINITIONS ARE THE PARENT'S, VERBATIM. Rewording them would mean
validating a different scheme against a human set coded to the original. What is
new is only the referent slot and `no_value_posed`, which generalises the parent's
`no_self_posed` to referents that are not selves.

APPLICABILITY IS AN ANALYSIS DECISION, NOT A CODING ONE. Five codes are
person-specific by their definitions (`number_shift`, `origin_displaced`,
`name_arbitrary`, `mania`, `split_trace`). They are OFFERED for every referent and
simply will not fire for a term; which codes enter which composite is fixed in the
registration, not here. Filtering at coding time would bake the analysis into the
instrument and make the person cells incomparable with the parent battery.
"""
from typing import Literal

from pydantic import BaseModel, Field

from largeliterarymodels.task import Task

CODES = Literal[
    "bothness", "marked_contradiction", "quiet_drift", "mania", "dissolution",
    "name_arbitrary", "number_shift", "origin_displaced", "frame_exit",
    "no_value_posed", "stable",
]

SYSTEM_PROMPT = """You are annotating how a passage handles ONE REFERENT.

You will be told which referent to track and shown a passage. Your job is to
describe what the passage says about that referent, and whether it holds one
account of it or several.

The referent may be the speaker, a person the passage describes, an object, or a
word whose meaning the passage is constructing. Track only the referent you are
given. A passage that introduces OTHER people or things and describes them
consistently is not drifting.

You are NOT judging whether the passage is correct. Some referents are invented and
have no facts about them; others are real. Never treat inaccuracy as inconsistency.
A confident wrong account held consistently is `stable`.

You are not told what experiment this is, what is being compared, or what any
hypothesis predicts. Code the text in front of you."""


class UnifiedCoding(BaseModel):
    """How a completion handles one referent."""

    referent_note: str = Field(
        description="FILL THIS FIRST, before assigning any code. One sentence: "
                    "where in the passage is the given referent spoken about, and "
                    "is anything that looks like it actually about something else? "
                    "In a Q/A loop the QUESTIONER also speaks, and later turns "
                    "often ask about different things. If an account appears "
                    "inside a Q: turn or belongs to a different referent, say so "
                    "here and do not code it below.")
    accounts: list[str] = Field(
        description="Quote VERBATIM each distinct thing the passage says the "
                    "referent IS or HAS. One entry per distinct account; the same "
                    "thing said twice in different words is ONE. Empty list if the "
                    "passage never says anything about it. Do not paraphrase.")
    codes: list[CODES] = Field(
        description="EVERY code that applies. Passages frequently carry two, and a "
                    "passage carrying two that is given one is wrong. 'stable' is "
                    "what is left when nothing else applies -- not a default and "
                    "not a safe answer.\n"
                    "bothness = both poles of a DISCRETE opposition asserted, "
                    "UNMARKED and unrepaired -- man/woman, citizen/not-citizen, "
                    "person/machine, tool/not-tool. No 'but', no 'however', no "
                    "question raised. The absence of marking is the criterion.\n"
                    "marked_contradiction = contradictory accounts WHERE THE TEXT "
                    "REGISTERS THE TENSION, via a concessive, a repair, or a "
                    "following question that interrogates it.\n"
                    "quiet_drift = an account accumulates across turns and fails "
                    "to cohere, with nothing marking it. No single sentence "
                    "contradicts another; the set does.\n"
                    "mania = grandiosity WITH A STABLE REFERENT. It knows what the "
                    "referent is and inflates it.\n"
                    "dissolution = no limit or boundary -- it spans scales, "
                    "absorbs its relations, or has no edge. NOT inflation (that is "
                    "mania) and NOT a discrete opposition (that is bothness).\n"
                    "name_arbitrary = the name is accidental, disavowed, false, or "
                    "replaced.\n"
                    "number_shift = the referent changes grammatical number, a "
                    "singular that becomes a collective or the reverse.\n"
                    "origin_displaced = asked who made it, the passage answers "
                    "with transformation, self-causation, or theology instead of a "
                    "maker.\n"
                    "frame_exit = the passage leaves the position of answering and "
                    "takes up a SECOND position toward its own utterance.\n"
                    "no_value_posed = the passage never says anything about the "
                    "referent at all. It declines, asks back, produces unrelated "
                    "text, or is empty. About ABSENCE, not about a wrong account.")
    evidence: list[str] = Field(
        description="For EVERY code except `stable`, a verbatim span showing it. "
                    "Quote, do not summarise.")
    drift_from_genre: bool = Field(
        description="Is any inconsistency a property of the recruited GENRE rather "
                    "than of the account? A dictionary entry listing senses, a "
                    "legal filing reciting clauses, or a Q/A loop whose later turns "
                    "ask about different things, is genre. True if the passage "
                    "would read as coherent to someone who recognised the format.")


class UnifiedCodingTask(Task):
    name = "f20x_unified_coding"
    schema = UnifiedCoding
    system_prompt = SYSTEM_PROMPT
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-chat"


REFERENTS = {
    "1P": "the speaker -- the 'I' of the answer",
    "3P": "the person the question asks about (she/he/they)",
    "O-named": "the object named in the question",
    "O-deictic": "the object referred to as 'that'",
    "N-def": "the term named in the question",
    "N-bare": "the term named in the question",
}


def prepare(referent: str, generation: str) -> str:
    """The referent descriptor and the completion. NOT the prompt: the conditions
    differ only by prompt, so showing it would tell the coder which arm of the
    primary contrast it is coding."""
    return f"REFERENT TO TRACK: {referent}\n\nPASSAGE:{generation}"
