"""The coder with the prompt in place -- the LLM counterpart of the human-sheet fix.

WHY. RH coded the human sheets and found they did not show the prompt (docket
[183]). The serious part was the third: **in the stipulated conditions the PROMPT
CONTAINS AN ACCOUNT** -- `A adze is a kind of tool.` -- so a coder judging whether
the passage holds one consistent account, while not seeing the account the passage
was handed, is judging less than the object. That argument applies identically to
the LLM coder, which is also blind to the prompt.

THIS REVERSES A CHOICE I MADE AND THE REASON MATTERS. `code_nonce` withholds the
prompt so the annotator cannot see which condition it is coding. That rationale was
already weak -- a one-word rule ('says tool') recovers the condition from the
completion 84.5% of the time -- and it is now superseded: withholding does not buy
blindness, and it costs part of the object. **What the annotator must stay blind to
is the ARM**, which it is either way, because base and aligned differ in no field
it is shown.

WHAT IS ASKED IS FITTED TO WHAT WAS ASKED. `the referent` is not a phrase that can
be applied to `that` without saying what `that` picked out, so the referent is
named per condition exactly as on the human sheet. One construct, sited wording.

WHETHER THIS REPLACES THE BLIND CODER IS AN AMENDMENT DECISION, NOT A SILENT
SWITCH. It is smoke-tested against the blind coder on the same passages first, and
the comparison is expected to move the STIPULATED conditions most -- that is where
the prompt carries an account and therefore where blindness costs something.
"""
from typing import Literal

from pydantic import BaseModel, Field

from largeliterarymodels.task import Task

CODES = Literal[
    "bothness", "marked_contradiction", "quiet_drift", "mania", "dissolution",
    "name_arbitrary", "number_shift", "origin_displaced", "frame_exit",
    "no_value_posed", "stable",
]

REFERENT = {
    "1P": "the speaker -- the 'I' of the answer",
    "3P": "the person the question asks about",
    "O-named": "the {w}",
    "O-deictic": "whatever the answer takes 'that' to refer to",
    "N-def": "the invented word '{w}'",
    "N-bare": "the invented word '{w}'",
}
STIPULATED = {"O-named", "N-def"}

SYSTEM_PROMPT = """You are annotating how a passage handles ONE REFERENT.

You are shown the referent, the prompt the model received, and the answer it gave.

THE PROMPT IS PART OF THE OBJECT. Where the prompt already states something about
the referent -- for example "A froe is a kind of tool." -- that is one of the
accounts on the table, and an answer that contradicts it does not hold one
consistent account.

Track only the referent you are given. A passage that introduces other people or
things and describes them consistently is not drifting. Topic drift is not referent
drift: the referent itself must acquire incompatible descriptions.

You are NOT judging whether the answer is correct. Some referents are invented and
have no facts about them. A confident wrong account held consistently is `stable`.

You are not told what is being compared or what any hypothesis predicts."""


class SitedCoding(BaseModel):
    """How a completion handles one referent, with the prompt visible."""

    accounts: list[str] = Field(
        description="FILL THIS FIRST. Quote VERBATIM each distinct thing said "
                    "about the referent, INCLUDING anything the prompt states "
                    "about it. One entry per distinct account; the same thing "
                    "twice in different words is one. Empty list if nothing is "
                    "said about it anywhere.")
    referent_note: str = Field(
        description="One sentence: does the answer actually take up the referent "
                    "the question asked about? In a Q/A loop the model often "
                    "writes further questions of its own, and later turns may be "
                    "about something else. If the referent is never picked up, say "
                    "so here.")
    codes: list[CODES] = Field(
        description="EVERY code that applies; passages frequently carry two. "
                    "'stable' is what is left when nothing else applies.\n"
                    "no_value_posed = nothing is said about the referent at all: "
                    "it declines, asks back, produces unrelated text, or never "
                    "picks the referent up. About ABSENCE, not a wrong account.\n"
                    "stable = one account, held throughout, however thin or wrong.\n"
                    "quiet_drift = two or more INCOMPATIBLE accounts, with nothing "
                    "marking the inconsistency -- including incompatibility with "
                    "what the prompt stated.\n"
                    "marked_contradiction = incompatible accounts WHERE THE TEXT "
                    "REGISTERS THE TENSION, via a concessive, a repair, or a "
                    "question interrogating it.\n"
                    "bothness = both poles of a DISCRETE opposition asserted, "
                    "unmarked.\n"
                    "dissolution = the referent has no boundary; it spans scales "
                    "or kinds without limit.\n"
                    "mania = grandiosity with a stable referent.\n"
                    "name_arbitrary = the name is accidental, disavowed, false or "
                    "replaced.\n"
                    "number_shift = the referent changes grammatical number.\n"
                    "origin_displaced = asked who made it, answers with "
                    "transformation, self-causation or theology instead of a "
                    "maker.\n"
                    "frame_exit = the passage leaves the position of answering and "
                    "takes up a second position toward its own utterance.")
    evidence: list[str] = Field(
        description="For every code except `stable`, a verbatim span. Quote.")
    drift_from_genre: bool = Field(
        description="Is any inconsistency a property of the recruited GENRE rather "
                    "than of the account -- a multiple-choice item the model wrote, "
                    "a glossary listing senses, a Q/A loop asking about different "
                    "things? True if it would read as coherent to someone who "
                    "recognised the format.")


class SitedCodingTask(Task):
    name = "f20x_sited_coding"
    schema = SitedCoding
    system_prompt = SYSTEM_PROMPT
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-chat"


def prepare(condition: str, word: str, prompt: str, generation: str) -> str:
    ref = REFERENT[condition].format(w=word)
    mark = ("\n(The prompt states an account of the referent.)"
            if condition in STIPULATED else "")
    return (f"REFERENT TO TRACK: {ref}{mark}\n\n"
            f"PROMPT THE MODEL RECEIVED:\n{prompt.rstrip()}\n\n"
            f"ITS ANSWER:{generation}")
