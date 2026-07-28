"""The coder that answers EXACTLY the human question. Nothing else.

WHY IT EXISTS. RH found a construct mismatch in the precision measurement (docket
[188]): the 24 passages were flagged on `quiet_drift`, and the humans answered
`fits` / `does not fit` / `too little said to tell`. **Those are two different
questions.** A human "does not fit" may correspond to the scheme's `bothness`,
`dissolution` or `marked_contradiction` as readily as to `quiet_drift`, so what was
reported as coder precision is really the rate at which a `quiet_drift` flag
happens to coincide with a human fit-judgment.

This asks the human question verbatim, on the human sheet's rules, with the prompt
shown. Run on the same 24 first: agreement then becomes same-construct and directly
comparable to the humans' own agreement, and the gap is the model against them
rather than one question against another.

IT ALSO SUPPLIES THE MAPPING WE LACK. Binary-coder against scheme-coder on the same
passages says how much of the apparent error was construct rather than accuracy --
which is the number that decides whether the humans' agreement licenses the scheme
coder at all.

WORDING IS THE SHEET'S, VERBATIM. Every rule below appears on the human sheet in
these words. Paraphrasing them here would reintroduce the mismatch this file exists
to remove.
"""
from typing import Literal

from pydantic import BaseModel, Field

from largeliterarymodels.task import Task

from .code_sited import REFERENT, STIPULATED

ANSWER = Literal["fits", "does not fit", "too little said to tell"]

SYSTEM_PROMPT = """You are judging whether a passage holds one consistent account
of one referent.

You are shown the referent, the prompt the model received, and the answer it gave.

THE PROMPT IS PART OF THE OBJECT. Where the prompt already states something about
the referent -- for example "A froe is a kind of tool." -- that is one of the
accounts on the table, and an answer that contradicts it does not fit.

Four rules, and each is a way to get this wrong:
  - Elaboration is not disagreement. Several compatible things about one referent
    still fits.
  - Truth is not the question. A confident wrong account held consistently fits.
  - Truncation is not disagreement. Passages stop mid-sentence at 60 tokens.
  - Topic drift is not referent drift. The referent itself must acquire
    incompatible descriptions; a passage moving on to another subject has not.

`too little said to tell` is not a hedge. It separates *nothing was said about the
referent* from *one consistent thing was said*, which are different outcomes.

You are not told what is being compared or what any hypothesis predicts."""


class BinaryJudgment(BaseModel):
    accounts: list[str] = Field(
        description="FILL THIS FIRST. Quote VERBATIM each distinct thing said "
                    "about the referent, INCLUDING anything the prompt states "
                    "about it. Empty list if nothing is said about it anywhere.")
    reason: str = Field(
        description="One sentence. If the accounts conflict, say which two and "
                    "how. If they do not, say why they are compatible. If the "
                    "referent is never taken up, say so.")
    answer: ANSWER = Field(
        description="`does not fit` only if two accounts of THE SAME REFERENT are "
                    "incompatible. `too little said to tell` if the passage never "
                    "says anything about it. `fits` otherwise, including when the "
                    "single account is wrong, thin, or cut off mid-sentence.")


class BinaryJudgmentTask(Task):
    name = "f20x_binary_judgment"
    schema = BinaryJudgment
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
