"""Locate the two conflicting accounts in a drifting passage, and say whether the
passage's own later questions keep the referent.

FOR THE BOUNDARY TEST, docket [246]-[256], [333]. The test asks whether drift falls
ACROSS a self-written turn boundary more than WITHIN a turn. Deciding that requires
knowing WHERE in the passage the two conflicting accounts sit, which no existing
coder emits -- the scheme codes are passage-level.

THE CODER IS BLIND TO THE COMPARISON. It is not told that boundaries exist, that
anything is being compared across them, or what any hypothesis predicts. It quotes
the two accounts; the location analysis happens downstream in code, from the quotes.
That is deliberate: the turn boundary is visible in the text the coder reads and
[249] booked its legibility as a property of the object rather than something
blinding can remove, so the least this design can do is not name it.

THE SAME-REFERENT FIELD is the [247] fix (b). A model that writes `Q: Where does
'The Starry Night' come from?` has changed the subject, and a difference across THAT
boundary is topic drift, not referent drift -- RH's own rule from the binary-set
revision. Without this field the across-cell is inflated by the model asking itself
about something else.
"""
from typing import Literal

from pydantic import BaseModel, Field

REFERENT_STATUS = Literal["all_keep_referent", "some_change_subject",
                          "all_change_subject", "no_further_questions"]

SYSTEM_PROMPT = """You are reading a passage that has been judged to contain two
incompatible accounts of one referent. Your job is to find them and quote them.

You are told which referent to track. The passage may contain further questions the
writer posed to itself; treat those as part of the passage.

THREE THINGS, in order.

1. THE TWO ACCOUNTS. Quote VERBATIM the two shortest spans of the passage that
   cannot both be true of the referent. Quote exactly -- character for character,
   copied from the passage, not paraphrased and not corrected. The quotes are used
   to locate the accounts in the text, so a paraphrase makes the item useless.

   If you cannot find two spans that genuinely conflict, say so in `found` and leave
   the quotes empty. A prior judgment that the passage drifts is not binding on you.

2. WHICH REFERENT the later questions concern. If the passage poses further
   questions after its opening answer, do they still ask about the same referent, or
   do they move to some other subject? A question about a different person, object
   or topic has changed the subject.

3. WHETHER THE CONFLICT IS MARKED. If the passage itself notices, hedges or repairs
   the inconsistency, say so. An unmarked conflict is one the passage lets stand.

You are not told what is being compared or what any hypothesis predicts."""


class ConflictLocation(BaseModel):
    found: bool = Field(
        description="Are there two spans that genuinely cannot both be true of the "
                    "referent? False is a legitimate answer.")
    quote_1: str = Field(
        description="First conflicting span, VERBATIM from the passage. Empty if "
                    "found is false.")
    quote_2: str = Field(
        description="Second conflicting span, VERBATIM. Empty if found is false.")
    attribute: str = Field(
        description="What property of the referent the two spans disagree about, "
                    "one or two words. Empty if found is false.")
    referent_status: REFERENT_STATUS = Field(
        description="Do the passage's later questions still ask about the same "
                    "referent? `no_further_questions` if it poses none.")
    marked: bool = Field(
        description="Does the passage itself notice, hedge or repair the conflict?")


try:
    from largeliterarymodels.task import Task

    class ConflictLocationTask(Task):
        name = "f20x_conflict_locate"
        schema = ConflictLocation
        system_prompt = SYSTEM_PROMPT
        retries = 2
        temperature = 0.0
        model = "deepseek/deepseek-chat"
except ImportError:  # pragma: no cover
    ConflictLocationTask = None


def prepare(question: str, passage: str) -> str:
    return (f"REFERENT TO TRACK: whatever the question asks about\n"
            f"THE QUESTION THE WRITER WAS ANSWERING: {question}\n\n"
            f"PASSAGE:\n{passage.strip()}")
