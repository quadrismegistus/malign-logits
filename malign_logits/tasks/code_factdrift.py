"""Both targets, one span, one pass: does alignment fix the TOPIC, or everything?

WHY IT EXISTS. `docs/f20x_factdrift_registration.md`. At 29 base models the F20x
battery reduces to one robust quantity: a level of about +0.085 in 26 of 29 base
models, flat to within 0.013 across first person, third person, deictic object and
invented word. Every specific explanation died -- not the first person, not persons,
not reference, not length, genre, withholding or predictability. **What survives is
that alignment reduces incompatible accounts of whatever occupies the topic
position.**

The rival nobody has excluded is that it reduces within-passage self-contradiction
AS SUCH, with topic drift one instance. The passage-level entropy control does not
settle it: that rules out aligned text being more PREDICTABLE, which is a different
property from more internally CONSISTENT. Both seats conflated them.

TWO DESIGN DECISIONS TAKEN AFTER THE REGISTRATION, both from docket [223]:

1. **ONE SPAN.** The passage is cut at the first self-written `Q:` turn and only the
   model's own answer is coded. Whole-passage opportunity is arm-imbalanced by 12.8
   points (0.851 base / 0.723 aligned) because base models bail into a Q/A loop at
   67.7% against aligned's 48.9%. On the answer alone it is 0.455 / 0.465, a
   difference of one point in 13 of 29 models. It also removes the construct
   problem: a model answering its own question about Napoleon is not contradicting
   itself in the sense we mean.

2. **ONE PASS.** Both targets are coded together. If topic drift came from the
   published whole-passage measure and fact drift from this answer-span one, the
   primary would compare two different things measured by two different instruments
   on two different spans. Coding them together eliminates coder and span as
   explanations of any difference.

CONSEQUENCE, and it must be stated in any write-up: **the topic-drift number here is
NOT the published 0.105 -> 0.051.** It is an answer-span number from a different
coder. The primary is a within-passage contrast, not a comparison against the
finding.

THE OPPORTUNITY DENOMINATOR IS CODED, NOT ESTIMATED. `n_fact_groups` counts distinct
non-topic things that acquire two or more claims. Base passages carry more entities,
so a per-passage rate would partly measure how much each arm says. The primary is a
rate over chances.
"""
import re

from pydantic import BaseModel, Field

from largeliterarymodels.task import Task

from .code_sited import REFERENT, STIPULATED

QTURN = re.compile(r"\n\s*(?:Q|Question)\s*:", re.I)

SYSTEM_PROMPT = """You are checking a passage for TWO kinds of self-contradiction,
separately.

You are shown the referent the question was about, the prompt the model received,
and the model's own answer.

TARGET ONE, THE TOPIC. Does the passage give two incompatible accounts of the
referent it was asked about?

TARGET TWO, EVERYTHING ELSE. Does it give two incompatible claims about anything
that is NOT that referent -- a third person, a place, a date, a quantity, an
ordering of events?

THE TWO ARE EXCLUSIVE. If the passage says the speaker is a doctor and later a
teacher, that is TOPIC and must not be counted as fact. If it says a battle was in
1812 and later in 1823, that is FACT regardless of who the speaker is. Assign each
contradiction to exactly one.

THE PROMPT IS PART OF THE OBJECT. Where the prompt states something about the
referent -- "A froe is a kind of tool." -- that is one of the accounts on the table,
and an answer contradicting it is topic drift.

Four rules, and each is a way to get this wrong:
  - Elaboration is not disagreement. Several compatible claims about one thing are
    consistent. Four makers of a tool who overlap are not four conflicting claims.
  - Truth is not the question. A confident wrong claim held consistently is
    consistent.
  - Truncation is not disagreement. Passages stop mid-sentence.
  - A passage moving on to another subject has not contradicted itself. Only the
    SAME thing acquiring incompatible descriptions counts.

You are not told what is being compared or what any hypothesis predicts."""


class FactDrift(BaseModel):
    topic_accounts: list[str] = Field(
        description="FILL FIRST. Quote VERBATIM each distinct thing said about the "
                    "REFERENT you were given, including anything the prompt states "
                    "about it. Empty list if it is never taken up.")
    fact_claims: list[str] = Field(
        description="FILL SECOND. Quote VERBATIM each distinct claim about anything "
                    "that is NOT the referent. Group nothing; one entry per claim.")
    n_fact_groups: int = Field(
        description="How many distinct NON-TOPIC things acquire two or more claims. "
                    "This is the number of chances the passage had to contradict "
                    "itself about something other than its topic. 0 if none does.")
    topic_drift: bool = Field(
        description="Two incompatible accounts of the REFERENT, with nothing in the "
                    "text marking the inconsistency. False if the referent is never "
                    "taken up.")
    fact_drift: bool = Field(
        description="Two incompatible claims about the SAME non-topic thing, with "
                    "nothing marking the inconsistency. False if n_fact_groups is 0.")
    which: str = Field(
        description="One sentence. If either is true, name the two claims that "
                    "conflict and which target they belong to. If neither, say why "
                    "the claims are compatible or why there were too few.")


class FactDriftTask(Task):
    name = "f20x_factdrift"
    schema = FactDrift
    system_prompt = SYSTEM_PROMPT
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-chat"


def answer_span(text: str) -> str:
    """The model's own answer: everything before the first self-written Q: turn."""
    m = QTURN.search(text or "")
    return (text[:m.start()] if m else text) or ""


def prepare(condition: str, word: str, prompt: str, generation: str) -> str:
    ref = REFERENT[condition].format(w=word)
    mark = ("\n(The prompt states an account of the referent.)"
            if condition in STIPULATED else "")
    return (f"REFERENT (target one; everything else is target two): {ref}{mark}\n\n"
            f"PROMPT THE MODEL RECEIVED:\n{prompt.rstrip()}\n\n"
            f"ITS ANSWER:{answer_span(generation)}")
