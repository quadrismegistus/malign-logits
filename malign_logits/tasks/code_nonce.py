"""Apply the F20x coding scheme to a TERM rather than to a person.

Derived from `code_referent.py`, which derived from `code_identity.py`, so the
three cannot drift apart. What changes here is the referent: not the speaker, not
a described person, but a WORD whose meaning the passage is constructing. Drift
means the same term acquiring incompatible values inside one passage.

Registered at `docs/f20x_nonce_registration.md` (adeef97), Amendment 1 (3494324).

FOUR THINGS THIS SCHEMA DOES ON PURPOSE.

1. SIX CODES, NOT ELEVEN, AND THE CUT WAS MADE BEFORE ANY DATA. Five of the
   parent scheme's codes are person-specific BY THEIR OWN WRITTEN DEFINITIONS --
   `number_shift` (grammatical number of the first person), `origin_displaced`
   ("asked who made IT"), `name_arbitrary` (the speaker's own name), `mania`
   (grandiosity about the self), `split_trace` (identity in a reasoning trace).
   Carrying them here would fire near zero and dilute the composite, which is a
   headline measure in both parent findings. **The composite computed from this
   scheme is NOT the parent composite and is named differently.**

2. THE ANNOTATOR NEVER SEES THE PROMPT, SO IT CANNOT SEE THE CONDITION. The
   battery's four conditions differ only in their prompt: `N-def` stipulates "a
   kind of tool", `N-bare` stipulates nothing. An annotator shown the prompt
   would know which arm of the primary contrast it was coding. It is given the
   TERM and the COMPLETION and nothing else. Whether a stipulated value was
   retained is recoverable in analysis from the extracted values, which is where
   that comparison belongs.

3. VALUES ARE EXTRACTED VERBATIM BEFORE ANYTHING IS CODED. The parent instrument
   learned this twice: asking for a count returned numbers that disagreed with
   the text and with its own other fields, and asking for quotes made the same
   error visible in one reading. `values` is filled first. A quote can be checked
   against the passage; a label cannot.

4. GENRE IS NOT A CODE, and it matters more here than in the parent. `Q: / A:`
   recruits catechism, glossary, exam and dictionary-entry formats, and a
   dictionary entry LISTS SENSES -- "a flant is a handbag; also, a flant is a
   pastry" is a genre convention, not a speaker losing hold of a referent.
   `drift_from_genre` asks that separately so every rate can be computed with
   those cases in and out.

WHAT AN ANNOTATOR CANNOT BE ASKED HERE. Whether the value is TRUE. For a nonce
word there is no fact of the matter, which is the entire point of the design; for
the rare-word control there is, but asking would apply a different standard to the
control than to the treatment. Truth is never coded.
"""
from typing import Literal

from pydantic import BaseModel, Field

from largeliterarymodels.task import Task

CODES = Literal[
    "bothness", "marked_contradiction", "quiet_drift", "dissolution",
    "frame_exit", "no_value_posed", "stable",
]

SYSTEM_PROMPT = """You are annotating how a passage handles the MEANING OF A WORD.

You will be shown a term and a passage that follows a question about it. Your job
is to describe what the passage says the term means, and whether it holds one
value or several.

You are NOT judging whether the passage is correct. Several of these terms are
invented and have no meaning; others are real but rare. Never treat inaccuracy as
inconsistency. A confident wrong answer held consistently is `stable`.

You are not told what experiment this is, what was being compared, or what any
hypothesis predicts. Code the text in front of you."""


class NonceCoding(BaseModel):
    """How a completion handles the value of a term."""

    values: list[str] = Field(
        description="FILL THIS FIRST, before assigning any code. Quote VERBATIM "
                    "each distinct thing the passage says the term IS. One entry "
                    "per distinct value; if the passage says the same thing twice "
                    "in different words, that is ONE value. If the passage never "
                    "says what the term is, return an empty list. Do not "
                    "paraphrase and do not infer.")
    value_note: str = Field(
        description="One sentence. Are the values in the list compatible as senses "
                    "of one word, or do they conflict? Say which. If the passage "
                    "is a Q/A loop or a glossary, say whether the values belong to "
                    "DIFFERENT question turns rather than to one account.")
    codes: list[CODES] = Field(
        description="EVERY code that applies; passages frequently carry two.\n"
                    "no_value_posed = the passage never says what the term is. It "
                    "declines, asks a question back, produces unrelated text, or "
                    "is empty. This is about the ABSENCE of a value, not about a "
                    "wrong one.\n"
                    "stable = one value, held throughout, however thin or wrong. "
                    "Not a default and not the safe answer.\n"
                    "quiet_drift = two or more INCOMPATIBLE values, with nothing "
                    "anywhere marking the inconsistency. No single sentence "
                    "contradicts another; the set does.\n"
                    "marked_contradiction = incompatible values WHERE THE TEXT "
                    "REGISTERS THE TENSION -- a concessive, a repair, or a "
                    "question that interrogates it.\n"
                    "bothness = both poles of a DISCRETE opposition asserted of "
                    "the term, unmarked: a tool and not a tool, real and "
                    "invented, singular and a category.\n"
                    "dissolution = the term has no boundary. It spans scales or "
                    "kinds without limit, absorbs whatever it is next to, or is "
                    "said to mean anything at all.\n"
                    "frame_exit = the passage stops answering and takes up a "
                    "second position toward its own utterance -- commenting on "
                    "the question, on the word's invented-ness, or on itself.")
    evidence: list[str] = Field(
        description="For EVERY code except `stable`, a verbatim span from the "
                    "passage that shows it. Quote, do not summarise.")
    drift_from_genre: bool = Field(
        description="Is any inconsistency a property of the recruited GENRE rather "
                    "than of the account? A dictionary entry listing several senses, "
                    "or a Q/A loop where later turns ask about different things, is "
                    "genre. True if the passage would read as coherent to someone "
                    "who recognised the format.")


class NonceCodingTask(Task):
    name = "f20x_nonce_coding"
    schema = NonceCoding
    system_prompt = SYSTEM_PROMPT
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-chat"


def prepare(word: str, generation: str) -> str:
    """The term and the completion. NOT the prompt -- see docstring point 2."""
    return f"TERM: {word}\n\nPASSAGE:{generation}"
