"""The licit-category coder for the M05 syntax curve (registered secondary 5).

One call per PROMPT — the coder never sees a candidate word, a model name,
or a training stage; blindness is structural. It answers: which grammatical
classes of word could a competent speaker of English naturally continue
this text with, as the VERY NEXT word?

THE WITNESS DISCIPLINE IS THE INSTRUMENT. English is permissive enough
that a lazy judge licenses every class and the downstream curve loses all
dynamic range. So a class counts as licit ONLY with a produced witness --
an actual word of that class that could follow -- and every witness is
mechanically cross-checked downstream: appended to the prompt and run
through the same spaCy tagger that tagged the store's words
(data/m05_syntax_tags.parquet). A claimed NOUN witness that does not tag
NOUN in context is a flagged disagreement, not a silent one.

The class inventory is exactly the artifact's `pos_class` values, minus
the format band (PUNCT / X / SYM), which the curve reports separately --
a fill-in-the-blank token is format, not grammar, and must not be counted
against syntax. SCONJ is folded into ADP by the artifact's tag table (IN),
so subordinators are judged under ADP.

Judge FORM, not meaning: if a word of the class fits grammatically, the
class is licit even if the result is strange or unpleasant in meaning
("She slowly took off her scream" is grammatical). Meaning is tier 3's
question, deliberately not this one.
"""
from typing import Literal

from pydantic import BaseModel, Field

from largeliterarymodels.task import Task

POSClass = Literal["NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "PRON",
                   "DET", "ADP", "PART", "CCONJ", "NUM", "INTJ"]

SYSTEM_PROMPT = """You judge what kinds of word can grammatically come NEXT.

You are shown a text that stops mid-stream, often mid-phrase. Your job:
for each grammatical class below, decide whether a competent speaker of
English could NATURALLY continue the text with a single word of that class
as the very next word.

The classes, with examples:
  NOUN   common noun (dog, idea, table)      PROPN  proper name (Tokyo, Maria)
  VERB   verb, any form (run, took, located) AUX    auxiliary/modal (is, can, must)
  ADJ    adjective (red, strange)            ADV    adverb (slowly, very, now)
  PRON   pronoun (she, it, hers)             DET    determiner (the, a, this, each)
  ADP    preposition or subordinator (in, of, that, because)
  PART   particle (to, off, up as in "gave up", possessive 's)
  CCONJ  coordinator (and, or, but)          NUM    numeral (three, 1969)
  INTJ   interjection (oh, well, yes)

Rules, each a way to get this wrong:
  - JUDGE FORM, NOT MEANING. If a word of the class fits the grammar, the
    class is licit even when the result is semantically odd or unpleasant.
    Strangeness of meaning is not your question; ungrammaticality is.
  - A CLASS NEEDS A WITNESS. Call a class licit only if you can produce an
    actual example word of that class that could follow. If you cannot
    write the word, the class is not licit.
  - NATURAL means a speaker could just say it. If a continuation needs
    heavy strain, an unusual register, or an imagined quotation to work,
    the class is MARGINAL, not licit.
  - THE NEXT WORD ONLY. Do not license a class because some longer
    continuation could eventually reach it.

You are not told what is being compared or what any hypothesis predicts."""


class WitnessedClass(BaseModel):
    pos: POSClass
    example: str = Field(
        description="One actual word of this class that could follow the "
                    "text as the very next word. The word alone, no "
                    "punctuation, no quotation marks.")


class LicitSet(BaseModel):
    frame: str = Field(
        description="FILL THIS FIRST. One sentence naming the syntactic "
                    "position the next word would occupy (e.g. 'after a "
                    "possessive determiner, heading the object NP').")
    licit: list[WitnessedClass] = Field(
        description="Every class a competent speaker could naturally "
                    "continue with, each with its witness word.")
    marginal: list[WitnessedClass] = Field(
        description="Classes grammatical only with strain or special "
                    "context, each with its witness word. Empty list if "
                    "none.")


class LicitSetTask(Task):
    name = "m05_licit_v1"
    schema = LicitSet
    system_prompt = SYSTEM_PROMPT
    retries = 2
    temperature = 0.0
    #: pinned to the resolved id, not the retired `deepseek-chat` alias, on
    #: the harness's own warning: the alias resolves server-side to a
    #: DIFFERENT model name, and the model of record must be the real one.
    #: Stability probe (plan): second family, anthropic/claude-haiku-4-5.
    model = "deepseek/deepseek-v4-flash"
