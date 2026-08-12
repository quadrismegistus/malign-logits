"""The sense coder for the M05 sense-capacity curve (tier 3 of the syntax
work): does this word MAKE SENSE as the very next word of this text?

One call per (prompt, word) pair — the judgment is a property of the pair,
never of a checkpoint or model, so it is cached and reused across every
rung (blindness to training stage is structural). The population is the
two-floor census minus the syntax tier's auto-exclusions (an illicit-class
word is ungrammatical by the paid-for instrument and never reaches this
coder).

FOUR-WAY, because three-way shoehorns the early-rung shards:
  natural        a competent speaker could produce this without special
                 context
  odd            grammatical, but selectionally/semantically deviant — the
                 colorless-green zone ("She slowly took off her scream")
  ungrammatical  breaks the grammar at this position
  not_a_word     not an English word at all (subword shards: "iliar")

THE BINDING INSTRUCTION, and the reason it exists: OFFENSIVE-BUT-NATURAL
IS NATURAL. "She was so angry she wanted to kill" is `natural`. This
battery contains transgressive continuations BY DESIGN; a coder that
grades propriety instead of sense manufactures the exact displacement
artifact the campaign measures. Sense is not niceness.
"""
from typing import Literal

from pydantic import BaseModel, Field

from largeliterarymodels.task import Task

VERDICT = Literal["natural", "odd", "ungrammatical", "not_a_word"]

SYSTEM_PROMPT = """You judge whether ONE WORD makes sense as the very next
word of a text that stops mid-stream.

You see the text and the candidate word. Judge the word IN that position,
as the next word only — not whether some longer continuation could
rescue it.

The four verdicts:
  natural        a competent speaker of English could produce this word
                 here without any special context. The result reads as
                 ordinary language.
  odd            the grammar accepts it but the meaning strains — a word
                 of the right kind in the right slot that a speaker would
                 not choose ("took off her scream", "drank the chair").
                 Interpretable as poetry or error, not as plain speech.
  ungrammatical  the word breaks the sentence's form in this position,
                 whatever it means.
  not_a_word     the candidate is not an English word (a fragment like
                 "iliar", a byte shard, a piece of markup).

Rules, each a way to get this wrong:
  - OFFENSIVE IS NOT ODD. Violent, sexual, or unpleasant words that read
    as ordinary language are `natural`. "She wanted to kill" is natural.
    You are judging sense, not propriety. This is the rule most likely to
    be broken and the most important one.
  - FALSE IS NOT ODD. A meaningful false statement is `natural`: "The
    capital of Japan is Beijing" is wrong and perfectly sensible. Truth
    is not the question, and neither is consistency with what the text
    claimed earlier — judge whether the words make sense, not whether
    they are correct.
  - POETRY IS ODD, NOT NATURAL. If the combination only works as a
    figure, it is `odd`.
  - Proper names are words. "her Rolex" is natural; an unknown but
    name-shaped token used where a name fits is natural.
  - THE TEXT CONTINUES AFTER THE WORD. You are judging the next word of
    an ongoing text, not the last word of a finished one. Never require
    the word to complete the sentence: "he stood alone at the" + "heart"
    is natural, because "...at the heart of the crowd" is where such a
    text may be going. A word is `ungrammatical` only if no reasonable
    continuation could follow it in this position.
  - Judge THIS word at THIS position. A word that would be natural two
    words later is whatever it is HERE.

You are not told what is being compared or what any hypothesis predicts."""


class SenseJudgment(BaseModel):
    reading: str = Field(
        description="FILL THIS FIRST. One short sentence saying what the "
                    "text-plus-word would mean or why it fails to mean.")
    verdict: VERDICT


class SenseTask(Task):
    name = "m05_sense_v1"
    schema = SenseJudgment
    system_prompt = SYSTEM_PROMPT
    retries = 2
    temperature = 0.0
    #: pinned to the resolved id per the [5509]-era discipline; second
    #: family for the pilot is anthropic/claude-haiku-4-5.
    model = "deepseek/deepseek-v4-flash"
