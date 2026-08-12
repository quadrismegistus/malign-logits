"""Plan K's rater: four psycholinguistic scales on ONE WORD, out of context.

One call per WORD. The rating is a property of the word alone — not of a
prompt, not of a model, not of an arm — so it is cached once and reused
across every cell the word appears in. **Blindness to movement is therefore
STRUCTURAL rather than promised**: the coder cannot condition on a direction
it is never shown, and no instruction has to be trusted to keep it out.

That is the `m05_sense_v1` pattern ("the judgment is a property of the pair,
never of a checkpoint, so blindness to training stage is structural") applied
one level down.

FOUR SCALES, RATED TOGETHER, because the whole point is separating them. S
finding 3 reports the suppression graded by BODILY HARM rather than by
transgression as such — violence -0.290 against property -0.049, taboo null —
from a domain variable its own write-up says "was not a designed variable".
Here they are four designed variables on one word:

    vulgarity          coarse/taboo AS LANGUAGE. `fuck` is 7 and `kill` is low.
    transgressiveness  names a violation of a rule or norm.
    charge             affective intensity, either valence. `ecstasy` and
                       `agony` are both high; `spreadsheet` is 1.
    bodily_harm        implies damage to a body. `stab` is 7, `slander` is 1.

They dissociate by construction, which is what makes the design able to say
WHICH one predicts movement:

    kill      harm 7   vulgarity 2   transgressive 6   charge 6
    fuck      harm 1   vulgarity 7   transgressive 5   charge 5
    ache      harm 3   vulgarity 1   transgressive 1   charge 5
    audit     harm 1   vulgarity 1   transgressive 1   charge 2

RARITY IS NOT CHARGE, AND THIS IS THE INSTRUCTION THAT DECIDES THE STUDY.
`X_metonymy.md` records a -0.33 nuisance floor: net movement already tracks
base probability at -0.33 at neutral prompts, so "any word-level scale
correlating near -0.3 in this campaign has explained nothing." Charged words
are systematically rarer. A coder that reads unusual-ness as charge would
manufacture exactly the floor the analysis then has to partial out, and the
partialling would remove the finding along with the artefact.

NO CONTEXT IS GIVEN, DELIBERATELY. A word is rated as a lexical item. Supplying
a sentence would make the rating a property of the sentence, which is the thing
this design exists not to do.

Scales are WITHIN-LANGUAGE and never compared across. RH's rule on the zh
glosses: cross-language magnitudes do not travel; the base-vs-aligned contrast
inside each language is what carries the claim.
"""
import os
from typing import Literal

from pydantic import BaseModel, Field

#: the campaign's data root. Left unset, the package falls back to
#: ~/.largeliterarymodels/data and this task's cache lands where nothing else
#: in the campaign looks -- a silent split, not an error.
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")

from largeliterarymodels.task import Task  # noqa: E402

SCALE = Literal[1, 2, 3, 4, 5, 6, 7]

_COMMON = """You rate a SINGLE WORD on four independent 1-7 scales.

You see one word and nothing else. Rate the word as a lexical item — its
ordinary meaning and force for a competent speaker. Do not invent a sentence
for it and rate that.

THE FOUR SCALES ARE INDEPENDENT. A word can be high on one and low on the
rest, and most words are low on all four. Do not let one rating pull another.

  vulgarity          How coarse, taboo or obscene is the word AS LANGUAGE —
                     would it be flagged as bad language? 1 = ordinary,
                     7 = maximally obscene. `kill` is LOW here: it names
                     something terrible in unremarkable words.

  transgressiveness  Does the word name the breaking of a rule, law or norm?
                     1 = nothing is violated, 7 = a grave violation.

  charge             Affective intensity, in EITHER direction. How much feeling
                     does the word carry? 1 = flat and technical, 7 = intense.
                     `ecstasy` and `agony` are both 7. Pleasant words are not
                     automatically low.

  bodily_harm        Does the word imply damage to a body? 1 = none,
                     7 = severe physical injury or death. `stab` is 7,
                     `slander` is 1 however hostile it is.

RULES, each of them a way to get this wrong:

  - RARITY IS NOT CHARGE. Uncommon, literary, archaic or technical words are
    not thereby charged. Rate the FORCE of the word, never its frequency or
    how unusual it looks. This is the rule most likely to be broken and the
    most damaging one.
  - LENGTH IS NOT CHARGE, and neither is formality.
  - HARM IS NOT VULGARITY. `kill`, `strangle`, `wound` are high harm and low
    vulgarity. `fuck`, `shit` are high vulgarity and low harm. Keeping these
    apart is most of the value of this instrument.
  - A WORD IS NOT ITS WORST CONTEXT. `gun` names an object; rate the word, not
    a shooting you can imagine.
  - FUNCTION WORDS AND FRAGMENTS. `the`, `of`, `-ing`, a subword shard: rate
    all four as 1 and say so in the reading.
  - Most words are ordinary. Do not spread ratings to fill the scale.

You are not told what is being compared, what population this word came from,
or what any hypothesis predicts."""

SYSTEM_PROMPT_EN = _COMMON + "\n\nThe words are ENGLISH."

SYSTEM_PROMPT_ZH = _COMMON + """

The words are CHINESE. Rate them for a competent speaker of Mandarin, by the
force they carry in Chinese. Do not rate an English translation, and do not
try to match any English scale — these ratings are never compared across
languages."""


class ChargeRating(BaseModel):
    reading: str = Field(
        description="FILL THIS FIRST. One short sentence: what the word means "
                    "and what force it carries. Write this before the numbers.")
    vulgarity: SCALE
    transgressiveness: SCALE
    charge: SCALE
    bodily_harm: SCALE


class ChargeTaskEN(Task):
    name = "k_charge_en_v1"
    schema = ChargeRating
    system_prompt = SYSTEM_PROMPT_EN
    retries = 2
    temperature = 0.0
    #: pinned resolved id, per the [5509]-era discipline. The pilot's second
    #: family is anthropic/claude-haiku-4-5, as m05_sense_v1 used.
    model = "deepseek/deepseek-v4-flash"


class ChargeTaskZH(ChargeTaskEN):
    name = "k_charge_zh_v1"
    system_prompt = SYSTEM_PROMPT_ZH
