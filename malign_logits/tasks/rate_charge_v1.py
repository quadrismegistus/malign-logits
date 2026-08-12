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


# ── v2: SEVEN SCALES, for the pilot that decides which survive ───────────
#
# Three added on RH's questions, 2026-08-12, each for a stated reason:
#
#   register     Vulgarity is UNIPOLAR -- it sees only the coarse end. Register
#                is bipolar (clinical <-> coarse) and `penis` scoring vulgarity 2
#                is not the same fact as `penis` being CLINICAL. S finding 4 (at
#                marked sites the act moves into speech) and T (movement onto
#                perception/cognition/speech) are register claims, and vulgarity
#                can only see their absence.
#   valence      `fields.py` carries Warriner V/A/D -- ENGLISH ONLY, which is the
#                wall registration O hit: it could not separate "the affective
#                mechanism is absent in Chinese" from "the Chinese norm join is
#                too thin to see it". Rating it here gives Chinese an affective
#                measure it has never had, AND lets the English half be
#                calibrated against human norms.
#   concreteness Brysbaert, same story: English only, and T's central claim is
#                movement off contact/motion onto cognition/perception, which is
#                the concreteness axis.
#
# NOT ADDED: a `safety` scale. "Would a safety-tuned model avoid this word" asks
# the coder to predict the outcome under measurement -- the stimulus naming the
# construct. Transgressiveness is the honest version: it asks about the norm,
# not about the model. Refused deliberately, recorded so the absence is a
# decision rather than an oversight.
#
# NOT ADDED: arousal. `charge` is arousal, and registration D found arousal
# movement-general rather than site-targeted.
#
# SEVEN IS PROBABLY TOO MANY and this exists to find out. The pilot rates the
# whole set, then the correlation matrix drops whatever fails to dissociate --
# collinear scales buy noise and multiple comparisons, not information.

_V2_EXTRA = """

  register_level     HOW ELEVATED IS THE WORD -- lowbrow to highbrow.
                     1 = colloquial, slang, street (`telly`, `grub`, `mate`,
                     `shit`), 4 = ordinary everyday usage (`dog`, `table`,
                     `dead`), 7 = formal, learned, literary or technical
                     (`notwithstanding`, `utilise`, `deceased`, `faeces`).

                     THIS IS NOT VULGARITY, and the difference is the point.
                     `telly` and `grub` are lowbrow and not remotely obscene;
                     `faeces` and `deceased` are highbrow words for unpleasant
                     things. A word's elevation and its coarseness are separate
                     facts and a great many words are low on one and neutral on
                     the other. Rate cultural register, not offensiveness.

                     Runs low-to-high like every other scale here: a bigger
                     number means MORE elevated.

  valence            PLEASANT vs UNPLEASANT. 1 = very negative (`torture`),
                     4 = neutral (`ledger`), 7 = very positive (`joy`). This is
                     direction, where `charge` is intensity: `agony` and
                     `ecstasy` share a charge of 7 and sit at opposite ends here.

  concreteness       Can the word's referent be perceived by the senses?
                     1 = wholly abstract (`justice`, `whereas`), 7 = directly
                     perceptible (`toes`, `knife`). Independent of every other
                     scale."""


class ChargeRating7(BaseModel):
    reading: str = Field(
        description="FILL THIS FIRST. One short sentence: what the word means "
                    "and what force it carries. Write this before the numbers.")
    vulgarity: SCALE
    #: NOT `register`: that name is a classmethod on pydantic's BaseModel (via
    #: ABCMeta registration), so the field validates fine and lands in
    #: model_fields while attribute access returns the METHOD. It raised here
    #: rather than returning a plausible number, which is the only reason it was
    #: cheap to find.
    register_level: SCALE
    transgressiveness: SCALE
    charge: SCALE
    valence: SCALE
    bodily_harm: SCALE
    concreteness: SCALE


class ChargeTask7EN(Task):
    name = "k_charge_en_v2"
    schema = ChargeRating7
    system_prompt = (_COMMON.replace(
        "You rate a SINGLE WORD on four independent 1-7 scales.",
        "You rate a SINGLE WORD on seven independent 1-7 scales.")
        .replace("  bodily_harm        Does the word imply damage to a body?",
                 _V2_EXTRA.strip("\n") +
                 "\n\n  bodily_harm        Does the word imply damage to a body?")
        + "\n\nThe words are ENGLISH.")
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-v4-flash"


class ChargeTask7ZH(ChargeTask7EN):
    name = "k_charge_zh_v2"
    system_prompt = ChargeTask7EN.system_prompt.replace(
        "The words are ENGLISH.",
        "The words are CHINESE. Rate them for a competent speaker of Mandarin, "
        "by the force they carry in Chinese. Do not rate an English "
        "translation, and do not try to match any English scale -- these "
        "ratings are never compared across languages.")
