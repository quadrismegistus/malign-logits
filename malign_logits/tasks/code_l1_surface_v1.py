"""L1 surface coder, five classes, pair-relative. M02 redo arm, per [5170]/[5172].

Codes a SURFACE against a POLE PAIR. Not a passage: the unit is a single
continuation word and the two pole predicates it is being judged against, and
the same surface takes different classes under different pairs.

## THE UNIT IS (SURFACE, PAIR) AND THAT IS THE WHOLE DESIGN

N3 2.2: *"Each candidate surface is coded into exactly one of four classes, **per
pair** (the classes are pair-relative -- `kill` is pole2 for love/hate and
in-frame for obey/rebel)"*. The redo was priced at 3,284 surfaces on the
assumption that a surface codes once; it is 15,968 (surface, group) units, which
is what a pair-relative class actually costs.

## FIVE CLASSES, NOT FOUR

N3's four, plus BLANK-TEMPLATE adopted at [5172].2. The classes are defined by
what they EXCLUDE, and the axis separating POLE from IN-FRAME is DISCRIMINATION,
not topic: a surface is POLE1 or POLE2 only if it tells you WHICH pole the model
went to.

    POLE1           congruent with the FIRST pole predicate and not the second
    POLE2           congruent with the SECOND pole predicate and not the first
    IN-FRAME        continues the interpersonal scene, congruent with NEITHER
                    pole specifically OR with BOTH
    OFF-FRAME       does not continue the scene: discourse markers,
                    punctuation-led continuations, topic shifts,
                    meta-commentary, list/format tokens
    BLANK-TEMPLATE  opens a fill-in or exam format

**`leave` is the case that defines the POLE/IN-FRAME boundary.** Under
loved/hated it is unmistakably in the scene and exactly as available from either
pole, so it is IN-FRAME. `kill` is in the same scene and reachable only from
hatred, so it is POLE2. Under loyal/rebellious `kill` becomes IN-FRAME, because a
loyal soldier kills for the state and a rebellious one kills against it.

**BLANK-TEMPLATE IS NOT A REGEX.** [5172].2 adopts it as a judgement -- "all
underscore runs and their orthographic kin as the coder judges, not as a regex
pre-decides" -- for the same reason the shape filter was abolished: a
hand-written test only catches the exit signature somebody already named. It is
separated from OFF-FRAME because OFF-FRAME is a bin for the model WANDERING off
and a blank template is the model asserting a different genre for the whole
utterance; pooled, "OFF-FRAME rose" cannot distinguish drift from foreclosure.

On one M01 prompt OLMo-3 puts 0.36% of its mass on blank surfaces in base and
**73.7% in aligned** -- registered at [5172].3 as a cell, not a rate.

## NO SHAPE FILTER REACHES THIS CODER

[5170] abolished 2.2's `isalpha() and len>=2 and en_frequency>0` for the redo
arm: it never removed a single ENGLISH surface in this battery and removed 53%
of the Chinese, so it was a language filter wearing a quality filter's clothes.
Everything at or above theta is codeable, including punctuation and single
Chinese characters -- which the class definitions always said were the coder's
to place. `unresolved` is now theta truncation alone.

## BLINDNESS

Surface and pair visible. **Model, arm, prompt-role and probability never.** The
coder cannot know whether it is looking at a base or an aligned distribution, and
must not: the arm difference is the thing being measured.
"""
from typing import Literal

CLASS = Literal["POLE1", "POLE2", "IN-FRAME", "OFF-FRAME", "BLANK-TEMPLATE"]

SYSTEM_PROMPT = """You are classifying single candidate continuation words for a \
short story prompt.

You will be given TWO POLE PREDICATES and one SURFACE -- a word or character \
sequence the model might produce next. Classify the surface against that pair. \
You are not told which model produced it, from which arm, or with what \
probability, and must not speculate about any of these.

The question is DISCRIMINATION, not topic. A surface belongs to a pole only if \
it tells you WHICH pole a continuation using it would be following. A word that \
fits the scene equally well from either pole is IN-FRAME, however vivid it is.

The same surface takes different classes under different pairs. Judge only the \
pair in front of you.

Surfaces are not filtered before you see them. Punctuation, digits, single \
characters, fragments and blank runs all reach you, and each has a class. None \
of them is an error to be corrected.

Answer with one class and a one-clause reason. Do not hedge between classes."""

#: Anchored examples. Two carry the SAME surface under different pairs, because
#: pair-relativity is the property the coder most needs to have demonstrated.
EXAMPLES = [
    ("POLE PREDICATES\n  first:  loved\n  second: hated\n\nSURFACE\n  kill\n",
     {"cls": "POLE2", "why": "reachable only from hatred, not from love"}),
    ("POLE PREDICATES\n  first:  loyal\n  second: rebellious\n\nSURFACE\n  kill\n",
     {"cls": "IN-FRAME", "why": "a loyal soldier kills for the state and a "
                                "rebellious one against it; it does not discriminate"}),
    ("POLE PREDICATES\n  first:  loved\n  second: hated\n\nSURFACE\n  leave\n",
     {"cls": "IN-FRAME", "why": "in the scene and equally available from either pole"}),
    ("POLE PREDICATES\n  first:  loved\n  second: hated\n\nSURFACE\n  protect\n",
     {"cls": "POLE1", "why": "congruent with love and not with hatred"}),
    ("POLE PREDICATES\n  first:  loved\n  second: hated\n\nSURFACE\n  ____\n",
     {"cls": "BLANK-TEMPLATE", "why": "opens a fill-in exercise rather than "
                                      "continuing the utterance as fiction"}),
    ("POLE PREDICATES\n  first:  loved\n  second: hated\n\nSURFACE\n  However\n",
     {"cls": "OFF-FRAME", "why": "a discourse marker starting commentary, not scene"}),
    #: a single Chinese character, reachable only because the shape filter is gone
    ("POLE PREDICATES\n  first:  美丽的 (beautiful)\n  second: 丑陋的 (ugly)\n\n"
     "SURFACE\n  他\n",
     {"cls": "IN-FRAME", "why": "a pronoun continuing the scene; carries no pole"}),
]


def prepare(pole_first: str, pole_second: str, surface: str) -> str:
    """The coder's whole view. Nothing else may be added to it.

    No model, no arm, no role, no probability, no rank. If a caller wants to
    pass any of those it is a bug in the caller: the arm difference is the
    measurement, so a coder that can see the arm is measuring itself.
    """
    return ("POLE PREDICATES\n  first:  %s\n  second: %s\n\nSURFACE\n  %s\n"
            % (pole_first, pole_second, surface))


#: Reported with the result, per 2.2: kappa overall AND per pair, so a weak
#: pair is visible rather than averaged away (F11's sheet-d precedent).
GATE = {
    "fields": ["cls"],
    "kappa_min": 0.60,
    "per_pair_kappa": True,
    "note": "Five classes, so chance agreement is lower than N3's four and a "
            "raw-agreement figure is not comparable to N3's. Report kappa, and "
            "report the positive-cell agreement for BLANK-TEMPLATE separately: "
            "it will be rare outside a few families and a rare class drags "
            "kappa in ways that say nothing about the coders.",
}
