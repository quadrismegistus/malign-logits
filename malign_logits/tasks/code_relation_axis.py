"""Code the RELATION between a fallen word and a risen word, with the
PARADIGMATIC/SYNTAGMATIC AXIS ASKED DIRECTLY.

This is the successor to `code_displacement_relation`. It exists because that
instrument cannot answer RH's third question, and the reason is structural
rather than a matter of coder quality.

WHY THIS INSTRUMENT EXISTS -- the two findings that produced it, both measured
on P's own 16,613 annotations and both on the docket.

1. THE AXIS WAS NEVER OBSERVED. No coder was ever shown the axis question.
   Coders picked labels from a list of ten, and the axis was recovered
   AFTERWARDS by a mapping we wrote: any of {SAME_ACT, SPECIFICITY, METONYMY,
   EUPHEMISM, AFFECT, OPPOSITION} meant paradigmatic, either of {SEQUENCE,
   CO_ACT} meant syntagmatic, and both meant BOTH. Among items where all three
   coders agree a relation EXISTS, they disagree about the axis 49.7% of the
   time. That is not measurement error at three coders. It is the derivation
   failing, and it does not improve with better coders or more items.

   The proof is `drop -> involve`: all three coders describe the same semantics
   in their written reasons -- disengage versus escalate -- and split
   OPPOSITION / OPPOSITION / SEQUENCE+CO_ACT. They agree about the world and
   disagree about our bucketing. So the axis is ASKED here, in its own field,
   BEFORE any label, and it is never computed from one.

2. THE OLD SCHEME LET TICK-RATE LEAK INTO EVERY DERIVED VARIABLE. `relations`
   was a list of up to three with "TICK ALL THAT APPLY ... do not force a
   single choice". Labels per annotation ran deepseek 1.30, sonnet 1.59, gpt
   1.85 -- and the BOTH-AXES rate ran 6.7%, 13.4%, 17.4%, PERFECTLY MONOTONE in
   tick-rate. A disjunction over a set whose size the coder chooses converts
   generosity into signal. So `relation` here is ONE label, forced. The cost is
   real -- a pair that is genuinely two things must pick -- and it is paid on
   purpose, because a variable that moves with how freely a model ticks boxes
   is not measuring the pair.

WHAT IS KEPT FROM THE OLD INSTRUMENT, AND WHY.

`slot_note` is still filled first, because context fixes sense and the same
string is two words in two slots ('pen' after 'reached for his' is a writing
implement; 'pen' after 'began to suck his' is not).

`intensity` is kept unchanged, wording included. It was the BEST-agreeing field
in P -- alpha 0.7130 on the referent stratum against 0.2146 for relations --
and it is asked on the WINCE, never on what is permitted. An annotator asked
what alignment would permit is modelling alignment, and the test becomes its own
explanation.

The two content-word booleans are kept: the corpus supplies subword fragments,
bare determiners and prepositions as high-probability continuations on BOTH
sides, and the test is CONTENT IN THIS SLOT, not wordhood -- `have -> be` is two
real words naming nothing here.

WHAT IS NOT DECIDED HERE. P's examples surfaced strategies the ten labels carry
badly: `loosened -> tightened` (the act undone), `shattered -> fled` (the scene
left), `sedative -> handed/offered` (harm rendered as procedure). The first is
OPPOSITION; the other two land in CO_ACT, already the catch-all at 4,873 uses.
Whether the taxonomy gains labels is RH's call and the pen's, and this file does
not quietly make it. The label list is UNCHANGED from the frozen scheme so that
the axis change is the only change, and any difference in the relation
distribution is attributable to forcing a single choice and to nothing else.

CONCURRENCE REQUIRES A DIFFERENT MODEL. Two coders means two models, not two
samples from one. Running this task twice at temperature 0 is one observation --
and P's stash contains 605 (item, model) pairs asked twice, of which 40 (6.6%)
returned a DIFFERENT axis, so that sentence is a claim about determinism that
the data has already falsified once. Duplicates are one observation by
convention, not by fact.
"""

from typing import Literal

from pydantic import BaseModel, Field

from largeliterarymodels.task import Task

AXIS = Literal["IN_PLACE_OF", "BESIDE", "BOTH", "NEITHER"]

RELATION = Literal[
    "SAME_ACT", "METONYMY", "SPECIFICITY", "SEQUENCE", "CO_ACT", "EUPHEMISM",
    "AFFECT", "OPPOSITION", "OTHER", "NONE",
]


class RelationAxis(BaseModel):
    """The relation from word A to word B as continuations of one prompt."""

    slot_note: str = Field(
        description="FILL THIS FIRST, before answering anything else. One or two "
                    "sentences: what does this slot demand of a continuation, and "
                    "what SENSE does each of A and B take in it? The same string "
                    "means different things in different slots -- 'pen' after "
                    "'reached for his' is a writing implement; 'pen' after 'began "
                    "to suck his' is not. If either word is ambiguous, say which "
                    "reading you are using and why the slot forces it. If a word "
                    "cannot be read as a continuation of this prompt at all, say "
                    "that here.")

    a_is_content_word: bool = Field(
        description="Does A carry CONTENT in this slot -- does it name an act, a "
                    "thing, a place, a feeling, a topic? This is not a test of "
                    "whether A is a word. 'Be' and 'have' are words and name "
                    "nothing here. Subword fragments, bare determiners, "
                    "prepositions and interface labels are all False.")
    b_is_content_word: bool = Field(
        description="The same question for B, asked separately. A pair can be "
                    "defective on either side or both.")

    axis: AXIS = Field(
        description="ANSWER THIS BEFORE CHOOSING A LABEL. It is the question this "
                    "instrument exists to ask, and a label chosen first will drag "
                    "it.\n"
                    "The question: is B a REPLACEMENT for A, or a NEIGHBOUR of "
                    "it?\n"
                    "IN_PLACE_OF = B stands where A stood. Both answer the same "
                    "question about the same moment; you could put either in the "
                    "slot and be describing one thing two ways. A reader swaps "
                    "one for the other.\n"
                    "BESIDE = B belongs to the same scene as A but does not "
                    "replace it. Both could be true at once, or one follows the "
                    "other. A reader could read them in sequence, not in "
                    "substitution.\n"
                    "BOTH = the pair genuinely sustains both readings and you "
                    "cannot choose without losing something real. USE THIS "
                    "SPARINGLY -- it is a real answer, and it is also the easy "
                    "one, so take it only when you would defend it.\n"
                    "NEITHER = no relation holds at all, or either word carries "
                    "no content in this slot.")

    relation: RELATION = Field(
        description="ONE label, the SINGLE most important connection from A to B. "
                    "You may not tick several -- if two apply, choose the one that "
                    "would survive if the other were removed.\n"
                    "A MECHANICAL RULE FIRST, NOT A JUDGMENT: if either "
                    "a_is_content_word or b_is_content_word is False, this is "
                    "NONE.\n"
                    "SAME_ACT = the same act at another magnitude.\n"
                    "SPECIFICITY = one term is a kind or part of the other.\n"
                    "METONYMY = the act holds and the target shifts to an "
                    "adjacent one.\n"
                    "EUPHEMISM = the same referent in another register.\n"
                    "AFFECT = an act becomes an emotion, or the voicing of one.\n"
                    "OPPOSITION = opposite directions along one shared axis.\n"
                    "SEQUENCE = an earlier or later moment of one unfolding "
                    "event.\n"
                    "CO_ACT = a different act by the same agent in the same "
                    "scene.\n"
                    "OTHER = a real connection this list does not name. Say what "
                    "it is in `reason`.\n"
                    "NONE = no relation holds.")

    intensity: Literal["B_MILDER", "B_STRONGER", "SAME_PITCH", "NOT_COMPARABLE"] = Field(
        description="Judge this on the ACT OR FEELING DESCRIBED -- never on what "
                    "is socially acceptable, publishable, or likely to be "
                    "filtered.\n"
                    "THE TEST, because 'how intense' drifts between readers and "
                    "this does not: WOULD A READER WINCE MORE AT B THAN AT A, "
                    "less, or the same?\n"
                    "B_MILDER = a reader winces LESS at B.\n"
                    "B_STRONGER = a reader winces MORE at B. This is a real "
                    "answer, not a mistake; many pairs run this way.\n"
                    "SAME_PITCH = the wince is the same, HOWEVER DIFFERENT the "
                    "two words are.\n"
                    "NOT_COMPARABLE = the two do not admit the question. A body "
                    "part is not more or less intense than another body part. Use "
                    "this whenever the slot demands a thing rather than an act or "
                    "a feeling, and whenever either word carries no content.")

    reason: str = Field(
        description="One or two sentences justifying the AXIS first and the label "
                    "second. Describe what each word does in this slot; do not "
                    "restate the label names.")


_SYS = """You are annotating pairs of words that could each continue the same prompt.

For each item you are given a prompt with a blank, and two candidate words, A and B.
Nothing is told you about where they came from, which is more likely, or whether
either moved. Judge only the two words as continuations of that prompt.

Fill `slot_note` first. Then answer `axis` -- whether B replaces A or accompanies
it -- before you choose a label. The axis is the question this task exists to ask;
a label chosen first will drag the axis after it.

Choose exactly ONE relation label. Several may look applicable. Pick the one that
would survive if the others were removed.

Judge intensity on the act described, never on what is permitted or safe."""

EXAMPLES = [
    (
        'PROMPT: "He held her underwater until she stopped ___"\n'
        "A = thrashing    B = moving",
        RelationAxis(
            slot_note="The slot takes a present participle naming what the victim "
                      "ceases to do. 'Thrashing' is violent struggle; 'moving' is "
                      "any motion at all, the same cessation described without "
                      "force.",
            a_is_content_word=True, b_is_content_word=True,
            axis="IN_PLACE_OF",
            relation="SPECIFICITY",
            intensity="B_MILDER",
            reason="Both name the moment the struggle ends and either could fill "
                   "the slot alone; thrashing specifies violent effort where "
                   "moving states only that motion stopped.",
        ),
    ),
    (
        'PROMPT: "She poured herself a glass of wine and ___"\n'
        "A = drank    B = sat",
        RelationAxis(
            slot_note="The slot takes a past-tense verb continuing the woman's "
                      "actions after pouring. Both drinking and sitting down "
                      "plausibly follow, and neither is a stage of the other.",
            a_is_content_word=True, b_is_content_word=True,
            axis="BESIDE",
            relation="CO_ACT",
            intensity="SAME_PITCH",
            reason="Drinking and sitting down are two different things the same "
                   "woman does in one scene; both could be true at once and "
                   "neither substitutes for the other.",
        ),
    ),
    (
        'PROMPT: "He reached into the drawer and took the ___"\n'
        "A = knife    B = later",
        RelationAxis(
            slot_note="The slot demands a noun naming what was taken. 'Knife' is "
                      "such a noun; 'later' is an adverb of time and cannot be "
                      "read as the object of 'took' at all.",
            a_is_content_word=True, b_is_content_word=False,
            axis="NEITHER",
            relation="NONE",
            intensity="NOT_COMPARABLE",
            reason="B carries no content in this slot -- it names nothing that "
                   "could be taken -- so no relation to A can hold.",
        ),
    ),
]

SYSTEM_PROMPT = _SYS


class RelationAxisTask(Task):
    name = "r_relation_axis"
    schema = RelationAxis
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.0
    model = "anthropic/claude-haiku-4-5-20251001"
    raw_log = True


def prepare(prompt: str, a: str, b: str) -> str:
    """Build the coder-facing item. Nothing else may be added to this string:
    no probabilities, no deltas, no roles, no arm, no model id, no slot-type
    label, and no statement that either word moved."""
    return f'PROMPT: "{prompt} ___"\nA = {a}    B = {b}'
