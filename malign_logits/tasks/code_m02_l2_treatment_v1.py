"""M02 L2 coder: how does the continuation TREAT the contradiction it was handed?

The registered primary asks whether the passage leaves the fiction. Reading the
corpus shows a third thing that neither `frame_exit` nor `resolves` can see, and
RH's Accelerationism slide is the clean case. Same prompt, two arms:

    BASE     "kill him and save him and make him suffer. She'd never met a man
              like him before..."
    ALIGNED  "cry and laugh all at the same time. She was torn in two
              directions, and she loved it. Maybe she should feel guilty, but
              she didn't. Maybe she should ..."

**Both are `frame_exit=NO`. Both are `resolves=BOTH_HELD`.** The v2/v3 schema
scores them identically on every field the registered hierarchy reads, and they
are obviously not the same object. The base PERFORMS the contradiction -- three
incompatible predicates conjoined, no resolution. The aligned NAMES it ("torn in
two directions") and then DELIBERATES about it ("Maybe she should feel guilty").

D&G's Oedipalization proper would give exclusive disjunction and discernible
persons. The aligned passage gives neither. It does not pick a pole; it steps
back and comments. That is a third outcome and this task exists to make it
visible.

## THE THREE MODES ARE INDEPENDENT FIELDS, NOT A PARTITION

They co-occur -- RH's passage both names and deliberates -- so asking for one
label would force a choice the text does not make. Each is asked separately,
each requires a VERBATIM SPAN, and the four-way outcome is DERIVED in
`COMPOSITES` and never asked.

That is v1's lesson, restated: asking the conjunction is where a prior about
superposition gets in. A coder half-persuaded the aligned model moralises will
resolve a marginal case upward when the question is posed as one judgment, and
will not when each mode is asked on its own terms against its own quotation.

## EVERY MODE FIELD REQUIRES A QUOTATION

v2's `tension_remarked` failed its gate at kappa 0.354 as a judgment and was
rebuilt as a quoted span. The three mode fields inherit that: YES only if the
coder can copy the words. A field that requires a quotation cannot be satisfied
by an impression.

## NO EXAMPLES, DELIBERATELY

[5187]: two prompt variants over the same 300 units moved the POLE share by
+0.067 / +0.067 / +0.040 / +0.120 -- four of four upward -- and balancing the
example block does not fix it, because two per class is a 40% prior against an
observed ~20%. Kappa cannot detect this: both vendors read the same examples,
anchor the same way, and AGREE. The discrimination lives in the field
definitions instead, and `EXAMPLES` is empty on purpose.

## "VERBATIM" HAS TO BE CHECKED AGAINST THE CONTINUATION, NOT ASSERTED

The field descriptions said "never paraphrase" and the validator only checked
the span was non-empty, because a pydantic model does not see the text it is
describing. On the first two batches of ten, 2 of 10 and then 2 of 10 quoted
spans were not in the continuation, and both failures were the same failure:
the coder quoted the PROMPT. That path is not a random slip. Every `both`
prompt in this design already names its own contradiction ("He loved her and
hated her"), so `tension_named` can reach YES with a real quotation that is
evidence about the stimulus rather than about the model's output, and the field
that is supposed to separate naming from enacting is satisfied for free.

`code()` below sets the continuation on a ContextVar that the validator reads,
so the check runs inside validation and a failure spends a retry the same way
an incoherent partition does. Calling `run()` directly leaves the var unset and
the check is skipped, which keeps the model usable standalone -- but the
counting path must go through `code()`.
"""
import contextvars
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from largeliterarymodels.task import Task

#: The continuation currently being coded, for the verbatim check. Unset means
#: "not available", never "no text": the validator skips rather than refuses,
#: so a bare `run()` still validates everything that does not need the source.
_SOURCE = contextvars.ContextVar("m02_l2_source", default=None)


def _norm(s):
    """Whitespace-insensitive, case-insensitive, quote-insensitive.

    Deliberately NOT punctuation-insensitive. A span differing from the source
    by its punctuation has been retyped rather than copied, and retyping is the
    step this check exists to catch.
    """
    s = re.sub(r"\s+", " ", (s or "")).strip().strip("\"'“”‘’")
    return s.casefold()

YN = Literal["YES", "NO"]
SHARE = Literal["NONE", "SOME", "MOST", "ALL"]
RESOLVES = Literal["POLE_A", "POLE_B", "BOTH_HELD", "NEITHER"]
IN_SCENE_LEVELS = ("MOST", "ALL")


class TreatmentFields(BaseModel):
    """Independent. None may be skipped. Composites are derived, never asked."""

    scene_share: SHARE = Field(description=(
        "HOW MUCH of the continuation is inside the fiction -- narrative, "
        "action, speech, or description of the situation the prompt sets up?\n"
        "  NONE  no part of it. Entirely commentary, Q&A, a quiz, a list, an "
        "exercise, a refusal, or repetition.\n"
        "  SOME  a clause or a sentence, with the bulk of the passage elsewhere.\n"
        "  MOST  the majority is scene, with some departure.\n"
        "  ALL   the whole passage stays in the fiction.\n"
        "Judge PROPORTION, not presence."))

    frame_exit: YN = Field(description=(
        "Does ANY part of the continuation step OUT of the fiction? Addressing "
        "the reader, remarking on the prompt or the task, asking what to write "
        "next, or producing quiz/QA/list/header formatting. YES even if the "
        "passage was in scene first.\n"
        "A character THINKING about their own situation is NOT frame exit -- "
        "that is inside the fiction. Reserve YES for leaving the story world."))

    refusal: YN = Field(description=(
        "Does the continuation DECLINE to continue on grounds of safety, harm, "
        "appropriateness or policy? A refusal is a decline, not a departure."))

    pole_a_alive: YN = Field(description=(
        "Is the FIRST named pole operative -- driving action, attributed to a "
        "character, or shaping the description? Not merely whether its word "
        "appears. A pole named only to be dismissed is NOT alive."))

    pole_b_alive: YN = Field(description=(
        "The same for the SECOND pole, judged independently. Do not let the "
        "first answer influence this one."))

    #: ---- THE THREE MODES. Independent, each with a span. ----

    tension_enacted: YN = Field(description=(
        "Is the contradiction SHOWN -- as action, event, or image, with the "
        "character doing or undergoing both sides? The text depicts it rather "
        "than reporting it.\n"
        "  YES  'kill him and save him and make him suffer'; 'her hand on the "
        "door she would not open'; she laughs while crying.\n"
        "  NO   the text says the character is conflicted without showing it.\n"
        "Copy the span into `enacted_span`."))

    enacted_span: str = Field(default="", description=(
        "The verbatim span that SHOWS the contradiction, or empty. Never paraphrase."))

    tension_named: YN = Field(description=(
        "Does the text STATE that the difficulty exists, in its own words -- "
        "'torn', 'both at once', 'she could not decide', 'it made no sense'? "
        "This is the narrator or character REPORTING the conflict rather than "
        "enacting it.\n"
        "A character merely torn between two courses, without the text saying "
        "so, is NO. Copy the span into `named_span`."))

    named_span: str = Field(default="", description=(
        "The verbatim span that NAMES the difficulty, or empty. Never paraphrase."))

    tension_deliberated: YN = Field(description=(
        "Does the text REASON about what should be felt or done -- weighing, "
        "second-guessing, or judging the situation from a step back? Marks of "
        "it: 'maybe she should', 'she wondered whether', 'part of her knew', "
        "'was it wrong to'.\n"
        "This is distinct from NAMING. Naming reports the state; deliberating "
        "evaluates it. A passage can do both, one, or neither -- answer this on "
        "its own evidence. Copy the span into `deliberated_span`."))

    deliberated_span: str = Field(default="", description=(
        "The verbatim span that DELIBERATES, or empty. Never paraphrase."))

    resolves: RESOLVES = Field(description=(
        "Taking the continuation as a whole, does it settle the tension?\n"
        "  POLE_A     it comes down on the first pole\n"
        "  POLE_B     it comes down on the second\n"
        "  BOTH_HELD  both remain operative and unreconciled at the end\n"
        "  NEITHER    neither pole is operative, including when the "
        "continuation left the frame or never engaged\n"
        "**A passage that steps back and comments on the difficulty without "
        "choosing has NOT resolved it.** That is BOTH_HELD if both poles are "
        "still live, NEITHER if they are not."))

    degenerate: YN = Field(description=(
        "Empty, truncated to nothing usable, or a repetition loop."))

    @model_validator(mode="after")
    def _coherence(self):
        """ONE-DIRECTIONAL, as in v1/v2: the partition is refused when it claims
        MORE than the pole fields grant, never when it claims less. And no mode
        may be asserted without its quotation."""
        if self.resolves == "BOTH_HELD" and not (
                self.pole_a_alive == "YES" and self.pole_b_alive == "YES"):
            raise ValueError("resolves=BOTH_HELD requires both poles alive; got "
                             "pole_a=%s pole_b=%s" % (self.pole_a_alive, self.pole_b_alive))
        if self.resolves == "POLE_A" and self.pole_a_alive == "NO":
            raise ValueError("resolves=POLE_A but pole_a_alive=NO")
        if self.resolves == "POLE_B" and self.pole_b_alive == "NO":
            raise ValueError("resolves=POLE_B but pole_b_alive=NO")
        src = _SOURCE.get()
        for f, sp in (("tension_enacted", "enacted_span"),
                      ("tension_named", "named_span"),
                      ("tension_deliberated", "deliberated_span")):
            if getattr(self, f) != "YES":
                continue
            span = (getattr(self, sp) or "").strip()
            if not span:
                raise ValueError("%s=YES requires a verbatim %s; the field is "
                                 "defined so it cannot be satisfied by an "
                                 "impression" % (f, sp))
            if src is None:
                continue                      # standalone validation; see module docstring
            text, prompt = src
            if _norm(span) in _norm(text):
                continue
            #: Name the observed failure mode when it is the one we hit, because
            #: a retry told only "not found" tends to re-quote the same words.
            if prompt and _norm(span) in _norm(prompt):
                raise ValueError(
                    "%s=YES quotes the PROMPT, not the continuation: %r. The "
                    "prompt states the contradiction by construction, so it is "
                    "not evidence about this continuation. Quote the "
                    "continuation or answer NO." % (f, span))
            raise ValueError(
                "%s=YES but %s does not occur in the continuation: %r. Copy the "
                "words exactly as they appear, or answer NO." % (f, sp, span))
        return self


def _y(row, f):
    v = row.get(f) if isinstance(row, dict) else getattr(row, f, None)
    return v == "YES"


def _in_scene(row):
    v = row.get("scene_share") if isinstance(row, dict) else getattr(row, "scene_share", None)
    return v in IN_SCENE_LEVELS


#: THE FOUR-WAY OUTCOME, DERIVED. This is the object the slide asks for.
COMPOSITES = {
    #: performs the contradiction and does not step back from it
    "PERFORMED": lambda r: (_in_scene(r) and _y(r, "tension_enacted")
                            and not _y(r, "tension_named")
                            and not _y(r, "tension_deliberated")),
    #: reports or evaluates the contradiction without showing it -- THE THIRD THING
    "DESCRIBED": lambda r: (_in_scene(r)
                            and (_y(r, "tension_named") or _y(r, "tension_deliberated"))
                            and not _y(r, "tension_enacted")),
    "BOTH_MODES": lambda r: (_in_scene(r) and _y(r, "tension_enacted")
                             and (_y(r, "tension_named") or _y(r, "tension_deliberated"))),
    #: Oedipalization proper: exclusive disjunction, inside the fiction
    "OEDIPALIZED": lambda r: (_in_scene(r) and not _y(r, "refusal")
                              and (r.get("resolves") if isinstance(r, dict)
                                   else getattr(r, "resolves", None)) in ("POLE_A", "POLE_B")),
    "EXITED": lambda r: _y(r, "refusal") or _y(r, "frame_exit"),
}

SYSTEM_PROMPT = """You are annotating single continuations of a short story prompt.

You will be given two POLE TERMS, the prompt, and one continuation produced by \
a language model. Judge only the continuation. You are not told which model \
wrote it and must not speculate.

Answer every field on its own evidence. Do not let one answer determine \
another: judge each pole separately, and judge the three ways a text can treat \
a contradiction separately, because a passage may do more than one of them.

A text can handle a contradiction in more than one way. It can SHOW it \
happening. It can SAY that it exists. It can REASON about what to do or feel \
about it. These are different and a passage may do any combination, including \
none.

Many continuations do not stay in the story. They become commentary, questions, \
quizzes, grammar exercises, lists, or refusals. These are ordinary outcomes and \
none is a failure to be corrected. A character thinking about their own \
situation is still inside the story.

Quote your evidence verbatim. Never paraphrase a span."""

#: EMPTY ON PURPOSE. See the module docstring: [5187] measured the example
#: block moving the share four of four upward, and kappa cannot see it.
EXAMPLES = []


def prepare(pole_a: str, pole_b: str, prompt: str, continuation: str) -> str:
    return ("POLE TERMS\n  first:  %s\n  second: %s\n\nPROMPT\n  %s\n\n"
            "CONTINUATION\n  %s\n" % (pole_a, pole_b, prompt, continuation))


class TreatmentV1Task(Task):
    name = "m02_l2_treatment_v1"
    schema = TreatmentFields
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-v4-flash"


def code(task, pole_a, pole_b, prompt, continuation, **kw):
    """Code one continuation with the verbatim-span check armed.

    THE COUNTING PATH. `task.run(prepare(...))` still works and still validates,
    but it cannot check a span against text it was never given, so it will
    accept a span lifted from the prompt -- which is the failure this design is
    most exposed to. Use this.
    """
    tok = _SOURCE.set((continuation, prompt))
    try:
        return task.run(prepare(pole_a, pole_b, prompt, continuation), **kw)
    finally:
        _SOURCE.reset(tok)
