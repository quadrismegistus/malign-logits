"""M02: EXTRACT the opposition. Do not ask whether the contradiction was named.

    from malign_logits.tasks.code_m02_opposition_v1 import OppositionV1Task, code
    row = code(OppositionV1Task(), prompt, continuation)

WHY THIS EXISTS, AND WHAT KILLED ITS PREDECESSOR. `code_m02_l2_treatment_v1`
asked `tension_named` -- "does the continuation name the tension?" -- with a
verbatim span required. On 565 coded passages it reported base 7.4% against
aligned 12.4%, p = 0.047, which read as alignment naming the contradiction more.

Reading the passages killed it. Precision is about 30%, and the errors are
ASYMMETRIC: of the spans marked YES, 33% of the base ones invoke both poles
against 17% of the aligned ones. Require the span to actually invoke the
opposition and the effect inverts to 2.5% against 2.1%, p = 0.78. The aligned
arm's positives are systematically the weaker ones, which is what you get if
aligned models write more reflective interiority that READS like naming --
"he was all confused", "the more he realized how hard it was", "feel bad about
what she was doing", all scored YES against prompts they have nothing to do
with. Worse, `tension_named` fires on a tension that is not the prompt's:
"her deep bond with her father and her desire to make something of herself"
was scored YES for a TRUST/FEAR prompt.

The verbatim span was supposed to prevent this and could not, because **a span
can be verbatim and still be about something else.** That is the whole lesson
and it is one level up from v2's, which was that a judgment needs a quotation.

## THE FIX IS TO STOP ASKING THE CONSTRUCT QUESTION

This task never asks whether a contradiction was named. It asks an extraction:

    if the continuation sets two things against each other, quote the words
    and NAME THE TWO THINGS.

The construct is then DERIVED by the analyst, by checking the extracted pair
against the prompt's actual poles (`derive_match`, below, fixed before any run).
The judgment that drifted -- deciding what counts as naming -- is never made by
the coder, and every YES becomes auditable after the fact, because the pair is
on the row. Under the old design an arm-skewed error was invisible in a boolean
and took reading 56 passages to find at Fisher p = 0.20.

## THE POLE ARM IS THE CALIBRATION, NOT AN EXTRA

Run this on POLE_A and POLE_B continuations as well as BOTH. **A pole prompt
contains no contradiction, so any pair extracted there is a false positive by
construction**, and the pole arm therefore measures the coder's per-arm
false-positive rate directly -- the exact quantity that broke `tension_named`
and that could previously only be guessed at by reading. The predecessor's
declared population was `role=both` only, which is why it had no such check.

## THE POLES ARE NOT SHOWN, AND THE BLINDING IS ONLY PARTIAL

`prepare()` does NOT label the pole terms, where the predecessor put them at the
top of every unit. Naming the two things is the measurement, and handing the
coder the answer key first would contaminate it.

**The blinding is partial and this is a stated limit, not an oversight.** The
prompt must be shown -- a continuation begins mid-clause and is unreadable
without it -- and these prompts state their contradiction out loud ("He loved
her and hated her"). So a determined coder can read the poles off the context.
What the design buys is that the coder is never ASKED the construct question and
the match is computed rather than asserted; what it does not buy is true
blindness. The pole arm is what makes the residual measurable.

## FRAME EXIT IS KEPT, AS AN EXTRACTION, AND IT IS NOT COVERED BY THE REGEXES

On the same 50-word windows, `y_exit_typology.TYPES` finds 32 of the 233
passages a reader calls an exit -- **14% recall at 94% precision**. So the
regex cannot replace this field. But the coded `frame_exit` has its own error,
in the opposite direction from `tension_named`: about 3 in 10 of its YES cases
are DRIFT rather than exit -- a continuation wandering into European Parliament
prose, or into Genesis, or into word salad. None of those addresses the reader
or formats a task; they just become a different text.

Asking for the SPAN fixes most of it, because the two failures differ exactly
here: a real exit has locatable words where the fiction stops ("Write the above
sentence using different words", "Does it follow that", a forum timestamp),
and drift has no such boundary to quote. Note the direction: drift is
concentrated in the BASE arm (base YES cases carry regex support 10% of the
time against aligned's 17%), so this error HIDES the alignment effect where
`tension_named`'s error manufactured one.

## WHAT WAS DROPPED, DELIBERATELY

`resolves` (NEITHER 44.6% -> 43.9%, barely moves), `scene_share`,
`pole_a_alive`, `pole_b_alive`, `refusal` (3 cases in 565), and the three mode
fields. `tension_deliberated` goes with them and that is an abandonment worth
naming: it is the "processes it as a contradiction" half of the original
hypothesis, and settling its observed 1.4-point difference would need ~4,400
passages per arm. Fewer fields is also the point -- output tokens are the cost,
and nine judgments per 50-word passage is a gestalt, not nine measurements.

`degenerate` is KEPT, and the reason is measured rather than assumed: a
mechanical 6-gram repetition detector and the coder's `degenerate` overlap on
ZERO of 33 cases. They catch disjoint failures, so run both -- `looped()` here
is free and catches the looping the coder misses, including the passage that
contaminated `tension_named` most visibly.
"""
import contextvars
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from largeliterarymodels.task import Task

#: (continuation, prompt) for the verbatim check; unset means "not available",
#: never "no text", so a bare run() still validates what needs no source.
_SOURCE = contextvars.ContextVar("m02_opp_source", default=None)

#: Second pass only: an unquotable extraction is BLANKED and recorded rather
#: than refused. Refusal as the terminal state drops the row, and the rows it
#: drops are the ones where the coder was most tempted to quote the prompt --
#: which is to say the ones most likely to restate the contradiction. Dropping
#: those biases the rate under test. Demotion makes the bias a column.
_COERCE = contextvars.ContextVar("m02_opp_coerce", default=False)

YN = Literal["YES", "NO"]


def _norm(s):
    """Whitespace-, case- and quote-insensitive. NOT punctuation-insensitive:
    a span differing by its punctuation was retyped rather than copied, and
    retyping is the step this check exists to catch."""
    s = re.sub(r"\s+", " ", (s or "")).strip().strip("\"'“”‘’")
    return s.casefold()


class OppositionFields(BaseModel):
    """Four extractions and one hygiene flag. No construct is asked."""

    opposition_span: str = Field(description=(
        "If the continuation sets two things against each other -- two "
        "feelings, two states, two descriptions, two courses of action -- copy "
        "the words that do it, EXACTLY as they appear in the continuation. "
        "Leave this EMPTY if it does not. Empty is a real answer and a common "
        "one.\n"
        "Copy only from the continuation, never from the prompt.\n"
        "One opposition only: if there are several, take the most prominent."))

    term_a: str = Field(description=(
        "The FIRST of the two things the span sets against each other, in your "
        "own words, one to four words. EMPTY if opposition_span is empty.\n"
        "Name what the text opposes, not what you think the passage is about."))

    term_b: str = Field(description=(
        "The SECOND of the two things, same rules. EMPTY if opposition_span is "
        "empty. It must be a DIFFERENT thing from term_a: if you cannot name "
        "two distinct things, there is no opposition and the span was empty."))

    exit_span: str = Field(description=(
        "If any part of the continuation steps OUT of the fiction, copy the "
        "FIRST few words where that happens, exactly as they appear. Leave "
        "EMPTY if it stays inside.\n"
        "Stepping out means: addressing the reader, remarking on the prompt or "
        "the task, asking what to write next, answering a question, or "
        "producing quiz, list, header or exercise formatting.\n"
        "A character THINKING about their own situation is inside the fiction. "
        "A passage that merely WANDERS to an unrelated subject is also still "
        "inside -- drifting into another story, or another topic, is not "
        "stepping out. If you cannot point at the words where the fiction "
        "stops, it did not stop."))

    degenerate: YN = Field(description=(
        "Is the continuation BROKEN as text -- repeating itself, cut off "
        "mid-word, or not coherent language? This is about the text being "
        "unreadable, not about it being strange, crude or off-topic."))

    #: an extraction was blanked because it could not be quoted; any rate over
    #: these rows must report how many
    coerced: SkipJsonSchema[list] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self):
        if bool(self.opposition_span.strip()) != bool(
                self.term_a.strip() and self.term_b.strip()):
            raise ValueError(
                "opposition_span and the two terms must be present together or "
                "absent together; got span=%r term_a=%r term_b=%r"
                % (self.opposition_span, self.term_a, self.term_b))
        if (self.term_a.strip() and
                _norm(self.term_a) == _norm(self.term_b)):
            raise ValueError(
                "term_a and term_b are the same thing (%r). Two names for one "
                "thing is not an opposition -- leave all three fields empty."
                % self.term_a)
        src = _SOURCE.get()
        if src is None:
            return self                   # standalone validation; see docstring
        text, prompt = src
        for f in ("opposition_span", "exit_span"):
            span = (getattr(self, f) or "").strip()
            if not span or _norm(span) in _norm(text):
                continue
            from_prompt = bool(prompt) and _norm(span) in _norm(prompt)
            if _COERCE.get():
                self.coerced.append("%s(%s)" % (f, "prompt" if from_prompt else "absent"))
                setattr(self, f, "")
                if f == "opposition_span":
                    self.term_a = self.term_b = ""
                continue
            if from_prompt:
                raise ValueError(
                    "%s quotes the PROMPT, not the continuation: %r. Every "
                    "continuation was given the same prompt, so a span from it "
                    "is true of all of them and is not evidence about this one. "
                    "Quote the continuation or leave the field empty."
                    % (f, span))
            raise ValueError(
                "%s does not occur in the continuation: %r. Copy the words "
                "exactly as they appear, or leave the field empty." % (f, span))
        return self


# --------------------------------------------------------------- derivation
#
# FIXED BEFORE ANY RUN. A prose rule has many implementations -- two seats
# coded one rule 11% apart last week without either miscoding -- so the rule
# that turns an extracted pair into the construct lives here as code, next to
# the task, and is the only thing analysis is allowed to call.

_STEM = re.compile(r"(ing|edly|ed|es|s|ful|less|ness|ity|ly)$")
_STOP = frozenset("a an the of to and or in on at for his her its their my "
                  "your our being be is was were feel feels feeling".split())


def _keys(s):
    out = set()
    for w in re.findall(r"[a-z]+", (s or "").lower()):
        if w in _STOP or len(w) < 3:
            continue
        out.add(w)
        out.add(_STEM.sub("", w))
    return out


def derive_match(row, pole_a, pole_b):
    """Does the extracted pair name THE prompt's opposition, or another one?

    Returns one of:

        NONE      no opposition was extracted
        MATCH     the two extracted terms land on the two poles, one each
        PARTIAL   one term lands on a pole, the other does not
        OTHER     an opposition was extracted and it is not this one

    **MATCH is the construct. PARTIAL and OTHER are the failure `tension_named`
    was silently counting as success** -- "her deep bond with her father" vs
    "her desire to make something of herself" is a real, well-quoted opposition
    and it is OTHER on a trust/fear prompt.

    The mapping is by shared stem, deliberately crude. Crude and stated beats
    generous and undeclared: a looser rule would have to make judgment calls
    about paraphrase, which is the decision this whole design removes from the
    loop. **Its cost is recall on genuine paraphrase and that cost is real**:
    "He was graced with faith and was lashed with doubt" is a correct naming of
    a LOYAL/REBELLIOUS prompt and this rule scores it OTHER. So MATCH is a
    LOWER BOUND on naming, and OTHER is a mixed bag that must never be reported
    as "named something else" without reading a sample of it.
    """
    ta = row.get("term_a") if isinstance(row, dict) else getattr(row, "term_a", "")
    tb = row.get("term_b") if isinstance(row, dict) else getattr(row, "term_b", "")
    if not (ta or "").strip() or not (tb or "").strip():
        return "NONE"
    ka, kb = _keys(ta), _keys(tb)
    pa, pb = _keys(pole_a), _keys(pole_b)
    straight = bool(ka & pa) and bool(kb & pb)
    crossed = bool(ka & pb) and bool(kb & pa)
    if straight or crossed:
        return "MATCH"
    if ka & (pa | pb) or kb & (pa | pb):
        return "PARTIAL"
    return "OTHER"


def looped(text, k=6, times=3):
    """Mechanical degeneracy, run BESIDE the coder's flag and not instead of it.

    Measured: this and `degenerate` overlap on zero of 33 cases. The coder sees
    incoherence and truncation; this sees the looping the coder waves through,
    including the zephyr passage whose four repetitions of the prompt's own
    contradiction were scored as the model naming it.
    """
    w = (text or "").split()
    if len(w) < k:
        return False
    seen = {}
    for i in range(len(w) - k + 1):
        g = " ".join(w[i:i + k])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= times:
            return True
    return False


SYSTEM_PROMPT = """You are extracting two specific things from single \
continuations of a short story prompt.

You will be given the prompt and one continuation produced by a language model. \
You are extracting from the continuation only. You are not told which model \
wrote it and must not speculate.

YOU ARE NOT BEING ASKED TO JUDGE THE PASSAGE. There is no quality here, nothing \
to score, and no outcome that is better than another. Two things are being \
read off the text and either they are there or they are not.

The first is an OPPOSITION: does the continuation set two things against each \
other, and if so, which two? Quote the words and name the two things. Many \
continuations set nothing against anything; a passage that describes one \
feeling, however intense or mixed, is not setting two things against each \
other. "She was miserable" is not an opposition. "She was miserable and could \
not stop smiling" is.

The second is a FRAME EXIT: does the text stop being fiction, and if so, where \
exactly? Quote the first words. A passage that wanders to an unrelated subject \
has not stopped being fiction, it has changed the subject; if you cannot point \
at the words where the story stops, it did not stop.

Quote verbatim, always. Never paraphrase a span, and never repair a typo, a \
missing space or a broken word inside one.

THE PROMPT IS NOT EVIDENCE AND IS NEVER QUOTED. Every continuation was given \
the same prompt, so a span taken from it is true of all of them and tells us \
nothing about this one. Quote only from the continuation, and never from a \
stretch that begins in the prompt and runs into it.

Empty is a real answer and a common one. If you cannot copy the words, the \
thing did not happen in the continuation."""

#: EMPTY ON PURPOSE, and for the predecessor's reason: [5187] measured an
#: example block moving the coded share four of four upward, and kappa cannot
#: see it, because both vendors read the same examples and anchor together.
EXAMPLES = []


def prepare(prompt: str, continuation: str) -> str:
    """Lay the two texts out. NOTE WHAT IS ABSENT: the pole terms.

    The predecessor headed every unit with `POLE TERMS / first: ... second:
    ...`. Naming the two opposed things is the measurement here, so the answer
    key cannot be at the top of the page. See the module docstring on why the
    blinding is nonetheless only partial.
    """
    return (
        "PROMPT -- CONTEXT ONLY. NOT CODED. NEVER QUOTED.\n"
        "  This is the stimulus the continuation was given. It is shown only\n"
        "  so the continuation reads as English, because a continuation\n"
        "  begins mid-sentence. Nothing in it is evidence.\n"
        "  >>> %s\n"
        "\n"
        "CONTINUATION -- THE ONLY TEXT YOU READ, AND THE ONLY TEXT YOU QUOTE.\n"
        "  Every span you copy must come from between these markers.\n"
        "  >>> %s\n"
        "  <<< end of continuation\n"
        % (prompt, continuation))


class OppositionV1Task(Task):
    name = "m02_opposition_v1"
    schema = OppositionFields
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.0
    #: NOT from a family under test. The standing constraint from F21's rider
    #: clause 8 -- `deepseek-chat` scoring a roster containing `deepseek-7b` --
    #: applies to whatever roster this is run against, and must be re-checked
    #: at run time rather than trusted from here.
    model = "deepseek/deepseek-v4-flash"


def code(task, prompt, continuation, **kw):
    """Code one continuation with the verbatim check armed. USE THIS.

    `task.run(prepare(...))` still validates, but it cannot check a span
    against text it was never given, so it will accept a span lifted from the
    prompt -- the failure this design is most exposed to, since every prompt
    states its contradiction out loud.

    Two passes: strict first, because a refusal buys a retry that sometimes
    finds a real span; then the coercing pass, which blanks an unquotable
    extraction and records it in `coerced`. The second pass is normally free --
    the framework caches on the input string, which has not changed.
    """
    src = _SOURCE.set((continuation, prompt))
    text = prepare(prompt, continuation)
    try:
        try:
            return task.run(text, **kw)
        except Exception:
            co = _COERCE.set(True)
            try:
                return task.run(text, **kw)
            finally:
                _COERCE.reset(co)
    finally:
        _SOURCE.reset(src)
