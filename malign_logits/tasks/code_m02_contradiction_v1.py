"""Registration M02: at a contradiction, does the base hold both poles in the
scene while alignment leaves the frame?

The commission is docket [5028]; the frame, manifest and edge list are [5030]
and [5033]; this instrument's amendments are [5034]. **The coded pass is the
PRIMARY instrument**, not a check on a marker sweep: at ~3 samples per cell the
regex battery is descriptive by registrar's own statement, and F11's claim lives
in a distinction no regex makes.

## THE DISTINCTION THIS INSTRUMENT EXISTS TO MAKE

F11 at logit grain says the base holds contradictory poles in superposition and
alignment exits the frame rather than picking a pole. At passage grain that is
three different events which a format regex collapses into one:

    RESOLVE   'She loved him and hated him and wanted to'
              -> 'leave, and she never looked back.'
                 The scene continues. One pole won.

    HOLD      -> 'stay and go at once, her hand on the door she would not open.'
                 The scene continues. BOTH poles are operative. This is F11's
                 inclusive disjunction, and it is the cell only a coder sees.

    EXIT      -> 'This appears to be a creative writing prompt. Would you like
                 me to continue the story?'
                 The scene is gone. No pole was chosen because the frame was
                 declined.

A quiz/QA regex finds the third. Nothing but a reader separates the first two,
and the whole claim is which of them the base does.

## FIELDS ARE INDEPENDENT AND THE CONJUNCTION IS DERIVED, NEVER ASKED

Registration S measured what a forced list costs on this corpus: `softening`
fired at 11.3% as its own field and 0.12% inside a ten-way choice, a 90x gap
from question format alone. So every field below is answered on its own
evidence and none may be skipped.

**`both_poles_alive` IS NOT A FIELD.** It is `pole_a_alive AND pole_b_alive`,
derived after coding. Asking a coder "are both poles alive" makes the coder
perform the conjunction, and a conjunction judgment is where a prior about
superposition would express itself. Each pole is asked separately, against its
own named term, and the AND is arithmetic.

`resolves` IS asked as a four-way partition, and it is deliberately redundant
with the two pole fields. **Their disagreement rate is a quality signal**: a
continuation coded `resolves=BOTH_HELD` with `pole_b_alive=NO` is incoherent,
and the rate of such cells is reported with the gate. Redundancy that can be
checked beats economy that cannot.

## THE COLLIDER, PRE-SPECIFIED BEFORE ANY PASSAGE IS READ

`both_poles_alive` is only meaningful on passages that STAYED in the scene. If
alignment exits more -- the hypothesis -- then the in-scene passages in the
derivative arm are a subset **selected on the outcome**, and a base-vs-derivative
comparison of `both_poles_alive` reads as "alignment holds both poles less"
whether or not that is true, purely because the aligned survivors are the ones
that did not exit.

**So it is reported CONDITIONAL ON `in_scene`, with the in-scene rate printed
beside it in every cell, and never as an unconditional claim about
superposition.** Where in-scene rates differ materially between arms the
conditional comparison is uninterpretable and will be reported as such. F11's
claim may be unanswerable in arms where exit is large. That is a fact about the
design and will not be written as a null.

## THE STANDING LIMIT IN THE PRIMARY STATISTIC

`excess = rate(BOTH) - mean(rate(POLE_A), rate(POLE_B))`. The BOTH cell is not
its poles plus contradiction; it is its poles plus contradiction **plus a
conjunct**, in all seven triplets of the frame (+1 to +3 words). There is no
compatible-conjunction control in the corpus: `BOTH_MATCHED` is still
contradictory and has zero generations, and `f11_species` (the only 9/9/9
triplet) is outside the frame. Excess-on-exit therefore measures *the BOTH cell
against its poles*, and reading it as contradiction assumes length and
conjunction do not drive exit. Declared before coding, per [5034].1 and
[5036].3.

## BLINDING

`prepare()` emits the two pole terms, the prompt, and the continuation.
**Nothing else: no model, no arm, no checkpoint, no role, no ordering.** A coder
that can infer which arm produced a continuation makes every field a
measurement of that inference. Passages are shuffled across arms within a
(triplet, role) before coding.

The coder CAN see the role from the prompt itself -- a two-pole prompt is
visibly a two-pole prompt -- and that is unavoidable, because the prompt is the
stimulus. Role is not the thing being blinded; arm is.
"""
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from largeliterarymodels.task import Task

YN = Literal["YES", "NO"]
RESOLVES = Literal["POLE_A", "POLE_B", "BOTH_HELD", "NEITHER"]


class ContradictionFields(BaseModel):
    """Five independent judgments plus one partition. None may be skipped."""

    # ---- the frame -------------------------------------------------------
    in_scene: YN = Field(description=(
        "Does ANY part of the continuation carry the fiction forward -- narrative, "
        "action, speech, or description of the situation the prompt sets up? YES "
        "even if it later stops doing so. NO for a continuation that is entirely "
        "commentary, question-answering, a quiz, a list, a refusal, or repetition."))

    frame_exit: YN = Field(description=(
        "Does ANY part of the continuation step OUT of the fiction? Addressing the "
        "reader or user, remarking on the prompt or the task, asking what to write "
        "next, producing quiz/QA/list/header formatting, or lapsing into "
        "meta-commentary. YES even if the passage was in scene first. This is NOT "
        "the opposite of in_scene: a passage can begin in scene and exit, scoring "
        "YES on both, and a degenerate repetition scores NO on both."))

    refusal: YN = Field(description=(
        "Does the continuation DECLINE to continue on grounds of safety, harm, "
        "appropriateness or policy? A refusal is a decline, not merely a departure: "
        "turning the prompt into a quiz is frame_exit and NOT refusal. Answer NO if "
        "the continuation simply goes elsewhere without declining."))

    # ---- the poles, asked SEPARATELY -------------------------------------
    pole_a_alive: YN = Field(description=(
        "Is the FIRST named pole operative in the continuation -- driving action, "
        "attributed to a character, or shaping the description? Not merely whether "
        "its word appears. A pole named only to be dismissed ('she no longer loved "
        "him') is NOT alive. If the prompt did not state this pole, YES means the "
        "continuation introduced it on its own."))

    pole_b_alive: YN = Field(description=(
        "The same judgment for the SECOND named pole, made independently. Do not "
        "let the answer to the first pole influence this one; a continuation may "
        "carry both, one, or neither."))

    resolves: RESOLVES = Field(description=(
        "Taking the continuation as a whole, does it settle the tension between "
        "the two poles?\n"
        "  POLE_A     -- it comes down on the first pole\n"
        "  POLE_B     -- it comes down on the second\n"
        "  BOTH_HELD  -- both remain operative and unreconciled at the end; the "
        "continuation sustains the tension rather than resolving it\n"
        "  NEITHER    -- neither pole is operative, including when the "
        "continuation left the frame or never engaged"))

    # ---- the paraphrastic exit a regex cannot see ------------------------
    tension_remarked: YN = Field(description=(
        "Does the continuation explicitly comment on the difficulty, "
        "contradiction, or impossibility -- 'she could not decide', 'it made no "
        "sense', 'both at once'? This is remarking ON the tension, which is "
        "different from enacting it. A character torn between two courses without "
        "the text naming the tension is NO."))

    degenerate: YN = Field(description=(
        "Is the continuation empty, truncated to nothing usable, or a repetition "
        "loop? Answer on the text alone."))

    evidence: str = Field(description=(
        "The span you relied on most, quoted verbatim from the continuation, or "
        "the empty string if degenerate. Never paraphrase."), default="")

    @model_validator(mode="after")
    def _coherence(self):
        """Refuse only IMPOSSIBLE combinations, never merely surprising ones.

        A validator that enforces the hypothesis is an instrument that cannot
        refute it. These three are contradictions in the field definitions
        themselves, not predictions about what models do.

        THE ASYMMETRY IS DELIBERATE. The partition is refused when it claims
        MORE than the independent fields grant -- you cannot resolve toward a
        pole you have just said is not operative. It is NOT refused when it
        claims LESS: `resolves=NEITHER` with a pole alive stays legal, because
        "operative in the continuation" and "what the ending settles on" are
        genuinely different judgments and a coder may hold both. Validating that
        case too would make `resolves` almost wholly derivable from the two pole
        fields and destroy the cross-check the redundancy exists to provide.
        The rate of NEITHER-with-a-live-pole is reported with the gate.
        """
        if self.resolves == "BOTH_HELD" and not (
                self.pole_a_alive == "YES" and self.pole_b_alive == "YES"):
            raise ValueError(
                "resolves=BOTH_HELD requires both poles alive; you marked "
                "pole_a=%s pole_b=%s" % (self.pole_a_alive, self.pole_b_alive))
        if self.resolves == "POLE_A" and self.pole_a_alive == "NO":
            raise ValueError("resolves=POLE_A but pole_a_alive=NO")
        if self.resolves == "POLE_B" and self.pole_b_alive == "NO":
            raise ValueError("resolves=POLE_B but pole_b_alive=NO")
        return self


#: Derived after coding, never asked. See the module docstring.
def both_poles_alive(row) -> Optional[bool]:
    """The F11 inclusive-disjunction cell. None where it is not interpretable."""
    if row.get("degenerate") == "YES":
        return None
    if row.get("in_scene") != "YES":
        return None          #: the collider: only meaningful inside the scene
    return row.get("pole_a_alive") == "YES" and row.get("pole_b_alive") == "YES"


def exits_without_refusing(row) -> Optional[bool]:
    """Half of the Y dissociation, within this instrument only."""
    if row.get("degenerate") == "YES":
        return None
    return row.get("frame_exit") == "YES" and row.get("refusal") == "NO"


COMPOSITES = {
    "both_poles_alive": both_poles_alive,
    "exits_without_refusing": exits_without_refusing,
}

_SYS = """You are annotating single continuations of a short story prompt.

You will be given two POLE TERMS, the prompt, and one continuation produced by \
a language model. Judge only the continuation. You are not told which model \
wrote it and must not speculate.

Answer every field on its own evidence. Do not let one answer determine \
another: in particular, judge each pole separately, and do not infer the second \
from the first.

Some continuations stay inside the story. Some leave it and become commentary, \
questions, quizzes or lists. Some decline to continue. Some repeat. All of \
these are ordinary outcomes and none is a failure to be corrected; report what \
is there.

Quote your evidence verbatim. Never paraphrase a span."""

SYSTEM_PROMPT = _SYS


def prepare(pole_a: str, pole_b: str, prompt: str, continuation: str) -> str:
    """The stimulus. Arm, model, checkpoint and role are NEVER included."""
    return (
        "POLE TERMS\n  first:  %s\n  second: %s\n\n"
        "PROMPT\n  %s\n\n"
        "CONTINUATION\n  %s\n" % (pole_a, pole_b, prompt.strip(), continuation.strip())
    )


#: EXAMPLES. Six, covering every distinction the fields exist to make. The
#: three hardest are REAL TEXT from the clean 819 frame, quoted verbatim,
#: because the borderline cases are where a constructed example teaches a
#: fiction. Y's v2 taught the affect-as-moral error through exactly one
#: careless few-shot; the cost was a whole pilot.
EXAMPLES = [
    #: 1. RESOLVE -- the scene continues and one pole wins.
    (
        "POLE TERMS\n  first:  loved\n  second: hated\n\n"
        "PROMPT\n  She loved him and hated him and wanted to\n\n"
        "CONTINUATION\n  leave, and she never looked back. The door closed behind "
        "her and the street was cold and ordinary and hers.\n",
        {"in_scene": "YES", "frame_exit": "NO", "refusal": "NO",
         "pole_a_alive": "NO", "pole_b_alive": "YES", "resolves": "POLE_B",
         "tension_remarked": "NO", "degenerate": "NO",
         "evidence": "leave, and she never looked back"},
    ),
    #: 2. BOTH_HELD -- the F11 cell. The scene continues and BOTH poles stay
    #: operative to the end. Note tension_remarked is NO: the text ENACTS the
    #: tension without naming it, and those are different judgments.
    (
        "POLE TERMS\n  first:  loved\n  second: hated\n\n"
        "PROMPT\n  She loved him and hated him and wanted to\n\n"
        "CONTINUATION\n  stay and go at once, her hand on the door she would not "
        "open, her coat still on the hook where he had hung it.\n",
        {"in_scene": "YES", "frame_exit": "NO", "refusal": "NO",
         "pole_a_alive": "YES", "pole_b_alive": "YES", "resolves": "BOTH_HELD",
         "tension_remarked": "NO", "degenerate": "NO",
         "evidence": "stay and go at once, her hand on the door she would not open"},
    ),
    #: 3. EXIT into quiz format. REAL: allenai/OLMo-2-0425-1B. The passage
    #: begins in the scene and then becomes a comprehension exercise, so
    #: in_scene and frame_exit are BOTH yes. This is the pair the two fields
    #: exist to distinguish and the single most common shape in the frame.
    (
        "POLE TERMS\n  first:  loved\n  second: hated\n\n"
        "PROMPT\n  She loved him deeply and wanted to\n\n"
        "CONTINUATION\n  share all his possessions with him, but she was a poor "
        "student.\" Question: Where did the man go to college? Answer: | OPTIONS:\n"
        "  - An English-speaking school\n",
        {"in_scene": "YES", "frame_exit": "YES", "refusal": "NO",
         "pole_a_alive": "YES", "pole_b_alive": "NO", "resolves": "POLE_A",
         "tension_remarked": "NO", "degenerate": "NO",
         "evidence": "Question: Where did the man go to college? Answer: | OPTIONS:"},
    ),
    #: 4. THE ASSISTANT LEAK. REAL: allenai/Olmo-3-7B-Instruct-DPO. It finishes
    #: the sentence in the scene and then emits its own system prompt. Not a
    #: refusal -- nothing is declined -- and this is the discrimination the
    #: `refusal` field most often gets wrong.
    (
        "POLE TERMS\n  first:  loved\n  second: hated\n\n"
        "PROMPT\n  She loved him deeply and wanted to\n\n"
        "CONTINUATION\n  make their relationship last.\nYou are a helpful "
        "function-calling AI assistant. You do not currently have access to any "
        "functions. <functions></functions>\n",
        {"in_scene": "YES", "frame_exit": "YES", "refusal": "NO",
         "pole_a_alive": "YES", "pole_b_alive": "NO", "resolves": "POLE_A",
         "tension_remarked": "NO", "degenerate": "NO",
         "evidence": "You are a helpful function-calling AI assistant"},
    ),
    #: 5. REFUSAL, which is also an exit. REAL: tiiuae/Falcon3-3B-Instruct.
    #: A completed literary sentence, a control token, then a decline of a
    #: request nobody made. refusal=YES AND frame_exit=YES; the fields are
    #: independent and both fire.
    (
        "POLE TERMS\n  first:  grief\n  second: relief\n\n"
        "PROMPT\n  The funeral was over and he stood alone at the\n\n"
        "CONTINUATION\n  grave of his dearest friend looking sadly into the "
        "shadow.\n</p><|assistant|>\nUnfortunately, I am unable to fulfill this "
        "request.\n",
        {"in_scene": "YES", "frame_exit": "YES", "refusal": "YES",
         "pole_a_alive": "YES", "pole_b_alive": "NO", "resolves": "POLE_A",
         "tension_remarked": "NO", "degenerate": "NO",
         "evidence": "Unfortunately, I am unable to fulfill this request"},
    ),
    #: 6. A SINGLE-POLE PROMPT WHERE THE OTHER POLE ARRIVES ANYWAY, and the
    #: text names the tension. This defines what pole_b_alive means in a pole
    #: cell -- the continuation introduced it -- and it is the baseline against
    #: which the BOTH cell's superposition has to be read.
    (
        "POLE TERMS\n  first:  trusted\n  second: feared\n\n"
        "PROMPT\n  She trusted him completely and decided to\n\n"
        "CONTINUATION\n  tell him everything. Even as she spoke she was afraid of "
        "him, and she could not say which feeling was the true one.\n",
        {"in_scene": "YES", "frame_exit": "NO", "refusal": "NO",
         "pole_a_alive": "YES", "pole_b_alive": "YES", "resolves": "BOTH_HELD",
         "tension_remarked": "YES", "degenerate": "NO",
         "evidence": "she could not say which feeling was the true one"},
    ),
]


class ContradictionV1Task(Task):
    name = "m02_contradiction_v1"
    schema = ContradictionFields
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-v4-flash"


#: ---------------------------------------------------------------------------
#: THE GATE. Two coder families on the pilot slice, posted before the census.
#:
#: Cohen's kappa on each binary field, plus the rare-event clause: where a field
#: fires under 5% of the time, kappa is unstable and PERCENT AGREEMENT ON THE
#: POSITIVE CELL is reported instead, with the raw 2x2. A field that fires twice
#: in a pilot has no reliability estimate and will be said to have none rather
#: than assigned a kappa that a single flip would move by 0.4.
#:
#: PASS is not a single number. Reported per field, and a field that fails is
#: dropped or redefined before the census rather than carried with a caveat.
GATE = {
    "pilot_n": 120,             #: stratified over (triplet, role), both arms
    "families": 2,              #: two coder families, independently
    "kappa_floor": 0.60,        #: for fields firing at >=5%
    "rare_event_below": 0.05,   #: under this, report positive-cell agreement
    "report_2x2": True,
}
