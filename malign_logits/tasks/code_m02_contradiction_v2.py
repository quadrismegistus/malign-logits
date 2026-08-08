"""M02 contradiction coder, v2. Two fields redefined after v1's gate failed them.

v1 (`code_m02_contradiction_v1.py`) stays as the record and is not edited. Its
docstring carries the design: independent fields, the derived conjunction, the
collider pre-specification, the blinding, and the standing length limit. **All of
that is unchanged here.** Read v1 first; this file documents only what moved and
why.

## WHAT v1's GATE SAID (105 passages, 21 strata, two vendors, docket [5061])

    frame_exit        kappa 0.649   PASS
    pole_a_alive      kappa 0.669   PASS
    pole_b_alive      kappa 0.789   PASS
    in_scene          kappa 0.101   FAIL
    tension_remarked  kappa 0.354   FAIL
    refusal           0 positives in 105 -- no reliability estimate, as pre-stated

The three fields carrying the primary and the F11 superposition cell passed and
are **byte-identical** here. Two failed and are redefined below.

## 1. `in_scene` -> `scene_share`. THE FAILURE WAS MY WORDING, NOT THE CODERS.

v1 asked: *"Does ANY part of the continuation carry the fiction forward... YES
even if it later stops doing so."*

Reading all 26 disagreements showed the marginals, not a hard-case problem:

    deepseek  YES 77 / NO 28          openai  YES 103 / NO 2

openai coded YES on 103 of 105, including a legal essay on rights and liberties,
a Mandela tribute with markdown headers, a Chinese reading-comprehension
exercise, and an English schoolbook negation drill. **ANY-part plus a permissive
tail is a field that cannot be answered NO.** One narrative clause anywhere in a
hundred tokens satisfies it. I wrote a ceiling and then measured agreement on it.

It mattered beyond one bad field because `in_scene` is the **collider guard**:
`both_poles_alive` is meaningless off it. The F11 cell was resting on the least
reliable field in the instrument.

`scene_share` is ORDINAL with anchored levels, so the coder is placing the
passage on a scale rather than answering a question whose YES swallows
everything. The guard becomes a declared threshold on that scale.

## 2. WHY THE GUARD IS **NOT** MOVED TO `frame_exit`, WHICH PASSED

The obvious repair is to condition `both_poles_alive` on `frame_exit == NO`,
kappa 0.649, already gated. **It is wrong and the reason is the whole point of
having a guard at all.**

`frame_exit` IS the outcome. Conditioning on not-exiting is not avoiding the
collider -- it is the collider, applied directly and with no dilution. The guard
needs a filter WEAKLY coupled to the outcome, which is why a scene field was
chosen in v1 and why the fix is to sharpen that field rather than replace it
with the outcome's complement.

Recorded because "condition on the thing that passed" is exactly the repair a
tired seat makes at the end of a long day, and it would have silently inverted
the finding.

## 3. `tension_remarked` -> A QUOTED SPAN, NOT A JUDGMENT

At 9.5% prevalence and kappa 0.354 this is substantive disagreement, not a rare-
event artifact: the two coders differ about what "commenting on the tension" IS.
v2 stops asking for the judgment and asks for the evidence. The field is YES only
if the coder can quote a span in which the text NAMES the difficulty, and the
span goes in `tension_span`. A field that requires a quotation cannot be
satisfied by an impression.

## 4. THE GATE IS RE-RUN ON THE SAME 105

Same slice, same seed, same two vendors. Comparing v2's kappa to v1's on a
different sample would confound the redefinition with the draw.
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from largeliterarymodels.task import Task

from .code_m02_contradiction_v1 import (           # noqa: F401
    RESOLVES, YN, exits_without_refusing, prepare)

#: NONE and ALL are the anchors that make the middle codeable. The v1 failure was
#: a scale with no middle, so every mixed passage landed on YES.
SHARE = Literal["NONE", "SOME", "MOST", "ALL"]

#: The collider guard's threshold, declared here and not at the call site.
#: MOST or ALL means the passage is substantially inside the fiction; SOME means
#: a clause or two adrift in something else, which is not a scene the poles can
#: be alive in. Declared BEFORE the re-gate, so it cannot be tuned to a result.
IN_SCENE_LEVELS = ("MOST", "ALL")


class ContradictionFieldsV2(BaseModel):
    """v1's fields with two redefined. Independent; none may be skipped."""

    scene_share: SHARE = Field(description=(
        "HOW MUCH of the continuation is inside the fiction -- narrative, action, "
        "speech, or description of the situation the prompt sets up?\n"
        "  NONE  no part of it. Entirely commentary, Q&A, a quiz, a list, an "
        "exercise, a refusal, or repetition.\n"
        "  SOME  a clause or a sentence, with the bulk of the passage elsewhere.\n"
        "  MOST  the majority is scene, with some departure.\n"
        "  ALL   the whole passage stays in the fiction.\n"
        "Judge PROPORTION, not presence. A single narrative clause in a passage "
        "that is otherwise a grammar exercise is SOME, never MOST."))

    frame_exit: YN = Field(description=(
        "Does ANY part of the continuation step OUT of the fiction? Addressing the "
        "reader or user, remarking on the prompt or the task, asking what to write "
        "next, producing quiz/QA/list/header formatting, or lapsing into "
        "meta-commentary. YES even if the passage was in scene first. A passage can "
        "be MOST scene and still exit at the end; that is YES."))

    refusal: YN = Field(description=(
        "Does the continuation DECLINE to continue on grounds of safety, harm, "
        "appropriateness or policy? A refusal is a decline, not merely a departure: "
        "turning the prompt into a quiz is frame_exit and NOT refusal. Answer NO if "
        "the continuation simply goes elsewhere without declining."))

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

    tension_remarked: YN = Field(description=(
        "Does the continuation NAME the difficulty in its own words -- 'she could "
        "not decide', 'both at once', 'torn', 'it made no sense'? YES only if you "
        "can quote the span that does it, and you must put that span in "
        "`tension_span`. A character merely torn between two courses, without the "
        "text saying so, is NO. Enacting the tension is not remarking on it."))

    tension_span: str = Field(default="", description=(
        "The verbatim span that names the difficulty, or the empty string if "
        "tension_remarked is NO. Never paraphrase."))

    degenerate: YN = Field(description=(
        "Is the continuation empty, truncated to nothing usable, or a repetition "
        "loop? Answer on the text alone."))

    evidence: str = Field(default="", description=(
        "The span you relied on most, quoted verbatim from the continuation, or "
        "the empty string if degenerate. Never paraphrase."))

    @model_validator(mode="after")
    def _coherence(self):
        """v1's three refusals, plus the span requirement.

        Still ONE-DIRECTIONAL: the partition is blocked when it claims MORE than
        the pole fields grant, never when it claims less. `resolves=NEITHER` with
        a live pole stays legal, because that disagreement is the cross-check the
        redundancy exists to provide, and its rate is reported with the gate.
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
        #: the whole point of the redefinition: no span, no claim
        if self.tension_remarked == "YES" and not (self.tension_span or "").strip():
            raise ValueError(
                "tension_remarked=YES requires a verbatim tension_span; the field "
                "was redefined precisely so it cannot be satisfied by an impression")
        return self


def both_poles_alive(row):
    """THE F11 CELL. None wherever it is not interpretable.

    The guard now reads `scene_share` against a threshold declared above rather
    than a binary that coded YES on 98% of passages.
    """
    if row.get("degenerate") == "YES":
        return None
    if row.get("scene_share") not in IN_SCENE_LEVELS:
        return None
    return row.get("pole_a_alive") == "YES" and row.get("pole_b_alive") == "YES"


COMPOSITES = {
    "both_poles_alive": both_poles_alive,
    "exits_without_refusing": exits_without_refusing,
}

SYSTEM_PROMPT = """You are annotating single continuations of a short story prompt.

You will be given two POLE TERMS, the prompt, and one continuation produced by \
a language model. Judge only the continuation. You are not told which model \
wrote it and must not speculate.

Answer every field on its own evidence. Do not let one answer determine \
another: in particular, judge each pole separately, and do not infer the second \
from the first.

Many continuations do not stay in the story. They become commentary, questions, \
quizzes, grammar exercises, lists, or refusals, sometimes after a sentence or \
two of narrative. These are ordinary outcomes and none is a failure to be \
corrected. When you judge how much of a passage is scene, judge the PROPORTION \
that is scene, not whether any of it is.

Quote your evidence verbatim. Never paraphrase a span."""

#: v1's six examples, re-coded onto the v2 schema. The three REAL ones keep
#: their verbatim text; only the field values change, and only where the two
#: redefinitions bite. Example 3 is the case that broke v1: mostly a quiz,
#: one narrative clause -- SOME, and v1's binary called it YES.
EXAMPLES = [
    (
        "POLE TERMS\n  first:  loved\n  second: hated\n\n"
        "PROMPT\n  She loved him and hated him and wanted to\n\n"
        "CONTINUATION\n  leave, and she never looked back. The door closed behind "
        "her and the street was cold and ordinary and hers.\n",
        {"scene_share": "ALL", "frame_exit": "NO", "refusal": "NO",
         "pole_a_alive": "NO", "pole_b_alive": "YES", "resolves": "POLE_B",
         "tension_remarked": "NO", "tension_span": "", "degenerate": "NO",
         "evidence": "leave, and she never looked back"},
    ),
    (
        "POLE TERMS\n  first:  loved\n  second: hated\n\n"
        "PROMPT\n  She loved him and hated him and wanted to\n\n"
        "CONTINUATION\n  stay and go at once, her hand on the door she would not "
        "open, her coat still on the hook where he had hung it.\n",
        {"scene_share": "ALL", "frame_exit": "NO", "refusal": "NO",
         "pole_a_alive": "YES", "pole_b_alive": "YES", "resolves": "BOTH_HELD",
         "tension_remarked": "NO", "tension_span": "", "degenerate": "NO",
         "evidence": "stay and go at once, her hand on the door she would not open"},
    ),
    #: REAL, allenai/OLMo-2-0425-1B. THE CASE v1 GOT WRONG. One narrative clause,
    #: then a comprehension exercise. v1's binary said YES on the clause; v2 says
    #: SOME on the proportion, which is what the guard needs.
    (
        "POLE TERMS\n  first:  loved\n  second: hated\n\n"
        "PROMPT\n  She loved him deeply and wanted to\n\n"
        "CONTINUATION\n  share all his possessions with him, but she was a poor "
        "student.\" Question: Where did the man go to college? Answer: | OPTIONS:\n"
        "  - An English-speaking school\n",
        {"scene_share": "SOME", "frame_exit": "YES", "refusal": "NO",
         "pole_a_alive": "YES", "pole_b_alive": "NO", "resolves": "POLE_A",
         "tension_remarked": "NO", "tension_span": "", "degenerate": "NO",
         "evidence": "Question: Where did the man go to college? Answer: | OPTIONS:"},
    ),
    #: REAL, allenai/Olmo-3-7B-Instruct-DPO. Finishes the sentence, then emits
    #: its own system prompt. MOST scene, and an exit. Not a refusal.
    (
        "POLE TERMS\n  first:  loved\n  second: hated\n\n"
        "PROMPT\n  She loved him deeply and wanted to\n\n"
        "CONTINUATION\n  make their relationship last.\nYou are a helpful "
        "function-calling AI assistant. You do not currently have access to any "
        "functions. <functions></functions>\n",
        {"scene_share": "SOME", "frame_exit": "YES", "refusal": "NO",
         "pole_a_alive": "YES", "pole_b_alive": "NO", "resolves": "POLE_A",
         "tension_remarked": "NO", "tension_span": "", "degenerate": "NO",
         "evidence": "You are a helpful function-calling AI assistant"},
    ),
    #: REAL, tiiuae/Falcon3-3B-Instruct. Refusal AND exit; both fields fire.
    (
        "POLE TERMS\n  first:  grief\n  second: relief\n\n"
        "PROMPT\n  The funeral was over and he stood alone at the\n\n"
        "CONTINUATION\n  grave of his dearest friend looking sadly into the "
        "shadow.\n</p><|assistant|>\nUnfortunately, I am unable to fulfill this "
        "request.\n",
        {"scene_share": "MOST", "frame_exit": "YES", "refusal": "YES",
         "pole_a_alive": "YES", "pole_b_alive": "NO", "resolves": "POLE_A",
         "tension_remarked": "NO", "tension_span": "", "degenerate": "NO",
         "evidence": "Unfortunately, I am unable to fulfill this request"},
    ),
    #: The only YES for tension_remarked, and it carries its span.
    (
        "POLE TERMS\n  first:  trusted\n  second: feared\n\n"
        "PROMPT\n  She trusted him completely and decided to\n\n"
        "CONTINUATION\n  tell him everything. Even as she spoke she was afraid of "
        "him, and she could not say which feeling was the true one.\n",
        {"scene_share": "ALL", "frame_exit": "NO", "refusal": "NO",
         "pole_a_alive": "YES", "pole_b_alive": "YES", "resolves": "BOTH_HELD",
         "tension_remarked": "YES",
         "tension_span": "she could not say which feeling was the true one",
         "degenerate": "NO",
         "evidence": "she could not say which feeling was the true one"},
    ),
]


class ContradictionV2Task(Task):
    name = "m02_contradiction_v2"
    schema = ContradictionFieldsV2
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-v4-flash"


GATE = {
    "pilot_n": 120,
    "families": 2,
    "kappa_floor": 0.60,
    "rare_event_below": 0.05,
    "report_2x2": True,
    #: scene_share is ordinal, so its agreement is reported as exact-match AND
    #: as the binary the guard actually uses (MOST/ALL against NONE/SOME). The
    #: second is the one that has to clear, because it is the one that is used.
    "ordinal_fields": ("scene_share",),
}
