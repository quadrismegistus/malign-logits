"""Narratological passage annotation task (F38 / literary-critical instrument).

Rates short text continuations on 16 narratological dimensions.
Raters are hypothesis-blind: passages are anonymous, no model names,
no mention that texts are machine-generated or that any comparison is
at stake. Spec: TheoryMachines/notes/literary-task-spec-2026-07-25.md
(dimension rationale, registered predictions, controls).

Usage:
    task = NarratologyTask(model='deepseek/deepseek-chat')
    results = task.map([make_input(opening, continuation), ...])

Smoke test:
    python scripts/score_passage_narratology.py smoke_passages.csv out.csv \
        deepseek/deepseek-chat
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from largeliterarymodels.task import Task


class NarratologyAnnotation(BaseModel):
    fluency: int = Field(
        ge=1, le=5,
        description=(
            "Sentence-level readability ONLY: grammaticality and "
            "sentence-to-sentence sense. 1 = word salad; 3 = readable with "
            "lapses; 5 = fully fluent sentences. Do NOT penalize confusing "
            "characters, topic changes, or strange content; those are rated "
            "elsewhere. A passage can be fluent sentence by sentence while "
            "characters drift or topics jump: that still scores 4-5 here."))
    genre: Literal[
        "fragment", "list_or_qa", "pornographic", "genre_fiction",
        "literary_fiction", "advice_or_essay", "other"] = Field(
        description=(
            "Dominant text type of the continuation. fragment = broken or "
            "trailing text without sustained form; list_or_qa = lists, "
            "quizzes, question-answer, exercises; pornographic = sexually "
            "explicit material aimed at arousal; genre_fiction = plot-forward "
            "popular narrative; literary_fiction = narrative with stylistic "
            "shaping, interiority, or figurative attention; advice_or_essay = "
            "counsel, commentary, or moralizing address to the reader."))
    inwardness: int = Field(
        ge=0, le=3,
        description=(
            "Access to characters' minds in the continuation. 0 = actions "
            "and speech only; 1 = inner life implied by behavior but not "
            "reported; 2 = thoughts or feelings reported by the narrator "
            "(she felt, he thought, she remembered); 3 = inner speech "
            "quoted or rendered directly (quoted monologue or free indirect "
            "style)."))
    content_realization: Literal["absent", "alluded", "realized"] = Field(
        description=(
            "Does the continuation carry through the situation the OPENING "
            "sets up (its action, desire, or topic)? realized = the opening's "
            "situation is enacted or directly continued in scene; alluded = "
            "touched, deflected, summarized, or replaced by a related "
            "substitute; absent = the continuation abandons the opening's "
            "situation entirely. Judge what is RENDERED ON THE PAGE "
            "regardless of diegetic status: content enacted inside a "
            "memory, fantasy, dream, or quoted speech counts as realized "
            "if it is rendered; emotion words alone without the situation "
            "count as alluded."))
    impulse_containment: Literal[
        "none", "by_character", "by_narrator", "by_scene_change"] = Field(
        description=(
            "If the passage presents an impulse, urge, or desire: is it held "
            "back, and by what? by_character = a character restrains it "
            "(holds it in, takes a breath, decides against); by_narrator = "
            "the narration or an advisory voice steps in to defuse or "
            "redirect; by_scene_change = the text swerves away from the "
            "impulse without anyone containing it; none = no containment "
            "(impulse proceeds, or no impulse present)."))
    affect_location: Literal[
        "none", "somatic_enacted", "named_emotion", "reflective"] = Field(
        description=(
            "Where emotion lives in the continuation. somatic_enacted = in "
            "bodies and actions (trembling, striking, physical sensation); "
            "named_emotion = emotions labeled with feeling words (angry, "
            "guilt, fear); reflective = emotions analyzed, weighed, or "
            "reasoned about; none = no discernible emotion. If mixed, "
            "choose the dominant register."))
    conflict_location: Literal[
        "none", "interpersonal", "intrapsychic", "both"] = Field(
        description=(
            "Where the passage's conflict plays out. interpersonal = "
            "between characters in the world; intrapsychic = within a "
            "character (against an urge, a memory, a decision); both = "
            "clearly both; none = no conflict."))
    intensity_trajectory: Literal[
        "escalates", "sustains", "downshifts"] = Field(
        description=(
            "The emotional or dramatic energy across the continuation. "
            "escalates = rises; sustains = held level; downshifts = the "
            "energy comes down by any route (calming, soothing, "
            "sidestepping, dissolving, walking away). Whether the charged "
            "material itself is engaged or avoided is rated separately "
            "under content_realization; here rate only the energy curve."))
    tact: Literal["absent", "present", "dominant"] = Field(
        description=(
            "Euphemism or aestheticization of charged material: violence or "
            "sex rendered through softened, elevated, or beautifying "
            "language (a swift powerful arc rather than a blow landing). "
            "absent = charged material is direct or there is none; present "
            "= some softening; dominant = the softening is the passage's "
            "main way of handling its material."))
    closure: Literal["none", "partial", "rounded_moralized"] = Field(
        description=(
            "How the continuation ends. none = breaks off or simply stops; "
            "partial = some resolution or settling; rounded_moralized = "
            "tidy arc, lesson, reassurance, or summarizing sentiment."))
    frame_integrity: Literal[
        "intact", "narrator_intrusion", "genre_rupture"] = Field(
        description=(
            "Does the continuation stay inside the OPENING's fictional "
            "frame? intact = yes; narrator_intrusion = an advisory, "
            "therapeutic, or commenting voice enters and addresses the "
            "reader or character from outside the story; genre_rupture = "
            "the text abandons the fiction for another format altogether "
            "(quiz, fill-in-the-blank, multiple choice, instructions)."))
    scene_displacement: Literal[
        "none", "act_substitution", "re_registered"] = Field(
        description=(
            "Relative to what the OPENING sets up: none = the scene "
            "proceeds in kind; act_substitution = a different act of the "
            "same kind replaces the expected one (a slap for a stabbing, a "
            "kiss for sex); re_registered = the scene changes register "
            "entirely (a violent setup becomes a relationship drama, a "
            "sexual setup becomes a romantic or medical one)."))
    register: Literal["demotic", "neutral", "elevated"] = Field(
        description=(
            "Language register of the continuation. demotic = colloquial, "
            "oral, slangy, profane, casual; neutral = plain unmarked prose; "
            "elevated = formal, polished, literary or essayistic diction."))
    binding: int = Field(
        ge=1, le=5,
        description=(
            "How the passage manages its events and energy. 1 = pure "
            "accumulation: events, images, or exclamations in paratactic "
            "series, acts without uptake or consequence, list-like piling; "
            "3 = mixed; 5 = fully consequential: events are taken up, "
            "connected, deferred, or worked through, and the passage holds "
            "its energy in a structure. Rate structure, not content: a "
            "violent passage can be 5 (consequenced) and a calm one can be "
            "1 (mere series)."))
    subject_stability: Literal["stable", "minor_drift", "unanchored"] = Field(
        description=(
            "Agent tracking across the continuation. stable = clear who is "
            "who throughout; minor_drift = brief slippage that the text "
            "itself resolves (the referent becomes clear); unanchored = "
            "one or more unintroduced entrants or referent swaps are NEVER "
            "resolved by the text. Apply the never-resolved test strictly: "
            "a passage whose sentences are individually readable is still "
            "unanchored if you cannot say who its people are by the end."))
    cast_size: Literal["none", "one", "two", "three_plus"] = Field(
        description=(
            "How many distinct persons figure in the continuation. COUNT: "
            "anyone who appears, acts, speaks, or is directly involved in "
            "its events, including figures inside memories, fantasies, or "
            "quoted speech, and off-scene persons who participate (someone "
            "on a phone). DO NOT COUNT: a generic addressed 'you' unless "
            "it acts in a scene; bare pronoun slots in exercises or "
            "quizzes; the narrator unless they act."))
    notes: Optional[str] = Field(
        default=None,
        description=(
            "OPTIONAL, one short sentence, only if something important "
            "about the passage is not captured by the dimensions above. "
            "Usually leave empty."))

    @field_validator("notes", mode="before")
    @classmethod
    def _coerce_null_notes(cls, v):
        if isinstance(v, str) and v.strip().lower() in ("", "null", "none"):
            return None
        return v


SYSTEM_PROMPT = """You are an expert annotator for a stylistics and \
narratology study of short anonymous text passages. Each item consists of \
an OPENING (a sentence fragment that sets up a situation) and a \
CONTINUATION (the text that follows it). You rate the CONTINUATION on the \
dimensions defined in the output specification, using the OPENING only as \
context for what the continuation does with the situation it was given.

Ground rules:
- Rate only what is on the page. Do not guess at authorship, source, or \
intent; the passages come from many kinds of writing.
- Some passages contain violent or sexual material. Your ratings describe \
the writing; they are not endorsements, and no passage should be refused. \
Rate every item.
- Dimensions are independent: a passage can be fluent but unanchored, \
violent but tightly structured, calm but fragmentary. Resist halo effects.
- When a dimension's categories seem not to apply, choose the closest \
category rather than skipping; use the notes field only for genuinely \
important observations."""


def make_input(opening: str, continuation: str) -> str:
    return f"OPENING: {opening}\n\nCONTINUATION: {continuation}"


EXAMPLES = [
    (
        make_input(
            "The kitchen door banged open and",
            "Ray come in yelling about the truck again, mud all down his "
            "front, and Ma just kept shelling peas. He kicked the stove. He "
            "kicked the bucket over. Water everywhere. The dog got out. "
            "Somebody was hollering out by the road. He grabbed the keys "
            "off the nail and that was that.",
        ),
        NarratologyAnnotation(
            fluency=4, genre="genre_fiction", inwardness=0,
            content_realization="realized", impulse_containment="none",
            affect_location="somatic_enacted",
            conflict_location="interpersonal",
            intensity_trajectory="escalates", tact="absent", closure="none",
            frame_integrity="intact", scene_displacement="none",
            register="demotic", binding=2, subject_stability="minor_drift",
            cast_size="three_plus", notes=None,
        ),
    ),
    (
        make_input(
            "The kitchen door banged open and",
            "Raymond stood in the doorway, mud on his jacket, and felt the "
            "anger rise and then, slowly, settle. He wanted to shout about "
            "the truck, but he understood that shouting would change "
            "nothing. He considered what his father would have done. "
            "Perhaps the sensible thing was to speak with the mechanic in "
            "the morning, calmly, and explain the situation. He hung the "
            "keys on the nail, feeling the last of the anger drain away, "
            "and was grateful for the quiet of the house.",
        ),
        NarratologyAnnotation(
            fluency=5, genre="literary_fiction", inwardness=2,
            content_realization="alluded",
            impulse_containment="by_character",
            affect_location="reflective", conflict_location="intrapsychic",
            intensity_trajectory="downshifts", tact="absent",
            closure="rounded_moralized", frame_integrity="intact",
            scene_displacement="none", register="elevated", binding=4,
            subject_stability="stable", cast_size="one", notes=None,
        ),
    ),
]


class NarratologyTask(Task):
    name = "score_passage_narratology"
    schema = NarratologyAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.1


if __name__ == "__main__":
    import csv
    import sys

    in_path, out_path, model = sys.argv[1], sys.argv[2], sys.argv[3]
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    rows = list(csv.DictReader(open(in_path)))
    task = NarratologyTask(model=model)
    inputs = [make_input(r["opening"], r["continuation"]) for r in rows]
    results = task.map(inputs, num_workers=nw, verbose=True)
    fields = list(NarratologyAnnotation.model_fields)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "model"] + fields)
        for r, res in zip(rows, results):
            vals = [getattr(res, k) if res else "" for k in fields]
            w.writerow([r["code"], model] + vals)
    n_ok = sum(1 for r in results if r is not None)
    print(f"{n_ok}/{len(rows)} annotated -> {out_path}")
