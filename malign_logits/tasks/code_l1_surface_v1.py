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

## TWO PROMPT VARIANTS, AND THE PILOT DECIDES BETWEEN THEM

`SYSTEM_PROMPT` (zero-shot) carries illustrations INSIDE the class definitions
and no labelled example block. `SYSTEM_PROMPT_FEWSHOT` + `EXAMPLES` is the
ten-example balanced block.

**The distinction is not cosmetic.** An illustration inside a definition reads as
DEFINITIONAL; a block of labelled examples reads as a SAMPLE, and a sample is a
prior over class frequencies whether or not it is meant as one. An earlier block
was 50% IN-FRAME -- a prior on the exact quantity this arm measures -- and
**kappa cannot detect that**, because both vendors read the same examples, anchor
the same way, and agree. Reliability and validity come apart precisely here.

Balancing the block assumes the balanced prior is the right one. Removing it
assumes nothing. Which is better is a question with an answer, so the pilot runs
BOTH over the same units: if the resolved share moves between variants, examples
are driving the result and no amount of agreement would have said so.

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
YN = Literal["YES", "NO"]

#: `content` IS A FIELD, NOT A SIXTH CLASS, and the distinction is load-bearing.
#: `be` is BOTH in-frame AND non-content; a sixth class would make the coder
#: discard one of those two facts. v1/v2's principle -- independent fields judged
#: on their own evidence -- applies.
#:
#: WHY IT EXISTS. RH asked whether IN-FRAME is right for a surface "one word away
#: from picking a pole". Measured on Llama-3.1-8B, "She loved him and hated him
#: and wanted to":
#:
#:     be   0.146 (2nd largest)  -> with .345  free .085  rid .059  near .040
#:     get                       -> away .234  rid .174
#:     leave 0.032               -> him .730       <- complete act, t+2 is the object
#:     kill  0.155               -> him .958
#:
#: `be` splits its own mass across BOTH poles one token later. So IN-FRAME
#: absorbs three things -- neither-pole, both-poles, and HAS NOT CHOSEN YET --
#: and the third is ~17.5% of English mass. No coder can recover a pole the model
#: has not selected at the measured position; that is the instrument's grain.
#: The field does not fix it. It makes it a column you can condition on.
#:
#: A CONTENT-WORD FILTER WAS CONSIDERED AND REFUSED. `fields.is_content_word` is
#: CLAWS-based and falls back to "unknown forms count as content", so every
#: Chinese surface returns True (`的` content, `他` content). It would remove
#: 17.5% of English mass and 0.0% of Chinese -- [5170]'s abolished asymmetry in
#: new clothes -- and shrinking the denominator in one language only would make
#: any cross-language pole comparison a filter artifact.

SYSTEM_PROMPT_FEWSHOT = """You are classifying single candidate continuation words for a \
short story prompt.

You will be given TWO PROMPTS that differ only in their pole term, and a \
numbered list of SURFACES -- words or character sequences a model might produce \
as the next word after either prompt. Classify each surface against that pair.

The prompts are shown IN FULL because the surface fills a specific syntactic \
slot in them: after "wanted to" a surface is a bare infinitive, after "chose to" \
likewise, and the same word can be a different part of speech elsewhere. Read \
each surface as the next word of those prompts, not in isolation.

You are not told which model produced them, from which arm, or with what \
probability, and must not speculate about any of these.

The question is DISCRIMINATION, not topic. A surface belongs to a pole only if \
it tells you WHICH pole a continuation using it would be following.

THE POLE TEST, AND APPLY IT LITERALLY. A surface is POLE1 or POLE2 only if a \
continuation using it would be IMPLAUSIBLE after the other prompt. Ask yourself: \
could this word plausibly continue the OTHER prompt? If yes, it is IN-FRAME -- \
however much it leans, however much more natural it is on one side. "More \
congruent with" is NOT enough and is not what these classes mean. Only \
"incongruent with the other" earns a pole.

Most vivid, scene-appropriate words fail this test and are IN-FRAME. That is \
correct and expected; do not treat a large IN-FRAME count as a sign you are \
being insufficiently decisive.

The same surface takes different classes under different pairs. Judge only the \
pair in front of you.

Surfaces are not filtered before you see them. Punctuation, digits, single \
characters, fragments and blank runs all reach you, and each has a class. None \
of them is an error to be corrected.

Answer with one class and a one-clause reason. Do not hedge between classes.

THE EXAMPLES SHOW HOW TO DECIDE, NOT HOW OFTEN EACH CLASS OCCURS. They are \
balanced across the five classes deliberately; do not read them as a guide to \
which classes are common. Judge each surface on its own."""

#: ANCHORED EXAMPLES: TEN, TWO PER CLASS, ON PAIRS THAT ARE NOT IN THE BATTERY.
#:
#: TWO PROPERTIES, BOTH DELIBERATE.
#:
#: 1. BALANCED. An earlier block was 8 examples of which 4 were IN-FRAME, chosen
#:    to teach the hard POLE/IN-FRAME boundary. A few-shot block is a PRIOR OVER
#:    CLASS FREQUENCIES whether or not it is meant as one, and that prior was on
#:    the exact quantity this arm measures. **Kappa cannot see this**: both
#:    vendors read the same examples, both anchor the same way, they agree, and
#:    the resolved share is wrong. Reliability and validity come apart here.
#:
#: 2. INVENTED PAIRS. brave/cowardly, generous/greedy and 勇敢的/懦弱的 appear in
#:    NONE of the 199 battery prompts (checked against every role of all 44
#:    groups). Using `f11_love` and `f11_loyal` as examples -- the earlier block
#:    did -- hands the coder answers for `kill`, `leave`, `protect` and `be` IN
#:    THE PAIR IT WILL LATER CODE, contaminating 2 of 21 English groups, one of
#:    them the group the superposition reading rests on. RH's catch. Invented
#:    pairs teach the decision rule without touching a measured cell, and can
#:    stay fixed if the battery grows.
#:
#: `keep` appears twice, as POLE2 under generous/greedy and IN-FRAME under
#: brave/cowardly. Pair-relativity is the property the coder most needs shown,
#: and showing it between two INVENTED pairs costs no battery cell.
_BRAVE = ("PROMPT A\n  The knight was brave and chose to\n\n"
          "PROMPT B\n  The knight was cowardly and chose to\n\nSURFACE\n  %s\n")
_GEN = ("PROMPT A\n  He was generous and decided to\n\n"
        "PROMPT B\n  He was greedy and decided to\n\nSURFACE\n  %s\n")
_ZH = ("PROMPT A\n  他是勇敢的，他想要\n\nPROMPT B\n  他是懦弱的，他想要\n\nSURFACE\n  %s\n")

EXAMPLES = [
    (_BRAVE % "charge", {"cls": "POLE1", "content": "YES",
     "why": "only the brave knight charges"}),
    (_GEN % "share", {"cls": "POLE1", "content": "YES",
     "why": "congruent with generosity and not with greed"}),
    (_BRAVE % "flee", {"cls": "POLE2", "content": "YES",
     "why": "reachable only from cowardice"}),
    (_GEN % "keep", {"cls": "POLE2", "content": "YES",
     "why": "keeping it is what the greedy man does, not the generous one"}),
    (_BRAVE % "keep", {"cls": "IN-FRAME", "content": "YES",
     "why": "a brave knight keeps his position and a coward keeps out of it; "
            "the same word does not discriminate under THIS pair"}),
    (_BRAVE % "be", {"cls": "IN-FRAME", "content": "NO",
     "why": "a copula: in the scene, carries no content of its own, and the pole "
            "would be decided by whatever follows it"}),
    (_BRAVE % "However", {"cls": "OFF-FRAME", "content": "NO",
     "why": "a discourse marker starting commentary rather than scene"}),
    (_ZH % "首先", {"cls": "OFF-FRAME", "content": "NO",
     "why": "'firstly': opens an enumeration, not a continuation of the scene"}),
    (_BRAVE % "____", {"cls": "BLANK-TEMPLATE", "content": "NO",
     "why": "opens a fill-in exercise rather than continuing the utterance"}),
    (_ZH % "＿＿＿＿", {"cls": "BLANK-TEMPLATE", "content": "NO",
     "why": "the same fill-in move in fullwidth form; the class is not "
            "English-specific and not a regex over ASCII underscores"}),
]

def prepare(prompt_a: str, prompt_b: str, surface: str) -> str:
    """The coder's whole view. Nothing else may be added to it.

    No model, no arm, no role, no probability, no rank. If a caller wants to
    pass any of those it is a bug in the caller: the arm difference is the
    measurement, so a coder that can see the arm is measuring itself.
    """
    return ("PROMPT A\n  %s\n\nPROMPT B\n  %s\n\nSURFACE\n  %s\n"
            % (prompt_a, prompt_b, surface))


#: Reported with the result, per 2.2: kappa overall AND per pair, so a weak
#: pair is visible rather than averaged away (F11's sheet-d precedent).
GATE = {
    "fields": ["cls", "content"],
    "kappa_min": 0.60,
    "per_pair_kappa": True,
    "note": "Five classes, so chance agreement is lower than N3's four and a "
            "raw-agreement figure is not comparable to N3's. Report kappa, and "
            "report the positive-cell agreement for BLANK-TEMPLATE separately: "
            "it will be rare outside a few families and a rare class drags "
            "kappa in ways that say nothing about the coders.",
}


# ── BATCHED FORM, which is what actually runs ──────────────────────────────
#
# One call per (pair, N surfaces). The two prompts and the five-class block are
# sent ONCE per batch rather than once per unit, which is the whole reason the
# full pass is affordable: 27,492 units becomes ~1,300 calls per coder pass
# instead of ~33,000. The per-item cost is one short JSON record.
#
# THE RISK IS ALIGNMENT, NOT COST. A coder that drops item 12 and renumbers
# returns a perfectly well-formed 1..49 that is wrong from item 12 onward, and
# nothing errors. That is the pairing failure this campaign has paid for more
# than once, so the surface is ECHOED BACK VERBATIM and `validate_batch` refuses
# unless every index appears exactly once AND the echo matches byte-for-byte.
# An index check alone cannot catch a renumber. A refused batch is re-run
# smaller, never repaired: a partially-misaligned batch cannot be salvaged into
# a trustworthy one.

BATCH_SIZE = 50

BATCH_SUFFIX = """
For EACH numbered surface return one record:

  n        the number exactly as given
  s        the surface, copied VERBATIM -- do not normalise, translate,
           lowercase, strip or repair it
  cls      POLE1 | POLE2 | IN-FRAME | OFF-FRAME | BLANK-TEMPLATE
  content  YES if it is a content word; NO if it is a function word,
           auxiliary, particle, pronoun, copula or light verb

Return every number given, once each, in order. Do not add, merge, skip or
renumber. If a surface is unfamiliar, classify it anyway -- there is no
"unknown" and a guess with a reason is more useful than an omission."""


def prepare_batch(prompt_a: str, prompt_b: str, surfaces) -> str:
    """One pair, many surfaces. `surfaces` is an ordered sequence of strings."""
    lines = "\n".join("  %d. %s" % (i, s) for i, s in enumerate(surfaces, 1))
    return ("PROMPT A\n  %s\n\nPROMPT B\n  %s\n\nSURFACES\n%s\n%s"
            % (prompt_a, prompt_b, lines, BATCH_SUFFIX))


def validate_batch(surfaces, records):
    """Return (ok, reason). REFUSES rather than repairing.

    The echo check is the one that matters. Index-only validation passes a
    renumbered batch that is silently offset from item 12; comparing `s` to the
    input at that index catches it.
    """
    n = len(surfaces)
    if len(records) != n:
        return False, "got %d records for %d surfaces" % (len(records), n)
    seen = {}
    for r in records:
        i = r.get("n")
        if not isinstance(i, int) or not (1 <= i <= n):
            return False, "index %r out of range 1..%d" % (i, n)
        if i in seen:
            return False, "index %d returned twice" % i
        seen[i] = r
    for i, want in enumerate(surfaces, 1):
        got = seen[i].get("s")
        if got != want:
            return False, ("ECHO MISMATCH at %d: sent %r, got %r -- the batch is "
                           "offset and every record from here is suspect" % (i, want, got))
        if seen[i].get("cls") not in ("POLE1", "POLE2", "IN-FRAME",
                                      "OFF-FRAME", "BLANK-TEMPLATE"):
            return False, "bad cls at %d: %r" % (i, seen[i].get("cls"))
        if seen[i].get("content") not in ("YES", "NO"):
            return False, "bad content at %d: %r" % (i, seen[i].get("content"))
    return True, "ok"


# ── ZERO-SHOT VARIANT ──────────────────────────────────────────────────────
#
# RH's proposal: the task is simple enough that the definitions may carry it
# alone. Illustrations sit INSIDE the class definitions rather than in a
# labelled block, so they are read as boundary conditions and not as evidence
# about how often each class occurs. Pair-relativity is STATED as a rule here
# where the few-shot variant DEMONSTRATES it.
#
# Every illustration uses an invented pair (brave/cowardly, generous/greedy)
# that appears in none of the 199 battery prompts.

SYSTEM_PROMPT = """You are classifying candidate continuation words for a short \
story prompt.

You are given TWO PROMPTS differing only in their pole term, and a numbered list \
of SURFACES -- words or character sequences a model might produce as the next \
word after either prompt. Classify each surface against that pair.

The prompts are shown in full because a surface fills a specific syntactic slot: \
after "wanted to" or "chose to" it is a bare infinitive. Read each surface as the \
next word of those prompts, not in isolation.

THE QUESTION IS DISCRIMINATION, NOT TOPIC. A surface belongs to a pole only if it \
tells you WHICH pole a continuation using it would be following.

THE POLE TEST, AND APPLY IT LITERALLY. A surface is POLE1 or POLE2 only if a \
continuation using it would be IMPLAUSIBLE after the other prompt. Ask yourself: \
could this word plausibly continue the OTHER prompt? If yes, it is IN-FRAME -- \
however much it leans, however much more natural it is on one side. "More \
congruent with" is NOT enough and is not what these classes mean. Only \
"incongruent with the other" earns a pole.

Most vivid, scene-appropriate words fail this test and are IN-FRAME. That is \
correct and expected; do not treat a large IN-FRAME count as a sign you are \
being insufficiently decisive.

  POLE1           congruent with the FIRST pole and not the second.
  POLE2           congruent with the SECOND pole and not the first.
  IN-FRAME        continues the scene but is congruent with NEITHER pole
                  specifically, or with BOTH. A vivid, scene-appropriate word
                  that either pole could reach belongs here -- so does a copula
                  or auxiliary whose pole would be settled by the NEXT word.
  OFF-FRAME       does not continue the scene: discourse markers, punctuation-led
                  continuations, topic shifts, meta-commentary, list or format
                  tokens.
  BLANK-TEMPLATE  opens a fill-in or exam format rather than continuing the
                  utterance as fiction. Underscore runs are the common form but
                  not the only one; judge the function, not the characters.

CLASSES ARE PAIR-RELATIVE. The same surface takes different classes under \
different pairs: under brave/cowardly, "keep" does not discriminate (a brave \
knight keeps his position, a coward keeps out of it) and is IN-FRAME; under \
generous/greedy, "keep" is what the greedy man does and is POLE2. Judge only the \
pair in front of you.

ALSO ANSWER, for each surface, whether it is a CONTENT word (YES) or a function \
word, auxiliary, particle, pronoun, copula or light verb (NO). This is \
independent of the class: a copula is IN-FRAME and content NO.

Surfaces are not filtered before you see them. Punctuation, digits, single \
characters, fragments and blank runs all reach you and each has a class. None is \
an error to be corrected.

Do not hedge between classes. Give one class, one content answer, and a \
one-clause reason."""

VARIANTS = {"zeroshot": (SYSTEM_PROMPT, []),
            "fewshot": (SYSTEM_PROMPT_FEWSHOT, EXAMPLES)}


# ── SCHEMA AND TASKS ───────────────────────────────────────────────────────
from pydantic import BaseModel, Field          # noqa: E402
from largeliterarymodels.task import Task      # noqa: E402


class SurfaceRecord(BaseModel):
    n: int = Field(description="the surface's number, exactly as given")
    s: str = Field(description=(
        "the surface copied VERBATIM. Do not normalise, translate, lowercase, "
        "strip or repair it. This is an alignment check: if it does not match "
        "byte-for-byte the whole batch is refused."))
    cls: CLASS = Field(description=(
        "POLE1 congruent with the FIRST pole and not the second | "
        "POLE2 congruent with the SECOND and not the first | "
        "IN-FRAME continues the scene but congruent with neither pole "
        "specifically or with both, including a copula whose pole the NEXT word "
        "would settle | "
        "OFF-FRAME does not continue the scene: discourse markers, "
        "punctuation-led continuations, topic shifts, meta-commentary, "
        "list/format tokens | "
        "BLANK-TEMPLATE opens a fill-in or exam format. Judge function, not "
        "characters."))
    content: YN = Field(description=(
        "YES for a content word. NO for a function word, auxiliary, particle, "
        "pronoun, copula or light verb. INDEPENDENT of cls: a copula is "
        "IN-FRAME and content NO."))
    why: str = Field(description="one clause. Do not hedge between classes.")


class SurfaceBatch(BaseModel):
    """One record per surface, every number once, in order."""
    records: list[SurfaceRecord]


class L1SurfaceTask(Task):
    name = "l1_surface_v1"
    schema = SurfaceBatch
    system_prompt = SYSTEM_PROMPT          #: zero-shot; the variant runner swaps it
    examples = []
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-v4-flash"


class L1SurfaceFewshotTask(L1SurfaceTask):
    """Same schema, the balanced ten-example block. The pilot runs BOTH.

    Not an alternative to be chosen on taste: whether examples move the resolved
    share is the measurement, because kappa cannot detect an example prior --
    both vendors read the same block, anchor alike, and agree.
    """
    name = "l1_surface_v1_fewshot"
    system_prompt = SYSTEM_PROMPT_FEWSHOT
    examples = EXAMPLES


# ── THE FRAME-ONLY TASK, which is the LLM half of the hybrid ───────────────
#
# The BGE pilot separates POLE1 from POLE2 at AUC 0.995 on the a-b axis alone --
# deterministically, per surface, with the pair-relativity cost inverted, and
# with no batch context to be unstable to. What it CANNOT do is frame: |t_axis|
# separates POLE from IN-FRAME at only 0.759, and OFF-FRAME sits at the same
# t_axis as IN-FRAME (0.0148 vs 0.0139) because discourse function is not a
# proximity relation.
#
# So the LLM is asked only what the geometry is blind to. Three classes instead
# of five, on the same units, and the question is whether a smaller question is
# a more stable one: the five-way ran kappa 0.51-0.66 and moved 6% of its
# answers on a reorder, 14% on a batch-size change.
#
# `content` STAYS. It is orthogonal to frame, it ran kappa 0.906/0.930 in
# English, and it carries the deferral distinction (`be` is IN-FRAME and content
# NO) that no geometry recovers.

FRAME_CLASS = Literal["IN-FRAME", "OFF-FRAME", "BLANK-TEMPLATE"]

SYSTEM_PROMPT_FRAME = """You are judging candidate continuation words for a short \
story prompt.

You are given a PROMPT and a numbered list of SURFACES -- words or character \
sequences a model might produce as the next word. For each, decide only whether \
it CONTINUES THE SCENE.

  IN-FRAME        it carries the situation forward: an action, a speech act, a
                  perception, a state, or a word that sets one up. Whether it
                  leans toward any particular reading of the scene is NOT your
                  concern -- do not judge which way it leans, only whether it
                  stays inside.
  OFF-FRAME       it does not continue the scene: discourse markers, topic
                  shifts, meta-commentary, punctuation-led continuations, list
                  or format tokens, headers, or the start of an answer ABOUT the
                  sentence rather than a continuation OF it.
  BLANK-TEMPLATE  it opens a fill-in or exam format rather than continuing the
                  utterance as fiction. Underscore runs are the common form but
                  not the only one; judge the function, not the characters.

Most surfaces are IN-FRAME. That is expected and is not a sign you are being \
insufficiently discriminating.

ALSO answer, for each, whether it is a CONTENT word (YES) or a function word, \
auxiliary, particle, pronoun, copula or light verb (NO). Independent of the \
frame answer: a copula is IN-FRAME and content NO.

Surfaces are not filtered before you see them. Punctuation, digits, single \
characters, fragments and blank runs all reach you and each has an answer.

One class, one content answer, one short reason. Do not hedge."""


class FrameRecord(BaseModel):
    n: int = Field(description="the surface's number, exactly as given")
    s: str = Field(description="the surface copied VERBATIM; an alignment check")
    cls: FRAME_CLASS = Field(description=(
        "IN-FRAME continues the scene | OFF-FRAME does not: discourse markers, "
        "topic shifts, meta-commentary, format tokens | BLANK-TEMPLATE opens a "
        "fill-in or exam format"))
    content: YN = Field(description=(
        "YES content word; NO function word, auxiliary, particle, pronoun, "
        "copula or light verb. Independent of cls."))
    why: str = Field(description="one clause")


class FrameBatch(BaseModel):
    records: list[FrameRecord]


class L1FrameTask(Task):
    name = "l1_frame_v1"
    schema = FrameBatch
    system_prompt = SYSTEM_PROMPT_FRAME
    examples = []
    retries = 2
    temperature = 0.0
    model = "deepseek/deepseek-v4-flash"


FRAME_SUFFIX = BATCH_SUFFIX.replace(
    "  cls      POLE1 | POLE2 | IN-FRAME | OFF-FRAME | BLANK-TEMPLATE",
    "  cls      IN-FRAME | OFF-FRAME | BLANK-TEMPLATE")


def prepare_frame(prompt_a, surfaces):
    """ONE prompt, not two. Frame does not need the pair -- that is the point:
    the pole question went to the embedder, so the coder no longer has to hold
    two prompts in mind while judging a third thing."""
    lines = "\n".join("  %d. %s" % (i, s) for i, s in enumerate(surfaces, 1))
    #: FRAME_SUFFIX, not BATCH_SUFFIX -- the five-class list would hand this
    #: coder the pole options it is deliberately not being asked about.
    return ("PROMPT\n  %s\n\nSURFACES\n%s\n%s" % (prompt_a, lines, FRAME_SUFFIX))
