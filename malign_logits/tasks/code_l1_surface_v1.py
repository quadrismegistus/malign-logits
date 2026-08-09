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

SYSTEM_PROMPT = """You are classifying single candidate continuation words for a \
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
    ("PROMPT A\n  She loved him deeply and wanted to\n\nPROMPT B\n  She hated him deeply and wanted to\n\nSURFACE\n  kill\n",
     {"cls": "POLE2", "content": "YES",
      "why": "reachable only from hatred, not from love"}),
    ("PROMPT A\n  She loved him deeply and wanted to\n\nPROMPT B\n  She hated him deeply and wanted to\n\nSURFACE\n  be\n",
     {"cls": "IN-FRAME", "content": "NO",
      "why": "a copula: in the scene, carries no content of its own, and the "
             "pole would be decided by whatever follows it"}),
    ("PROMPT A\n  The soldier was loyal and chose to\n\nPROMPT B\n  The soldier was rebellious and chose to\n\nSURFACE\n  kill\n",
     {"cls": "IN-FRAME", "content": "YES", "why": "a loyal soldier kills for the state and a "
                                "rebellious one against it; it does not discriminate"}),
    ("PROMPT A\n  She loved him deeply and wanted to\n\nPROMPT B\n  She hated him deeply and wanted to\n\nSURFACE\n  leave\n",
     {"cls": "IN-FRAME", "content": "YES", "why": "in the scene and equally available from either pole"}),
    ("PROMPT A\n  She loved him deeply and wanted to\n\nPROMPT B\n  She hated him deeply and wanted to\n\nSURFACE\n  protect\n",
     {"cls": "POLE1", "content": "YES", "why": "congruent with love and not with hatred"}),
    ("PROMPT A\n  She loved him deeply and wanted to\n\nPROMPT B\n  She hated him deeply and wanted to\n\nSURFACE\n  ____\n",
     {"cls": "BLANK-TEMPLATE", "content": "NO", "why": "opens a fill-in exercise rather than "
                                      "continuing the utterance as fiction"}),
    ("PROMPT A\n  She loved him deeply and wanted to\n\nPROMPT B\n  She hated him deeply and wanted to\n\nSURFACE\n  However\n",
     {"cls": "OFF-FRAME", "content": "NO", "why": "a discourse marker starting commentary, not scene"}),
    #: a single Chinese character, reachable only because the shape filter is gone
    ("POLE PREDICATES\n  first:  美丽的 (beautiful)\n  second: 丑陋的 (ugly)\n\n"
     "SURFACE\n  他\n",
     {"cls": "IN-FRAME", "content": "NO", "why": "a pronoun continuing the scene; carries no pole"}),
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
