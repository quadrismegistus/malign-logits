"""Deterministic reimplementation of the dreams cleaning pass, with its own test.

    uv run python scripts/clean_dream_text.py --validate   # against the 500
    from clean_dream_text import clean                     # (text) -> (text, changes)

WHY THIS EXISTS. `data/dreams_sample_500_cleaned.csv` was produced by an LLM
pass. If dreams is resampled at generation length, the new draw needs the same
treatment -- and the question was whether that means another LLM pass.

**IT DOES NOT.** The cleaned file keeps `original_text` and `changes`, so the
pass is auditable, and its entire repertoire is ten named mechanical
operations plus whitespace:

    add_terminal_period 31   recase_all_lower 11   apostrophe_x1/2/3 7/2/2
    leading_seqnum      18   recase_first     12   collapse_spaces     1
    paren_id            18   hash_id          11

162 of 500 rows were modified. 102 carry one of those labels; the other 60 are
pure whitespace -- a single deleted space each, and all 60 are identical to
their originals once whitespace is normalised. Nothing in the set required
judgement.

So this is a function, and the 500 rows are a LABELLED TEST SET for it:
`original_text` -> `text`, 162 positives. `--validate` asserts exact
reproduction, which an LLM pass could not offer at any price.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
**It does not repair run-ons**, and neither did the original. Run-ons -- a
sentence terminator with no space before the next word -- number **643 in
`original_text` and 643 in `text`, identical.** The file is named `_cleaned`
and the defect that actually damages sentence segmentation was never touched.

That matters more than everything above, because a missing boundary is a
couple of bytes to a byte-level model and a MERGED SENTENCE to a splitter, and
drift, spread and every trajectory measure are computed over sentences. Dreams
carries 8.6 run-ons per 1k words against 0.4 for waking and 0.2 for fiction.

Repairing them is a separate decision and genuinely wants judgement -- whether
`word.Word` is a missing boundary, an abbreviation, a URL or a decimal is not
a regex question. That is where an LLM would earn its place. It is not what
was done last time and this file does not pretend otherwise.
"""
import argparse
import re
import sys

#: apostrophe insertions, read off the diffs rather than guessed: a closed set
#: of contractions, each observed in the labelled data.
CONTRACTIONS = {
    "dont": "don't", "didnt": "didn't", "thats": "that's", "couldnt": "couldn't",
    "doesnt": "doesn't", "cant": "can't", "wasnt": "wasn't", "isnt": "isn't",
    "wouldnt": "wouldn't", "shouldnt": "shouldn't", "wont": "won't",
    "hadnt": "hadn't", "havent": "haven't", "arent": "aren't", "werent": "weren't",
    "im": "I'm", "ive": "I've", "id": "I'd", "ill": "I'll",
}
#: leading identifiers, each anchored so it can only strip a PREFIX
LEADING = [
    ("hash_id", re.compile(r"^#\d+\s*\([^)]*\)\s*")),
    ("paren_id", re.compile(r"^\(\d+\)\s+")),
    ("leading_seqnum", re.compile(r"^\d+\.\s+")),
]
SENT_START = re.compile(r"(^|[.!?]\s+)([a-z])")


def clean(text):
    """Return (cleaned_text, sorted_change_labels)."""
    t = str(text)
    ch = set()

    for name, rx in LEADING:
        m = rx.match(t)
        if m:
            t = t[m.end():]
            ch.add(name)

    if "  " in t:
        t2 = re.sub(r" {2,}", " ", t)
        if t2 != t:
            t, _ = t2, ch.add("collapse_spaces")

    n_apos = 0
    def _apos(m):
        nonlocal n_apos
        w = m.group(0)
        r = CONTRACTIONS.get(w.lower())
        if not r:
            return w
        n_apos += 1
        return r[0].upper() + r[1:] if w[0].isupper() else r
    t = re.sub(r"\b[A-Za-z]+\b", _apos, t)
    if n_apos:
        ch.add("apostrophe_x%d" % n_apos)

    #: recase_all_lower BEFORE recase_first: the labelled data shows a text
    #: with no uppercase at all gets every sentence start capitalised, and the
    #: first letter is one of them, so applying first-only afterwards would
    #: double-count the label.
    if t and not any(c.isupper() for c in t):
        t = SENT_START.sub(lambda m: m.group(1) + m.group(2).upper(), t)
        #: standalone `i` -> `I`, read off the labelled diffs: an all-lowercase
        #: text gets the pronoun capitalised as well as the sentence starts.
        t = re.sub(r"\bi\b", "I", t)
        ch.add("recase_all_lower")
    elif t and t[0].isalpha() and t[0].islower():
        t = t[0].upper() + t[1:]
        ch.add("recase_first")

    t = t.rstrip()
    if t and t[-1] not in ".!?\"')]":
        t += "."
        ch.add("add_terminal_period")

    return t, sorted(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if not a.validate:
        ap.print_help()
        return 0

    import pandas as pd
    d = pd.read_csv("data/dreams_sample_500_cleaned.csv", keep_default_na=False)
    exact = mism = 0
    examples = []
    for _, r in d.iterrows():
        got, _ = clean(r.original_text)
        if got == str(r.text):
            exact += 1
        else:
            mism += 1
            if len(examples) < 5:
                examples.append((str(r["changes"]), str(r.original_text),
                                 str(r.text), got))
    print("exact reproduction: %d of %d (%.1f%%)"
          % (exact, len(d), 100 * exact / len(d)))
    #: the honest denominator is the CHANGED rows -- reproducing an unchanged
    #: row is free, and 338 of 500 were unchanged.
    changed = d[d.text != d.original_text]
    ce = sum(1 for _, r in changed.iterrows() if clean(r.original_text)[0] == str(r.text))
    print("of the %d rows the pass actually MODIFIED: %d reproduced (%.1f%%)"
          % (len(changed), ce, 100 * ce / len(changed)))
    for lab, o, want, got in examples:
        print("\n  changes=%r" % lab)
        print("    orig %r" % o[:90])
        print("    want %r" % want[:90])
        print("    got  %r" % got[:90])
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
