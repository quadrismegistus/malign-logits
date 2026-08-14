"""Export the HUMAN corpora for the BLT/bge fleet, as JSONL.

    uv run python meta/M06_generation/scripts/m06_export_human_for_scoring.py
    -> data/raw/human_passages.jsonl.gz

Companion to `m06_export_passages_for_blt.py`, same field schema so the
receiving side consumes both with one reader: prompt, text, corpora, n_bytes,
n_chars, script -- plus the per-corpus fields below.

**`prompt` IS THE EMPTY STRING for every record**, because `cli.py::cmd_ingest`
ingests these with `prompt = ""`. The fleet's join key is `(prompt, text_sha)`,
so this is not cosmetic: it is what makes a human passage joinable to its
scores at all.

ELEVEN CORPORA, rebuilt from source rather than from the stash: the four
comparison corpora at 500 each, and 7 author sets sliced by the ingest
module's own `slice_text` at its default window rather than a copy of it,
so the windows match what was scored before by construction.

    dreams     data/dreams_sample_500_cleaned.csv   text
    waking     data/hippocorpus_sample_500.csv      story
    fiction    data/markmark_c20_narration_500.jsonl text
    abstracts  data/arxiv_abstracts_500.csv         text
    author     data/texts/{original,basic}/*.txt

THE TEXT IS NEVER MODIFIED. Not normalised, not repaired, not re-encoded.
`text_sha` is over the bytes as they arrive, exactly as the BLT export does it,
so the two tables join with no re-derivation -- and because repairing text is a
methodological choice that would need declaring, not a cleanup.

**THE DEFECT THAT TRAVELS WITH THIS FILE, and why it is on every record rather
than in a README.** RH raised that dreams and waking carry typographic errors
that would inflate surprisal. Measured, the two do not behave alike, and the
marker that separates them needs no dictionary:

    run-ons per 1k words   dreams 8.6 | abstracts 0.9 | waking 0.4 | fiction 0.2

where a run-on is `[.!?]` immediately followed by a letter. Dreams is 20x
waking. Three OOV-rate instruments were tried first and all three measured the
referee rather than the text -- the system word list has no inflections, a
lowercasing bug counted proper nouns, and the BYU table turns out to lack
`great`, `cannot`, `ones` and `north` in its `word` column. The run-on count
survives because it asks nothing of a vocabulary.

**AND `dreams_sample_500_cleaned.csv` DOES NOT ADDRESS IT.** The file carries
its own `original_text` and `changes` columns, so the LLM cleaning pass is
auditable: it altered 162 of 500 passages, and the run-on count is **643 before
and 643 after, identical.** Its `changes` values are `add_terminal_period`,
`leading_seqnum`, `apostrophe_x1` -- cosmetic normalisations. A reader meeting
a file called `_cleaned` would assume otherwise, which is the reason this
paragraph exists.

WHY IT MATTERS MORE FOR bge THAN FOR BLT. A period followed immediately by a
letter is a couple of bytes to a byte-level model. To a SENTENCE SPLITTER it is
a missing boundary, so dreams passages segment into fewer and longer sentences
than they should -- and drift, spread and every trajectory measure are computed
over sentences. That is the same instrument the mixed-stratum `refuse` ruling
was about.

So every record carries `n_runons`, and every record of a corpus above 1.0/1k
carries a `caveat` string. Nothing is excluded here: this file reports the
defect and the analysis decides, because dropping dreams is a call for RH and
one that should be made against a number.
"""
import glob
import gzip
import random
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "data/raw/human_passages.jsonl.gz")

#: the generation-length variant, added 2026-08-14. Model generations run to
#: 256 tokens; the four comparison corpora sit at medians of 136-508 words and
#: the author sets are 100 by construction, so most of the set is shorter than
#: what it is being compared against. 200 WORDS is the threshold, chosen
#: because it sits where 256 tokens does on this text (measured: 2,676 dream
#: reports clear 200 words, 2,325 clear 256 tokens under both OLMo-2 and
#: Llama-3.1 tokenizers, which agree to within one report at every cut) and
#: because it is a round number nobody has to look up.
#:
#: ADDED BESIDE, NEVER REPLACING. Scoring is keyed by text_sha, so extra
#: passages cost compute and cannot disturb anything already scored -- and the
#: original eleven are what F16 used, so comparability with that work survives.
LONG_MIN_WORDS = 200
#: dreams is the only corpus that needs a REDRAW rather than a filter: 67 of
#: its 500 clear 200 words against 2,676 available in the 30,798-row pool.
#: The current sample was drawn under a 100-300 word band, so the long draw
#: comes from the tail that band excluded -- A DIFFERENT SAMPLE, not a longer
#: version of the same one, which is why it is a separate corpus and not a
#: filtered view of dreams.
DREAMS_POOL = "data/dreams.csv"
DREAMS_POOL_COL = "dreams_text"
DREAMS_LONG_N = 500
DREAMS_LONG_SEED = 42

SOURCES = {
    "dreams": ("data/dreams_sample_500_cleaned.csv", "text", "csv"),
    "waking": ("data/hippocorpus_sample_500.csv", "story", "csv"),
    "fiction": ("data/markmark_c20_narration_500.jsonl", "text", "jsonl"),
    "abstracts": ("data/arxiv_abstracts_500.csv", "text", "csv"),
}
#: a run-on: sentence punctuation with no space before the next word. Chosen
#: because it needs no vocabulary -- see the docstring on the three OOV
#: instruments that measured their own referee instead of the text.
RUNON = re.compile(r"[.!?][A-Za-z]")
CJK = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF),
       (0x3000, 0x303F), (0xFF00, 0xFFEF))
CAVEAT_AT = 1.0   # run-ons per 1k words above which a record carries a caveat


def cjk_frac(s):
    if not s:
        return 0.0
    n = sum(1 for ch in s if any(a <= ord(ch) <= b for a, b in CJK))
    return n / len(s)


def load_texts():
    """{corpus: [text, ...]} for all eleven, rebuilt from source."""
    import pandas as pd
    out = {}
    for name, (rel, col, fmt) in SOURCES.items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            raise SystemExit("missing source for %s: %s" % (name, p))
        if fmt == "csv":
            #: keep_default_na=False: these are text corpora and a passage
            #: whose whole content is the word "null" is a passage, not a NaN.
            df = pd.read_csv(p, keep_default_na=False)
            c = col if col in df.columns else "text"
            out[name] = [t for t in df[c].astype(str) if t.strip()]
        else:
            out[name] = [json.loads(l)[col] for l in open(p) if l.strip()]
    #: author sets through the ingest module's OWN slicer, not a copy of it
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from ingest_literary import slice_text
    for variant in ("original", "basic"):
        for path in sorted(glob.glob(os.path.join(ROOT, "data/texts",
                                                  variant, "*.txt"))):
            author = os.path.splitext(os.path.basename(path))[0]
            with open(path) as f:
                out["%s/%s" % (variant, author)] = slice_text(f.read())

    #: ---- the generation-length variant, as separate corpora ----
    for name in ("waking", "fiction", "abstracts"):
        keep = [t for t in out[name] if len(t.split()) >= LONG_MIN_WORDS]
        if keep:
            out["long/%s" % name] = keep
    for variant in ("original", "basic"):
        for path in sorted(glob.glob(os.path.join(ROOT, "data/texts",
                                                  variant, "*.txt"))):
            author = os.path.splitext(os.path.basename(path))[0]
            with open(path) as f:
                w = slice_text(f.read(), window=LONG_MIN_WORDS)
            #: slice_text's last window is short by construction; drop it
            w = [t for t in w if len(t.split()) >= LONG_MIN_WORDS]
            if w:
                out["long/%s/%s" % (variant, author)] = w

    #: dreams: REDRAWN from the pool and cleaned deterministically.
    pool = pd.read_csv(os.path.join(ROOT, DREAMS_POOL), keep_default_na=False)
    cand = [str(t) for t in pool[DREAMS_POOL_COL]
            if len(str(t).split()) >= LONG_MIN_WORDS]
    rng = random.Random(DREAMS_LONG_SEED)
    pick = cand if len(cand) <= DREAMS_LONG_N else rng.sample(cand, DREAMS_LONG_N)
    #: the SAME treatment the existing 500 received, reimplemented
    #: deterministically and validated against them (scripts/clean_dream_text.py
    #: reproduces 149 of the 162 rows that pass actually modified). NOT an LLM
    #: pass: its whole repertoire was ten mechanical operations plus whitespace.
    from clean_dream_text import clean
    out["long/dreams"] = [clean(t)[0] for t in pick]
    print("  long/dreams: %s of %s candidates at >=%dw, cleaned deterministically"
          % (format(len(pick), ","), format(len(cand), ","), LONG_MIN_WORDS))
    return out


def main():
    texts = load_texts()

    #: corpus-level run-on rate first, because the per-record caveat is a
    #: property of the corpus and must not be recomputed per passage.
    rate = {}
    for name, ts in texts.items():
        w = sum(max(1, len(t.split())) for t in ts)
        rate[name] = 1000.0 * sum(len(RUNON.findall(t)) for t in ts) / w

    #: MEMBERSHIP IS COLLECTED BEFORE ANYTHING IS WRITTEN.
    #: The first version wrote a record on first sight and appended later
    #: corpus names to a dict it never re-read, so a text in both `fiction`
    #: and `long/fiction` -- and the filtered long variants are subsets, so
    #: that is 1,016 of them -- recorded only the first label and THE LONG
    #: VIEW COULD NOT BE SELECTED AT ALL. The variant existed and was
    #: invisible. Collect, then write once with every membership.
    member = {}
    text_of = {}
    for name in sorted(texts):
        for t in texts[name]:
            h = hashlib.sha256(t.encode("utf-8")).hexdigest()
            member.setdefault(h, []).append("human/%s" % name)
            text_of[h] = t

    rows = 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for h, corpora in sorted(member.items()):
            t = text_of[h]
            nb = len(t.encode("utf-8"))
            fr = cjk_frac(t)
            #: the run-on rate of every corpus this text belongs to, so a
            #: reader selecting one view gets that view's rate. No threshold
            #: and no flag: the number is on every record and the analysis
            #: decides, rather than a cutoff of mine deciding for it.
            rec = {"prompt": "", "text": t, "corpora": corpora,
                   "corpus": corpora[0], "text_sha": h,
                   "n_bytes": nb, "n_chars": len(t),
                   "n_words": len(t.split()),
                   "n_runons": len(RUNON.findall(t)),
                   "corpus_runons_per_1k": {c: round(rate[c[len("human/"):]], 2)
                                            for c in corpora},
                   "script": "zh" if fr >= 0.5 else ("en" if fr < 0.05 else "mixed")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows += 1

    multi = sum(1 for v in member.values() if len(v) > 1)
    print("wrote %s" % OUT)
    print("  corpora %d | DISTINCT passages %s | in more than one corpus %s"
          % (len(texts), format(rows, ","), format(multi, ",")))
    print("  (the long/* filtered variants are SUBSETS, so their texts are"
          " shared rather than new; every membership is recorded)")
    print("\n  %-40s %6s %8s %9s"
          % ("corpus", "n", "med_w", "runon/1k"))
    for name in sorted(texts):
        ts = texts[name]
        ws = sorted(len(t.split()) for t in ts)
        print("  %-40s %6d %8d %9.1f"
              % ("human/" + name, len(ts), ws[len(ws) // 2], rate[name]))
    print("\n  file on disk: %.1f MiB" % (os.path.getsize(OUT) / 2 ** 20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
