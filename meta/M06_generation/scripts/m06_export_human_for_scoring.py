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
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "data/raw/human_passages.jsonl.gz")

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
    return out


def main():
    texts = load_texts()

    #: corpus-level run-on rate first, because the per-record caveat is a
    #: property of the corpus and must not be recomputed per passage.
    rate = {}
    for name, ts in texts.items():
        w = sum(max(1, len(t.split())) for t in ts)
        rate[name] = 1000.0 * sum(len(RUNON.findall(t)) for t in ts) / w

    seen, rows = {}, 0
    dropped_dup = 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for name in sorted(texts):
            for t in texts[name]:
                #: DEDUPLICATE ON THE TEXT, as the BLT export does, because
                #: surprisal is a property of the string. `corpora` would carry
                #: the provenance of a collision, but there are none across
                #: these eleven; asserted below rather than assumed.
                h = hashlib.sha256(t.encode("utf-8")).hexdigest()
                if h in seen:
                    seen[h].append(name)
                    dropped_dup += 1
                    continue
                seen[h] = [name]
                nb = len(t.encode("utf-8"))
                fr = cjk_frac(t)
                rec = {"prompt": "", "text": t, "corpora": ["human/%s" % name],
                       "corpus": "human/%s" % name, "text_sha": h,
                       "n_bytes": nb, "n_chars": len(t),
                       "n_words": len(t.split()),
                       "n_runons": len(RUNON.findall(t)),
                       "script": "zh" if fr >= 0.5 else ("en" if fr < 0.05 else "mixed")}
                if rate[name] > CAVEAT_AT:
                    rec["caveat"] = (
                        "%s carries %.1f run-ons per 1k words ([.!?] followed "
                        "directly by a letter). Sentence splitting under-segments "
                        "here, so any per-sentence measure (drift, spread, "
                        "trajectory) is affected; byte-level surprisal is not. "
                        "For dreams the _cleaned pass did NOT address this: 643 "
                        "run-ons before and after." % (name, rate[name]))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows += 1

    print("wrote %s" % OUT)
    print("  corpora %d | passages %s | cross-corpus duplicate texts %d"
          % (len(texts), format(rows, ","), dropped_dup))
    print("\n  %-22s %6s %8s %9s %10s"
          % ("corpus", "n", "med_w", "runon/1k", "caveat"))
    for name in sorted(texts, key=lambda k: -rate[k]):
        ts = texts[name]
        ws = sorted(len(t.split()) for t in ts)
        print("  %-22s %6d %8d %9.1f %10s"
              % ("human/" + name, len(ts), ws[len(ws) // 2], rate[name],
                 "YES" if rate[name] > CAVEAT_AT else ""))
    print("\n  file on disk: %.1f MiB" % (os.path.getsize(OUT) / 2 ** 20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
