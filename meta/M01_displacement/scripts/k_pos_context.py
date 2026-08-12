"""In-context POS for every (prompt, candidate word) pair, cached.

    uv run python meta/M01_displacement/scripts/k_pos_context.py en
    -> results/k/pos_context_<lang>.tsv     prompt_sha16 \\t word \\t tag \\t coarse

WHY THIS EXISTS. `fields._byu()` returns the MOST FREQUENT reading of a word
form, out of context, so the POS decomposition in `k_armclf_bands` was not
measuring POS. Its "noun" band 0-50 contains `fall break kiss punch strike stroke
touch change work sign dance tear phone love`, which are verbs at sites like
"She began to ___", plus the gerunds `turning walking`. Its "adjective" band
contains `twisted sliced crumpled burnt chopped fixed shared extended burning
passing answering continuing` -- participles, not adjectives. Only the adverb
band was clean.

That defect is invisible from the aggregate: the noun band classified at AUC
0.922 on 0.35 coverage, which reads as a surprising finding rather than as
mislabelled verbs. It surfaced by RH reading the word list.

WHAT THIS TAGS. Every distinct (prompt, word) pair where the word appears in some
model's top-20 at that prompt: 365,892 pairs over 2,229 prompts and 21,871 word
types. The text tagged is the prompt with the candidate appended, and the tag
taken is the LAST token's -- which is the position the model was predicting.

THE SAME WORD GETS DIFFERENT TAGS AT DIFFERENT PROMPTS, which is the entire
point. `kiss` is a verb after "She began to" and a noun after "He gave her a".
Caching by word alone would rebuild the defect being fixed, so the key is the
PAIR.

PROMPTS ARE KEYED BY sha16 OF THEIR TEXT, not by prompt_id. The campaign's
standing rule is that prompt ids are not trustworthy across tables; the text is
the join key everywhere else in K and it is the join key here.

RESUMABLE. The TSV is appended and flushed per batch; on start the existing pairs
are read and skipped, so an interrupted run costs only what it had not written.
"""
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import k_analysis as A

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
TOPN = 20
BATCH = 2000

#: PTB tags collapsed to the four content classes plus a residue. The verb series
#: includes participles and gerunds (VBG, VBN) BECAUSE THAT IS THE POINT -- BYU
#: files those under adjective and noun respectively.
COARSE = {}
for t in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"):
    COARSE[t] = "verb"
for t in ("NN", "NNS", "NNP", "NNPS"):
    COARSE[t] = "noun"
for t in ("JJ", "JJR", "JJS"):
    COARSE[t] = "adjective"
for t in ("RB", "RBR", "RBS", "WRB"):
    COARSE[t] = "adverb"


def sha16(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main(lang="en"):
    import spacy
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x

    out = os.path.join(K, "pos_context_%s.tsv" % lang)
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            for ln in fh:
                p = ln.rstrip("\n").split("\t")
                if len(p) >= 2:
                    done.add((p[0], p[1]))
        print("resuming: %d pairs already cached" % len(done))

    print("fetching (prompt, word) pairs in any model's top-%d ..." % TOPN, flush=True)
    #: FINAL AND A COLLAPSE TO (model, prompt, word), because `source` -- the
    #: ingest batch -- is in the sorting key and `FINAL` therefore leaves the
    #: analysis unit undeduplicated. Without it a repeated word takes two of the
    #: twenty slots and the selection is not a top-20 at all: 17.76% of cells
    #: held fewer than 20 distinct words. See [5657]/[5659].
    #:
    #: THE CACHE BUILT BEFORE THIS FIX IS NOT WRONG, IT IS SHORT. Its 365,892
    #: pairs are all real (prompt, word) pairs and their tags stand; what it
    #: lacks is the 12,850 pairs that dedup PROMOTES into the top-20. Those are
    #: exactly the words any re-run needs, so a consumer reading the old cache
    #: silently drops the words the fix surfaced. Resumability makes this cheap:
    #: existing pairs are skipped and only the promoted ones are tagged.
    rows = A.q("""
      SELECT prompt, word FROM (
        SELECT prompt, word,
               row_number() OVER (PARTITION BY model, prompt ORDER BY p DESC) rk
        FROM (SELECT model, prompt, word, avg(p) p
              FROM %s.twp_words FINAL
              WHERE prompt IN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                               WHERE status='ACTIVE' AND language='%s')
              GROUP BY model, prompt, word))
      WHERE rk <= %d
      GROUP BY prompt, word""" % (A.DB, A.DB, lang, TOPN))
    byp = collections.defaultdict(list)
    for r in rows:
        h = sha16(r["prompt"])
        if (h, r["word"]) not in done:
            byp[r["prompt"]].append(r["word"])
    todo = sum(len(v) for v in byp.values())
    print("  %s pairs over %d prompts, %s still to tag"
          % (f"{len(rows):,}", len({r['prompt'] for r in rows}), f"{todo:,}"), flush=True)
    if not todo:
        print("  nothing to do"); return 0

    #: only the tagger is needed, and the pipeline is a third of the cost without
    #: the parser and NER
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "lemmatizer",
                                                "attribute_ruler", "senter"])
    print("  spacy pipeline: %s" % ", ".join(nlp.pipe_names), flush=True)

    texts, keys = [], []
    for prompt, words in byp.items():
        h = sha16(prompt)
        base = prompt if prompt.endswith((" ", "\n")) else prompt + " "
        for w in words:
            texts.append(base + w.strip())
            keys.append((h, w))

    fh = open(out, "a", encoding="utf-8")
    n_by = collections.Counter()
    with tqdm(total=len(texts), unit="pair", desc="tagging") as bar:
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            kk = keys[i:i + BATCH]
            for (h, w), doc in zip(kk, nlp.pipe(chunk, batch_size=256)):
                #: the LAST token is the predicted position. A word that spacy
                #: splits into several tokens is read at its final piece, which
                #: is the head for the hyphenated and clitic cases that occur.
                tag = doc[-1].tag_ if len(doc) else "XX"
                c = COARSE.get(tag, "other")
                n_by[c] += 1
                fh.write("%s\t%s\t%s\t%s\n" % (h, w, tag, c))
            fh.flush()
            bar.update(len(chunk))
    fh.close()

    print("\n  tagged %s pairs" % f"{sum(n_by.values()):,}")
    for c, n in n_by.most_common():
        print("    %-11s %8s  %5.1f%%" % (c, f"{n:,}", 100 * n / sum(n_by.values())))
    print("\n  -> %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en"))
