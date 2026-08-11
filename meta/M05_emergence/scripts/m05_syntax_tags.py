#!/usr/bin/env python
"""The syntax artifact, tier 1: spaCy-tag every unique (prompt, word) pair
on the M05 battery, both ladders, in context.

    uv run python meta/M05_emergence/scripts/m05_syntax_tags.py

RH's word, 2026-08-11 ("start by spacy-tagging all of the completions").
The dedup object: the 13.8M (checkpoint x prompt x word) battery rows across
the OLMo-3 and Pythia ladders collapse to ~338k unique (prompt, word) pairs
once the checkpoint dimension is dropped — the tag of a word as the
continuation of a prompt does not depend on which rung proposed it, which
is also what keeps any later judge blind to training stage by construction.

Each pair is tagged by parsing "<prompt> <word>" and reading the tokens
that span the appended word: UPOS, fine tag, dependency relation, and the
head's text. Deterministic (spaCy en_core_web_sm, version recorded in the
parquet), local, free; re-runnable byte-identically. This artifact does NOT
decide licitness — that needs the licit-category set (registered secondary
5's frozen artifact, LLM-coded per prompt, pending RH's word) or an
endpoint-derived set. It supplies the POS layer both would consume, and
already supports the POS-composition-over-training curve on its own.

Writes data/m05_syntax_tags.parquet:
    prompt, word, n_tokens (spaCy tokens spanning the word),
    upos, tag, dep, head, pos_class, spacy_model, spacy_version

KNOWN ARTIFACT, and why pos_class exists: a determiner with no following
noun (the battery's prompts end mid-phrase, so trailing "the"/"a" is a
common candidate) keeps the correct fine tag DT but UPOS-maps to PRON.
`pos_class` re-derives the coarse class from the PTB fine tag by an
explicit table and is the column the composition curve should use; `upos`
is kept as spaCy emitted it.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
DB = os.environ.get("MALIGN_CH_DB", "malign_logits")
OUT = "data/m05_syntax_tags.parquet"
MODELS = "(model LIKE 'EleutherAI/pythia-6.9b%' OR model LIKE 'allenai/Olmo-3%')"

#: explicit PTB fine tag -> coarse class; the artifact's own mapping, so
#: the truncated-context UPOS distortion (DT -> PRON) cannot reach the curve.
POS_CLASS = {
    "DT": "DET", "PDT": "DET", "WDT": "DET",
    "NN": "NOUN", "NNS": "NOUN",
    "NNP": "PROPN", "NNPS": "PROPN",
    #: VB*/MD deliberately ABSENT: verb fine tags defer to spaCy's UPOS,
    #: which distinguishes AUX from VERB contextually ("is being" vs
    #: "being difficult"). Forcing VB* -> VERB destroyed that distinction
    #: and cost 3 witness disagreements in the first smoke.
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV", "WRB": "ADV",
    "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "WP$": "PRON",
    "EX": "PRON",
    "IN": "ADP", "CC": "CCONJ", "CD": "NUM", "UH": "INTJ",
    "RP": "PART", "TO": "PART", "POS": "PART",
    "FW": "X", "LS": "X", "XX": "X", "ADD": "X", "NIL": "X", "GW": "X",
    "SYM": "SYM", "$": "SYM", "#": "SYM",
}


def pos_class(tag, upos):
    if tag in POS_CLASS:
        return POS_CLASS[tag]
    if tag in (".", ",", ":", "``", "''", "-LRB-", "-RRB-", "HYPH",
               "NFP", "_SP"):
        return "PUNCT"
    return upos  # fall back to spaCy's own mapping for anything else


def battery_texts():
    b = json.load(open("data/m05_battery.json"))
    texts = []
    for blk in b["blocks"].values():
        for t in blk["texts"]:
            texts.append(t if isinstance(t, str) else
                         t.get("text", t.get("prompt")))
    return list(dict.fromkeys(texts))


def unique_pairs(texts):
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    inlist = ",".join(f"'{esc(t)}'" for t in texts)
    q = (f"SELECT DISTINCT prompt, word FROM {DB}.twp_words "
         f"WHERE {MODELS} AND abs(theta - 0.001) < 1e-9 "
         f"AND prompt IN ({inlist}) FORMAT TSV")
    r = subprocess.run([CH, "client", "--query", q],
                       capture_output=True, text=True, check=True)
    pairs = []
    for line in r.stdout.splitlines():
        p, _, w = line.partition("\t")
        # TSV escapes from clickhouse
        #: ClickHouse TSV escapes apostrophes too (`Arendt\'s`) -- missing
        #: that unescape poisoned 46 prompts in the first build (8% of all
        #: mass read UNTAGGED, flat across rungs). ch_read's own docstring
        #: warns about exactly this string (`didn\'t`).
        for esc_seq, ch in (("\\t", "\t"), ("\\n", "\n"), ("\\'", "'"),
                            ("\\\\", "\\")):
            p = p.replace(esc_seq, ch)
            w = w.replace(esc_seq, ch)
        pairs.append((p, w))
    return pairs


def main():
    import pandas as pd
    import spacy

    texts = battery_texts()
    pairs = unique_pairs(texts)
    print(f"{len(texts)} battery prompts, {len(pairs)} unique "
          f"(prompt, word) pairs to tag")

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    docs_in = [f"{p} {w}" for p, w in pairs]
    rows = []
    for (p, w), doc in zip(pairs,
                           nlp.pipe(docs_in, batch_size=512)):
        start = len(p) + 1  # the appended word begins after "<prompt> "
        toks = [t for t in doc if t.idx >= start]
        if not toks:  # word merged into the prompt's last token (rare)
            toks = [doc[-1]]
        t0 = toks[0]
        rows.append(dict(
            prompt=p, word=w, n_tokens=len(toks),
            upos=t0.pos_, tag=t0.tag_, dep=t0.dep_,
            head=t0.head.text, pos_class=pos_class(t0.tag_, t0.pos_)))
        if len(rows) % 50000 == 0:
            print(f"  {len(rows)}/{len(pairs)}", flush=True)
    df = pd.DataFrame(rows)
    df["spacy_model"] = "en_core_web_sm"
    df["spacy_version"] = spacy.__version__
    df.to_parquet(OUT)
    print(f"wrote {OUT}: {len(df)} rows")
    print("\npos_class composition of the unique-pair vocabulary:")
    print(df.pos_class.value_counts().to_string())
    dis = (df.pos_class != df.upos).mean()
    print(f"\npos_class differs from spaCy UPOS on {dis:.1%} of pairs "
          "(mostly truncated-context determiners)")
    print("\nmulti-token words (spaCy split the appended word):",
          (df.n_tokens > 1).mean().round(4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
