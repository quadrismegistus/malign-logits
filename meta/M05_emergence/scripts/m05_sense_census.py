#!/usr/bin/env python
"""The sense-capacity judging census: every (prompt, word) pair the tier-3
coder will ever be asked about, built reproducibly and bucketed.

    uv run python meta/M05_emergence/scripts/m05_sense_census.py

Design fixed with RH (2026-08-12), each element his: CENSUS, not a sample;
two declared floors — max p >= 0.003 at any rung of either ladder
(CANONICAL's own eligibility constant) plus an early-window top-up at
p >= 0.002 for steps < 2,000 (Pythia) / stage1 < 4,000 (OLMo), where the
colorless-green claim lives; minus the syntax tier's auto-exclusions
(a pair illicit under BOTH coder families is `ungrammatical` by the
paid-for instrument; PUNCT/X/SYM is format). Auto-assigned pairs still
enter the curve — their mass lands in the ungrammatical/format bands from
tier 2 — the coder is just never paid for them.

Writes data/m05_sense_census.parquet: prompt, word, pos_class, floor
(core|early), bucket (JUDGE | ungrammatical_auto | format_auto), with
input shas in the sidecar metadata per [5468]'s rule.
"""
import hashlib
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
OUT = "data/m05_sense_census.parquet"
MODELS = "(model LIKE 'EleutherAI/pythia-6.9b%' OR model LIKE 'allenai/Olmo-3%')"
EARLY = ("(model LIKE 'EleutherAI/pythia-6.9b@step%' AND "
         " toUInt32OrZero(replaceOne(splitByChar('@', model)[2], 'step','')) < 2000) OR "
         "(model LIKE 'allenai/Olmo-3-1025-7B@stage1-step%' AND "
         " toUInt32OrZero(extract(model, 'stage1-step(\\\\d+)')) < 4000)")
EQUIV = [{"ADP", "PART"}, {"NUM", "NOUN"}, {"AUX", "VERB"}]
FORMAT_BAND = {"PUNCT", "X", "SYM"}


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def unescape(s):
    for e, c in (("\\t", "\t"), ("\\n", "\n"), ("\\'", "'"), ("\\\\", "\\")):
        s = s.replace(e, c)
    return s


def battery_texts():
    b = json.load(open("data/m05_battery.json"))
    texts = []
    for blk in b["blocks"].values():
        for t in blk["texts"]:
            texts.append(t if isinstance(t, str) else
                         t.get("text", t.get("prompt")))
    return list(dict.fromkeys(texts))


def ch_pairs(q):
    r = subprocess.run([CH, "client", "--query", q + " FORMAT TSV"],
                       capture_output=True, text=True, check=True)
    out = []
    for line in r.stdout.splitlines():
        p, _, w = line.partition("\t")
        out.append((unescape(p), unescape(w)))
    return out


def expand(s):
    s = set(s)
    for g in EQUIV:
        if s & g:
            s |= g
    return s


def main():
    texts = battery_texts()
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    inlist = ",".join(f"'{esc(t)}'" for t in texts)

    core = ch_pairs(
        f"SELECT prompt, word FROM malign_logits.twp_words "
        f"WHERE {MODELS} AND abs(theta-0.001)<1e-9 AND prompt IN ({inlist}) "
        f"GROUP BY prompt, word HAVING max(p) >= 0.003")
    early = ch_pairs(
        f"WITH gk AS (SELECT prompt, word FROM malign_logits.twp_words "
        f"  WHERE {MODELS} AND abs(theta-0.001)<1e-9 AND prompt IN ({inlist}) "
        f"  GROUP BY prompt, word HAVING max(p) >= 0.003) "
        f"SELECT prompt, word FROM malign_logits.twp_words "
        f"WHERE ({EARLY}) AND abs(theta-0.001)<1e-9 AND prompt IN ({inlist}) "
        f"  AND (prompt, word) NOT IN gk "
        f"GROUP BY prompt, word HAVING max(p) >= 0.002")
    df = pd.concat([
        pd.DataFrame(core, columns=["prompt", "word"]).assign(floor="core"),
        pd.DataFrame(early, columns=["prompt", "word"]).assign(floor="early"),
    ], ignore_index=True)
    print(f"core {len(core)}, early top-up {len(early)}, total {len(df)}")

    tags = pd.read_parquet("data/m05_syntax_tags.parquet")[
        ["prompt", "word", "pos_class"]]
    df = df.merge(tags, on=["prompt", "word"], how="left")
    assert df.pos_class.notna().all(), "untagged pairs in census"

    lic_d = {p: expand({w["pos"] for w in v["licit"]})
             for p, v in json.load(open("data/m05_licit_sets.json"))
             ["prompts"].items()}
    lic_h = {p: expand({w["pos"] for w in v["licit"]})
             for p, v in json.load(open("data/m05_licit_sets_haiku.json"))
             ["prompts"].items()}

    def bucket(r):
        if r.pos_class in FORMAT_BAND:
            return "format_auto"
        if (r.pos_class in lic_d.get(r.prompt, set())
                or r.pos_class in lic_h.get(r.prompt, set())):
            return "JUDGE"
        return "ungrammatical_auto"

    df["bucket"] = df.apply(bucket, axis=1)
    print(df.bucket.value_counts().to_string())

    df.attrs = {}
    meta = dict(
        _invocation="m05_sense_census.py (no flags)",
        _inputs={
            "data/m05_syntax_tags.parquet": sha16("data/m05_syntax_tags.parquet"),
            "data/m05_licit_sets.json": sha16("data/m05_licit_sets.json"),
            "data/m05_licit_sets_haiku.json": sha16("data/m05_licit_sets_haiku.json"),
            "data/m05_battery.json": sha16("data/m05_battery.json"),
        },
        _floors="core: max p >= 0.003 any rung either ladder; early: "
                "p >= 0.002 at Pythia step<2000 / OLMo stage1 step<4000",
        _rule="auto-exclude only pairs illicit under BOTH coder families; "
              "format band PUNCT/X/SYM auto",
    )
    df.to_parquet(OUT)
    json.dump(meta, open(OUT + ".meta.json", "w"), indent=1)
    print(f"wrote {OUT} ({len(df)} rows, sha {sha16(OUT)}) + sidecar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
