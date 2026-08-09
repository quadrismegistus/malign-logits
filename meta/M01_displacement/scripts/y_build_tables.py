#!/usr/bin/env python
"""Build the two Y analysis tables. Everything downstream is a groupby on these.

    python y_build_tables.py              # ~20 minutes, both tables
    python y_build_tables.py --files 4    # subsample, for a smoke

    results/y_tokens/*.parquet    one shard per raw file, ~10.6M rows total
    results/y_passages.parquet    41,596 rows

Both are gitignored: rebuildable from `data/raw/y_y-*/` plus
`results/y_confirmatory_coded.jsonl`, and a few hundred MB.

## WHY THIS EXISTS

Every Y analysis was re-deriving the same four joins -- load the raw
generations, load the coding, match on the 5-tuple key, load 64 tokenisers,
relocate every span -- and paying twenty minutes before it could answer
anything. Six separate analyses in one session would each have been one groupby
on this table.

**The stronger reason is not speed.** Five scripts were locating spans
independently, each with its own MIN_CHARS, its own leading-space handling, its
own tolerance. They can silently disagree about which tokens a span covers, and
three mutually incompatible estimators in one session came out of exactly that.
Locating ONCE and storing `l2_span_id` means every downstream analysis is
provably talking about the same spans.

## SPAN LOCATION

Parse the span text out of `tagged` with lxml, re-encode it with the model's own
tokeniser, find that id sequence in `tokens`. **The match IS the alignment** --
no character offsets, no drift bands, no tolerance to choose. A span that cannot
be found excludes itself, and the count is recorded per passage in
`spans_total` / `spans_located` so coverage is a column rather than something
rediscovered later. Located rate is about 85%.

## THE TWO LAYERS ARE SEPARATE COLUMNS

Layer 2 nests inside layer 1 by design, so a token is routinely in both
`<story>` and `<guilt>`. One `span` column cannot represent that. `layer1` and
`layer2` are independent; `l2_span_id` and `l2_pos` identify the layer-2
occurrence and the token's position within it, which is what per-span extrema
and within-span profiles need.

## WHAT IS NOT STORED

`surprisal_diff`. It is `aligned_surprisal - base_surprisal`, one expression,
and a stored derived column is how a table acquires two sources of truth when
somebody recomputes one and not the other.

## THE INVARIANT THAT MUST HOLD

`len(scored_by_base) == len(scored_by_aligned) == len(tokens)`, and
`len(full_ids) == plen + len(tokens)`. Verified on 10,200 sequences. The score
arrays are the CONTINUATION ONLY -- do not slice them by `plen`, which is the
defect that invalidated the first version of the span-surprisal analysis.
Sequences failing the invariant are dropped and counted.
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

LAYER1 = ("story", "refusal", "noise", "meta", "web")
LAYER2 = ("sexual", "moral", "guilt", "consent", "resist")
COMPOSITES = ("SUPEREGO_IN_SCENE", "CLEAN_SCENE", "EXIT", "MORAL_UTTERED")
FIELDS = ("sexual_scene", "consummation", "guilt_or_shame", "moralisation_in_scene",
          "consent_hesitation", "assistant_refusal", "frame_exit", "noise_present",
          "continues_narrative", "degenerate")
MIN_CHARS = 12


def find_all(hay, needle):
    n = len(needle)
    return [i for i in range(len(hay) - n + 1) if hay[i:i + n] == needle] if n and n <= len(hay) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=0)
    a = ap.parse_args()

    import pandas as pd
    from lxml import etree
    from transformers import AutoTokenizer

    outdir = os.path.join(CAMP, "results", "y_tokens")
    os.makedirs(outdir, exist_ok=True)

    coded = {}
    for l in open(os.path.join(CAMP, "results", "y_confirmatory_coded.jsonl")):
        r = json.loads(l)
        coded[(r["pair"], r["role"], r["prompt_id"], r.get("word"), r["seq_i"])] = r
    print("coded rows: %s" % format(len(coded), ","))

    files = [x for x in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*", "*.jsonl")))
             if "FAILED" not in os.path.basename(x)]
    if a.files:
        files = files[:a.files]

    P = etree.XMLParser(recover=True)
    passages, led = [], collections.Counter()
    span_uid = 0
    ntok = 0

    for fi, f in enumerate(files):
        recs = []
        for line in open(f):
            try:
                recs.append(json.loads(line))
            except Exception:
                led["unreadable line"] += 1
        models = {r.get("model") for r in recs}
        toks_cache = {}
        for m in models:
            try:
                toks_cache[m] = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
            except Exception:
                toks_cache[m] = None
                led["tokenizer unavailable"] += 1
        rows = collections.defaultdict(list)
        for r in recs:
            model = r.get("model")
            T = toks_cache.get(model)
            if T is None:
                continue
            enc = {}
            for i, s in enumerate(r.get("sequences") or []):
                key = (r.get("pair"), r.get("role"), r.get("prompt_id"), r.get("word"), i)
                cr = coded.get(key)
                if cr is None or cr.get("pass") != "A" or not cr.get("parsed"):
                    continue
                b, al, tk = s.get("scored_by_base"), s.get("scored_by_aligned"), s.get("tokens")
                if not (b and al and tk) or not (len(b) == len(al) == len(tk)):
                    led["invariant failed"] += 1
                    continue
                fi_ = s.get("full_ids") or []
                if fi_ and len(fi_) != (s.get("plen") or 0) + len(tk):
                    led["full_ids != plen + tokens"] += 1
                    continue
                root = etree.fromstring("<r>" + (cr.get("tagged") or "") + "</r>", P)
                if root is None:
                    led["unparseable tagged"] += 1
                    continue
                n = len(tk)
                l1 = [None] * n
                l2 = [None] * n
                sid = [-1] * n
                spos = [-1] * n
                stot = sloc = 0
                for tag in LAYER1 + LAYER2:
                    for el in root.iter(tag):
                        t = "".join(el.itertext()).strip()
                        if len(t) < MIN_CHARS:
                            continue
                        stot += 1
                        hit = None
                        for cand in (" " + t, t):
                            ids = enc.get(cand)
                            if ids is None:
                                ids = tuple(T(cand, add_special_tokens=False)["input_ids"])
                                enc[cand] = ids
                            h = find_all(tk, list(ids))
                            if h:
                                hit = (h[0], len(ids))
                                break
                        if hit is None:
                            continue
                        sloc += 1
                        i0, ln = hit
                        if tag in LAYER1:
                            for j in range(i0, min(i0 + ln, n)):
                                l1[j] = tag
                        else:
                            span_uid += 1
                            for j in range(i0, min(i0 + ln, n)):
                                l2[j] = tag
                                sid[j] = span_uid
                                spos[j] = j - i0
                mid = cr.get("mid")
                for j in range(n):
                    rows["mid"].append(mid)
                    rows["token_num"].append(j)
                    rows["token_id"].append(tk[j])
                    rows["token"].append(T.decode([tk[j]]))
                    rows["layer1"].append(l1[j])
                    rows["layer2"].append(l2[j])
                    rows["l2_span_id"].append(sid[j])
                    rows["l2_pos"].append(spos[j])
                    rows["base_surprisal"].append(-b[j])
                    rows["aligned_surprisal"].append(-al[j])
                ntok += n
                p = {"mid": mid, "pair": r.get("pair"), "prompt_id": r.get("prompt_id"),
                     "model": model, "arm": r.get("role"), "forced_word": r.get("word"),
                     "seq_i": i, "num_tokens": n, "rt_band": cr.get("rt_band"),
                     "spans_total": stot, "spans_located": sloc}
                for k in FIELDS:
                    p[k] = cr.get(k)
                for k in COMPOSITES:
                    p[k] = bool(cr.get(k))
                passages.append(p)
        if rows:
            D = pd.DataFrame(rows)
            for c in ("token", "layer1", "layer2"):
                D[c] = D[c].astype("category")
            D["token_num"] = D.token_num.astype("int16")
            D["l2_pos"] = D.l2_pos.astype("int16")
            for c in ("base_surprisal", "aligned_surprisal"):
                D[c] = D[c].astype("float32")
            D.to_parquet(os.path.join(outdir, "%03d.parquet" % fi), index=False)
        print("  [%2d/%d] %-46s %s tokens" % (fi + 1, len(files), os.path.basename(f),
                                              format(ntok, ",")), flush=True)

    Pdf = pd.DataFrame(passages)
    for c in ("pair", "prompt_id", "model", "arm", "forced_word", "rt_band") + FIELDS:
        if c in Pdf:
            Pdf[c] = Pdf[c].astype("category")
    pp = os.path.join(CAMP, "results", "y_passages.parquet")
    Pdf.to_parquet(pp, index=False)

    print("\nTOKENS   %s rows across %d shards -> %s"
          % (format(ntok, ","), len(files), os.path.relpath(outdir, ROOT)))
    print("PASSAGES %s rows, %d cols -> %s"
          % (format(len(Pdf), ","), len(Pdf.columns), os.path.relpath(pp, ROOT)))
    print("  spans located %s of %s (%.1f%%)"
          % (format(int(Pdf.spans_located.sum()), ","), format(int(Pdf.spans_total.sum()), ","),
             100 * Pdf.spans_located.sum() / max(Pdf.spans_total.sum(), 1)))
    if led:
        print("  dropped:")
        for k, v in led.most_common():
            print("     %-28s %s" % (k, format(v, ",")))
    print("\n  read with:  pd.read_parquet('%s')" % os.path.relpath(outdir, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
