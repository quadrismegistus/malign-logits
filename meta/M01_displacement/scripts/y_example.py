#!/usr/bin/env python
"""Pull example passages out of the Y corpus by any combination of filters.

    y_example.py --tag guilt --arm aligned --n 3
    y_example.py --field SUPEREGO_IN_SCENE --word shoes --prompt liminal
    y_example.py --field sexual_scene=YES --field guilt_or_shame=YES --full
    y_example.py --tag guilt --arm aligned --tokens        # per-token surprisal
    y_example.py --list-fields                             # what you can filter on

## WHY THIS EXISTS

`Y_examples.md` holds examples; it does not hold the means of getting more. The
knowledge of where the raw generations live, that `plen` locates the forced word,
which fields are composites and how a coded span maps to tokens was carried in a
chat transcript, which is exactly the condition that made `Y_superego.md` §6's
numbers unrecoverable. Writing prose from this corpus needs examples on demand,
so the retrieval is a script.

## WHERE THE DATA IS, since finding it took a while

    meta/M01_displacement/results/y_confirmatory_coded.jsonl   the coding, 41,596
                                                               pass-A rows. Has
                                                               `tagged` but NO ids.
    data/raw/y_y-00 .. y_y-08 / y__<model>.jsonl               the generations, 53
                                                               files. full_ids,
                                                               plen, tokens, text,
                                                               scored_by_base,
                                                               scored_by_aligned.

The producer wrote `y__<model>.jsonl` into a box path (`vllm_y_run.py:203`), and
the rsync landed them under `data/raw/y_y-*/`. Searching for the producer's
filename finds nothing; the consumer (`y_run_manifest.py:74`) is what names the
real location.

**THE FORCED WORD SITS AT `full_ids[plen - len(word_ids) : plen]`.** `plen`
counts prompt + forced word, and the score arrays are the CONTINUATION ONLY:
`len(scored_by_*) == len(tokens)` and `len(full_ids) == plen + len(tokens)`,
verified on 10,200 sequences. Do not slice the scores by `plen`.

## SPAN LOCATION

Parse the span text out of `tagged` with lxml, re-encode it with the model's own
tokeniser, and find that id sequence in `tokens`. The match IS the alignment.
Locates 85.1% of spans, 0 ambiguous. Do not map characters to token offsets --
the coder's reproduction drifts from the source (`rt_band`) and the repo's
`tok_char_offsets` rebuilds text in a way `cache.py` warns against.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

COMPOSITES = ("SUPEREGO_IN_SCENE", "CLEAN_SCENE", "EXIT", "MORAL_UTTERED")
CODED = os.path.join(CAMP, "results", "y_confirmatory_coded.jsonl")


def load_coded():
    return [json.loads(l) for l in open(CODED)]


def raw_index(models=None):
    """(pair, role, prompt_id, word, seq_i) -> (model, sequence dict)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*", "*.jsonl"))):
        if "FAILED" in os.path.basename(f):
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if models and r.get("model") not in models:
                continue
            for i, s in enumerate(r.get("sequences") or []):
                out[(r.get("pair"), r.get("role"), r.get("prompt_id"),
                     r.get("word"), i)] = (r.get("model"), s)
    return out


def match(r, a):
    if a.arm and r.get("role") != a.arm:
        return False
    if a.word and str(r.get("word")) != a.word:
        return False
    if a.prompt and a.prompt not in (r.get("prompt_id") or ""):
        return False
    if a.model and a.model.lower() not in (r.get("model") or "").lower():
        return False
    if a.tag and ("<%s>" % a.tag) not in (r.get("tagged") or ""):
        return False
    for f in a.field or []:
        k, _, want = f.partition("=")
        v = r.get(k)
        if k in COMPOSITES:
            if bool(v) is not (want.upper() != "NO" if want else True):
                return False
        elif v != (want.upper() or "YES"):
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", help="story refusal noise meta web sexual moral guilt consent resist")
    p.add_argument("--arm", choices=["base", "aligned"])
    p.add_argument("--word", help="forced word, or 'None' for undisturbed")
    p.add_argument("--prompt", help="substring, e.g. liminal or explicit_1")
    p.add_argument("--model", help="substring")
    p.add_argument("--field", action="append",
                   help="FIELD or FIELD=YES/NO, repeatable. e.g. --field guilt_or_shame")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--chars", type=int, default=460)
    p.add_argument("--full", action="store_true", help="whole passage, untruncated")
    p.add_argument("--tokens", action="store_true",
                   help="per-token base/aligned surprisal around the tag's span")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list-fields", action="store_true")
    a = p.parse_args()

    rows = load_coded()
    if a.list_fields:
        keys = sorted(rows[0].keys())
        print("FILTERABLE FIELDS  (--field NAME  or  --field NAME=NO)")
        for k in keys:
            v = rows[0][k]
            kind = "composite bool" if k in COMPOSITES else type(v).__name__
            print("   %-24s %s" % (k, kind))
        return 0

    ok = [r for r in rows if r.get("pass") == "A" and r.get("parsed") and match(r, a)]
    print("%s passages match" % format(len(ok), ","))
    if not ok:
        return 1
    random.Random(a.seed).shuffle(ok)
    sel = ok[:a.n]

    idx = raw_index({r["model"] for r in sel}) if a.tokens else {}
    for r in sel:
        print("=" * 96)
        print("  %s | %s | prompt=%s | word=%r" % (r["role"], r["model"], r["prompt_id"], r.get("word")))
        flags = " ".join("%s=%s" % (k, r.get(k)) for k in
                         ("sexual_scene", "guilt_or_shame", "consent_hesitation",
                          "moralisation_in_scene", "assistant_refusal", "frame_exit")
                         if r.get(k) == "YES")
        comp = " ".join(k for k in COMPOSITES if r.get(k))
        print("  coded: %s %s | rt_band=%s | mid=%s" % (flags or "-", comp, r.get("rt_band"), r.get("mid")))
        txt = re.sub(r"</?[a-z_]+>", "", r.get("tagged") or "").strip()
        print(textwrap.fill(txt if a.full else txt[:a.chars], 96,
                            initial_indent="    ", subsequent_indent="    "))
        if a.tokens and a.tag:
            show_tokens(r, idx, a.tag)
        print()
    return 0


def show_tokens(r, idx, tag):
    from lxml import etree
    from transformers import AutoTokenizer
    k = (r["pair"], r["role"], r["prompt_id"], r.get("word"), r["seq_i"])
    if k not in idx:
        print("    (no scored sequence for this row)")
        return
    model, s = idx[k]
    toks, b, al = s["tokens"], s["scored_by_base"], s["scored_by_aligned"]
    root = etree.fromstring("<r>" + (r.get("tagged") or "") + "</r>",
                            etree.XMLParser(recover=True))
    T = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    for el in root.iter(tag):
        t = "".join(el.itertext()).strip()
        if len(t) < 12:
            continue
        for cand in (" " + t, t):
            ids = T(cand, add_special_tokens=False)["input_ids"]
            hit = [i for i in range(len(toks) - len(ids) + 1)
                   if toks[i:i + len(ids)] == ids]
            if hit:
                i0, n = hit[0], len(ids)
                print("    <%s> at tokens [%d,%d)   base | aligned surprisal" % (tag, i0, i0 + n))
                for j in range(max(0, i0 - 4), min(len(toks), i0 + n + 4)):
                    mark = "|" if i0 <= j < i0 + n else " "
                    print("    %s %4d %-16s %7.2f %7.2f"
                          % (mark, j, repr(T.decode([toks[j]]))[:16], -b[j], -al[j]))
                return
    print("    (span not locatable in the token stream -- ~15%% are not)")


if __name__ == "__main__":
    sys.exit(main())
