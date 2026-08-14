#!/usr/bin/env python3
"""Is a candidate prompt REDUNDANT with the items we already have?

    x_slot_novelty.py "He slipped something into her"
    x_slot_novelty.py --corpus a.yaml,b.yaml "prompt"   [--top 5]

Prints the nearest existing items by SLOT COSINE and a verdict.

WHY SLOT COSINE AND NOT PROMPT TEXT. Two prompts can read completely
differently and be one measurement: if `She unzipped his ___` and
`He unzipped her ___` put mass on the same words in the same proportions, the
pair measures one thing twice. Conversely two near-identical sentences with
different slots are not redundant at all. The distribution is the measurement,
so the distribution is what gets compared.

MEASURED ON THE EXISTING 51 ITEMS, 1,275 pairs:
    > 0.85    6 pairs, and FIVE OF THEM ARE DELIBERATE GENDER PAIRS -- which
              are not redundancy but the design, since a controlled pair should
              differ only in gender and therefore SHOULD have near-identical
              slots.
    the real redundancy was a cluster of four -- reached for his / squeezed her
              / grabbed his / grabbed her -- at 0.83-0.95 with pole-Jaccard
              0.18-0.22. LOW POLE OVERLAP WITH HIGH SLOT COSINE IS THE
              SIGNATURE: different words tagged, same distribution underneath.

So the threshold here is 0.75 and it is advisory. A near-miss that is the
INTENDED twin of an existing item is fine; a near-miss that is an accident is
not, and only the author knows which. The tool reports, it does not refuse.
"""
import argparse, json, math, os, sys, urllib.parse, urllib.request

ROOT = "/Users/rj416/github/malign-logits"
SERVER = os.environ.get("MALIGN_SERVER", "http://127.0.0.1:8421")
CORPUS = ["pair_drafts/round3/round3_slots.yaml",
          "pair_drafts/round3/round3_agent.yaml"]
NEAR = 0.75


def slot(prompt, k=60):
    u = SERVER + "/api/slot?" + urllib.parse.urlencode({"prompt": prompt, "k": k})
    d = json.loads(urllib.request.urlopen(u, timeout=900).read())
    if d.get("skipped"):
        return None
    return {w["word"]: w["p"] for w in d["words"]}


def cos(a, b):
    ks = set(a) | set(b)
    n = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in ks)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return n / (da * db) if da and db else 0.0


def load(paths):
    import yaml
    out = []
    for p in paths:
        f = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if not os.path.exists(f):
            continue
        for i in yaml.safe_load(open(f)) or []:
            if isinstance(i, dict) and "prompt" in i:
                out.append((i.get("item_id") or i.get("pair_id") or "?", i["prompt"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--corpus", default=",".join(CORPUS))
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    cand = slot(a.prompt)
    if cand is None:
        print("  the instrument REFUSED this prompt"); return 1
    rows = []
    for iid, p in load(a.corpus.split(",")):
        if p.strip() == a.prompt.strip():
            print("  EXACT DUPLICATE of %s" % iid); return 1
        d = slot(p)
        if d:
            rows.append((cos(cand, d), iid, p))
    rows.sort(reverse=True)
    print("  candidate: %r" % a.prompt)
    print("  nearest existing items by slot cosine:")
    for c, iid, p in rows[:a.top]:
        mark = "  NEAR-DUPLICATE" if c > NEAR else ""
        print("     %.3f  %-32s %s%s" % (c, iid[:32], p[:40], mark))
    top = rows[0][0] if rows else 0.0
    print("\n  max %.3f against %d items -> %s"
          % (top, len(rows),
             "REDUNDANT unless it is a deliberate twin" if top > NEAR else "NOVEL"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
