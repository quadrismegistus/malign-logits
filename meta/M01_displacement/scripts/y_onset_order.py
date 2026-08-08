#!/usr/bin/env python
"""The ORDER of the three onsets: break, noise, refusal.

    python y_onset_order.py

Each is a quote; this locates it in the continuation and reports which subset
fires and in what sequence. The order is the thing no single field could carry
and the reason there are three.

TIES ARE REAL AND ARE KEPT AS TIES. Two onsets at the same word means the same
token both stopped the story and stopped being language -- `sap.ReLU(1)` is
exactly that. Collapsing a tie into an arbitrary order would invent a sequence.
Tolerance is 0 words: same word index, same event.

UNIT IS THE ANNOTATION, NOT THE ITEM. Each item is coded twice and the two
coders can locate differently; pooling to the item would need a rule for
disagreement that I would have to invent. Per-annotation double-counts by
construction and the arm comparison is unaffected because both arms are
double-coded equally. Stated rather than hidden.
"""
import collections
import difflib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")


def norm(s):
    return " ".join((s or "").lower().split())


def locate(quote, text):
    q, nt = norm(quote), norm(text)
    if not q or not nt:
        return None
    if q in nt:
        return len(nt[:nt.index(q)].split())
    sm = difflib.SequenceMatcher(None, nt, q).find_longest_match(0, len(nt), 0, len(q))
    if sm.size >= max(12, 0.6 * len(q)):
        return len(nt[:sm.a].split())
    return None


def main():
    import y_pilot_coder as Y
    rng = random.Random(Y.SEED)
    G = Y.load()
    texts, metas = Y.build_items(G, 10, rng)
    k2t = {}
    for t, m in zip(texts, metas):
        k2t[(m["pair"], m["role"], m["word"], m["seq_i"])] = \
            t.split('CONTINUATION: "', 1)[1].rsplit('"', 1)[0]

    rows = [json.loads(l) for l in open(os.path.join(CAMP, "results", "y_pilot_coded.jsonl"))]
    if "break_onset" not in rows[0]:
        print("this file predates the three onsets -- rerun y_pilot_coder.py")
        return 1

    LED = collections.Counter()
    obs = []
    for d in rows:
        k = (d["pair"], d["role"], d["word"], d["seq_i"])
        if k not in k2t:
            LED["no text"] += 1
            continue
        t = k2t[k]
        p = {}
        for name, fld in (("B", "break_onset"), ("N", "noise_onset"), ("R", "refusal_onset")):
            q = (d.get(fld) or "").strip()
            if not q:
                continue
            pos = locate(q, t)
            if pos is None:
                LED["quote not in text: %s" % fld] += 1
                continue
            p[name] = pos
        LED["annotations"] += 1
        obs.append((p, d["role"], d["cls"], d["pair"]))

    print("LEDGER")
    for k2, v in LED.most_common():
        print("   %-28s %d" % (k2, v))

    def seq(p):
        """e.g. 'B=N<R', 'N<B<R', 'B only', 'none'."""
        if not p:
            return "none"
        items = sorted(p.items(), key=lambda x: x[1])
        out, prev = [items[0][0]], items[0][1]
        for name, pos in items[1:]:
            out.append("=" if pos == prev else "<")
            out.append(name)
            prev = pos
        s = "".join(out)
        return s + (" only" if len(p) == 1 else "")

    print("\n" + "=" * 84)
    print("ORDER OF ONSETS   B=break  N=noise  R=refusal   (%d annotations)" % len(obs))
    print("%-14s %7s %7s %8s %8s   %s" % ("pattern", "base", "aligned", "n", "% of all", ""))
    print("-" * 84)
    cnt = collections.Counter((seq(p), r) for p, r, c, pr in obs)
    tot = collections.Counter(r for p, r, c, pr in obs)
    pats = sorted({s for s, r in cnt}, key=lambda s: -(cnt[(s, "base")] + cnt[(s, "aligned")]))
    for s in pats:
        b, a = cnt[(s, "base")], cnt[(s, "aligned")]
        print("%-14s %6.1f%% %6.1f%% %8d %7.1f%%" %
              (s, 100 * b / tot["base"], 100 * a / tot["aligned"], b + a,
               100 * (b + a) / len(obs)))

    print("\n" + "=" * 84)
    print("AMONG ANNOTATIONS WITH TWO OR MORE ONSETS -- what precedes what")
    multi = [(p, r) for p, r, c, pr in obs if len(p) >= 2]
    print("  %d of %d annotations have 2+ onsets (%.1f%%)" % (len(multi), len(obs), 100 * len(multi) / len(obs)))
    for x, y in (("B", "N"), ("B", "R"), ("N", "R")):
        sel = [(p, r) for p, r in multi if x in p and y in p]
        if not sel:
            continue
        first = sum(1 for p, r in sel if p[x] < p[y])
        tie = sum(1 for p, r in sel if p[x] == p[y])
        last = len(sel) - first - tie
        print("  %s vs %s  n=%-4d  %s first %3d (%2.0f%%) | tied %3d (%2.0f%%) | %s first %3d (%2.0f%%)"
              % (x, y, len(sel), x, first, 100 * first / len(sel), tie, 100 * tie / len(sel),
                 y, last, 100 * last / len(sel)))

    print("\n  GAPS in words, median, where both are present and not tied")
    for x, y in (("B", "N"), ("B", "R"), ("N", "R")):
        g = sorted(abs(p[y] - p[x]) for p, r, c, pr in obs if x in p and y in p and p[x] != p[y])
        if g:
            import statistics
            print("    %s to %s   n=%-4d median %3d  max %3d" % (x, y, len(g), statistics.median(g), max(g)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
