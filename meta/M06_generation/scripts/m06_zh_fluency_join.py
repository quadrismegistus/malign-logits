"""Join blind fluency verdicts back to models, tiers and character statistics.

    uv run python meta/M06_generation/scripts/m06_zh_fluency_join.py

Reads the workflow's verdicts (key -> verdict), the local truth map
(key -> model), the registry (model -> cjk_tier) and recomputes the
character statistics on the SAME passages that were judged, so the
comparison is on one population rather than three.
"""
import collections
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
BATCHES = os.path.join(OUTD, "zh_fluency_batches")
ORDER = ["fluent", "flawed", "broken", "not_chinese"]
SCORE = {"fluent": 3, "flawed": 2, "broken": 1, "not_chinese": 0}


def cjk(s):
    return [c for c in s if "一" <= c <= "鿿"]


def charstats(text):
    """TTR, commonest-char share and repeated-bigram share, or None if short."""
    c = cjk(text)
    if len(c) < 20:
        return None
    bs = [tuple(c[i:i + 2]) for i in range(len(c) - 1)]
    return {
        "cjk_frac": len(c) / max(1, len(text)),
        "ttr": len(set(c)) / len(c),
        "bigrep": 1 - len(set(bs)) / max(1, len(bs)),
    }


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        OUTD, "zh_fluency_verdicts.json")
    verdicts = json.load(open(src))
    if isinstance(verdicts, dict):
        verdicts = verdicts.get("verdicts", verdicts)
    v = {r["key"]: r["verdict"] for r in verdicts if r.get("key")}
    truth = json.load(open(os.path.join(
        OUTD, "zh_fluency_sample.json")))["truth"]
    #: VIA `Checkpoint`, not a hand-rolled parse of the registry file --
    #: 1 of the 18 consumers named at [6321]. `Checkpoint(mid).record` is `{}`
    #: for an unknown id, which is what the `or {}` at the call sites relied on
    #: the raw dict for, so present and absent models both behave as before.
    from malign_logits.checkpoint import Checkpoint
    #: the judged text, so character stats are on the SAME passages
    text = {}
    for i in range(12):
        p = os.path.join(BATCHES, "batch_%02d.json" % i)
        for it in json.load(open(p)):
            text[it["key"]] = it["continuation"]

    def tier(m):
        return Checkpoint(m).record.get("cjk_tier") or "-"

    missing = [k for k in truth if k not in v]
    print("judged %d of %d passages (%d missing)"
          % (len(v), len(truth), len(missing)))
    if missing:
        print("  missing keys:", missing[:10])
    print()

    #: ---- per model ----
    per = collections.defaultdict(list)
    for k, ver in v.items():
        per[truth[k]["model"]].append((ver, text.get(k, "")))

    rows = []
    for m, items in per.items():
        c = collections.Counter(x[0] for x in items)
        n = len(items)
        mean = sum(SCORE[x[0]] for x in items) / n
        cs = [charstats(t) for _, t in items]
        cs = [x for x in cs if x]
        rows.append({
            "model": m, "n": n, "mean": mean, "tier": tier(m),
            "vocab": Checkpoint(m).record.get("cjk_chars"),
            "ok": (c["fluent"] + c["flawed"]) / n,
            "counts": [c[o] for o in ORDER],
            "ttr": st.median([x["ttr"] for x in cs]) if cs else float("nan"),
            "bigrep": st.median([x["bigrep"] for x in cs]) if cs else float("nan"),
            "cjkf": st.median([x["cjk_frac"] for x in cs]) if cs else float("nan"),
        })
    rows.sort(key=lambda r: -r["mean"])

    print("%-44s %3s %5s %-19s %6s %6s %6s %-8s %6s"
          % ("model", "n", "score", "flu/flw/brk/notzh",
             "ok", "TTR", "bigrep", "tier", "vocab"))
    for r in rows:
        print("%-44s %3d %5.2f %-19s %6.2f %6.3f %6.3f %-8s %6s"
              % (r["model"][:44], r["n"], r["mean"],
                 "/".join(str(x) for x in r["counts"]),
                 r["ok"], r["ttr"], r["bigrep"], r["tier"],
                 r["vocab"] if r["vocab"] is not None else "-"))

    #: ---- does the registry tier predict the judged verdict? ----
    print("\nBY REGISTRY TIER")
    bt = collections.defaultdict(list)
    for r in rows:
        bt[r["tier"]].append(r)
    print("  %-9s %6s %7s %7s %7s" % ("tier", "models", "score", "ok", "TTR"))
    for t in ("FLUENT", "PARTIAL", "NOMINAL", "MARGINAL", "-"):
        g = bt.get(t)
        if not g:
            continue
        print("  %-9s %6d %7.2f %7.2f %7.3f"
              % (t, len(g), st.mean([x["mean"] for x in g]),
                 st.mean([x["ok"] for x in g]),
                 st.mean([x["ttr"] for x in g if x["ttr"] == x["ttr"]])))

    #: ---- SIGN OF THE CHARACTER STATISTICS, the open question ----
    print("\nDOES LOW TTR / HIGH BIGREP MEAN FLUENT OR BROKEN?")
    good = [r for r in rows if r["mean"] >= 2.5 and r["ttr"] == r["ttr"]]
    bad = [r for r in rows if r["mean"] <= 1.5 and r["ttr"] == r["ttr"]]
    for label, g in (("judged FLUENT (score>=2.5)", good),
                     ("judged BROKEN  (score<=1.5)", bad)):
        if not g:
            print("  %-28s none" % label)
            continue
        print("  %-28s n=%2d  TTR %.3f  bigrep %.3f  cjk_frac %.3f"
              % (label, len(g), st.mean([x["ttr"] for x in g]),
                 st.mean([x["bigrep"] for x in g]),
                 st.mean([x["cjkf"] for x in g])))

    #: correlation across models, which is the actual test of the sign
    try:
        from scipy import stats as sps
        ok = [r for r in rows if r["ttr"] == r["ttr"]]
        for f in ("ttr", "bigrep", "cjkf"):
            rho, p = sps.spearmanr([r[f] for r in ok], [r["mean"] for r in ok])
            print("  spearman(%-6s, judged score) = %+.3f  p=%.2g  (n=%d models)"
                  % (f, rho, p, len(ok)))
    except Exception as e:
        print("  scipy unavailable:", e)

    #: ---- what this means for the 25 arms pairs ----
    print("\nTHE 25 ARMS PAIRS, BY JUDGED FLUENCY OF BOTH MEMBERS")
    try:
        import pandas as pd
        pq = os.path.join(ROOT, "meta/M06_generation/results/crosslingual_arms_pairs.parquet")
        used = {tuple(s.split(">")) for s in set(pd.read_parquet(pq)["pair"])}
        sc = {r["model"]: r["mean"] for r in rows}
        out = []
        for b, a in sorted(used):
            if b in sc and a in sc:
                out.append((min(sc[b], sc[a]), b, a, sc[b], sc[a]))
        out.sort(reverse=True)
        print("  %-6s %-40s %-38s" % ("min", "base", "aligned"))
        for mn, b, a, sb, sa in out:
            print("  %5.2f  %-40s %.2f  %-32s %.2f" % (mn, b[:40], sb, a[:32], sa))
        for thr in (2.5, 2.0, 1.5):
            print("  pairs where BOTH members score >= %.1f : %d of %d"
                  % (thr, sum(1 for x in out if x[0] >= thr), len(out)))
    except Exception as e:
        print("  (pairs join unavailable: %s)" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
