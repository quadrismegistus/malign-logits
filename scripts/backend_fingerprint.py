#!/usr/bin/env python3
"""Measure the MPS-vs-cloud backend difference UPSTREAM of the rater.

lacan's route: the sanity pair is TinyLlama generated on both backends with the
same prompts and sampling parameters, fully labelled and NOT blind material. So
the backend effect can be measured in the text directly, with no codes, no
rubric and no key -- which sidesteps custody entirely and also measures the
effect before it inherits any rater noise.

Three things fall out of one pass:
  (a-replication) same-vs-different by ARM, on a rater-free path
  (b)             per-prompt divergence correlated with base entropy
  (fingerprint)   does any text statistic separate the two labelled sets?
"""
import json, math, collections, statistics as st
from malign_logits.cache import get_cache

MPS_BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
MPS_ALGN = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
N_DRAWS = 8


def toks(t):
    return [w for w in "".join(c.lower() if c.isalnum() or c.isspace() else " "
                              for c in t).split() if w]


def js(p, q):
    keys = set(p) | set(q)
    tp, tq = sum(p.values()) or 1, sum(q.values()) or 1
    d = 0.0
    for k in keys:
        a, b = p.get(k, 0) / tp, q.get(k, 0) / tq
        m = (a + b) / 2
        if a: d += 0.5 * a * math.log2(a / m)
        if b: d += 0.5 * b * math.log2(b / m)
    return d


def main():
    cm = get_cache()
    prompts = json.load(open("data/confirmation_battery_prompts.json"))
    ent = {r["prompt"]: float(r["entropy_bits"])
           for r in __import__("csv").DictReader(open("data/tinyllama_base_entropy.csv"))}

    cloud = collections.defaultdict(list)
    for line in open("data/tinyllama_cloud_sanity.jsonl"):
        r = json.loads(line)
        cloud[(r["role"], r["prompt"])].append(r["text"])

    try:
        from wordfreq import zipf_frequency
        rare = lambda ws: sum(1 for w in ws if zipf_frequency(w, "en") < 3.0) / max(len(ws), 1)
    except ImportError:
        rare = None

    rows = []
    for role, model in (("base", MPS_BASE), ("aligned", MPS_ALGN)):
        for p in prompts:
            mps = [cm.get_generation(model, p, temp=1.0, idx=i) for i in range(N_DRAWS)]
            mps = [t for t in mps if t]
            cld = cloud.get((role, p), [])
            if len(mps) < 4 or len(cld) < 4:
                continue
            tm = [toks(t) for t in mps]
            tc = [toks(t) for t in cld]
            fm = collections.Counter(w for t in tm for w in t)
            fc = collections.Counter(w for t in tc for w in t)
            rows.append(dict(
                role=role, prompt=p, entropy=ent.get(p, float("nan")),
                js=js(fm, fc),
                len_mps=st.mean(len(t) for t in tm), len_cloud=st.mean(len(t) for t in tc),
                ttr_mps=st.mean(len(set(t)) / max(len(t), 1) for t in tm),
                ttr_cloud=st.mean(len(set(t)) / max(len(t), 1) for t in tc),
                rare_mps=(st.mean(rare(t) for t in tm) if rare else float("nan")),
                rare_cloud=(st.mean(rare(t) for t in tc) if rare else float("nan")),
            ))

    import csv
    with open("data/backend_fingerprint.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"{len(rows)} prompt x arm cells\n")
    print("(a-REPLICATION) token-JS divergence MPS vs cloud, by arm:")
    for role in ("base", "aligned"):
        v = [r["js"] for r in rows if r["role"] == role]
        print(f"  {role:8s} n={len(v):3d}  mean JS {st.mean(v):.4f}  median {st.median(v):.4f}")
    b = [r["js"] for r in rows if r["role"] == "base"]
    a = [r["js"] for r in rows if r["role"] == "aligned"]
    print(f"  base/aligned ratio: {st.mean(b)/st.mean(a):.2f}   "
          f"(desktop's rated ratios were 2.26 and 2.63)")

    print("\n(b) correlation of per-prompt JS with base entropy:")
    for role in ("base", "aligned"):
        s = [(r["entropy"], r["js"]) for r in rows if r["role"] == role and r["entropy"] == r["entropy"]]
        n = len(s)
        mx, my = st.mean(x for x, _ in s), st.mean(y for _, y in s)
        num = sum((x - mx) * (y - my) for x, y in s)
        den = math.sqrt(sum((x - mx) ** 2 for x, _ in s) * sum((y - my) ** 2 for _, y in s))
        r_ = num / den if den else float("nan")
        t = r_ * math.sqrt((n - 2) / max(1e-12, 1 - r_ ** 2))
        print(f"  {role:8s} n={n:3d}  Pearson r={r_:+.3f}  t={t:+.2f}")

    print("\n(FINGERPRINT) does any statistic separate the labelled sets?")
    for nm, km, kc in (("length", "len_mps", "len_cloud"),
                       ("type-token ratio", "ttr_mps", "ttr_cloud"),
                       ("rare-word rate", "rare_mps", "rare_cloud")):
        d = [r[km] - r[kc] for r in rows if r[km] == r[km] and r[kc] == r[kc]]
        if not d: print(f"  {nm:18s} unavailable"); continue
        m, sd = st.mean(d), (st.stdev(d) if len(d) > 1 else 0)
        se = sd / math.sqrt(len(d)) if sd else 0
        pos = sum(1 for x in d if x > 0)
        print(f"  {nm:18s} mean diff {m:+.4f}  t={m/se if se else float('nan'):+.2f}  "
              f"{pos}/{len(d)} positive")


if __name__ == "__main__":
    main()
