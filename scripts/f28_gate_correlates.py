#!/usr/bin/env python3
"""What predicts gate height?

The 19-family scale-up of F28 found that resistance spikes at the first
generated token and that the spike's HEIGHT is a family property varying from
-0.58 (llama) to +14.90 (map-neo). This asks what that height tracks.

Two candidates from the existing findings:
  displacement (JS base->aligned)   -- how far the distribution MOVED
  entropy drop (H_base - H_aligned) -- how much the distribution NARROWED
and one downstream question: does gating substitute for narrative work?
"""
import csv, collections, json, math, statistics as st
import numpy as np
from malign_logits.cache import get_cache
from malign_logits import MODEL_FAMILIES


def corr(a, b):
    n = len(a); ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    r = num / den if den else float("nan")
    return r, r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))


def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); rk = [0] * len(v)
    for i, j in enumerate(s): rk[j] = i
    return rk


def entropy(lg):
    x = np.asarray(lg, dtype=np.float64); x = x - x.max()
    q = np.exp(x); q /= q.sum(); q = q[q > 0]
    return float(-(q * np.log2(q)).sum())


def main():
    cm = get_cache()
    prompts = json.load(open("data/confirmation_battery_prompts.json"))

    g = collections.defaultdict(list)
    for r in csv.DictReader(open("data/f28_scaled_trajectories.csv")):
        if r["role"] != "reinforced" and int(r["pos"]) == 0:
            g[r["family"]].append(float(r["resistance"]))
    gate = {f: st.mean(v) for f, v in g.items()}

    rows = []
    for fk in sorted(gate):
        f = MODEL_FAMILIES.get(fk)
        if not f or not f.base: continue
        al = getattr(f, "superego", None) or getattr(f, "ego", None)
        if not al: continue
        hb, ha = [], []
        for p in prompts:
            lb, la = cm.get_logits(f.base, p), cm.get_logits(al, p)
            if lb is not None and la is not None:
                hb.append(entropy(lb)); ha.append(entropy(la))
        if len(hb) < 20: continue
        rows.append(dict(family=fk, gate=gate[fk],
                         entropy_drop=st.mean(hb) - st.mean(ha),
                         entropy_base=st.mean(hb), n_prompts=len(hb)))

    with open("data/f28_gate_correlates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    a = [r["gate"] for r in rows]; b = [r["entropy_drop"] for r in rows]
    r1, t1 = corr(a, b); rs1, _ = corr(rank(a), rank(b))
    print(f"gate vs ENTROPY DROP : Pearson {r1:+.3f} (t={t1:+.2f})  Spearman {rs1:+.3f}  n={len(a)}")

    js = collections.defaultdict(list)
    for r in csv.DictReader(open("data/battery_results.csv")):
        try: js[r["family"]].append(float(r["js_base_superego"]))
        except (ValueError, KeyError): pass
    JS = {f: st.mean(v) for f, v in js.items()}
    common = sorted(set(JS) & set(gate))
    a2 = [gate[f] for f in common]; b2 = [JS[f] for f in common]
    r2, t2 = corr(a2, b2); rs2, _ = corr(rank(a2), rank(b2))
    print(f"gate vs JS DISPLACEMENT: Pearson {r2:+.3f} (t={t2:+.2f})  Spearman {rs2:+.3f}  n={len(a2)}")
    print(f"\n{len([x for x in a if x > 1.0])} of {len(a)} families gate at all (pos0 > 1 bit)")


if __name__ == "__main__":
    main()
