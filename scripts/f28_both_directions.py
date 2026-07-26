#!/usr/bin/env python3
"""Resistance in BOTH directions.

FORWARD  (base generates, aligned scores): log2(p_base / p_aligned)
  positive = the aligned model finds the base's own continuation surprising.
  "How much does alignment resist what the base wanted to say?"

REVERSE  (aligned generates, base scores):  log2(p_aligned / p_base)
  positive = the base finds the aligned model's continuation surprising.
  "How much of what alignment says would the base never have said?"

The asymmetry is the interesting quantity. Forward-only would be suppression
without substitution; reverse-only would be new material introduced without
blocking anything; both would be replacement.
"""
import collections, csv, json, math, statistics as st
from malign_logits.cache import get_cache
from malign_logits import MODEL_FAMILIES
import malign_logits.experiments as E

MAXPOS = 10


def norm(x):
    return str(x).split("/")[-1].replace("-", "_").replace(".", "_")


def main():
    cm = get_cache(); s = cm._stash("beams")
    dp = E.DEFAULT_PROMPTS
    cats = {t: l.rsplit("_", 1)[0] for l, t in dp.items()} if isinstance(dp, dict) else {}
    for l, t in getattr(E, "INSTITUTIONAL_PROMPTS", {}).items():
        cats[t] = "institutional"

    who = {}
    for fk, f in MODEL_FAMILIES.items():
        for rl, m in (("base", f.base), ("ego", getattr(f, "ego", None)),
                      ("superego", getattr(f, "superego", None)),
                      ("reinforced", getattr(f, "reinforced_superego", None))):
            if m:
                who.setdefault(norm(m), (fk, rl))

    acc = collections.defaultdict(list)   # (direction, family, pos) -> [bits]
    for k in s:
        if not isinstance(k, dict) or k.get("type") != "beam_cross_v1":
            continue
        gen = who.get(norm(k.get("source")))
        if gen is None or cats.get(k.get("prompt")) is None:
            continue
        try:
            beams = s[k]
        except Exception:
            continue
        for b in beams:
            src = b.get("base_token_probs") or []
            if not src:
                continue
            for sname, ann in (b.get("annotations") or {}).items():
                sc = (ann or {}).get("token_probs") or []
                tgt = who.get(norm(sname))
                if not sc or tgt is None or tgt[0] != gen[0]:
                    continue
                if gen[1] == "base" and tgt[1] != "base":
                    d = "forward"
                elif gen[1] != "base" and tgt[1] == "base":
                    d = "reverse"
                else:
                    continue
                for i in range(min(MAXPOS, len(src), len(sc))):
                    ps, pc = src[i], sc[i]
                    if ps and pc and ps > 0 and pc > 0:
                        acc[(d, gen[0], i)].append(math.log2(ps / pc))

    rows = [dict(direction=d, family=f, pos=i, n=len(v), resistance=st.mean(v))
            for (d, f, i), v in acc.items() if len(v) >= 20]
    with open("data/f28_both_directions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("MEAN TRAJECTORY BY DIRECTION (bits), pooled over families")
    print(f"{'direction':10s}" + "".join(f"{'p'+str(i):>7s}" for i in range(6)) + f"{'fams':>6s}")
    for d in ("forward", "reverse"):
        line = ""
        for i in range(6):
            v = [r["resistance"] for r in rows if r["direction"] == d and r["pos"] == i]
            line += f"{st.mean(v):>7.2f}" if v else f"{'-':>7s}"
        nf = len({r["family"] for r in rows if r["direction"] == d})
        print(f"{d:10s}{line}{nf:>6d}")

    fwd = {r["family"]: r["resistance"] for r in rows if r["direction"] == "forward" and r["pos"] == 0}
    rev = {r["family"]: r["resistance"] for r in rows if r["direction"] == "reverse" and r["pos"] == 0}
    both = sorted(set(fwd) & set(rev))
    print(f"\nPOS-0 GATE, BOTH DIRECTIONS ({len(both)} families)")
    print(f"{'family':14s}{'forward':>10s}{'reverse':>10s}{'asymmetry':>11s}")
    for f in sorted(both, key=lambda x: -(fwd[x] - rev[x])):
        print(f"{f:14s}{fwd[f]:>10.2f}{rev[f]:>10.2f}{fwd[f]-rev[f]:>11.2f}")


if __name__ == "__main__":
    main()
