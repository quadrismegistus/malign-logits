#!/usr/bin/env python3
"""Two controls on the positional-asymmetry residue.

The residue: forward resistance (base generates, aligned scores) peaks at pos0;
reverse resistance (aligned generates, base scores) peaks at pos1. Sharpening
explains the forward pos0 spike but not obviously the reverse pos1 one.

CONTROL A -- is the reverse pos1 spike content-blind?
  The forward pos0 spike turned out identical at neutral prompts, which is what
  a sharpening artifact predicts. If the reverse spike is content-blind too, it
  is generic typical-set divergence and not site-specific anything.

CONTROL B -- the base-vs-base null.
  Two UNRELATED base models scoring each other's storylines give the generic
  cross-model divergence profile by position. The aligned/base contrast has to
  beat THAT shape, not zero. Testing against zero (which is what the residue
  implicitly did) assumes any positional structure is alignment-specific, when
  it may just be what happens when any two models disagree.
"""
import collections, csv, math, statistics as st
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

    rev = collections.defaultdict(list)     # (category, pos) -> bits   [control A]
    revf = collections.defaultdict(list)    # (family, category, pos)
    null = collections.defaultdict(list)    # pos -> bits               [control B]
    nullf = collections.defaultdict(list)   # (pairkey, pos)
    fwd = collections.defaultdict(list)     # pos -> bits (for reference)

    for k in s:
        if not isinstance(k, dict) or k.get("type") != "beam_cross_v1":
            continue
        gen = who.get(norm(k.get("source")))
        cat = cats.get(k.get("prompt"))
        if gen is None or cat is None:
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
                if not sc or tgt is None:
                    continue
                same_fam = tgt[0] == gen[0]
                for i in range(min(MAXPOS, len(src), len(sc))):
                    ps, pc = src[i], sc[i]
                    if not (ps and pc and ps > 0 and pc > 0):
                        continue
                    bits = math.log2(ps / pc)
                    if same_fam and gen[1] == "base" and tgt[1] != "base":
                        fwd[i].append(bits)
                    elif same_fam and gen[1] != "base" and tgt[1] == "base":
                        rev[(cat, i)].append(bits); revf[(gen[0], cat, i)].append(bits)
                    elif not same_fam and gen[1] == "base" and tgt[1] == "base":
                        null[i].append(bits); nullf[(gen[0], tgt[0], i)].append(bits)

    print("=== CONTROL A: is the REVERSE pos1 spike content-blind? ===")
    cs = sorted({c for c, _ in rev})
    print(f"{'category':18s}" + "".join(f"{'p'+str(i):>7s}" for i in range(4)) + f"{'p1-p0':>8s}")
    spikes = {}
    for c in cs:
        v = [st.mean(rev[(c, i)]) if rev.get((c, i)) else float("nan") for i in range(4)]
        spikes[c] = v[1] - v[0]
        print(f"{c[:17]:18s}" + "".join(f"{x:>7.2f}" for x in v) + f"{v[1]-v[0]:>8.2f}")
    tv = [spikes[c] for c in spikes if c not in ("neutral", "institutional")]
    if "neutral" in spikes:
        print(f"\n  transgressive mean p1-p0 {st.mean(tv):+.2f} vs NEUTRAL {spikes['neutral']:+.2f}"
              f"   difference {st.mean(tv)-spikes['neutral']:+.2f} bits")
    # family as unit
    d = []
    fams = sorted({f for f, _, _ in revf})
    for f in fams:
        t = [st.mean(revf[(f, c, 1)]) - st.mean(revf[(f, c, 0)])
             for c in cs if c not in ("neutral", "institutional")
             and revf.get((f, c, 0)) and revf.get((f, c, 1))]
        n0, n1 = revf.get((f, "neutral", 0)), revf.get((f, "neutral", 1))
        if t and n0 and n1:
            d.append(st.mean(t) - (st.mean(n1) - st.mean(n0)))
    if len(d) > 2:
        m = st.mean(d); se = st.stdev(d) / math.sqrt(len(d))
        print(f"  FAMILY AS UNIT: mean {m:+.3f} bits, n={len(d)}, "
              f"{sum(1 for x in d if x > 0)}/{len(d)} positive, t={m/se:+.2f}")

    print("\n=== CONTROL B: the base-vs-base null ===")
    print(f"{'profile':22s}" + "".join(f"{'p'+str(i):>7s}" for i in range(6)))
    for nm, dd in (("forward (base->aligned)", fwd),
                   ("reverse (aligned->base)", {i: [x for c, j in rev for x in rev[(c, j)] if j == i]
                                                for i in range(6)}),
                   ("NULL (base->other base)", null)):
        line = "".join(f"{st.mean(dd[i]):>7.2f}" if dd.get(i) else f"{'-':>7s}" for i in range(6))
        print(f"{nm:22s}{line}")
    npairs = len({(a, b) for a, b, _ in nullf})
    print(f"\n  null built from {npairs} cross-family base pairs, "
          f"{len(null.get(0, [])):,} position-0 observations")
    if null.get(0) and null.get(1):
        print(f"  null p1-p0 = {st.mean(null[1])-st.mean(null[0]):+.2f} bits "
              f"(reverse p1-p0 for comparison: see above)")


if __name__ == "__main__":
    main()
