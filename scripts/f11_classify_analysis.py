#!/usr/bin/env python3
"""F11 behavioral classification: kappa, consensus rates, AWAY-excess, profiles."""

import csv
from collections import Counter, defaultdict

#: **THE FILE THIS NAMED HAS NEVER EXISTED IN ANY COMMIT.** `f11_key.csv`
#: appears in no git history; the blinding map is `f11_classify_key.csv`, added
#: at 7fc3149 with exactly the columns read below (code, prompt_type, layer).
#: So this script -- which produced F11's behavioural numbers -- COULD NOT RUN
#: AS COMMITTED. Repaired 2026-08-01; paths are relative, so run from data/.
key = {r["code"]: r for r in csv.DictReader(open("f11_classify_key.csv"))}

def load(files):
    d = {}
    for fn in files:
        for r in csv.DictReader(open(fn)):
            d[r["code"].strip()] = r["category"].strip().upper()
    return d

r1 = load([f"f11_r1{s}.csv" for s in "abcd"])
r2 = load([f"f11_r2{s}.csv" for s in "abcd"])
codes = sorted(set(key) & set(r1) & set(r2))
print(f"n = {len(codes)}")

OFF = {"META", "INCOH", "OTHER"}
def merge(c): return "OFF" if c in OFF else c

def kappa(pairs):
    n = len(pairs)
    if n == 0: return float("nan"), float("nan")
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    c1, c2 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((c1[k]/n)*(c2[k]/n) for k in set(c1)|set(c2))
    return po, (po-pe)/(1-pe) if pe < 1 else float("nan")

fine = [(r1[c], r2[c]) for c in codes]
mrg = [(merge(r1[c]), merge(r2[c])) for c in codes]
po_f, k_f = kappa(fine); po_m, k_m = kappa(mrg)
print(f"fine:   agreement {po_f:.3f}, kappa {k_f:.3f}")
print(f"merged: agreement {po_m:.3f}, kappa {k_m:.3f}")

# per-sheet kappa (sheets by code ranges: a=1-399, b=400-798, c=799-1197, d=1198-1596)
for s, lo, hi in [("a",1,399),("b",400,798),("c",799,1197),("d",1198,1596)]:
    cs = [c for c in codes if lo <= int(c[1:]) <= hi]
    _, kf = kappa([(r1[c], r2[c]) for c in cs])
    _, km = kappa([(merge(r1[c]), merge(r2[c])) for c in cs])
    print(f"  sheet {s}: fine k={kf:.3f}  merged k={km:.3f}")

# exclude N items for the substantive kappa
non_n = [c for c in codes if key[c]["prompt_type"] != "N"]
_, k_nn = kappa([(merge(r1[c]), merge(r2[c])) for c in non_n])
print(f"merged kappa excluding N items: {k_nn:.3f} (n={len(non_n)})")

# consensus set under merge
cons = {c: merge(r1[c]) for c in codes if merge(r1[c]) == merge(r2[c])}
print(f"consensus coverage (merged): {len(cons)}/{len(codes)} = {len(cons)/len(codes):.2f}")

CATS = ["D1", "D2", "BOTH", "AWAY", "OFF"]

def rates(subset):
    n = len(subset)
    cnt = Counter(subset.values())
    return {k: cnt.get(k, 0)/n if n else 0 for k in CATS}, n

def show(label, subset):
    r, n = rates(subset)
    print(label.ljust(34) + "".join(f"{r[k]:6.2f}" for k in CATS) + f"  n={n}")

print("\n== consensus rates (cols: " + " ".join(CATS) + ") ==")
for pt in ["AB", "A", "B", "N"]:
    for lay in ["base", "aligned"]:
        sub = {c: v for c, v in cons.items()
               if key[c]["prompt_type"] == pt and key[c]["layer"] == lay}
        show(f"{pt} x {lay}", sub)

# AWAY-excess: AB away-rate minus pooled A/B away-rate, per layer
print("\n== AWAY-excess (AB minus single-pole), consensus ==")
for lay in ["base", "aligned"]:
    ab = {c: v for c, v in cons.items() if key[c]["prompt_type"] == "AB" and key[c]["layer"] == lay}
    sp = {c: v for c, v in cons.items() if key[c]["prompt_type"] in ("A","B") and key[c]["layer"] == lay}
    ra, _ = rates(ab); rs, _ = rates(sp)
    print(f"{lay:8s} AB_away={ra['AWAY']:.3f} SP_away={rs['AWAY']:.3f} excess={ra['AWAY']-rs['AWAY']:+.3f}"
          f" | BOTH: AB={ra['BOTH']:.3f} SP={rs['BOTH']:.3f}")

# per-family aligned AB profiles + away-excess
print("\n== per-family ALIGNED AB profiles (consensus) ==")
fams = sorted({key[c]["family"] for c in codes})
for fam in fams:
    ab = {c: v for c, v in cons.items() if key[c]["family"] == fam
          and key[c]["prompt_type"] == "AB" and key[c]["layer"] == "aligned"}
    sp = {c: v for c, v in cons.items() if key[c]["family"] == fam
          and key[c]["prompt_type"] in ("A","B") and key[c]["layer"] == "aligned"}
    ra, na = rates(ab); rs, ns = rates(sp)
    print(f"{fam:12s}" + "".join(f"{ra[k]:6.2f}" for k in CATS) +
          f"  away_ex={ra['AWAY']-rs['AWAY']:+.3f}  n={na}/{ns}")

# BOTH delta base->aligned on AB (superposition held?)
print("\n== BOTH rate on AB, base -> aligned, per family (consensus) ==")
for fam in fams:
    row = []
    for lay in ["base", "aligned"]:
        ab = {c: v for c, v in cons.items() if key[c]["family"] == fam
              and key[c]["prompt_type"] == "AB" and key[c]["layer"] == lay}
        r, n = rates(ab)
        row.append((r["BOTH"], n))
    print(f"{fam:12s} {row[0][0]:.2f} (n={row[0][1]}) -> {row[1][0]:.2f} (n={row[1][1]})  d={row[1][0]-row[0][0]:+.2f}")

# Deleuzian vs classical pairs, aligned AB
DELEUZIAN = {"desire_disgust","sacred_profane","man_woman","human_animal","create_destroy","free_captive"}
def isdel(p): return p in DELEUZIAN or p.replace("/","_") in DELEUZIAN or any(d in p for d in ["man","animal","create","free","sacred","desire_dis"])
print("\n== pairs (aligned AB, consensus): classical vs Deleuzian ==")
pairs = sorted({key[c]["pair"] for c in codes if key[c]["prompt_type"]=="AB"})
for p in pairs:
    ab = {c: v for c, v in cons.items() if key[c]["pair"] == p
          and key[c]["prompt_type"] == "AB" and key[c]["layer"] == "aligned"}
    if not ab: continue
    r, n = rates(ab)
    tag = "DEL" if isdel(p) else "cls"
    print(f"{tag} {p:22s}" + "".join(f"{r[k]:6.2f}" for k in CATS) + f"  n={n}")

# collapse baseline: INCOH-ish (OFF) on N items per family
print("\n== OFF rate on NEUTRAL items per family (consensus) ==")
for fam in fams:
    for lay in ["base", "aligned"]:
        sub = {c: v for c, v in cons.items() if key[c]["family"] == fam
               and key[c]["prompt_type"] == "N" and key[c]["layer"] == lay}
        r, n = rates(sub)
        print(f"{fam:12s} {lay:8s} OFF={r['OFF']:.2f} n={n}", end="   ")
    print()
