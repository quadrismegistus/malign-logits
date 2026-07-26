#!/usr/bin/env python3
"""Registered SELECTION RULE for the tier-2 positive-control gate.

Adopted by lacan ruling 2026-07-26, replacing author-chosen gate values with a
rule that quantifies over the whole (floor, k) family. This file DECLARES THE
GRID; it is committed before it is run.

RULE
  Among all (floor_percentile, k) in the declared grid, choose the combination
  MINIMISING false-certification under the null, SUBJECT TO power >= 0.80 at
  d = the corpus's own median chain-pair MDE. Tie-break by power (higher wins),
  then by lower k.

X ANCHOR (lacan amendment). The power constraint is NOT set at d=0.10. That is
1.105x, below hh's median chain-pair MDE of 1.19x, and would demand the
INSTRUMENT CHECK resolve an effect smaller than the MAIN TEST can -- which can
fail an adequate instrument and produce an insensitivity finding that is an
artifact of the gate. X is therefore anchored to the MDE table, which is derived
from counts alone and so inherits no contamination from the disclosed chain-D
leak.

  hh_rlhf      median MDE(D_excess) 1.19x  ->  d = ln(1.19) = 0.174
  pku_saferlhf median MDE(D_excess) 1.50x  ->  d = ln(1.50) = 0.405

RESOLUTION (defect found on execution, disclosed rather than papered over).
The first run used NSIM=200_000. The winning cells have false-certification
rates near 1e-4, so their Monte Carlo standard errors EXCEEDED the differences
between them and the ranking was decided by RNG noise: at 200k the rule returned
hh p65/6-of-7; at 4M it returns a different cell. A selection rule whose output
depends on simulation seed is not a rule. Two fixes, both applied:
  1. NSIM raised to 4_000_000, giving SE ~1e-5 at fc ~1e-4.
  2. TIE BAND: cells whose false-certification differs from the minimum by less
     than the 95% CI of that difference are treated as TIED, and the tie is
     broken by power (higher wins), then lower k. Without this the gate is
     picked by a point estimate finer than the method can resolve.
Fix 2 is an amendment to the rule's IMPLEMENTATION, not its logic; it is
disclosed to the lacan seat, which rules thresholds.

REGISTERED IN ADVANCE: whatever the search returns is the gate, INCLUDING if it
is neither p75/3-of-7 (author) nor p90/2-of-7 (blind re-derivation). If no
combination satisfies the constraint in a corpus, that corpus has NO VIABLE GATE
and its control result is descriptive -- already registered for pku.
"""
import csv, math, random, statistics as st

# ---- THE GRID, DECLARED BEFORE THE SEARCH RUNS ----
FLOOR_PCTS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
K_VALUES   = [1, 2, 3, 4, 5, 6, 7]
N_CAND     = 7
POWER_MIN  = 0.80
D_ANCHOR   = {"hh_rlhf": math.log(1.19), "pku_saferlhf": math.log(1.50)}
NSIM       = 4_000_000   # see RESOLUTION note below
# ---------------------------------------------------

random.seed(20260726)
exec(open('scripts/tier2_power_check.py').read().split('chains, chain_words')[0])

chains, chain_words = [], set()
for r in csv.DictReader(open("data/d2_modal_pairs.csv")):
    s, t = r["source"].strip().lower(), r["modal_target"].strip().lower()
    chain_words.update([s, t])
    if s not in STOP and t not in STOP:
        chains.append((s, t))
K = 20


def decoy_pool(corp):
    pc, pr = CORPORA[corp]
    ch, rj = load(pc), load(pr)
    Nc, Nr = sum(ch.values()), sum(rj.values())
    vocab = {w: ch[w] + rj.get(w, 0) for w in ch
             if ch[w] >= 20 and rj.get(w, 0) >= 20 and w not in STOP and w not in chain_words}
    items = sorted(vocab.items(), key=lambda x: x[1])
    nearest = lambda f, k: [w for w, _ in sorted(items, key=lambda x: abs(math.log(x[1]) - math.log(f)))[:k]]
    pool = []
    for s, t in chains:
        cs, rs, ct, rt = ch.get(s,0), rj.get(s,0), ch.get(t,0), rj.get(t,0)
        if min(cs, rs, ct, rt) < 20: continue
        ds, dt = nearest(cs+rs, K), nearest(ct+rt, K)
        for off in range(1, len(dt)):
            prs = [(a, b) for a, b in zip(ds, dt[off:]+dt[:off]) if a != b]
            if len(prs) >= 5: break
        for a, b in prs:
            pool.append(log_or(ch[b],rj[b],Nc,Nr) - log_or(ch[a],rj[a],Nc,Nr))
    return pool


def sim(pool, floor, k, delta, _cache={}):
    """Vectorised: 4M x 7 draws per cell is 3.9e9 in pure Python, seconds in numpy.

    Draws are shared across all (floor, k) cells for a given delta, which is
    both faster and removes between-cell Monte Carlo noise from the comparison
    -- the ranking is then decided by the gates, not by independent seeds.
    """
    import numpy as np
    key = (id(pool), delta)
    if key not in _cache:
        rng = np.random.default_rng(20260726)
        arr = np.asarray(pool)
        _cache[key] = rng.choice(arr, size=(NSIM, N_CAND), replace=True) + delta
    d = _cache[key]
    fires = (d > 0) & (np.abs(d) > floor)
    return float((fires.sum(axis=1) >= k).mean())


for corp in CORPORA:
    pool = decoy_pool(corp)
    absd = sorted(abs(x) for x in pool)
    d = D_ANCHOR[corp]
    print(f"=== {corp} ===  n_decoy={len(pool):,}  anchor d={d:.3f} "
          f"({math.exp(d):.2f}x)  constraint power >= {POWER_MIN}")
    rows = []
    for pct in FLOOR_PCTS:
        f = absd[min(int(pct/100*len(absd)), len(absd)-1)]
        for k in K_VALUES:
            fc = sim(pool, f, k, 0.0)
            pw = sim(pool, f, k, d)
            rows.append((pct, k, f, fc, pw))
    ok = [r for r in rows if r[4] >= POWER_MIN]
    print(f"  {len(ok)} of {len(rows)} grid points satisfy the power constraint")
    if not ok:
        print("  NO VIABLE GATE in this corpus -> control result is DESCRIPTIVE\n")
        continue
    # tie band: fc indistinguishable from the minimum -> break by power
    fmin = min(r[3] for r in ok)
    se = lambda v: math.sqrt(max(v, 1/NSIM) * (1 - v) / NSIM)
    tied = [r for r in ok if r[3] - fmin < 1.96 * math.sqrt(2) * max(se(r[3]), 1e-7)]
    tied.sort(key=lambda r: (-r[4], r[1]))
    ok = tied + sorted([r for r in ok if r not in tied], key=lambda r: (r[3], -r[4]))
    print(f"  {len(tied)} cell(s) tied at the false-cert minimum; tie broken by power")
    print(f"  {'floor':>7s}{'k':>3s}{'|D|':>9s}{'false-cert':>12s}{'power':>8s}")
    for pct, k, f, fc, pw in ok[:6]:
        print(f"  p{pct:<6d}{k:>3d}{f:>9.4f}{fc:>12.3f}{pw:>8.3f}")
    w = ok[0]
    print(f"  >>> SELECTED: p{w[0]} floor, {w[1]}-of-{N_CAND}, "
          f"|D|>{w[2]:.4f}, false-cert {w[3]:.3f}, power {w[4]:.3f}")
    for lbl, pct, k in [("author p75/3-of-7", 75, 3), ("blind p90/2-of-7", 90, 2)]:
        m = [r for r in rows if r[0] == pct and r[1] == k][0]
        print(f"      vs {lbl}: false-cert {m[3]:.3f}, power {m[4]:.3f}, "
              f"{'MEETS' if m[4] >= POWER_MIN else 'FAILS'} constraint")
    print()
