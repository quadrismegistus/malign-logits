#!/usr/bin/env python3
"""Construct-level selection grid for the tier-2 gate. COMMITTED BEFORE RUNNING.

The v2 slate was withdrawn because its seven pairs were three correlated
constructs (blocks of 3,3,1), which overstates power by 0.21 and understates
false certification by 74x. The fix is one marker per construct, which restores
independence by construction -- so a construct-capped slate needs no blockwise
correction, and the grid below is honest as written.

Earlier cells quoted in the amendment (p50/3-of-3 at power 0.804; p80/2-of-3 at
0.900) are different COLUMNS of a floor x k space and neither decides. This
searches the space.

TWO OBJECTIVES REPORTED, because the registered one is under audit:
  R1 (registered): minimise false certification subject to power >= 0.80
  R2 (proposed A7): maximise power subject to false certification <= 0.10
R1 is what chose p80/5-of-7. A7 argues R1 is backwards for this gate, since a
failed gate produces a substantive insensitivity finding rather than a null, so
power is the consequential axis. The grid does not decide between them; it
reports what each selects so the choice stays visible.

DECLARED BEFORE EXECUTION: floors = deciles p50..p95, k in {1,2,3}, N=4e6,
anchor = each corpus's own median chain-pair MDE.
"""
import csv, math
import numpy as np

FLOORS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
KS = [1, 2, 3]
N = 4_000_000
ANCHOR = {"hh_rlhf": math.log(1.19), "pku_saferlhf": math.log(1.50)}
FC_MAX, PW_MIN = 0.10, 0.80


def decoy_pool(corp):
    exec(open("scripts/tier2_power_check.py").read().split("chains, chain_words")[0], globals())
    chains, chain_words = [], set()
    for r in csv.DictReader(open("data/d2_modal_pairs.csv")):
        s, t = r["source"].strip().lower(), r["modal_target"].strip().lower()
        chain_words.update([s, t])
        if s not in STOP and t not in STOP:
            chains.append((s, t))
    pc, pr = CORPORA[corp]
    ch, rj = load(pc), load(pr)
    Nc, Nr = sum(ch.values()), sum(rj.values())
    vocab = {w: ch[w] + rj.get(w, 0) for w in ch
             if ch[w] >= 20 and rj.get(w, 0) >= 20 and w not in STOP and w not in chain_words}
    items = sorted(vocab.items(), key=lambda x: x[1])
    near = lambda f, k: [w for w, _ in sorted(items, key=lambda x: abs(math.log(x[1]) - math.log(f)))[:k]]
    pool = []
    for s, t in chains:
        cs, rs, ct, rt = ch.get(s, 0), rj.get(s, 0), ch.get(t, 0), rj.get(t, 0)
        if min(cs, rs, ct, rt) < 20:
            continue
        ds, dt = near(cs + rs, 20), near(ct + rt, 20)
        for off in range(1, len(dt)):
            prs = [(a, b) for a, b in zip(ds, dt[off:] + dt[:off]) if a != b]
            if len(prs) >= 5:
                break
        for a, b in prs:
            pool.append(log_or(ch[b], rj[b], Nc, Nr) - log_or(ch[a], rj[a], Nc, Nr))
    return np.asarray(pool)


def main():
    for corp, anch in ANCHOR.items():
        pool = decoy_pool(corp)
        absd = np.sort(np.abs(pool))
        rng = np.random.default_rng(20260726)
        z = rng.choice(pool, size=(N, 3))
        d = rng.choice(pool, size=(N, 3)) + anch
        rows = []
        for pct in FLOORS:
            f = absd[min(int(pct / 100 * len(absd)), len(absd) - 1)]
            for k in KS:
                fc = float((((z > 0) & (np.abs(z) > f)).sum(1) >= k).mean())
                pw = float((((d > 0) & (np.abs(d) > f)).sum(1) >= k).mean())
                rows.append((pct, k, f, fc, pw))
        print(f"\n=== {corp} ===  anchor d={anch:.3f}  n_decoy={len(pool)}")
        print(f"{'floor':>7s}{'k':>3s}{'|D|':>9s}{'false-cert':>12s}{'power':>8s}")
        for pct, k, f, fc, pw in rows:
            if k >= 2:
                print(f"p{pct:<6d}{k:>3d}{f:>9.4f}{fc:>12.5f}{pw:>8.3f}")
        r1 = [r for r in rows if r[4] >= PW_MIN]
        r2 = [r for r in rows if r[3] <= FC_MAX]
        s1 = min(r1, key=lambda r: (r[3], -r[4])) if r1 else None
        s2 = max(r2, key=lambda r: (r[4], -r[3])) if r2 else None
        print(f"  R1 (registered, min fc s.t. power>=0.80): "
              f"{'p%d/%d-of-3  fc %.5f  power %.3f' % (s1[0], s1[1], s1[3], s1[4]) if s1 else 'none'}")
        print(f"  R2 (proposed A7, max power s.t. fc<=0.10): "
              f"{'p%d/%d-of-3  fc %.5f  power %.3f' % (s2[0], s2[1], s2[3], s2[4]) if s2 else 'none'}")


if __name__ == "__main__":
    main()
