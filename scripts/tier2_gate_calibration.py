"""Calibrate the tier-2 positive-control gate.

lacan audit, rev 4: the FIRED definition (D>0 AND |D| > median|D_decoy|) uses a
dispersion CENTRE as a null threshold, so half of null pairs clear it by
construction. Gate rates are computed here by simulation from the REAL decoy D
distributions rather than from a symmetry argument, and both directions are
reported: false certification under the null AND gate power under a true effect.
"""
import csv, math, random, statistics as st

random.seed(20260726)
exec(open('scripts/tier2_power_check.py').read().split('chains, chain_words')[0])

chains, chain_words = [], set()
for r in csv.DictReader(open("data/d2_modal_pairs.csv")):
    s, t = r["source"].strip().lower(), r["modal_target"].strip().lower()
    chain_words.update([s, t])
    if s not in STOP and t not in STOP:
        chains.append((s, t))

K, NSIM = 20, 200_000

for corp, (pc, pr) in CORPORA.items():
    ch, rj = load(pc), load(pr)
    Nc, Nr = sum(ch.values()), sum(rj.values())
    vocab = {w: ch[w] + rj.get(w, 0) for w in ch
             if ch[w] >= 20 and rj.get(w, 0) >= 20 and w not in STOP and w not in chain_words}
    items = sorted(vocab.items(), key=lambda x: x[1])

    def nearest(f, k):
        return [w for w, _ in sorted(items, key=lambda x: abs(math.log(x[1]) - math.log(f)))[:k]]

    pool = []          # pooled decoy D values across all eligible chain pairs
    for s, t in chains:
        cs, rs, ct, rt = ch.get(s,0), rj.get(s,0), ch.get(t,0), rj.get(t,0)
        if min(cs, rs, ct, rt) < 20: continue
        ds, dt = nearest(cs+rs, K), nearest(ct+rt, K)
        for off in range(1, len(dt)):
            dd = [log_or(ch[b],rj[b],Nc,Nr) - log_or(ch[a],rj[a],Nc,Nr)
                  for a, b in zip(ds, dt[off:]+dt[:off]) if a != b]
            if len(dd) >= 5: break
        pool += dd

    absd = sorted(abs(x) for x in pool)
    def q(p): return absd[min(int(p*len(absd)), len(absd)-1)]
    floors = {"median": q(.50), "p75": q(.75), "p90": q(.90), "p95": q(.95)}
    frac_pos = sum(1 for x in pool if x > 0) / len(pool)
    print(f"=== {corp} ===  pooled decoy D: n={len(pool):,}  "
          f"P(D>0)={frac_pos:.3f}  median={st.median(pool):+.4f}")
    print(f"  |D| floors: " + "  ".join(f"{k}={v:.4f}" for k, v in floors.items()))

    def rate(floor, need, delta):
        """P(gate passes): 3 candidates drawn from decoy D shifted by delta."""
        hit = 0
        for _ in range(NSIM):
            n = sum(1 for _ in range(3)
                    if (d := random.choice(pool) + delta) > 0 and abs(d) > floor)
            hit += (n >= need)
        return hit / NSIM

    print(f"  {'gate':22s}{'FALSE-CERT':>12s}{'power d=.05':>13s}{'power d=.10':>13s}{'power d=.20':>13s}")
    for fname, fv in floors.items():
        for need in (1, 2, 3):
            fc = rate(fv, need, 0.0)
            mark = "  <- clears .10" if fc <= 0.10 else ""
            print(f"  {fname+' floor, '+str(need)+'-of-3':22s}{fc:12.3f}"
                  f"{rate(fv,need,0.05):13.3f}{rate(fv,need,0.10):13.3f}"
                  f"{rate(fv,need,0.20):13.3f}{mark}")
    print()

# Does enlarging the candidate set buy a better gate than 3 allows?
print("=== candidate-set size sweep (hh_rlhf, p75 floor) ===")
ch, rj = load(CORPORA["hh_rlhf"][0]), load(CORPORA["hh_rlhf"][1])
