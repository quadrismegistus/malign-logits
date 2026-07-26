"""Is pku's ~3x decoy dispersion length-driven or scarcity-driven?

lacan, rev 6: a response-level instrument removes response-length variability
from the variance but CANNOT create occurrences. So the registered wording turns
on which component dominates.

Test: under pure counting (Poisson/binomial), SE(D) = sqrt(1/a+1/b+1/c+1/d) from
the four cell counts alone. Compare the observed dispersion of D across decoy
pairs with that prediction. Ratio ~1 => dispersion is entirely scarcity/counting
and no counting scheme helps. Ratio >>1 => overdispersion, which is where
length variability and within-response clustering live, and which a
response-level design can remove.
"""
import csv, math, statistics as st
exec(open('scripts/tier2_power_check.py').read().split('chains, chain_words')[0])

chains, chain_words = [], set()
for r in csv.DictReader(open("data/d2_modal_pairs.csv")):
    s, t = r["source"].strip().lower(), r["modal_target"].strip().lower()
    chain_words.update([s, t])
    if s not in STOP and t not in STOP:
        chains.append((s, t))

K = 20
out = {}
for corp, (pc, pr) in CORPORA.items():
    ch, rj = load(pc), load(pr)
    Nc, Nr = sum(ch.values()), sum(rj.values())
    vocab = {w: ch[w] + rj.get(w, 0) for w in ch
             if ch[w] >= 20 and rj.get(w, 0) >= 20 and w not in STOP and w not in chain_words}
    items = sorted(vocab.items(), key=lambda x: x[1])
    nearest = lambda f, k: [w for w, _ in sorted(items, key=lambda x: abs(math.log(x[1]) - math.log(f)))[:k]]

    Ds, pred = [], []
    for s, t in chains:
        cs, rs, ct, rt = ch.get(s,0), rj.get(s,0), ch.get(t,0), rj.get(t,0)
        if min(cs, rs, ct, rt) < 20: continue
        ds, dt = nearest(cs+rs, K), nearest(ct+rt, K)
        for off in range(1, len(dt)):
            pairs = [(a, b) for a, b in zip(ds, dt[off:]+dt[:off]) if a != b]
            if len(pairs) >= 5: break
        for a, b in pairs:
            Ds.append(log_or(ch[b],rj[b],Nc,Nr) - log_or(ch[a],rj[a],Nc,Nr))
            pred.append(math.sqrt(1/ch[a]+1/rj[a]+1/ch[b]+1/rj[b]))
    obs = st.pstdev(Ds)
    pm = math.sqrt(st.mean([p**2 for p in pred]))   # RMS of the counting prediction
    out[corp] = (obs, pm, obs/pm, st.median([abs(x) for x in Ds]))
    print(f"=== {corp} ===  n_decoy={len(Ds):,}")
    print(f"  observed sd(D)                     {obs:.4f}")
    print(f"  predicted sd(D) from counts alone  {pm:.4f}   (RMS of sqrt(1/a+1/b+1/c+1/d))")
    print(f"  OVERDISPERSION FACTOR              {obs/pm:.2f}x")
    print(f"  median |D|                         {st.median([abs(x) for x in Ds]):.4f}\n")

h, p = out["hh_rlhf"], out["pku_saferlhf"]
print("=== decomposition of pku's excess dispersion over hh ===")
print(f"  observed dispersion ratio pku/hh        {p[0]/h[0]:.2f}x")
print(f"  counting-predicted ratio pku/hh         {p[1]/h[1]:.2f}x   <- SCARCITY component")
print(f"  residual overdispersion ratio pku/hh    {p[2]/h[2]:.2f}x   <- where LENGTH could live")
share = math.log(p[1]/h[1]) / math.log(p[0]/h[0])
print(f"\n  scarcity explains {share*100:.0f}% of the log dispersion gap; "
      f"overdispersion explains {(1-share)*100:.0f}%")

# CEILING: if a response-level design removed ALL overdispersion in pku,
# leaving only the irreducible counting component, what gate power results?
import random
random.seed(20260726)
