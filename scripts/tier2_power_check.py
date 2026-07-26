"""Design fix: nearest-k in log-frequency instead of a fixed +/-20% band."""
import csv, math, statistics as st
exec(open('/private/tmp/claude-502/-Users-rj416-github-malign-logits/4be0899b-7f9e-4e83-97c7-b68a7079afee/scratchpad/decoy_feasibility.py').read().split('# chain pairs')[0])

chains, chain_words = [], set()
for r in csv.DictReader(open("data/d2_modal_pairs.csv")):
    s, t = r["source"].strip().lower(), r["modal_target"].strip().lower()
    chain_words.update([s, t])
    if s not in STOP and t not in STOP:
        chains.append((s, t))

K = 20
for corp, (pc, pr) in CORPORA.items():
    ch, rj = load(pc), load(pr)
    Nc, Nr = sum(ch.values()), sum(rj.values())
    vocab = {w: ch[w] + rj.get(w, 0) for w in ch
             if ch[w] >= 20 and rj.get(w, 0) >= 20 and w not in STOP and w not in chain_words}
    items = sorted(vocab.items(), key=lambda x: x[1])
    print(f"=== {corp} === decoy vocab {len(vocab):,}")

    def nearest(f, k):
        return [w for w, _ in sorted(items, key=lambda x: abs(math.log(x[1]) - math.log(f)))[:k]]

    rows = []
    for s, t in chains:
        cs, rs, ct, rt = ch.get(s,0), rj.get(s,0), ch.get(t,0), rj.get(t,0)
        if min(cs, rs, ct, rt) < 20: continue
        D = log_or(ct,rt,Nc,Nr) - log_or(cs,rs,Nc,Nr)
        seD = math.sqrt(se_log_or(ct,rt)**2 + se_log_or(cs,rs)**2)
        fs, ft = cs+rs, ct+rt
        ds, dt = nearest(fs, K), nearest(ft, K)
        # worst-case log-frequency mismatch actually achieved
        mm = max(max(abs(math.log(vocab[w]/fs)) for w in ds),
                 max(abs(math.log(vocab[w]/ft)) for w in dt))
        # rotate the target list so identical-frequency pairs don't self-pair
        dd = []
        for off in range(1, len(dt)):
            dd = [log_or(ch[b],rj[b],Nc,Nr) - log_or(ch[a],rj[a],Nc,Nr)
                  for a, b in zip(ds, dt[off:]+dt[:off]) if a != b]
            if len(dd) >= 5: break
        sd = st.pstdev(dd); seM = 1.253*sd/math.sqrt(len(dd))
        seE = math.sqrt(seD**2 + seM**2)
        rows.append((s,t,fs,ft,len(dd),mm,D,st.median(dd),seD,seE))
    print(f"  eligible {len(rows)}; all have exactly {K} decoys by construction")
    infl = [math.exp(Z*r[9])/math.exp(Z*r[8]) for r in rows]
    mdeE = [math.exp(Z*r[9]) for r in rows]
    print(f"  MDE inflation D->D_excess: median {st.median(infl):.3f}, max {max(infl):.3f}")
    print(f"  MDE(D_excess): median {st.median(mdeE):.2f}x, max {max(mdeE):.2f}x, "
          f"n<=2.0x = {sum(1 for m in mdeE if m<=2.0)}/{len(rows)}")
    print(f"  worst achieved log-freq mismatch: median {st.median([r[5] for r in rows]):.3f}, "
          f"max {max(r[5] for r in rows):.3f}  (0.182 == +/-20%)")
    print(f"  decoy median |D| : median {st.median([abs(r[7]) for r in rows]):.4f}")
    # positive control frequency profile
    for a,b in [("sorry","unfortunately")]:
        f1, f2 = ch.get(a,0)+rj.get(a,0), ch.get(b,0)+rj.get(b,0)
        chainf = st.median([r[2] for r in rows] + [r[3] for r in rows])
        print(f"  CONTROL {a}({f1:,}) -> {b}({f2:,}); median chain-word freq {chainf:,.0f}")
        if min(ch.get(a,0),rj.get(a,0),ch.get(b,0),rj.get(b,0)) >= 20:
            Dc = log_or(ch[b],rj[b],Nc,Nr) - log_or(ch[a],rj[a],Nc,Nr)
            seC = math.sqrt(se_log_or(ch[b],rj[b])**2 + se_log_or(ch[a],rj[a])**2)
            print(f"    D_control={Dc:+.3f}  MDE={math.exp(Z*seC):.2f}x "
                  f"(vs median chain MDE {st.median(mdeE):.2f}x)")
        else:
            print(f"    UNDER THRESHOLD in this corpus")
    print()
