#!/usr/bin/env python3
"""The preference-corpus test, as registered in docs/preference_corpus_spec.md.

GATE FIRST. Seven marker pairs declared in docs/preference_corpus_markers.md
before any D was computed. A pair fires if D > 0 AND |D| > the p75 decoy floor.
Gate passes at 3/7. Fewer books the INSTRUMENT-INSENSITIVITY finding, not a
verdict on convention.

Statistic:  L(w) = log[(c_w/(N_c-c_w)) / (r_w/(N_r-r_w))]
            D    = L(target) - L(source)
            D_excess = D_chain - median(D_decoy)   <- the sign test runs on this
Decoys: nearest-20 in log frequency among non-chain, non-stopword words with
>=20 occurrences in each arm.
"""
import collections, csv, json, math, statistics as st

CORPORA = {
    "hh_rlhf": ("data/f37_corpus_unigrams_hh_rlhf_chosen_v2.csv",
                "data/f37_corpus_unigrams_hh_rlhf_rejected_v2.csv", 0.0903),
    "pku_saferlhf": ("data/f37_corpus_unigrams_pku_saferlhf_chosen_v2.csv",
                     "data/f37_corpus_unigrams_pku_saferlhf_rejected_v2.csv", 0.2657),
}
MARKERS = [("must","should"),("never","rarely"),("always","often"),
           ("wrong","incorrect"),("stupid","unclear"),("no","unfortunately"),
           ("obviously","perhaps")]
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been
being it its this that these those he she they them his her their i you we us our your me my
not no so do does did done have has had will would can could should may might must there here
what which who whom when where why how all any both each few more most other some such than too
very s t just don now up out off over under again further then once about into""".split())
K, MDE_CUT, Z = 20, 2.0, 2.49


def load(p): return {r["word"]: int(float(r["count"])) for r in csv.DictReader(open(p))}
def L(c, r, Nc, Nr): return math.log((c/(Nc-c))/(r/(Nr-r)))
def se(c, r): return math.sqrt(1/c + 1/r)


def run(corp, chains):
    pc, pr, floor = CORPORA[corp]
    ch, rj = load(pc), load(pr)
    Nc, Nr = sum(ch.values()), sum(rj.values())
    vocab = {w: ch[w]+rj.get(w,0) for w in ch
             if ch[w] >= 20 and rj.get(w,0) >= 20 and w not in STOP}
    items = sorted(vocab.items(), key=lambda x: x[1])

    def near(f, exclude):
        return [w for w,_ in sorted(items, key=lambda x: abs(math.log(x[1])-math.log(f)))
                if w not in exclude][:K]

    def pair_D(s, t):
        if min(ch.get(s,0), rj.get(s,0), ch.get(t,0), rj.get(t,0)) < 20:
            return None
        return (L(ch[t],rj[t],Nc,Nr) - L(ch[s],rj[s],Nc,Nr),
                math.sqrt(se(ch[t],rj[t])**2 + se(ch[s],rj[s])**2))

    # ---- GATE ----
    gate = []
    for s, t in MARKERS:
        d = pair_D(s, t)
        if d is None:
            gate.append((s, t, None, None, "EXCLUDED <20")); continue
        fired = d[0] > 0 and abs(d[0]) > floor
        gate.append((s, t, d[0], d[1], "FIRED" if fired else
                     ("directional, below floor" if d[0] > 0 else "wrong direction")))
    n_fired = sum(1 for g in gate if g[4] == "FIRED")

    # ---- CHAIN PAIRS ----
    cw = {w for p in chains for w in p}
    rows = []
    for s, t in chains:
        d = pair_D(s, t)
        if d is None: continue
        ds = near(ch.get(s,0)+rj.get(s,0), cw)
        dt = near(ch.get(t,0)+rj.get(t,0), cw)
        dd = []
        for off in range(1, len(dt)):
            dd = [L(ch[b],rj[b],Nc,Nr)-L(ch[a],rj[a],Nc,Nr)
                  for a,b in zip(ds, dt[off:]+dt[:off]) if a != b]
            if len(dd) >= 5: break
        if not dd: continue
        sd = st.pstdev(dd); sem = 1.253*sd/math.sqrt(len(dd))
        mde = math.exp(Z*math.sqrt(d[1]**2 + sem**2))
        rows.append(dict(source=s, target=t, D=d[0], D_excess=d[0]-st.median(dd),
                         mde=mde, informative=mde <= MDE_CUT))
    return gate, n_fired, rows, floor


def main():
    chains = []
    for r in csv.DictReader(open("data/d2_modal_pairs.csv")):
        s, t = r["source"].strip().lower(), r["modal_target"].strip().lower()
        if s not in STOP and t not in STOP:
            chains.append((s, t))

    out = {}
    for corp in CORPORA:
        gate, n_fired, rows, floor = run(corp, chains)
        out[corp] = dict(gate=gate, n_fired=n_fired, rows=rows, floor=floor)
        print(f"\n{'='*66}\n{corp}   (p75 decoy floor {floor:.4f})\n{'='*66}")
        print("GATE — all seven disclosed:")
        for s, t, d, sd, verdict in gate:
            ds = f"{d:+.4f}" if d is not None else "   —   "
            print(f"   {s:11s} -> {t:14s} D={ds}   {verdict}")
        print(f"   >>> {n_fired}/7 FIRED  (gate passes at 3)")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'gate'} for k, v in out.items()},
              open("data/preference_corpus_results.json", "w"), indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
