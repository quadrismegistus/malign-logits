#!/usr/bin/env python3
"""THE REGISTERED GATE. Committed before running.

Separate from scripts/preference_corpus_test.py, which implements the SUPERSEDED
v1 gate -- seven markers, p75 floor, 3-of-7 -- and which was once run by mistake
in place of the registered layer. That script is left untouched as the record of
what v1 was; it is not the gate and must not be invoked for one.

REGISTERED TERMS (docs/preference_corpus_spec.md + amendment 1, audited by lacan
2026-07-27, ruled by RH the same day):

    slate      3 markers, one per surviving construct, from
               docs/preference_corpus_markers_v2.md
                 require  -> prefer      (directive softening)
                 entirely -> mostly      (quantifier qualification)
                 angry    -> concerned   (emotional de-escalation)
    floor      p50 of |D| over the decoy pool -- THE SAME POOL the false
               certification and power were simulated on (scripts/
               tier2_construct_grid.py). hh: 0.0481.
    k          3 of 3
    corpus     hh_rlhf carries the verdict. pku_saferlhf is DESCRIPTIVE-ONLY:
               its chains are badly measured, its MDE threshold is 2.3x laxer,
               and it cannot certify anything the test can use.

    A marker fires iff  D > 0  AND  |D| > floor.

PROPERTIES OF THIS CELL, on the face because they are the disclosure:
  false certification 0.01414, 95% upper 0.0181 -- inside the 0.10 standard,
  so the margin clause is not triggered.
  power 0.80389 against a registered floor of 0.80: MET ON THE POINT ESTIMATE,
  by 0.004. NOT SHOWN at the 95% bootstrap lower bound, which is 0.759
  (sd 0.008 over 1,600 replicates, below 0.80 in all 8 disjoint blocks).
  The registration at 596213c names no estimator and its committed script
  evaluates the point estimate, so the floor is satisfied AS REGISTERED. It is
  not satisfied ON THE EVIDENCE. The gate therefore runs toward error B --
  condemning a live instrument -- more often than a shown-satisfied floor would
  allow. Reading two (shown at the 95% bound) was adopted as standing policy for
  gates registered after 2026-07-27 and deliberately not applied to this one.

TIER 2 IS REFUSED UNLESS THE GATE PASSES, in code and not by discipline. A
failed gate books the instrument-insensitivity finding; computing the chain-pair
sign test on a corpus whose instrument did not certify would be reading a
verdict off an instrument just shown not to work.
"""
import argparse, csv, importlib.util, json, math, statistics as st
import numpy as np

_s = importlib.util.spec_from_file_location("g", "scripts/tier2_construct_grid.py")
g = importlib.util.module_from_spec(_s)
_s.loader.exec_module(g)

MARKERS = [("require", "prefer"), ("entirely", "mostly"), ("angry", "concerned")]
FLOOR_PCT, K_REQUIRED = 50, 3
VERDICT_CORPUS = "hh_rlhf"
MDE_CUT, Z = 2.0, 2.49

CORPORA = {
    "hh_rlhf": ("data/f37_corpus_unigrams_hh_rlhf_chosen_v2.csv",
                "data/f37_corpus_unigrams_hh_rlhf_rejected_v2.csv"),
    "pku_saferlhf": ("data/f37_corpus_unigrams_pku_saferlhf_chosen_v2.csv",
                     "data/f37_corpus_unigrams_pku_saferlhf_rejected_v2.csv"),
}


def load(p):
    return {r["word"]: int(float(r["count"])) for r in csv.DictReader(open(p))}


def L(c, r, Nc, Nr):
    return math.log((c / (Nc - c)) / (r / (Nr - r)))


def se(c, r):
    return math.sqrt(1 / c + 1 / r)


def gate(corp):
    """Fire the three markers. Returns rows, n_fired, floor."""
    pool = g.decoy_pool(corp)
    absd = np.sort(np.abs(pool))
    floor = float(absd[min(int(FLOOR_PCT / 100 * len(absd)), len(absd) - 1)])
    ch, rj = load(CORPORA[corp][0]), load(CORPORA[corp][1])
    Nc, Nr = sum(ch.values()), sum(rj.values())

    rows = []
    for s, t in MARKERS:
        cs, rs, ct, rt = ch.get(s, 0), rj.get(s, 0), ch.get(t, 0), rj.get(t, 0)
        if min(cs, rs, ct, rt) < 20:
            rows.append(dict(source=s, target=t, D=None, se=None,
                             verdict="EXCLUDED <20"))
            continue
        d = L(ct, rt, Nc, Nr) - L(cs, rs, Nc, Nr)
        sd = math.sqrt(se(ct, rt) ** 2 + se(cs, rs) ** 2)
        fired = d > 0 and abs(d) > floor
        rows.append(dict(source=s, target=t, D=d, se=sd,
                         verdict="FIRED" if fired else
                         ("directional, below floor" if d > 0 else "wrong direction")))
    return rows, sum(1 for r in rows if r["verdict"] == "FIRED"), floor, ch, rj, Nc, Nr


def tier2(corp, ch, rj, Nc, Nr):
    """Chain-pair sign test on D_excess. ONLY reachable when the gate passed."""
    chain_words, chains = set(), []
    for r in csv.DictReader(open("data/d2_modal_pairs.csv")):
        s, t = r["source"].strip().lower(), r["modal_target"].strip().lower()
        chain_words.update([s, t])
        if s not in g.STOP and t not in g.STOP:
            chains.append((s, t))
    vocab = {w: ch[w] + rj.get(w, 0) for w in ch
             if ch[w] >= 20 and rj.get(w, 0) >= 20 and w not in g.STOP}
    items = sorted(vocab.items(), key=lambda x: x[1])
    near = lambda f: [w for w, _ in sorted(items, key=lambda x: abs(math.log(x[1]) - math.log(f)))
                      if w not in chain_words][:20]
    rows = []
    for s, t in chains:
        cs, rs, ct, rt = ch.get(s, 0), rj.get(s, 0), ch.get(t, 0), rj.get(t, 0)
        if min(cs, rs, ct, rt) < 20:
            continue
        d = L(ct, rt, Nc, Nr) - L(cs, rs, Nc, Nr)
        sd = math.sqrt(se(ct, rt) ** 2 + se(cs, rs) ** 2)
        ds, dt = near(cs + rs), near(ct + rt)
        dd = []
        for off in range(1, len(dt)):
            dd = [L(ch[b], rj[b], Nc, Nr) - L(ch[a], rj[a], Nc, Nr)
                  for a, b in zip(ds, dt[off:] + dt[:off]) if a != b]
            if len(dd) >= 5:
                break
        if not dd:
            continue
        sem = 1.253 * st.pstdev(dd) / math.sqrt(len(dd))
        mde = math.exp(Z * math.sqrt(sd ** 2 + sem ** 2))
        rows.append(dict(source=s, target=t, D=d, D_excess=d - st.median(dd),
                         mde=mde, informative=mde <= MDE_CUT))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier2", action="store_true",
                    help="also run the chain-pair verdict; REFUSED if the gate fails")
    a = ap.parse_args()

    out = {}
    for corp in CORPORA:
        rows, n, floor, ch, rj, Nc, Nr = gate(corp)
        tag = "" if corp == VERDICT_CORPUS else "   DESCRIPTIVE-ONLY — cannot certify"
        print(f"\n{'=' * 70}\n{corp}{tag}\n{'=' * 70}")
        print(f"floor = p{FLOOR_PCT} of |decoy D| = {floor:.4f}   k required = "
              f"{K_REQUIRED} of {len(MARKERS)}")
        for r in rows:
            ds = f"{r['D']:+.4f}" if r["D"] is not None else "   —   "
            ses = f"±{r['se']:.4f}" if r["se"] is not None else "       "
            print(f"   {r['source']:9s} -> {r['target']:11s} D={ds} {ses}   {r['verdict']}")
        passed = n >= K_REQUIRED
        print(f"   >>> {n}/{len(MARKERS)} FIRED — "
              f"{'GATE PASSES' if passed else 'GATE FAILS'}")
        out[corp] = dict(markers=rows, n_fired=n, floor=floor, passed=passed)

    v = out[VERDICT_CORPUS]
    print(f"\n{'=' * 70}")
    if not v["passed"]:
        print(f"VERDICT CORPUS {VERDICT_CORPUS}: GATE FAILED ({v['n_fired']}/3).")
        print("Books the INSTRUMENT-INSENSITIVITY finding. The chain-pair sign")
        print("test is NOT computed and no verdict on convention is available:")
        print("reading one off an instrument just shown not to certify is the")
        print("error the gate exists to prevent.")
    elif a.tier2:
        print(f"VERDICT CORPUS {VERDICT_CORPUS}: GATE PASSED. Running tier 2.")
        ch, rj = load(CORPORA[VERDICT_CORPUS][0]), load(CORPORA[VERDICT_CORPUS][1])
        rows = tier2(VERDICT_CORPUS, ch, rj, sum(ch.values()), sum(rj.values()))
        inf = [r for r in rows if r["informative"]]
        pos = sum(1 for r in inf if r["D_excess"] > 0)
        print(f"   chain pairs {len(rows)}, informative (MDE<={MDE_CUT}) {len(inf)}")
        print(f"   D_excess > 0 in {pos}/{len(inf)}")
        if inf:
            from math import comb
            p = 2 * sum(comb(len(inf), i) for i in range(max(pos, len(inf) - pos),
                                                         len(inf) + 1)) / 2 ** len(inf)
            print(f"   two-sided sign test p = {min(p, 1.0):.4g}")
            print(f"   median D_excess = {st.median(r['D_excess'] for r in inf):+.4f}")
        out["tier2"] = dict(rows=rows, n_informative=len(inf), n_positive=pos)
    else:
        print(f"VERDICT CORPUS {VERDICT_CORPUS}: GATE PASSED ({v['n_fired']}/3).")
        print("Tier 2 not run — re-invoke with --tier2. Reported separately so the")
        print("gate outcome is on the record before the verdict it licenses.")

    json.dump(out, open("data/preference_corpus_gate_v2.json", "w"),
              indent=1, default=str)
    print("\n-> data/preference_corpus_gate_v2.json")


if __name__ == "__main__":
    main()
