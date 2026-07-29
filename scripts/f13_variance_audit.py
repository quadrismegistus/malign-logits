"""Independent audit of [432]: is amber's weak L1 correlation a range-restriction
artefact of near-constant per-layer similarity?

    uv run .venv/bin/python scripts/f13_variance_audit.py

WHY THIS SCRIPT EXISTS. lacan refuted my [427].4 confound using the POOLED sd
([430].3), then retracted at [432] on the ground that the relevant quantity is the
PER-LAYER sd. registrar had already withdrawn a design on the strength of the
refutation. Three seats have now been wrong in sequence about one number, so it is
computed here from raw CSVs, touching no other seat's frame.

Section 1-3 reproduce [432]'s claims. Section 4 is the part neither seat has run:
a WITHIN-AMBER matched-band test. [432] establishes that amber's L1 has almost no
variance and infers that "L3 strongest" is what the variance profile predicts on its
own. That is an inference, not a measurement. It is measurable: restrict amber's L3
to L1's OWN similarity band -- matching location and spread together, within one
corpus, no cross-corpus comparison -- and recompute. If the restricted L3 collapses
toward L1's r, variance explains the within-amber gradient. If it stays strong, a
depth effect survives inside amber even at matched variance.

The cross-corpus band comparison is impossible (the corpora are disjoint at L1,
P3' stage 1). The within-corpus one is not, and it is the same question.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402

SIX = ["olmo", "olmo-tiny", "llama", "qwen", "zephyr", "tulu"]
MIN_PAIRS = 10  # the declared A4 parameter, kept so cells match the primary


def load(fam):
    d = pd.read_csv(os.path.join(PATH_DATA, f"taxonomy_{fam}.csv"))
    d = d.dropna(subset=["similarity", "syntagmatic_js", "layer", "prompt"])
    # layer index within family: the three conventional depths, ranked
    lv = sorted(d.layer.unique())
    d["L"] = d.layer.map({v: f"L{i+1}" for i, v in enumerate(lv)})
    return d


def within_prompt_r(d):
    """Median within-prompt Pearson r, prompt as unit, MIN_PAIRS floor.

    Mirrors the frozen primary: correlate inside each prompt, then take the
    median over prompts. Never pooled across prompts."""
    rs = []
    for _, g in d.groupby("prompt"):
        if len(g) < MIN_PAIRS:
            continue
        if g.similarity.std() == 0 or g.syntagmatic_js.std() == 0:
            continue
        rs.append(np.corrcoef(g.similarity, g.syntagmatic_js)[0, 1])
    return (np.median(rs) if rs else np.nan), len(rs)


def main():
    fams = SIX + ["amber"]
    data = {f: load(f) for f in fams}

    print("=" * 74)
    print("1. PER-LAYER sd(similarity) -- [432].1 reproduced")
    print("=" * 74)
    print(f"{'family':<11}{'L1':>9}{'L2':>9}{'L3':>9}{'POOLED':>10}")
    for f in fams:
        d = data[f]
        row = [d[d.L == L].similarity.std() for L in ("L1", "L2", "L3")]
        mark = "   <--" if f == "amber" else ""
        print(f"{f:<11}" + "".join(f"{v:>9.4f}" for v in row)
              + f"{d.similarity.std():>10.4f}{mark}")
    six_l1 = np.mean([data[f][data[f].L == "L1"].similarity.std() for f in SIX])
    am_l1 = data["amber"][data["amber"].L == "L1"].similarity.std()
    print(f"\namber L1 sd is {am_l1/six_l1:.0%} of the six families' mean L1 sd "
          f"({am_l1:.4f} vs {six_l1:.4f})")

    print("\n" + "=" * 74)
    print("2. IS THE SIX FAMILIES' sd FLAT ACROSS DEPTH? ([432].4 -- this is what")
    print("   protects the in-sample gradient from the same objection)")
    print("=" * 74)
    for f in SIX:
        d = data[f]
        v = [d[d.L == L].similarity.std() for L in ("L1", "L2", "L3")]
        print(f"  {f:<11} L1->L3 {v[0]:.4f} -> {v[2]:.4f}   "
              f"delta {v[2]-v[0]:+.4f}")

    print("\n" + "=" * 74)
    print("3. OVERLAP AT L1 -- are the corpora disjoint there? ([432].2)")
    print("=" * 74)
    a1 = data["amber"][data["amber"].L == "L1"].similarity
    print(f"  amber L1: min {a1.min():.3f}  median {a1.median():.3f}  "
          f"p95 {a1.quantile(.95):.3f}  n {len(a1):,}")
    print(f"  {'family':<11}{'% of L1 pairs > 0.75':>22}{'p95':>9}")
    for f in SIX:
        s = data[f][data[f].L == "L1"].similarity
        print(f"  {f:<11}{(s > 0.75).mean()*100:>21.1f}%{s.quantile(.95):>9.3f}")
    print(f"  {'amber':<11}{(a1 > 0.75).mean()*100:>21.1f}%{a1.quantile(.95):>9.3f}")

    print("\n" + "=" * 74)
    print("4. WITHIN-AMBER MATCHED-BAND TEST (new -- neither seat has run this)")
    print("   Restrict amber's L2/L3 to L1's own similarity band. Same corpus,")
    print("   location and spread matched together. Does the gradient survive?")
    print("=" * 74)
    am = data["amber"]
    lo, hi = a1.min(), a1.max()
    print(f"  L1 band = [{lo:.3f}, {hi:.3f}]  (L1's full observed range)\n")
    print(f"  {'axis':<13}{'layer':<7}{'n_pairs':>9}{'sd':>8}{'median r':>10}"
          f"{'prompts':>9}")
    for axis in sorted(am.axis.dropna().unique()):
        for L in ("L1", "L2", "L3"):
            g = am[(am.axis == axis) & (am.L == L)]
            gb = g[(g.similarity >= lo) & (g.similarity <= hi)]
            r_full, n_full = within_prompt_r(g)
            r_band, n_band = within_prompt_r(gb)
            print(f"  {axis:<13}{L:<7}{len(g):>9,}{g.similarity.std():>8.4f}"
                  f"{r_full:>10.3f}{n_full:>9}")
            print(f"  {'':<13}{'  band':<7}{len(gb):>9,}"
                  f"{gb.similarity.std() if len(gb) > 1 else float('nan'):>8.4f}"
                  f"{r_band:>10.3f}{n_band:>9}")
        print()

    print("=" * 74)
    print("5. SPREAD-MATCHED, REGION-FREE (§4 confounds the two: restricting L3")
    print("   to L1's band matches spread AND moves L3 into L1's region, so a")
    print("   nonlinearity would read as a variance effect. Here each layer is")
    print("   narrowed around its OWN median to L1's sd -- spread matched, each")
    print("   layer left where it sits.)")
    print("=" * 74)
    target = a1.std()
    print(f"  target sd = amber L1's own sd = {target:.4f}\n")
    print(f"  {'axis':<13}{'layer':<8}{'keep':>7}{'n':>8}{'sd':>8}"
          f"{'median':>8}{'r':>8}{'prompts':>9}")
    for axis in sorted(am.axis.dropna().unique()):
        for L in ("L1", "L2", "L3"):
            g = am[(am.axis == axis) & (am.L == L)]
            best = None
            for frac in np.arange(0.10, 1.01, 0.02):
                lo_q = max(0.0, 0.5 - frac / 2)
                hi_q = min(1.0, 0.5 + frac / 2)
                w = g[(g.similarity >= g.similarity.quantile(lo_q))
                      & (g.similarity <= g.similarity.quantile(hi_q))]
                if len(w) < MIN_PAIRS:
                    continue
                gap = abs(w.similarity.std() - target)
                if best is None or gap < best[0]:
                    best = (gap, frac, w)
            _, frac, w = best
            r, npr = within_prompt_r(w)
            print(f"  {axis:<13}{L:<8}{frac:>6.0%}{len(w):>8,}"
                  f"{w.similarity.std():>8.4f}{w.similarity.median():>8.3f}"
                  f"{r:>8.3f}{npr:>9}")
        print()

    print("=" * 74)
    print("6. CONTROL: does spread-matching destroy a gradient that is real?")
    print("   Same procedure on the six in-sample families, each matched to its")
    print("   OWN minimum per-layer sd. Their sd is already near-flat across")
    print("   depth, so a real gradient should survive nearly intact. If it")
    print("   collapses here too, the procedure is the artefact, not amber.")
    print("=" * 74)
    print(f"  {'family':<11}{'axis':<13}{'r L1':>8}{'r L2':>8}{'r L3':>8}"
          f"{'span':>8}{'matched span':>14}")
    for f in SIX:
        d = data[f]
        for axis in sorted(d.axis.dropna().unique()):
            raw, mat = [], []
            for L in ("L1", "L2", "L3"):
                g = d[(d.axis == axis) & (d.L == L)]
                if len(g) < MIN_PAIRS:
                    raw, mat = [], []
                    break
                raw.append(within_prompt_r(g)[0])
                tgt = min(d[(d.axis == axis) & (d.L == x)].similarity.std()
                          for x in ("L1", "L2", "L3"))
                best = None
                for frac in np.arange(0.10, 1.01, 0.02):
                    w = g[(g.similarity >= g.similarity.quantile(max(0, .5-frac/2)))
                          & (g.similarity <= g.similarity.quantile(min(1, .5+frac/2)))]
                    if len(w) < MIN_PAIRS:
                        continue
                    gap = abs(w.similarity.std() - tgt)
                    if best is None or gap < best[0]:
                        best = (gap, w)
                mat.append(within_prompt_r(best[1])[0])
            if not raw or any(np.isnan(x) for x in raw + mat):
                continue
            print(f"  {f:<11}{axis:<13}" + "".join(f"{v:>8.3f}" for v in mat)
                  + f"{abs(raw[2]-raw[0]):>8.3f}{abs(mat[2]-mat[0]):>14.3f}")
    print("\n  (r columns are MATCHED; 'span' is the unmatched |L3-L1|,")
    print("   'matched span' the same after spread-matching. amber for")
    print("   comparison: repression 0.401 -> 0.049, sublimation 0.416 -> 0.097.)")

    print("\n" + "=" * 74)
    print("READING. If band-restricted L3 falls toward L1's r, the within-amber")
    print("depth gradient is the variance profile. If it holds, a depth effect")
    print("survives at matched variance and [432].3's inference is too strong.")
    print("Composition first: any band cell below MIN_PAIRS in <10 prompts is")
    print("not usable and is visible in the prompts column above.")
    print("=" * 74)


if __name__ == "__main__":
    main()
