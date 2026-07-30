"""Draw the MASS-MATCHED item set for the attenuation test.

    uv run .venv/bin/python scripts/f13_draw_matched.py

WHAT CHANGED AND WHY, since this is the third draw and the differences are the point.

1. THE CONTROL IS NOW MASS-MATCHED ([710].1). Previously a decoy needed
   |delta| <= 0.0005 AND p_base >= 0.005, while a riser needed only delta >= 0.003
   and could start anywhere -- so decoys sat at median p_base 0.0072 against risers'
   0.0335 (p = 3.4e-22, non-overlapping at the bottom). Base mass proxies
   plausibility and plausibility proxies apparent relatedness, so the old contrast
   was confounded in the proposer's favour. Here each REAL riser gets the stationary
   word whose p_base is NEAREST its own, within a factor of TOL, and the achieved
   match is printed.

2. THE STATIONARY DEFINITION IS LOOSENED, and this is my correction rather than
   the docket's. EPS = 0.0005 was an arbitrary number I chose, and on amber it left
   14 decoys against 142 real items -- a control arm too small to test anything. The
   natural complement of "moved" is "did not move BY THE CRITERION", i.e.
   |delta| < DT. A word that gained 0.001 is not a riser under our own rule, so it
   is an admissible non-mover. That is a much larger pool and a defensible one.

3. UP TO K DECOYS PER REAL ITEM, to buy statistical power in the arm that had none.
   Each is matched independently; all are marked with their own match quality so a
   reader can restrict to the tightest.

4. THE CONTENT FILTER IS MECHANICAL AND UPSTREAM (RH's correction). A stopword
   list plus a non-alphabetic check reproduces three-coder consensus on
   b_is_content_word at 91.9%, so the coder's judgment is spent on the ~8% that is
   genuinely context-dependent instead of on `be` and `the`. Both sides must pass.

5. INTENSITY-COMPARABLE SLOTS ONLY for the primary. "Is B milder than A" is a
   coherent question for ACT and RESULT slots and is not one for REF (`mouth` vs
   `ears` has no intensity), NARR, SENSE or UTTER. Those are drawn and marked but
   excluded from the primary population, per [695]'s pooling ban.

THE TEST THIS EXISTS FOR: within a prompt, holding the faller fixed, is the word
that GAINED mass judged milder than a co-present word that did NOT gain mass?
That is (B) as attenuation. The paradigmatic-relatedness version returned null
because any two admissible fillers of a violence slot are related; this asks
something the slot's paradigm cannot answer for free.
"""
from __future__ import annotations
import argparse, math, os, sys, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from malign_logits.cache import get_cache
import malign_logits.taxonomy as T
from scripts.f13_draw_relation_items import (
    surface_probs, THETA, FLOOR, DT, DEV, TYPE_OF, FUNC, admissible)

TOL = 3.0          # decoy p_base within this factor of the real riser's
K_DECOY = 3        # up to this many matched decoys per real item
INTENSITY_SLOTS = {"ACT", "RESULT"}
OUT = "data/f13_matched_items.parquet"


def content(w: str) -> bool:
    """Mechanical content test: 91.9% against three-coder consensus."""
    w = (w or "").strip()
    if not w or not any(c.isalpha() for c in w):
        return False
    return w.lower() not in FUNC


def edges(seen):
    out = []
    for name, f in T.MODEL_FAMILIES.items():
        b = getattr(f, "base", None)
        if b not in seen:
            continue
        for arm in ("superego", "reinforced_superego", "ego"):
            a = getattr(f, arm, None)
            if a and a in seen and a != b:
                out.append((name, arm, b, a))
    return out


def main():
    cm = get_cache()
    s = cm._stash("true_word_probs")
    seen = {(dict(k) if not isinstance(k, dict) else k).get("model") for k in s}
    E = edges(seen)
    print(f"true_word_probs entries at read time: {len(s):,}")
    print(f"edges {len(E)}  TOL {TOL}x  K_DECOY {K_DECOY}  "
          f"FLOOR {FLOOR}  DT {DT}  stationary = |delta| < DT")

    rows, drop = [], collections.Counter()
    for fam, arm, bid, aid in E:
        for nm, p in ((k, v.rstrip()) for k, v in T.DEFAULT_PROMPTS.items()):
            pb, pa = (cm.get_true_word_probs(bid, p, theta=THETA),
                      cm.get_true_word_probs(aid, p, theta=THETA))
            if not pb or not pa:
                drop["prompt absent"] += 1
                continue
            B = surface_probs(pb, lambda w: admissible(w) and content(w))
            A = surface_probs(pa, lambda w: admissible(w) and content(w))
            W = set(B) | set(A)
            d = {w: A.get(w, 0.0) - B.get(w, 0.0) for w in W}
            fallers = [w for w in W if B.get(w, 0) >= FLOOR and d[w] <= -DT]
            risers = [w for w in W if d[w] >= DT]
            stat = [w for w in W if abs(d[w]) < DT and B.get(w, 0) > 0]
            if not (fallers and risers):
                drop["no faller or riser"] += 1
                continue
            a_w = max(fallers, key=lambda w: B[w])
            b_w = max(risers, key=lambda w: d[w])
            pbw = B.get(b_w, 0.0)
            meta = dict(family=fam, arm=arm, base_id=bid, aligned_id=aid,
                        prompt_name=nm, prompt=p, slot=TYPE_OF[nm],
                        intensity_slot=TYPE_OF[nm] in INTENSITY_SLOTS)
            rows.append(dict(item_class="REAL", a=a_w, b=b_w, pb_base=pbw,
                             pb_al=A.get(b_w, 0.0), delta_b=d[b_w],
                             match_log10=0.0, **meta))
            if pbw <= 0:
                drop["riser has no base mass (cannot match)"] += 1
                continue
            cand = sorted(((abs(math.log10(B[w] / pbw)), w) for w in stat
                           if w != b_w and w != a_w and B.get(w, 0) > 0))
            took = 0
            for r_, w in cand:
                if r_ > math.log10(TOL):
                    break
                rows.append(dict(item_class="NEAR-MISS", a=a_w, b=w,
                                 pb_base=B[w], pb_al=A.get(w, 0.0),
                                 delta_b=d[w], match_log10=round(r_, 4), **meta))
                took += 1
                if took >= K_DECOY:
                    break
            if not took:
                drop["no decoy within tolerance"] += 1

    df = pd.DataFrame(rows)
    n0 = len(df)
    df = df[[(r.prompt, r.a, r.b) not in DEV for r in df.itertuples()]].reset_index(drop=True)
    print(f"\nitems {n0}  dev removed {n0 - len(df)}  -> {len(df)}")
    for k, v in drop.most_common():
        print(f"    dropped: {k:<38} {v}")
    print("\nBY CLASS:")
    print(df.item_class.value_counts().to_string())
    mm = df[df.item_class == "NEAR-MISS"].match_log10
    if len(mm):
        print(f"\nACHIEVED MATCH  median |log10(p_decoy/p_real)| = {mm.median():.3f} "
              f"(factor {10**mm.median():.2f})   worst {mm.max():.3f} "
              f"(factor {10**mm.max():.2f})")
    print("\nMASS BALANCE -- the confound this draw exists to remove:")
    print(df.groupby("item_class").pb_base.agg(["median", "mean", "size"]).to_string())
    from scipy.stats import mannwhitneyu
    r = df[df.item_class == "REAL"].pb_base
    q = df[df.item_class == "NEAR-MISS"].pb_base
    if len(r) and len(q):
        print(f"  Mann-Whitney on p_base: p = {mannwhitneyu(r, q).pvalue:.3g}   "
              f"(was 3.4e-22 unmatched; want NON-significant)")
    print("\nPRIMARY POPULATION (intensity-comparable slots, both classes present):")
    pe = df[df.intensity_slot]
    print(pe.groupby(["slot", "item_class"]).size().to_string())
    cells = pe.groupby(["prompt", "base_id", "aligned_id"]).item_class.nunique().eq(2).sum()
    print(f"\n  prompt-edge cells holding BOTH a REAL and >=1 matched decoy: {cells}")
    print(f"  families {pe.family.nunique()}   prompts {pe.prompt_name.nunique()}")
    df.to_parquet(OUT, compression="zstd", index=False)
    print(f"\nwrote {len(df)} -> {OUT}")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
