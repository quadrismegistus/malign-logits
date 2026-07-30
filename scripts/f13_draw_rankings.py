"""Draw the intensity-ranking item set: one item per (prompt, edge), whole pool.

    uv run .venv/bin/python scripts/f13_draw_rankings.py

THE DESIGN, and it needs no decoy.

For each (prompt, base->aligned edge) the item is THE CANDIDATE POOL the base model
actually had: every admissible content word at or above FLOOR in the base
distribution, plus any riser that gained DT or more (a riser may start below FLOOR --
that is what rising means). Capped at POOL_MAX by base mass, with the faller and the
riser forced in so the item can never omit the two words the test is about.

The coder ranks that pool by intensity and never learns which word moved.

THE STATISTIC. Let n be the number of words the coder ranked and r the 1-indexed
position of a word, most intense first. Normalised rank = (r-1)/(n-1), so 0 is the
most intense word in the slot and 1 the least.

    NULL: alignment's riser is a random draw from the pool, so its expected
          normalised rank is 0.5.
    (B):  the riser sits BELOW the faller -- normalised rank strictly greater --
          and above 0.5 on average.

Both are permutation tests over the coder's own ordering. No probability enters the
statistic, so the mass confound that consumed the matched-decoy design cannot arise:
rank is scale-free.

WHY THE ORDER IS SHUFFLED, with a declared seed. A coder shown a list already sorted
by probability will anchor on it, and the pool is naturally built in mass order. The
seed is fixed so the item set is reproducible and so a later reader can verify the
presentation order was not chosen after the fact.

WHAT NEVER REACHES THE CODER: the probabilities, the deltas, the arm, the family, the
model id, the slot type, which word fell, which word rose, or that anything moved.
The item is a prompt and a shuffled vocabulary.

VALIDITY CHECK BUILT IN: referent slots (`blood poured from his ___`) should come
back with the pool in `not_rankable`, because one body part is not more intense than
another. Act slots should come back ranked. A coder that ranks body parts by
intensity is inventing a gradient, and that rate is the field's own audit.
"""
from __future__ import annotations
import argparse, os, random, sys, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from malign_logits.cache import get_cache
import malign_logits.taxonomy as T
from scripts.f13_draw_relation_items import (
    surface_probs, THETA, FLOOR, DT, TYPE_OF, FUNC, admissible)
from scripts.f13_setd_prompts import SETD

POOL_MAX = 15
SEED = 20260730
OUT = "data/f13_ranking_items_full.parquet"


def content(w: str) -> bool:
    w = (w or "").strip()
    return bool(w) and any(c.isalpha() for c in w) and w.lower() not in FUNC


PIPELINE = ["base", "ego", "superego", "reinforced_superego"]


def edges(seen):
    """EVERY ORDERED PAIR ALONG THE PIPELINE, not just base->X.

    The earlier draws emitted base->ego, base->superego and base->reinforced only,
    so every measured edge conflated the stages it spanned: base->superego mixes
    SFT with DPO and cannot say which did the work. The stage edges (ego->superego,
    superego->reinforced) are the decomposition, and four families have the arms
    for them: olmo-tiny, olmoe, amber, minicpm.

    `arm` becomes the edge label `from>to` so downstream groupby's read as stages.
    """
    out = []
    for name, f in T.MODEL_FAMILIES.items():
        present = [(a, getattr(f, a, None)) for a in PIPELINE]
        present = [(a, m) for a, m in present if m and m in seen]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                (fa, fm), (ta, tm) = present[i], present[j]
                if fm != tm:
                    out.append((name, f"{fa}>{ta}", fm, tm))
    return out


def main():
    cm = get_cache()
    s = cm._stash("true_word_probs")
    seen = {(dict(k) if not isinstance(k, dict) else k).get("model") for k in s}
    E = edges(seen)
    rng = random.Random(SEED)
    print(f"true_word_probs entries at read time: {len(s):,}")
    print(f"edges {len(E)}  POOL_MAX {POOL_MAX}  seed {SEED}  FLOOR {FLOOR}  DT {DT}")

    # canonical 73 keep their taxonomy names; Set D prompts are added with their
    # hand-assigned slot, pair_id and transgressive flag. Both sets are drawn.
    PROMPTS = [(nm, v.rstrip(), TYPE_OF[nm], None, None, "canonical")
               for nm, v in T.DEFAULT_PROMPTS.items()]
    for txt, (slot, pair, tr) in SETD.items():
        nm = "setd_" + (pair or txt.lower().split()[-1]) + ("_T" if tr else "_N")
        PROMPTS.append((nm, txt.rstrip(), slot, pair, tr, "setd"))
    print(f"prompts: {len(PROMPTS)}  "
          f"({sum(1 for x in PROMPTS if x[5]=='canonical')} canonical, "
          f"{sum(1 for x in PROMPTS if x[5]=='setd')} setd, "
          f"{len({x[3] for x in PROMPTS if x[3]})} matched pairs)")

    rows, drop = [], collections.Counter()
    for fam, arm, bid, aid in E:
        for nm, p, slotg, pair, tr, pset in PROMPTS:
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
            if not (fallers and risers):
                drop["no faller or riser"] += 1
                continue
            a_w = max(fallers, key=lambda w: B[w])
            b_w = max(risers, key=lambda w: d[w])
            pool = [w for w in W if B.get(w, 0) >= FLOOR or d[w] >= DT]
            pool = sorted(pool, key=lambda w: -B.get(w, 0.0))[:POOL_MAX]
            for forced in (a_w, b_w):          # the two words the test is about
                if forced not in pool:
                    pool = pool[:POOL_MAX - 1] + [forced]
            pool = list(dict.fromkeys(pool))
            if len(pool) < 5:
                drop["pool < 5 words"] += 1
                continue
            shown = pool[:]
            rng.shuffle(shown)
            rows.append(dict(
                family=fam, arm=arm, base_id=bid, aligned_id=aid, prompt_name=nm,
                prompt=p, slot=slotg, pair_id=pair, transgressive=tr,
                prompt_set=pset,
                intensity_slot=slotg in {"ACT", "RESULT"},
                faller=a_w, riser=b_w,
                pa_base=B.get(a_w, 0.0), pa_al=A.get(a_w, 0.0),
                pb_base=B.get(b_w, 0.0), pb_al=A.get(b_w, 0.0),
                delta_faller=d[a_w], delta_riser=d[b_w],
                n_pool=len(pool), pool="|".join(pool), shown="|".join(shown),
                n_fallers=len(fallers), n_risers=len(risers)))

    df = pd.DataFrame(rows)
    print(f"\nitems {len(df)}")
    for k, v in drop.most_common():
        print(f"    dropped: {k:<26} {v}")
    print(f"\npool size: median {df.n_pool.median():.0f}  "
          f"min {df.n_pool.min()}  max {df.n_pool.max()}")
    print(f"faller in pool: {df.apply(lambda r: r.faller in r.pool.split('|'), axis=1).mean():.1%}"
          f"   riser in pool: {df.apply(lambda r: r.riser in r.pool.split('|'), axis=1).mean():.1%}"
          f"   (both must be 100%)")
    print("\nBY SLOT:")
    print(df.groupby("slot").agg(n=("slot", "size"),
                                 median_pool=("n_pool", "median")).to_string())
    print(f"\nintensity-comparable (ACT/RESULT): {df.intensity_slot.sum()} items, "
          f"{df[df.intensity_slot].family.nunique()} families, "
          f"{df[df.intensity_slot].prompt_name.nunique()} prompts")
    print("\nSAMPLE ITEM (what the coder sees, and the metadata it never sees):")
    print("\nSET D COVERAGE:")
    sd = df[df.prompt_set == "setd"]
    if len(sd):
        print(sd.groupby(["pair_id", "transgressive"]).size().rename("items").to_string())
    r = df[df.prompt_name == "violence_liminal_3"].iloc[0]
    print(f"  PROMPT: {r.prompt} ___")
    print(f"  CANDIDATES: {', '.join(r.shown.split('|'))}")
    print(f"  [hidden] faller={r.faller} ({r.pa_base:.3f}->{r.pa_al:.3f})  "
          f"riser={r.riser} ({r.pb_base:.3f}->{r.pb_al:.3f})  {r.family}/{r.arm}")
    df.to_parquet(OUT, compression="zstd", index=False)
    print(f"\nwrote {len(df)} -> {OUT}")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
