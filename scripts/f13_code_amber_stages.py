"""Code the amber staged decomposition: base -> SFT -> DPO, one edge at a time.

    uv run .venv/bin/python scripts/f13_code_amber_stages.py [--limit N]

WHY AMBER AND WHY STAGED. Amber is the only family in `true_word_probs` with all
three arms present (LLM360/Amber, AmberChat = SFT, AmberSafe = DPO), 209 prompts in
common. So it is the one place the annotation can ask WHICH STAGE DOES THE
DISPLACING rather than only whether displacement happened. The SFT-primacy question
is a contrast between two edges on identical prompts, which is the only form that
avoids the level-read-as-a-contrast error.

DECLARED BEFORE ANY ITEM IS DRAWN
---------------------------------
EDGES. base->SFT = Amber -> AmberChat. SFT->DPO = AmberChat -> AmberSafe. Roles are
derived PER EDGE at read time; a word may be a riser on one and a faller on the
other, and that is a finding rather than an inconsistency.

MOVEMENT, as in the draw script: faller p_base >= 0.005 and delta <= -0.003;
riser delta >= +0.003; stationary p_base >= 0.005 and |delta| <= 0.0005.

MASS-MATCHED DECOY, per docket [710].1. The unmatched rule made decoys structurally
easier than real risers: a decoy had to clear p_base >= 0.005 while a riser could
start anywhere, giving median p_base(B) 0.0335 real against 0.0072 decoy
(p = 3.4e-22). Base mass proxies plausibility and plausibility proxies
findable-relations, so the confound inflated the primary. HERE the decoy is the
stationary word whose p_base is NEAREST the real riser's, TOLERANCE = within a
factor of 3. Items with no match inside tolerance are EXCLUDED AND COUNTED. The
achieved match is printed as the median absolute log10 ratio.

PRIMARY ELIGIBILITY, per [710].2: slot grammar = ACT and BOTH A and B content
words. Everything else is coded and reported as its own stratum, never pooled.

THE DEV SET IS EXCLUDED, per [707].2 -- the ten smoke items and the four France
pairs, barred from every statistic forever.

WHAT THE CODER SEES is `prepare(prompt, a, b)`. Probabilities, arm, stage, slot and
match quality travel in the output columns and are never joined into that string.
"""
from __future__ import annotations
import argparse, json, math, os, sys, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from malign_logits.cache import get_cache
import malign_logits.taxonomy as T
from malign_logits.tasks.code_displacement_relation import (
    DisplacementRelationTask, prepare)
from scripts.f13_draw_relation_items import (
    THETA, FLOOR, DT, EPS, DEV, TYPE_OF, FUNC, admissible, content_share,
    surface_probs)

TOL = 3.0            # decoy p_base within this factor of the real riser's
OUT = "data/f13_amber_stage_codings.parquet"
BASE, SFT, DPO = "LLM360/Amber", "LLM360/AmberChat", "LLM360/AmberSafe"
EDGES = [("base->SFT", BASE, SFT), ("SFT->DPO", SFT, DPO)]


def probs(cm, model, prompt):
    p = cm.get_true_word_probs(model, prompt, theta=THETA)
    if not p:
        return None
    # THE CANONICAL AGGREGATOR, NOT A REBUILT ONE. This was
    # `{r["word"].strip(): float(r["p"]) for r in p["rows"] if admissible(...)}`
    # -- the broken form `surface_probs`'s own docstring forbids rebuilding.
    # The payload is ONE ROW PER (word, FIRST TOKEN) and the rows are a
    # PARTITION, so a comprehension keeps the last token path and DROPS THE
    # REST. Measured over 2,937 amber cells: mean 3.4% of mass lost, median
    # 0.0%, MAX 99.9% -- 175 cells losing more than 5%. Heavy-tailed, so it
    # passes every spot check and fails where a surface has several token
    # paths, which is disproportionately Chinese.
    return surface_probs(p, keep=admissible)


def main(limit=0, workers=6, model=None, out_path=None):
    cm = get_cache()
    print(f"true_word_probs entries at read time: {len(cm._stash('true_word_probs')):,}")
    print(f"TOL = within a factor of {TOL} on p_base;  FLOOR={FLOOR} DT={DT} EPS={EPS}")

    items, drop = [], collections.Counter()
    for stage, m0, m1 in EDGES:
        for nm, p in ((k, v.rstrip()) for k, v in T.DEFAULT_PROMPTS.items()):
            B, A = probs(cm, m0, p), probs(cm, m1, p)
            if not B or not A:
                drop[f"{stage}: prompt absent"] += 1
                continue
            words = set(B) | set(A)
            d = {w: A.get(w, 0.0) - B.get(w, 0.0) for w in words}
            fallers = [w for w in words if B.get(w, 0) >= FLOOR and d[w] <= -DT]
            risers = [w for w in words if d[w] >= DT]
            stat = [w for w in words if B.get(w, 0) >= FLOOR and abs(d[w]) <= EPS]
            if not (fallers and risers):
                drop[f"{stage}: no faller/riser"] += 1
                continue
            a_w = max(fallers, key=lambda w: B[w])
            b_w = max(risers, key=lambda w: d[w])
            pb = B.get(b_w, 0.0)
            # mass-matched decoy: nearest p_base to the real riser's, within TOL
            cand = [w for w in stat if w != b_w and B.get(w, 0) > 0]
            best, ratio = None, None
            if cand and pb > 0:
                best = min(cand, key=lambda w: abs(math.log10(B[w] / pb)))
                ratio = abs(math.log10(B[best] / pb))
                if ratio > math.log10(TOL):
                    drop[f"{stage}: no decoy within tolerance"] += 1
                    best, ratio = None, None
            elif not cand:
                drop[f"{stage}: empty stationary pool"] += 1
            cs = content_share(B)
            ok = lambda w: (w.lower() not in FUNC) and any(c.isalpha() for c in w)
            meta = dict(stage=stage, arm_from=m0, arm_to=m1, prompt_name=nm,
                        prompt=p, slot=TYPE_OF[nm], content_share=round(cs, 4))
            items.append(dict(item_class="REAL", a=a_w, b=b_w,
                              pa_from=B.get(a_w, 0.), pa_to=A.get(a_w, 0.),
                              pb_from=pb, pb_to=A.get(b_w, 0.),
                              match_log10=None,
                              primary_eligible=bool(meta["slot"] == "ACT"
                                                    and cs >= 0.50
                                                    and ok(a_w) and ok(b_w)),
                              **meta))
            if best is not None:
                items.append(dict(item_class="NEAR-MISS", a=a_w, b=best,
                                  pa_from=B.get(a_w, 0.), pa_to=A.get(a_w, 0.),
                                  pb_from=B.get(best, 0.), pb_to=A.get(best, 0.),
                                  match_log10=round(ratio, 4),
                                  primary_eligible=bool(meta["slot"] == "ACT"
                                                        and cs >= 0.50
                                                        and ok(a_w) and ok(best)),
                                  **meta))

    d = pd.DataFrame(items)
    n0 = len(d)
    d = d[[(r.prompt, r.a, r.b) not in DEV for r in d.itertuples()]].reset_index(drop=True)
    print(f"\nitems built {n0}   dev-set removed {n0 - len(d)}   to code {len(d)}")
    for k, v in drop.most_common():
        print(f"    dropped: {k:<38} {v}")
    mm = d[d.item_class == "NEAR-MISS"].match_log10.dropna()
    if len(mm):
        print(f"\nACHIEVED MASS MATCH: median |log10(p_decoy/p_real)| = {mm.median():.3f} "
              f"(= a factor of {10**mm.median():.2f});  worst {mm.max():.3f}")
    print("\nby stage x class:")
    print(pd.crosstab(d.stage, d.item_class).to_string())

    if limit:
        d = (d.groupby(["stage", "item_class"], group_keys=False)
              .apply(lambda g: g.sample(min(len(g), limit), random_state=20260730))
              .reset_index(drop=True))
        print(f"\n--limit {limit} per (stage,class): coding {len(d)}")

    task = DisplacementRelationTask()
    if model:
        task.model = model
    print(f"\ncoding {len(d)} items on {task.model} at temperature {task.temperature}")
    anns = task.map([prepare(r.prompt, r.a, r.b) for r in d.itertuples()],
                    num_workers=workers, verbose=True)
    keep, fail = [], 0
    for r, ann in zip(d.itertuples(), anns):
        if ann is None:
            fail += 1
            continue
        keep.append({**{k: getattr(r, k) for k in d.columns}, **ann.model_dump()})
    out = pd.DataFrame(keep)
    dest = out_path or OUT
    out.to_parquet(dest, compression="zstd", index=False)
    print(f"\ncoded {len(out)}, failed {fail} -> {dest}")

    if not len(out) or "primary_eligible" not in out.columns:
        print("\n*** NO ITEMS CODED -- no summary is possible. "
              f"{fail} failures. Read the log for the provider error; a total "
              "failure is a configuration fault, not a result. ***")
        return
    print("\n" + "=" * 78)
    print("SPEECH_ACT BY STAGE, PRIMARY-ELIGIBLE ACT ITEMS ONLY (never pooled)")
    print("=" * 78)
    pe = out[out.primary_eligible]
    if len(pe):
        print(pd.crosstab([pe.stage, pe.item_class], pe.speech_act).to_string())
        print("\nDIRECTION:")
        print(pd.crosstab([pe.stage, pe.item_class], pe.direction).to_string())
        print("\nB IS AN ACT OF THE DRIVE (threat|exclamation) -- the [707].2 successor,")
        print("reported here as DESCRIPTION: these items are a fresh set, but this is")
        print("one family and the successor's confirmatory test needs the full draw.")
        pe = pe.assign(drive=pe.speech_act.isin(["THREAT", "EXCLAMATION"]))
        print(pe.groupby(["stage", "item_class"]).drive.agg(["mean", "size"]).to_string())
    else:
        print("  no primary-eligible items")
    print("\nREF STRATUM -- its own primary is METONYMY ([695].2):")
    ref = out[out.slot == "REF"]
    if len(ref):
        print(pd.crosstab([ref.stage, ref.item_class], ref.relation).to_string())
    else:
        print("  none")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.limit, a.workers, a.model, a.out)
