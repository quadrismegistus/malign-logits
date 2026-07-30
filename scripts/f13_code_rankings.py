"""Administer the intensity-ranking task over the drawn pools.

    uv run .venv/bin/python scripts/f13_code_rankings.py --model M --out F

The coder sees `shown` (the pool, shuffled at draw time under a declared seed) and
nothing else. Everything the statistic needs -- which word fell, which rose, their
probabilities -- is already in the draw and is never sent.

OUTCOMES ARE THREE, NOT TWO. A riser may come back RANKED, or in `not_rankable`
(foreclosure: the model exited the frame, e.g. `kill` -> `Options`), or missing
entirely (a coder failure). Those are recorded separately; collapsing the second
into the third would delete the foreclosure phenomenon.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from malign_logits.tasks.rank_intensity import IntensityRankingTask, prepare

SRC = "data/f13_ranking_items.parquet"


def main(model, out, workers=16, limit=0):
    d = pd.read_parquet(SRC)
    if limit:
        d = d.sample(limit, random_state=20260730).reset_index(drop=True)
    task = IntensityRankingTask()
    if model:
        task.model = model
    print(f"instrument sha256: {task.instrument_sha256()}")
    print(f"model {task.model}  items {len(d)}  workers {workers}")
    errs = {}
    anns = task.map([prepare(r.prompt, r.shown.split("|")) for r in d.itertuples()],
                    num_workers=workers, verbose=True, errors=errs, fail_fast=False)
    keep = []
    for r, a in zip(d.itertuples(), anns):
        if a is None:
            continue
        rank = [w for w in a.ranking]
        nr = [w for w in a.not_rankable]
        pool = r.pool.split("|")
        n = len(rank)
        def pos(w):
            return rank.index(w) if w in rank else None
        pf, pr = pos(r.faller), pos(r.riser)
        keep.append({**{k: getattr(r, k) for k in d.columns},
                     "ranking": "|".join(rank), "not_rankable": "|".join(nr),
                     "ties": "|".join(a.ties), "reason": a.reason,
                     "slot_note": a.slot_note, "n_ranked": n,
                     "faller_rank": pf, "riser_rank": pr,
                     "faller_nrank": (pf / (n - 1)) if (pf is not None and n > 1) else None,
                     "riser_nrank": (pr / (n - 1)) if (pr is not None and n > 1) else None,
                     "faller_unrankable": r.faller in nr, "riser_unrankable": r.riser in nr,
                     "coverage_ok": set(rank + nr) == set(pool)})
    o = pd.DataFrame(keep)
    o.to_parquet(out, compression="zstd", index=False)
    print(f"\ncoded {len(o)}, failed {len(errs)} -> {out}")
    if len(o):
        print(f"  coverage_ok {o.coverage_ok.mean():.1%}   "
              f"riser ranked {o.riser_rank.notna().mean():.1%}   "
              f"riser unrankable {o.riser_unrankable.mean():.1%}")
    try:
        print(task.usage.summary_line())
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.model, a.out, a.workers, a.limit)
