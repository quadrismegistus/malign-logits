"""Draw and FREEZE the held-out validation set BEFORE any examples are written.

lacan's condition on spending the adjudicated cases as few-shot examples ([223]):
draw the fresh held-out set FIRST. In that order it is a clean split; in the other
order it is the same injury that shrank the earlier held-out pool, with better
material, plus the temptation to draw a set that happens to exclude the hard cases.

WHY IT IS SAFE NOW AND WAS NOT BEFORE. When twelve of the first blind sixty became
examples the corpus was fixed and finite, so every example spent shrank a pool that
could not be replenished. Tonight's corpus is still generating; the held-out pool
grows faster than examples spend it.

FROZEN means: written once, committed, and never redrawn. If it turns out to
contain no hard cases that is a fact about the draw and gets reported, not fixed by
redrawing.

    uv run .venv/bin/python scripts/f20x_freeze_heldout.py
"""
import glob
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "data/f20x_heldout_frozen.parquet"
SEED = 20260728          # declared; not tuned
PER_CELL = 3             # 6 conditions x 2 arms = 36


def main():
    if os.path.exists(OUT):
        print(f"{OUT} exists — FROZEN, refusing to redraw."); return

    d = pd.read_parquet("data/f20x_nonce.parquet")
    d = d[d.text.fillna("").str.strip().str.len() > 0].copy()
    d["al"] = d.arm != "base"

    # exclude every passage already spent on a sheet, an example, or an adjudication
    spent = set()
    for f in glob.glob("data/f20x_*key.parquet"):
        k = pd.read_parquet(f)
        if "text" in k.columns:
            spent |= set(k.text.dropna())
    before = len(d)
    d = d[~d.text.isin(spent)]
    print(f"corpus {before:,} -> {len(d):,} after excluding {before-len(d)} already-spent passages")

    picked = pd.concat([g.sample(min(PER_CELL, len(g)), random_state=SEED)
                        for _, g in d.groupby(["condition", "al"])])
    picked = picked.sample(frac=1, random_state=SEED).reset_index(drop=True)
    picked.insert(0, "n", range(1, len(picked) + 1))
    cols = ["n", "condition", "word", "family", "arm", "al", "model_id",
            "base_model_id", "pid", "temperature", "prompt", "text"]
    picked[cols].to_parquet(OUT, index=False)

    h = hashlib.sha256("\n".join(sorted(picked.text)).encode()).hexdigest()[:16]
    print(f"{len(picked)} passages frozen -> {OUT}")
    print(f"  by condition: {picked.condition.value_counts().to_dict()}")
    print(f"  by arm:       {picked.al.value_counts().to_dict()}")
    print(f"  content hash: {h}")
    print(f"  seed {SEED}, {PER_CELL} per condition x arm")


if __name__ == "__main__":
    main()
