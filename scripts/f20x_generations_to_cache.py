"""Sync the F20x generations into the logits/generations cache.

    uv run .venv/bin/python scripts/f20x_generations_to_cache.py

WHY THIS EXISTS. `f20x_generate.py` writes a parquet and nothing else. It never
calls `set_generation`, so for the whole of the first pass the only copy of
~10,500 completions -- several GPU-hours, some of them from models that will not
load again on this machine -- was one file. RH caught it.

That is the same disease as the six instruments in /tmp, one layer up: not an
uncommitted script this time, but an uncached artifact whose producing pipeline
cannot be re-run cheaply. `olmo-32b` and the Mamba families are already
unreproducible here; the families that DID run should not be one `rm` from the
same status.

Idempotent, and safe to run mid-roster: it reads whatever is in the parquet now
and writes what is missing. Run it again when the run finishes.

KEY DESIGN. The cache key is (model, prompt, temp, idx), and `prompt` is the text
actually fed to the model -- `Q: {question}\nA:` -- not the prompt's short name.
A future reader asking the cache for that exact string gets these completions
back. `idx` is the row's position within its (model, prompt, temp) cell, ordered
as generated, so re-running this never duplicates and never renumbers.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from malign_logits.cache import get_cache

RUNG = "Q: {q}\nA:"
SOURCES = [
    "data/f20x_generations.parquet",
    "data/f20x_generations_partA_25fam.parquet",   # the killed pass, kept
]


def main():
    cm = get_cache()
    frames = []
    for p in SOURCES:
        if os.path.exists(p):
            d = pd.read_parquet(p)
            print(f"  {p}: {len(d):,} rows, {d.family.nunique()} families")
            frames.append(d)
    if not frames:
        print("nothing to sync")
        return

    # DO NOT dedup on content. The sources overlap by construction -- partA is a
    # snapshot of the main file at the kill -- but sampling at temperature 0.7
    # also produces genuinely identical short completions within one cell, and
    # those are two DRAWS, not one. A content dedup silently deleted 96 of them
    # on the first attempt and reported 10,464 written against 10,560 in the
    # parquet. A cache that drops repeated draws misrepresents the distribution
    # it exists to preserve, and it does so precisely at the high-probability
    # completions that repeat.
    #
    # Reconcile by CELL instead: for each (model, prompt, temp), take whichever
    # source carries the most draws. Order within a cell is generation order.
    cells: dict[tuple, list[str]] = {}
    for d in frames:
        for key, g in d.groupby(["model_id", "question", "temperature"], sort=False):
            texts = list(g.text)
            if len(texts) > len(cells.get(key, [])):
                cells[key] = texts
    total = sum(len(v) for v in cells.values())
    print(f"\n{total:,} draws across {len(cells):,} cells "
          f"(cell = model x prompt x temperature)")

    written = skipped = 0
    for (mid, q, temp), texts in cells.items():
        prompt = RUNG.format(q=q)
        for idx, text in enumerate(texts):
            if cm.get_generation(mid, prompt, temp=float(temp), idx=idx) == text:
                skipped += 1
                continue
            cm.set_generation(mid, prompt, text, temp=float(temp), idx=idx)
            written += 1

    print(f"\n  written {written:,}   already correct {skipped:,}")

    # VERIFY BY READING EVERY DRAW BACK, not by trusting the write count. The
    # first version of this script reported a write count that was 96 short of
    # the parquet and nothing checked it.
    bad = missing = 0
    for (mid, q, temp), texts in cells.items():
        prompt = RUNG.format(q=q)
        for idx, text in enumerate(texts):
            got = cm.get_generation(mid, prompt, temp=float(temp), idx=idx)
            if got is None:
                missing += 1
            elif got != text:
                bad += 1
    n_parquet = max(len(f) for f in frames)
    print(f"  READBACK: {total - missing - bad:,}/{total:,} exact, "
          f"{missing} missing, {bad} mismatched")
    print(f"  parquet rows in largest source: {n_parquet:,}  "
          f"{'MATCHES' if n_parquet == total else 'DIFFERS FROM'} cached draws")


if __name__ == "__main__":
    main()
