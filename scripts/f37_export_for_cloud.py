"""Export the generations stash to a self-contained parquet for cloud scoring.

    uv run .venv/bin/python scripts/f37_export_for_cloud.py

WHY NOT SHIP THE STASH. It is only 341 MB, so size is not the reason. The reason is
that reading it on the far end requires the same hashstash version AND the pinned
format options, and hashstash encodes serializer/compress/b64 into the on-disk path
— so an unpinned open resolves to a DIFFERENT, EMPTY store and raises nothing. That
trap produced two phantom stores in this project's data tree today. An instance is
the worst place to rediscover it: the failure is silent, and with resume-by-key
parity it either rescores everything or scores nothing.

So the pinned open happens HERE, where `cache.open_stash()` is enforced and the
entry count can be asserted against a known number. The instance receives one
parquet and one script and needs no hashstash at all. Scores come back keyed
identically and are merged into `reward_scores/` locally.

POPULATION: all 256,035 entries, per RH — the pretraining checkpoints and the
closed-API models are wanted for comparison. The stratum table governs which
CONTRASTS are licensed, never which items are scored.

KEYS are the stash's own dict keys `{idx, model, prompt, temp}`, flattened to
columns and re-joinable on exactly those four fields.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402
from malign_logits.cache import open_stash  # noqa: E402

SRC = os.path.join(PATH_DATA, "raw", "cache", "generations")
OUT = os.path.join(PATH_DATA, "f37_gens_for_scoring.parquet")
EXPECTED = 256_035


def main():
    g = open_stash(SRC)
    n = len(g)
    print(f"pinned open: {n:,} entries")
    assert n == EXPECTED, (
        f"expected {EXPECTED:,}, got {n:,} — an unpinned open resolves to an "
        f"empty store and raises nothing, so this assert is the guard")

    # `temp` is stored as BOTH int 1 and float 1.0 for 55 entries -- two distinct
    # stash keys holding two distinct generations, which collide the moment the
    # key is flattened to columns. Carrying the original type makes the round
    # trip exact; normalising would silently merge 55 pairs into one entry and
    # discard generations. The defect stays visible rather than papered over.
    rows = []
    for k in g.keys():
        v = g[k]
        t = k.get("temp")
        rows.append({"idx": k.get("idx"), "model": k.get("model"),
                     "prompt": k.get("prompt"), "temp": t,
                     "temp_type": type(t).__name__,
                     "text": v if isinstance(v, str) else str(v)})
    d = pd.DataFrame(rows)
    assert len(d) == n, f"exported {len(d)} of {n}"
    KEY = ["idx", "model", "prompt", "temp", "temp_type"]
    dup = d.duplicated(subset=KEY).sum()
    assert dup == 0, f"{dup} duplicate keys on {KEY} — join key is not unique"
    mixed = d.groupby(["idx", "model", "prompt"]).temp_type.nunique()
    print(f"  temp stored as both int and float for {(mixed > 1).sum()} items "
          f"(carried through, not merged)")

    d.to_parquet(OUT, compression="zstd", index=False)
    mb = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT}  {len(d):,} rows  {mb:.0f} MB")
    print(f"  models {d.model.nunique()} | prompts {d.prompt.nunique()}")
    w = d.text.fillna("").str.split().str.len()
    print(f"  words/item: median {w.median():.0f}  mean {w.mean():.0f}  "
          f"p99 {w.quantile(0.99):.0f}")
    print(f"  empty texts: {(d.text.fillna('').str.strip() == '').sum()}")


if __name__ == "__main__":
    main()
