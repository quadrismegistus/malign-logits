"""Run the same-construct binary coder over the FULL identity corpus.

    uv run .venv/bin/python scripts/f20x_binary_corpus.py [--limit N]

WHY. On 24 enriched passages the binary coder's does-not-fit rate by arm ran
OPPOSITE to lacan's: 0.167 base / 0.417 aligned against 0.417 / 0.167. At twelve
per arm that is a direction and not a result (Fisher p=0.371), and by the standard
both seats applied to `marked_contradiction` at 15/29 it must not be quoted as a
finding. But an inverted arm direction is the failure that puts a wrong SIGN on a
contrast, so it cannot be left unresolved either.

This settles it without humans and without a validated instrument, because it does
not ask which coder is right. It asks whether the disagreement survives scale:

    base > aligned across 29 base models  -> the n=12 inversion was noise
    aligned > base at scale               -> two independently-built coders
                                             disagree on the SIGN of the finding

The published effect is `quiet_drift` 0.103 base -> 0.042 aligned, 28 of 29 base
models, from the scheme coder. A second coder built to a different construct,
asking the humans' question rather than the scheme's, is as independent a
replication as this corpus allows.

Writes incrementally: this repository has lost finished work to interruption.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits.tasks.code_binary import BinaryJudgmentTask, prepare  # noqa: E402

OUT = "data/f20x_binary_corpus.parquet"
CHUNK = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    d = pd.read_parquet("data/f20x_codings.parquet",
                        columns=["family", "arm", "model_id", "base_model_id",
                                 "question", "temperature", "idx_in_cell", "text"])
    d = d[d.family != "olmo-think"].copy()
    d["text"] = d.text.fillna("")
    d = d[d.text.str.strip().str.len() > 0].reset_index(drop=True)
    d["key"] = (d.model_id + "|" + d.question + "|" + d.temperature.astype(str)
                + "|" + d.idx_in_cell.astype(str))
    if a.limit:
        d = d.head(a.limit)

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_parquet(OUT)["key"])
        print(f"resuming: {len(done):,} already coded")
    todo = d[~d.key.isin(done)].reset_index(drop=True)
    print(f"{len(todo):,} of {len(d):,} to code", flush=True)

    task = BinaryJudgmentTask()
    for i in range(0, len(todo), CHUNK):
        blk = todo.iloc[i:i + CHUNK]
        # The identity corpus is all first-person; the referent is the speaker.
        out = task.map([prepare("1P", "you", f"Q: {r.question}\nA:", r.text)
                        for r in blk.itertuples()], num_proc=24,
                       desc=f"binary {i//CHUNK + 1}")
        rec = blk.copy()
        rec["answer"] = [o.answer if o else None for o in out]
        rec["reason"] = [o.reason if o else None for o in out]
        prev = pd.read_parquet(OUT) if os.path.exists(OUT) else None
        pd.concat([prev, rec]) if prev is not None else rec
        (pd.concat([prev, rec]) if prev is not None else rec).to_parquet(
            OUT, compression="zstd", index=False)
        ok = rec.answer.notna().sum()
        print(f"  chunk {i//CHUNK + 1}: {ok}/{len(blk)} coded, "
              f"{len(done) + i + len(blk):,} total", flush=True)


if __name__ == "__main__":
    main()
