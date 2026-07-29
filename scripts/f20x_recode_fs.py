"""Re-code the 2x2 with the example-matched coder. Registered at
`docs/f20x_examplematch_registration.md`.

Tests whether Q1's flat rung contrasts are real or resolution. Same completions,
same coder family, same model, same temperature -- the ONLY difference is fifteen
few-shot examples, so any change is attributable to them.

    uv run .venv/bin/python scripts/f20x_recode_fs.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits.tasks.code_sited_fs import SitedFewShotTask  # noqa: E402
from malign_logits.tasks.code_sited import prepare  # noqa: E402

OUT = "data/f20x_nonce_coded_fs.parquet"


def _key(d):
    """Row identity. FAMILY IS REQUIRED: a base model is shared across families
    (Llama-3.1-8B backs 7, pythia-2.8b backs 4), and seeds increment across the
    run, so the same base in two families produces DIFFERENT completions under
    the same model_id/pid/temp/idx. Without family, 19.4% of rows collide and
    resume silently skips ~3,700 real passages."""
    return (d.family + "|" + d.model_id + "|" + d.arm + "|" + d.pid + "|"
            + d.temperature.astype(str) + "|" + d.idx_in_cell.astype(str))
CHUNK = 600


def main():
    d = pd.read_parquet("data/f20x_nonce.parquet")
    d = d[d.text.fillna("").str.strip().str.len() > 0].copy()
    d["key"] = _key(d)
    # never code an example passage with the coder those examples are inside
    ex = set(pd.read_parquet("data/f20x_coder_examples.parquet").text.str.strip())
    before = len(d)
    d = d[~d.text.str.strip().isin(ex)]
    print(f"corpus {before:,} -> {len(d):,} after removing {before-len(d)} example passages")

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_parquet(OUT)
        # DERIVE the key from the stored row; never read back the stored `key`
        # column. lacan added `family` to the key, resume compared new-format
        # keys against old-format stored ones, matched nothing, and recoded
        # 8,266 duplicates. A key is a function of the row: the moment its
        # definition changes, the persisted copy is a lie that resume trusts.
        done = set(_key(prev))
        print(f"resuming: {len(done):,} already coded")
    todo = d[~d.key.isin(done)].reset_index(drop=True)
    # A resumed run must ASSERT its todo count, not report it. The line that
    # should have stopped lacan's run said "35,650 of 35,650 to code" -- not an
    # error, and no monitor would flag it, but an impossible claim about a
    # resume. Only knowing what the number should be catches that.
    if done:
        assert len(todo) < len(d), (
            f"resume found {len(todo):,} of {len(d):,} to code with a non-empty "
            f"output file — the key definition has changed under the stored data")
    print(f"{len(todo):,} to code", flush=True)

    task = SitedFewShotTask()
    for i in range(0, len(todo), CHUNK):
        blk = todo.iloc[i:i+CHUNK]
        out = task.map([prepare(r.condition, r.word, r.prompt, r.text)
                        for r in blk.itertuples()], num_proc=24,
                       desc=f"fs {i//CHUNK+1}")
        rec = blk.copy()
        rec["codes"] = [list(o.codes) if o else None for o in out]
        rec["qd"] = [bool(o and "quiet_drift" in o.codes) for o in out]
        rec["nvp"] = [bool(o and "no_value_posed" in o.codes) for o in out]
        prev = pd.read_parquet(OUT) if os.path.exists(OUT) else None
        (pd.concat([prev, rec]) if prev is not None else rec).to_parquet(
            OUT, compression="zstd", index=False)
        print(f"  chunk {i//CHUNK+1}: {rec.codes.notna().sum()}/{len(blk)} "
              f"({i+len(blk):,}/{len(todo):,})", flush=True)


if __name__ == "__main__":
    main()
