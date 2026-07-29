"""Cross-provider replication of the 1P drift finding.

    uv run .venv/bin/python scripts/f20x_crossprovider.py [--model M] [--n 100]

WHAT IT TESTS. `quiet_drift` 0.103 -> 0.042, 28 of 29 base models, is the only
arm-direction evidence in this project that clears anything -- and every one of the
seven coders in this repository is `deepseek/deepseek-chat`. This re-codes the same
first-person completions with the SAME committed scheme, 15 examples intact, and
changes only the model. If the per-model direction holds, the finding is not an
artefact of one provider's vocabulary or failure modes.

WHAT IT DOES NOT TEST, and it must not be booked as more. Gemini is also an ALIGNED
model being asked to judge the effects of alignment. If aligned text is more legible
to an aligned reader, that bias is shared by every frontier coder and cross-provider
agreement confirms it rather than detects it. This buys "not deepseek-specific",
not "independent".

LICENCE. Gemini was validated first on the same 30 human-coded passages that
licensed deepseek: 0.895 against the two-human agreeing subset for gemini-2.5-flash
and gemini-3.6-flash alike, identical to deepseek's 0.895. It carries the scheme.

POWER, simulated at both seats before choosing n. 100 per cell gives 0.94 against
HALF the published gap; 30 per cell gives 0.52. The design is powered against the
alternative that motivates it -- that deepseek's magnitude is inflated -- rather
than against the estimate it is checking.

REGISTERED BEFORE RUNNING: under a perfectly real effect at this n the EXPECTED
count is 27.3 of 29, not 28 of 29. A lower win count from a smaller sample is not a
weaker effect, and without this line "26 of 29" reads as partial replication when
it is what full replication looks like here.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits.tasks.code_identity import IdentityCodingTask, prepare  # noqa: E402

OUT = "data/f20x_crossprovider.parquet"
SEED = 20260728
CHUNK = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemini-3.6-flash")
    ap.add_argument("--n", type=int, default=100, help="completions per base-model x arm")
    a = ap.parse_args()

    d = pd.read_parquet("data/f20x_codings.parquet",
                        columns=["family", "arm", "model_id", "base_model_id",
                                 "question", "temperature", "idx_in_cell", "text"])
    d = d[d.family != "olmo-think"].copy()
    d["text"] = d.text.fillna("")
    d = d[d.text.str.strip().str.len() > 0]
    d["al"] = d.arm != "base"
    # Terminal aligned arm only, matching the published analysis.
    ALIGNED = ("ego", "superego", "reinforced_superego")
    term = {}
    for f, g in d[d.al].groupby("family"):
        for s in reversed(ALIGNED):
            if s in set(g.arm):
                term[f] = s
                break
    keep = d[(d.arm == "base").to_numpy()
             | np.array([term.get(r.family) == r.arm for r in d.itertuples()])]
    samp = pd.concat([g.sample(min(a.n, len(g)), random_state=SEED)
                      for _, g in keep.groupby(["base_model_id", "al"])]).reset_index(drop=True)
    samp["key"] = (samp.model_id + "|" + samp.question + "|"
                   + samp.temperature.astype(str) + "|" + samp.idx_in_cell.astype(str))
    print(f"{len(samp):,} completions | {samp.base_model_id.nunique()} base models "
          f"| model {a.model}", flush=True)

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_parquet(OUT)["key"])
        print(f"resuming: {len(done):,} done", flush=True)
    todo = samp[~samp.key.isin(done)].reset_index(drop=True)

    task = IdentityCodingTask(model=a.model)
    for i in range(0, len(todo), CHUNK):
        blk = todo.iloc[i:i + CHUNK]
        out = task.map([prepare(r.question, r.text) for r in blk.itertuples()],
                       num_proc=24, desc=f"chunk {i//CHUNK + 1}")
        rec = blk.copy()
        rec["codes"] = [list(o.codes) if o else None for o in out]
        rec["quiet_drift"] = [bool(o and "quiet_drift" in o.codes) for o in out]
        prev = pd.read_parquet(OUT) if os.path.exists(OUT) else None
        (pd.concat([prev, rec]) if prev is not None else rec).to_parquet(
            OUT, compression="zstd", index=False)
        print(f"  chunk {i//CHUNK + 1}: {rec.codes.notna().sum()}/{len(blk)} "
              f"({i + len(blk):,}/{len(todo):,})", flush=True)


if __name__ == "__main__":
    main()
