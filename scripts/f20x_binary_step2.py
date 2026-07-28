"""Step 2: run the same-construct binary coder on the RANDOM 20, where recall and
precision are both computable.

    uv run .venv/bin/python scripts/f20x_binary_step2.py

WHY THIS FRAME AND NOT ANOTHER. Every other human-anchored set in this project is
enriched -- drawn from passages a machine already flagged -- which makes precision
computable and recall not, because the false negatives sit outside the frame. These
20 were sampled at random within condition. Nothing selected them, so the human
calls are a denominator as well as a numerator.

Step 1 ran `code_binary` on the 24 enriched passages and it disagreed with both
humans, inverting the arm direction (docket [189], [194]). That was n=12 per arm on
the hardest passages in the corpus by construction. This asks the same question on
material nothing selected, against two humans who agreed exactly on which two
passages drift ({9, 15}).

WHAT IT CAN SETTLE. Recall and precision for the binary coder against the human
consensus, on an unenriched frame. Two drift passages, so the recall denominator is
2 and every recall figure here is a direction rather than a rate -- the same
limitation malign stated when the sited and blind coders were compared on this set.
Precision has a larger denominator only if the coder over-flags.

WHAT IT CANNOT. Arm direction. Ten and ten by arm at these rates is not a test, and
the full-corpus run exists for that question.

RH's answer to #2 is blank on the returned sheet; that passage is excluded from
agreement counts rather than imputed.
"""
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits.tasks.code_binary import BinaryJudgmentTask, prepare  # noqa: E402

KEY = "data/f20x_binary_validation_key.parquet"
RH_SHEET = "data/f20x_binary_validation_set-RH.md"
LACAN = "data/f20x_binary_lacan.json"
OUT = "data/f20x_binary_step2.parquet"

CANON = {"fits": "fits", "does not fit": "does not fit",
         "too little": "too little said to tell",
         "too little said to tell": "too little said to tell"}


def parse_rh(path: str) -> dict[int, str]:
    """The returned sheet, section by section. Blank answers stay absent."""
    txt = open(path).read()
    out = {}
    for blk in re.split(r"\n## ", txt)[1:]:
        n = int(blk.split("\n", 1)[0].strip())
        m = re.search(r"\*\*Answer:\*\*[ \t]*(.*)", blk)
        if m and m.group(1).strip():
            out[n] = CANON[m.group(1).strip().lower()]
    return out


def main():
    k = pd.read_parquet(KEY).sort_values("n").reset_index(drop=True)
    rh = parse_rh(RH_SHEET)
    lac = {int(n): CANON[v.lower()]
           for n, v in json.load(open(LACAN))["answers"].items()}

    if os.path.exists(OUT):
        rec = pd.read_parquet(OUT)
        print(f"reusing {len(rec)} coded rows from {OUT}")
    else:
        task = BinaryJudgmentTask()
        prompts = []
        for r in k.itertuples():
            word = r.pid.split("_")[1] if r.condition.startswith(("O-n", "N-")) else ""
            prompts.append(prepare(r.condition, word, r.prompt, r.text))
        out = task.map(prompts, num_proc=8, desc="binary step2")
        rec = k.copy()
        rec["coder"] = [o.answer if o else None for o in out]
        rec["reason"] = [o.reason if o else None for o in out]
        rec.to_parquet(OUT, compression="zstd", index=False)

    rec["lacan"] = rec.n.map(lac)
    rec["RH"] = rec.n.map(rh)

    print("\n n  arm         condition   RH                lacan             coder")
    for r in rec.itertuples():
        print(f"{r.n:2d}  {r.arm[:10]:10s}  {r.condition:10s}  "
              f"{str(r.RH):16s}  {str(r.lacan):16s}  {str(r.coder)}")

    both = rec[rec.RH.notna() & rec.lacan.notna() & rec.coder.notna()]
    print(f"\nagreement, n={len(both)} (RH's #2 blank)")
    for a, b in [("lacan", "RH"), ("coder", "lacan"), ("coder", "RH")]:
        agr = (both[a] == both[b]).mean()
        print(f"  {a:5s} vs {b:5s}   {(both[a] == both[b]).sum():2d}/{len(both)} = {agr:.3f}")

    dnf = "does not fit"
    cons = both[(both.RH == both.lacan)]
    print(f"\nHUMAN CONSENSUS SET, n={len(cons)}")
    truth = set(cons[cons.RH == dnf].n)
    flag = set(cons[cons.coder == dnf].n)
    tp, fp, fn = len(truth & flag), len(flag - truth), len(truth - flag)
    print(f"  human does-not-fit: {sorted(truth)}")
    print(f"  coder does-not-fit: {sorted(flag)}")
    print(f"  TP {tp}  FP {fp}  FN {fn}")
    if truth:
        print(f"  recall    {tp/len(truth):.2f}  (denominator {len(truth)})")
    if flag:
        print(f"  precision {tp/len(flag):.2f}  (denominator {len(flag)})")

    print("\nMARGINALS (all 20, coder; humans where answered)")
    for who in ["RH", "lacan", "coder"]:
        vc = rec[who].value_counts()
        print(f"  {who:5s}  " + "  ".join(f"{k_}={v}" for k_, v in vc.items()))

    print("\nBY ARM, does-not-fit (direction only; 10/10 split is not a test)")
    rec["is_base"] = rec.arm.eq("base")
    for who in ["RH", "lacan", "coder"]:
        s = rec[rec[who].notna()]
        b = (s[s.is_base][who] == dnf).mean()
        a = (s[~s.is_base][who] == dnf).mean()
        print(f"  {who:5s}  base {b:.3f}  aligned {a:.3f}  delta {a-b:+.3f}")


if __name__ == "__main__":
    main()
