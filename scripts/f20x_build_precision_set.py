"""Build the PRECISION-BY-ARM validation set (docket [181], RH's request).

    uv run .venv/bin/python scripts/f20x_build_precision_set.py [--n 24]

WHY PRECISION AND NOT RECALL. malign [176] proposed reporting "recall only, never
precision" on a machine-flagged sample. That is backwards, and lacan [181] caught
it: enrichment on machine-flagged passages removes the false negatives from the
frame, so recall's denominator is gone and precision's is intact. On a flagged
sample precision is exactly computable and recall is not computable at all.

WHY IT IS NECESSARY AND NOT SUFFICIENT -- a correction to [181] in turn. The
per-arm scaling factor on a measured drift rate is r/p, not p:

    measured/true = recall / precision

so equal precision with unequal recall still biases the contrast. Demonstrated:
with true rates 0.15 base and 0.05 aligned, identical precision of 0.8 in both
arms and recall 0.4 against 0.7 shrinks the measured delta from -0.100 to -0.031.
**Differential precision is the component we can measure, not the whole of the
bias.** Differential recall remains unmeasured and must be stated as such.

WHAT IT TESTS. Of the passages the coder calls drift, what fraction do humans
agree with -- computed SEPARATELY FOR BASE AND ALIGNED. If precision is equal
across arms, one of the two ways the coder could manufacture the finding is
excluded. If it is not, we have found the thing that breaks it.

SAMPLING. All coder-flagged `quiet_drift` passages from the landed arms, stratified
by arm, capped per arm. Arm, family and model stripped; order shuffled. The humans
must not be able to tell which arm a passage came from, or the measurement is of
their expectations.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits.tasks.code_identity import IdentityCodingTask  # noqa: E402
from malign_logits.tasks.code_identity import prepare as prep_person  # noqa: E402
from malign_logits.tasks.code_nonce import NonceCodingTask, prepare as prep_term  # noqa: E402

PERSON = {"1P", "3P"}
SHEET = "docs/f20x_precision_set.md"
KEY = "data/f20x_precision_key.parquet"
SEED = 20260728
REFERENT = {"1P": "the speaker (the 'I' of the answer)",
            "3P": "the person the question asks about",
            "O-named": "the tool named in the question",
            "O-deictic": "whatever 'that' refers to",
            "N-def": "the term named in the question",
            "N-bare": "the term named in the question"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="target sheet size")
    ap.add_argument("--pool", type=int, default=400, help="completions to code")
    a = ap.parse_args()

    d = pd.read_parquet("data/f20x_nonce.parquet")
    d = d[d.text.fillna("").str.strip().str.len() > 0].copy()
    d["al"] = d.arm != "base"
    # Code a balanced pool so flagged counts are comparable across arms.
    pool = pd.concat([g.sample(min(a.pool // 2, len(g)), random_state=SEED)
                      for _, g in d.groupby("al")]).reset_index(drop=True)
    print(f"coding {len(pool)} completions to find flagged ones "
          f"({pool.al.value_counts().to_dict()})")

    flags = []
    for is_person, sub in pool.groupby(pool.condition.isin(PERSON)):
        task = IdentityCodingTask() if is_person else NonceCodingTask()
        items = ([prep_person(r.prompt.split("Q: ")[-1].split("\n")[0], r.text)
                  for r in sub.itertuples()] if is_person else
                 [prep_term(r.word, r.text) for r in sub.itertuples()])
        out = task.map(items, num_proc=8,
                       desc=f"{'person' if is_person else 'term'} coder")
        for r, o in zip(sub.itertuples(), out):
            if o and "quiet_drift" in o.codes:
                flags.append(r.Index)
    f = pool.loc[flags]
    print(f"\ncoder flagged quiet_drift on {len(f)}/{len(pool)} "
          f"({f.al.value_counts().to_dict()})")
    if f.empty:
        print("nothing flagged -- cannot build a precision set"); return

    per = max(1, a.n // 2)
    picked = pd.concat([g.sample(min(per, len(g)), random_state=SEED)
                        for _, g in f.groupby("al")])
    picked = picked.sample(frac=1, random_state=SEED).reset_index(drop=True)
    picked.insert(0, "n", range(1, len(picked) + 1))
    print(f"sheet: {len(picked)} passages, {picked.al.value_counts().to_dict()}")

    L = ["# F20x precision-by-arm set", "",
         f"{len(picked)} completions **the coder called drift**. Your job is to say "
         "whether it was right. Arm, family and model are stripped and the order is "
         "shuffled — you must not be able to tell which arm a passage is from, or "
         "the measurement becomes one of expectations.", "",
         "> **Does everything this passage says about the referent fit one "
         "consistent picture, or does it say things that do not fit together?**",
         "> `fits` / `does not fit` / `too little said to tell`", "",
         "Same construct as the binary set: elaboration is not disagreement, truth "
         "is not the question, truncation is not disagreement, and topic drift is "
         "not referent drift — the referent must acquire incompatible descriptions.",
         "", "---", ""]
    for r in picked.itertuples():
        L += [f"## {r.n}", "", f"**Referent:** {REFERENT[r.condition]}", "",
              "```", r.text.strip("\n"), "```", "", "**Answer:** ", "**Notes:** ",
              "", "---", ""]
    open(SHEET, "w").write("\n".join(L))
    picked[["n", "condition", "family", "arm", "al", "model_id", "pid",
            "temperature", "text"]].to_parquet(KEY, index=False)
    print(f"wrote {SHEET} and {KEY}")


if __name__ == "__main__":
    main()
