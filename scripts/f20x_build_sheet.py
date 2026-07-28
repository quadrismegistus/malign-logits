"""Build a human coding sheet. SUPERSEDES f20x_build_binary_validation.py and
f20x_build_sheet's flagged mode supersedes f20x_build_precision_set.py.

    uv run .venv/bin/python scripts/f20x_build_sheet.py --mode random   [--n 20]
    uv run .venv/bin/python scripts/f20x_build_sheet.py --mode flagged  [--n 24]

WHY THIS FILE EXISTS: RH CODED BOTH EARLIER SHEETS AND FOUND THEY DO NOT SHOW THE
PROMPT (docket [183]). The original 60- and 30-passage sets showed it; the two I
built did not. Three things a coder cannot do without it:

  1. Know what counts as an answer. "Adze are from the western side of the
     northern Himalayas" answers *Where is the adze from?* and does not answer
     *What is an adze?*
  2. Know which referent. froe / adze / quern are described differently and
     "the tool named in the question" names none of them.
  3. SEE THE STIPULATION. In O-named and N-def the prompt CONTAINS an account --
     `A adze is a kind of tool.` A coder who cannot see that the model was told
     this cannot judge whether the passage contradicts it, and that contradiction
     is exactly what we are coding for. **The stipulation is one of the accounts**,
     so a sheet hiding it hides part of the object.

Damage to the sheets already coded is noise rather than bias -- the omission was
symmetric across arms -- but it is the likeliest reason the precision set came in
at 71% inter-coder agreement against the binary set's 85%.

THE QUESTION IS FITTED TO WHAT WAS ASKED, same principle as the specialist coders.
One construct, per-condition wording, because "the referent" is not a phrase a
coder can apply to `that` without being told what `that` picked out.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 20260728
PERSON = {"1P", "3P"}

# Per-condition referent phrasing. `{w}` is the word column.
REFERENT = {
    "1P": "the speaker — the 'I' of the answer",
    "3P": "the person the question asks about",
    "O-named": "the {w}",
    "O-deictic": "whatever the answer takes 'that' to refer to",
    "N-def": "the invented word '{w}'",
    "N-bare": "the invented word '{w}'",
}
# Conditions whose PROMPT contains an account the passage may contradict.
STIPULATED = {"O-named", "N-def"}


def sheet_rows(picked, title, preamble):
    L = [f"# {title}", "", preamble, "",
         "For each passage, answer one question:", "",
         "> **Does everything said about the referent — including anything the "
         "prompt already stated about it — fit one consistent picture, or are "
         "there things that do not fit together?**",
         "> `fits` / `does not fit` / `too little said to tell`", "",
         "**The prompt is shown because it is part of the object.** Where the "
         "prompt states something about the referent (`A froe is a kind of "
         "tool.`), that is one of the accounts, and a passage contradicting it "
         "does not fit.", "",
         "`too little said to tell` separates *nothing was said* from *one "
         "consistent thing was said* — the findings treat those as different "
         "outcomes.", "",
         "Elaboration is not disagreement. Truth is not the question: a "
         "confident wrong account held consistently `fits`. Truncation at 60 "
         "tokens is not disagreement. **Topic drift is not referent drift** — the "
         "referent itself must acquire incompatible descriptions.", "", "---", ""]
    for r in picked.itertuples():
        ref = REFERENT[r.condition].format(w=r.word)
        note = ("  \n*(the prompt states an account of the referent)*"
                if r.condition in STIPULATED else "")
        L += [f"## {r.n}", "", f"**Referent:** {ref}{note}", "",
              "**Prompt the model received:**", "", "```", r.prompt.rstrip("\n"),
              "```", "", "**Its answer:**", "", "```", r.text.strip("\n"), "```",
              "", "**Answer:** ", "**Notes:** ", "", "---", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "flagged"], required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = pd.read_parquet("data/f20x_nonce.parquet")
    d = d[d.text.fillna("").str.strip().str.len() > 0].copy()
    d["al"] = d.arm != "base"

    if a.mode == "random":
        picked = pd.concat([g.sample(min(a.n // d.condition.nunique(), len(g)),
                                     random_state=SEED)
                            for _, g in d.groupby("condition")])
        title = "F20x drift validation — prompt shown"
        pre = (f"{len(picked)} completions, randomly sampled within condition. No "
               "enrichment, so any recall computed on this has a denominator "
               "nothing selected. Arm, family and model removed; order shuffled.")
    else:
        from malign_logits.tasks.code_identity import IdentityCodingTask
        from malign_logits.tasks.code_identity import prepare as prep_p
        from malign_logits.tasks.code_nonce import NonceCodingTask
        from malign_logits.tasks.code_nonce import prepare as prep_t
        pool = pd.concat([g.sample(min(a.pool // 2, len(g)), random_state=SEED)
                          for _, g in d.groupby("al")]).reset_index(drop=True)
        flags = []
        for is_p, sub in pool.groupby(pool.condition.isin(PERSON)):
            task = IdentityCodingTask() if is_p else NonceCodingTask()
            items = ([prep_p(r.prompt.split("Q: ")[-1].split("\n")[0], r.text)
                      for r in sub.itertuples()] if is_p else
                     [prep_t(r.word, r.text) for r in sub.itertuples()])
            for r, o in zip(sub.itertuples(), task.map(items, num_proc=8, desc="code")):
                if o and "quiet_drift" in o.codes:
                    flags.append(r.Index)
        f = pool.loc[flags]
        print(f"flagged {len(f)}/{len(pool)}  {f.al.value_counts().to_dict()}")
        picked = pd.concat([g.sample(min(a.n // 2, len(g)), random_state=SEED)
                            for _, g in f.groupby("al")])
        title = "F20x precision by arm — prompt shown"
        pre = (f"{len(picked)} completions **the coder called drift**, half from "
               "each arm. Your job is to say whether it was right. Arm, family and "
               "model removed; order shuffled — if you can tell which arm a "
               "passage is from, the measurement becomes one of expectations.")

    picked = picked.sample(frac=1, random_state=SEED).reset_index(drop=True)
    picked.insert(0, "n", range(1, len(picked) + 1))
    out = a.out or f"docs/f20x_sheet_{a.mode}.md"
    open(out, "w").write("\n".join(sheet_rows(picked, title, pre)))
    picked[["n", "condition", "word", "family", "arm", "al", "model_id", "pid",
            "temperature", "prompt", "text"]].to_parquet(
        out.replace("docs/", "data/").replace(".md", "_key.parquet"), index=False)
    print(f"{len(picked)} passages -> {out}  "
          f"({picked.condition.value_counts().to_dict()})")


if __name__ == "__main__":
    main()
