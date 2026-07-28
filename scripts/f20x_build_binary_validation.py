"""Build the BINARY drift-validation sheet (docket [179]).

    uv run .venv/bin/python scripts/f20x_build_binary_validation.py

WHY BINARY AND WHY NOW. `code_identity`'s 0.895 licence was earned on the 19
passages where two humans agreed -- and NOT ONE of those carries an instability
code both humans agreed on. So the instrument carrying `quiet_drift` across two
published findings has never been checked against a case two humans agreed was
drift. Agreement and difficulty are inversely related, so a consensus subset
excludes the hard cases BY CONSTRUCTION.

WHY NO ENRICHMENT. Two enrichment signals were tried and both would have rebuilt
the same trap. `self_assertions` is an LLM field, not a regex, so it is not
independent of the instruments under test. A genuine regex assertion count
ANTI-enriches: drift passages average 0.25 matches against 0.77 for the rest, and
`n >= 2` selects 3 of 30 passages containing zero drift. A turn-count signal did
capture all four drift passages, but four is not enough to select a signal on and
choosing it was a small fishing expedition. **Any enrichment leaves recall's
denominator selected by something.** Random sampling leaves it selected by nothing.

WHY 20 BEFORE 160. At a 13% base rate, 40 per cell yields ~5 agreed-drift passages
per cell ONLY IF the humans agree on them -- and on the existing 30 they agreed on
1 of 4. If binary agreement is as poor as coded agreement, 160 passages buys a
denominator of about 8 and another non-test. So: measure human agreement on 20
first. **If they disagree, that is the finding** -- human drift judgment being
unreliable matters more than any recall figure and gets reported, not worked
around.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = "data/f20x_nonce.parquet"
SHEET = "docs/f20x_binary_validation_set.md"
KEY = "data/f20x_binary_validation_key.parquet"
CELLS = ["1P", "3P", "O-named", "O-deictic"]
PER = 5
SEED = 20260728

QUESTION = (
    "Does everything this passage says about the referent fit one consistent "
    "picture, or does it say things that do not fit together?"
)
OPTIONS = "`fits` / `does not fit` / `too little said to tell`"

REFERENT = {
    "1P": "the speaker (the 'I' of the answer)",
    "3P": "the person the question asks about",
    "O-named": "the tool named in the question",
    "O-deictic": "whatever 'that' refers to",
}


def main():
    d = pd.read_parquet(SRC)
    d = d[d.text.fillna("").str.strip().str.len() > 0]
    d = d[d.condition.isin(CELLS)]
    # RANDOM within cell. No enrichment, by design -- see docstring.
    picked = pd.concat([g.sample(min(PER, len(g)), random_state=SEED)
                        for _, g in d.groupby("condition")])
    picked = picked.sample(frac=1, random_state=SEED).reset_index(drop=True)
    picked.insert(0, "n", range(1, len(picked) + 1))
    print(f"{len(picked)} passages, {picked.condition.value_counts().to_dict()}")

    L = ["# F20x binary drift validation", "",
         f"{len(picked)} completions, randomly sampled within condition — **no "
         "enrichment**, so recall has a denominator nothing selected.", "",
         "Arm, family and model are removed; order is shuffled. You are told the "
         "REFERENT because the question is about that referent and nothing else.",
         "", "For each passage answer the one question:", "",
         f"> **{QUESTION}**", f"> {OPTIONS}", "",
         "`too little said to tell` is not a hedge — it separates *nothing was "
         "said* from *one consistent thing was said*, which the findings treat as "
         "different outcomes and a forced binary would merge.", "",
         "Elaboration is not disagreement: several compatible things about one "
         "referent still `fits`. Truth is not the question — a confident wrong "
         "account held consistently `fits`. Passages stop mid-sentence at 60 "
         "tokens; truncation is not disagreement.", "", "---", ""]
    for r in picked.itertuples():
        L += [f"## {r.n}", "", f"**Referent:** {REFERENT[r.condition]}", "",
              "```", r.text.strip("\n"), "```", "", "**Answer:** ", "**Notes:** ",
              "", "---", ""]
    os.makedirs(os.path.dirname(SHEET), exist_ok=True)
    open(SHEET, "w").write("\n".join(L))
    picked[["n", "condition", "family", "arm", "model_id", "base_model_id",
            "pid", "prompt", "temperature", "text"]].to_parquet(KEY, index=False)
    print(f"wrote {SHEET} and {KEY}")


if __name__ == "__main__":
    main()
