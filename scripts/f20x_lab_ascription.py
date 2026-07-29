"""Does alignment install a LAB IDENTITY, and is it the model's own?

    uv run .venv/bin/python scripts/f20x_lab_ascription.py

WHY. The published AI-identification result is that alignment quadruples the rate at
which a model says it is an AI -- base 0.059 -> aligned 0.211, 24/29, weight-level,
on a bare `Q:`/`A:` rung with no chat template, no system prompt, no role tokens.
lacan noticed in passing that one aligned completion in a nine-passage sample was an
`allenai` model answering `My name is OpenAI Assistant`. This counts that.

WHAT IT IS NOT. **A regex over surface strings, not a coder.** It counts a model
MENTIONING a lab, not CLAIMING to be one, and `llama` / `claude` will catch
discussion as readily as self-ascription. The published AI-identification figure came
from a coder; these did not, and the two must not be quoted side by side as though
they were one instrument. Booked as a candidate finding pending a
`lab_self_ascription` enum on the four-level battery's coder pass.

THE TRAP THIS SCRIPT EXISTS TO EXPOSE RATHER THAN HIDE. The `own_lab` row has a
pooled mean of +0.050 and a sign count of 7/29 -- mean and per-model direction
pointing OPPOSITE ways, because a handful of families are trained to name themselves
while most do not move. **`own_lab` is not quotable as a mean.** Every row therefore
prints its sign count beside its mean, and a row whose two disagree is flagged in the
output rather than left for a reader to notice.

UNIT is the distinct base model (Rule 2), paired base vs aligned, sign test at n=29.
"""
import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402

GEN = os.path.join(PATH_DATA, "f20x_generations.parquet")
OUT = os.path.join(PATH_DATA, "f20x_lab_ascription.csv")
ALIGNED = ("ego", "superego", "reinforced_superego")

LABS = {
    "openai": r"\b(?:openai|chatgpt|gpt-?[34])\b",
    "anthropic": r"\b(?:anthropic|claude)\b",
    "google": r"\b(?:google|gemini|bard|deepmind)\b",
    "meta": r"\b(?:meta ai|llama)\b",
    "alibaba": r"\b(?:alibaba|qwen|tongyi)\b",
    "ai2": r"\b(?:allen institute|allenai|ai2|olmo|tulu)\b",
    "mistral": r"\b(?:mistral)\b",
    "deepseek": r"\b(?:deepseek)\b",
    "microsoft": r"\b(?:microsoft|bing|copilot)\b",
}
# HuggingFace org -> the lab key its models would be naming if they named themselves.
# Orgs with no lab key (EleutherAI, bigscience, ...) can only ever score `other_lab`,
# which is why `own_lab` is a minority-of-families quantity by construction.
ORG = {"allenai": "ai2", "meta-llama": "meta", "Qwen": "alibaba",
       "mistralai": "mistral", "deepseek-ai": "deepseek", "microsoft": "microsoft",
       "google": "google"}


def paired(d, col):
    """Mean delta AND sign count. A row where they disagree is not quotable."""
    p = d.pivot_table(index="base_model_id", columns="aligned", values=col,
                      aggfunc="mean").dropna()
    delta = p[True] - p[False]
    pos = int((delta > 0).sum())
    return {"metric": col, "base": p[False].mean(), "aligned": p[True].mean(),
            "delta": delta.mean(), "pos": pos, "n": len(delta),
            "p": stats.binomtest(pos, len(delta), 0.5).pvalue,
            # mean says one thing, per-model direction says another
            "incoherent": (delta.mean() > 0) != (pos > len(delta) / 2)}


def main():
    d = pd.read_parquet(GEN)
    d["aligned"] = d.arm.isin(ALIGNED)
    d["org"] = d.model_id.str.split("/").str[0].map(ORG).fillna("__none__")
    t = d.text.fillna("").str.lower()

    d["any_lab"] = False
    d["own_lab"] = False
    d["other_lab"] = False
    for lab, pat in LABS.items():
        hit = t.str.contains(pat, regex=True, na=False)
        d[f"lab_{lab}"] = hit
        d["any_lab"] |= hit
        d["own_lab"] |= hit & (d.org == lab)
        d["other_lab"] |= hit & (d.org != lab)

    rows = [paired(d, c) for c in ("any_lab", "other_lab", "own_lab")]
    res = pd.DataFrame(rows)
    print("1P corpus, lab naming, unit = base model, paired\n")
    for r in rows:
        flag = "   <-- MEAN AND SIGN COUNT DISAGREE; NOT QUOTABLE" if r["incoherent"] else ""
        print(f"  {r['metric']:10s} base {r['base']:.4f}  aligned {r['aligned']:.4f}"
              f"  delta {r['delta']:+.4f}  {r['pos']}/{r['n']}  p={r['p']:.4f}{flag}")

    # Is the openai result one family? The question that decides whether it is a
    # finding or a quirk -- asked here rather than left to a reader.
    p = d.pivot_table(index="base_model_id", columns="aligned", values="lab_openai",
                      aggfunc="mean").dropna()
    p["delta"] = p[True] - p[False]
    n = d[(d.aligned) & (d.lab_openai)]
    print(f"\nopenai self-naming: {len(n)} aligned completions across "
          f"{n.base_model_id.nunique()} base models, {n.family.nunique()} families; "
          f"delta positive {(p.delta > 0).sum()}/{len(p)}")
    print("\n  largest increases:")
    print(p.sort_values("delta", ascending=False).head(4).round(4).to_string())
    print("\n  REVERSALS -- base over-claims and alignment corrects it:")
    print(p.sort_values("delta").head(3).round(4).to_string())

    res.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
