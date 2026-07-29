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

THE `own_lab` ROW IS UNDETERMINED AT THIS ROSTER -- NOT "unquotable as a mean",
which is what an earlier version of this docstring said and which was wrong.

Only 10 of the 29 base models have an org that maps to a lab key at all, so for the
other 19 the own-lab outcome is UNDEFINED, not negative: they score zero in both arms
by construction. All seven positive deltas therefore fall among the ten, and 7/10 is
p=0.34 -- indistinguishable from half, the opposite of the "significantly below half"
the 7/29 appears to show. The ten also span only SIX labs, six of them from two
(3 allenai, 3 Qwen), so correcting the population turns an uninterpretable 7/29 into
an underpowered 7/10 over ~6 effective units.

That is a limit on the instrument's reach, not a statistic misbehaving. The
distinction matters because a false caveat costs what a false claim costs.

CONSEQUENCE FOR THE OTHER ROWS: `other_lab` is defined for all 29 (every model can
name someone else's lab) and `own_lab` for 10. The two are measured on different
rosters and must never be tabled as though the difference between them were
informative. The instrument sees misattribution across the whole roster and is blind
to CORRECT attribution for 19 of 29 -- Falcon naming Falcon, Pythia naming
EleutherAI, GLM naming Zhipu are definitionally absent.

THE FIX, for the coder pass: record the LAB NAMED, not a boolean. Own-versus-other is
then computed afterwards against a roster table that can be corrected without
recoding. Same argument as the tuple rule one level up -- the vocabulary is
provisional and the data format has to outlive it.

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
# An org absent here makes `own_lab` UNDEFINED for its models -- they score zero in
# both arms whatever they say, so they must not be counted as negatives. This map is
# the roster, and it is the thing to fix if the vocabulary above grows.
ORG = {"allenai": "ai2", "meta-llama": "meta", "Qwen": "alibaba",
       "mistralai": "mistral", "deepseek-ai": "deepseek", "microsoft": "microsoft",
       "google": "google"}


def paired(d, col):
    """Mean delta AND sign count, with TIES DROPPED FROM THE DENOMINATOR.

    An earlier version divided by the model count. A sign test's denominator is
    the NON-TIED pairs -- scoring ties as failures inflates n and deflates p, and
    it turned a p=0.26 into a p=0.03 elsewhere in this project. Ties are reported
    beside the count, never folded into either side: a model with zero events in
    both arms carries no directional information and both seats of this campaign
    once assigned such cells to whichever side they were arguing for.

    A row where the mean and the sign count disagree signals that the outcome is
    undefined for some units, not that the statistic misbehaves -- see the
    docstring.
    """
    p = d.pivot_table(index="base_model_id", columns="aligned", values=col,
                      aggfunc="mean").dropna()
    delta = p[True] - p[False]
    pos = int((delta > 0).sum())
    ties = int((delta == 0).sum())
    n_eff = len(delta) - ties
    return {"metric": col, "base": p[False].mean(), "aligned": p[True].mean(),
            "delta": delta.mean(), "pos": pos, "n": len(delta),
            "ties": ties, "n_eff": n_eff,
            "p": stats.binomtest(pos, n_eff, 0.5).pvalue if n_eff else float("nan"),
            # mean says one thing, per-model direction says another
            "incoherent": (delta.mean() > 0) != (pos > n_eff / 2 if n_eff else False)}


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

    eligible = sorted(d[d.org != "__none__"].base_model_id.unique())
    print(f"own_lab is DEFINED for {len(eligible)} of "
          f"{d.base_model_id.nunique()} base models; UNDEFINED for the rest.\n")

    rows = [paired(d, c) for c in ("any_lab", "other_lab", "own_lab")]
    rows.append(paired(d[d.base_model_id.isin(eligible)], "own_lab")
                | {"metric": "own_lab (eligible only)"})
    res = pd.DataFrame(rows)
    print("1P corpus, lab naming, unit = base model, paired\n")
    for r in rows:
        flag = ("   <-- UNDEFINED for most units; see the eligible-only row"
                if r["incoherent"] else "")
        print(f"  {r['metric']:10s} base {r['base']:.4f}  aligned {r['aligned']:.4f}"
              f"  delta {r['delta']:+.4f}  {r['pos']}/{r['n_eff']}"
              f"  (ties {r['ties']})  p={r['p']:.4f}{flag}")

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
