"""Same contrast at the honest unit. The Mann-Whitney above ran over ~1,000 BEAM
rows drawn from 4 benign_high PROMPTS; p=5e-56 is a statement about beam count,
not about prompts. Collapse to the prompt and see what is left.
"""
import numpy as np, pandas as pd

d = pd.read_csv("data/f36_sexual_beams.csv")
d["cond"] = np.where(d.transgression.isin(["sexual", "sexual_liminal"]), "sexual",
             np.where(d.transgression == "benign_high", "benign_high", "benign"))
pp = d.groupby(["family", "cond", "prompt"]).mean_resist.mean().reset_index()

print("distinct PROMPTS per condition:",
      pp[pp.family == pp.family.iloc[0]].groupby("cond").size().to_dict())
print(f"\n{'family':12s}{'benign':>9s}{'benign_high':>13s}{'sexual':>9s}   bh ranks above sexual")
for fam, g in pp.groupby("family"):
    m = g.groupby("cond").mean_resist.mean()
    bh = g[g.cond == "benign_high"].mean_resist.values
    sx = g[g.cond == "sexual"].mean_resist.values
    above = np.mean([(x > sx).mean() for x in bh])
    print(f"{fam:12s}{m['benign']:>9.3f}{m['benign_high']:>13.3f}{m['sexual']:>9.3f}"
          f"{above:>22.2f}")

print("\nPer benign_high prompt, resistance by family (4 prompts, unpaired -- this")
print("is the whole control arm, and it is why no p-value is quoted for it):")
bh = pp[pp.cond == "benign_high"].pivot_table(index="prompt", columns="family",
                                              values="mean_resist")
print(bh.round(3).to_string())
