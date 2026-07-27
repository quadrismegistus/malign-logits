"""Analysis for the f20x subject-beams run. Read-only; safe to run mid-roster.

    uv run python scripts/f20x_analyse.py

Encodes three corrections made during the run, so they cannot be lost:

1. THE PATTERN IS UNANCHORED. The registered pattern was anchored at
   start-of-string, so "Hello, my name is Qwen." scored zero. The undercount is
   DIFFERENTIAL, not constant -- qwen-tiny base moved 0.093 -> 0.856 while llama
   moved 0.001 -- so it reverses individual families in both directions. The
   beams are stored, so the pattern is recomputed here rather than at write time.

2. REASONING FAMILIES ARE INSTRUMENT-LIMITED. Models that open with thinking
   scaffolding ("Okay, the user is asking...") never reach a self-predication
   inside a 10-token window, so they score a spurious zero. Detected from the
   output rather than from the registry, whose has_reasoning flag does not
   identify the affected families. Reported separately, excluded from tallies.

3. smol IS EXCLUDED FROM POOLED TALLIES. Its smoke numbers motivated the
   control-set change, so including it is circular even on a fresh run.
   Reported per-family, excluded from means.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

BEAMS = "data/f20x_beams.parquet"

# Correction 1: allow an optional greeting/interjection before the predication.
# REPAIRED 2026-07-27. This file used to carry its own P_self pattern, which had
# two defects the producing instrument did not: it anchored on `^` rather than
# `^\s*` (invisible to the 83-87% of plain-rung beams beginning with a space)
# and matched only the ASCII apostrophe (base models emit U+2019 far more than
# aligned ones, so the miss was differential). Both were repaired in
# f20x_kinship_analyse.py and never here.
#
# Rather than patch two regexes into a third variant, this now imports the
# COMMITTED classifier -- the instrument that actually produced the addendum's
# table, recovered from a scratchpad and committed at 8463b8f, and verified by a
# second seat as behaviourally identical to the recovered original.
from f20x_identity import flags_for, _norm  # noqa: E402
# Correction 2: thinking scaffolding, detected from text.
THINK = re.compile(r"^\s*(okay|alright|let me|let's see|first,|the user)\b|<think", re.I)
EXCLUDE_POOLED = {"smol"}          # correction 3

# Correction 4: families sharing a base model produce IDENTICAL base rows
# (Llama-3.1-8B is the base for llama, tulu and every tulu-sft variant; Olmo-3
# for olmo and olmo-think). Pooling by family weights the base mean by how many
# aligned variants a lab shipped -- the pseudo-replication malign caught in the
# confirmation roster. Base arms are deduplicated by model_id below.


ALIGNED_ARMS = ("ego", "superego", "reinforced_superego")


def is_aligned(s):
    """The post-training stages, collectively.

    There is no arm called "aligned". This file assumed one at THREE separate
    sites; repairing only the one that crashed first left the other two, and the
    script then died further along on an empty frame. Defined once here so a
    fourth site cannot reintroduce it.
    """
    return s.isin(ALIGNED_ARMS)


def self_predicates(t):
    """Delegated to the committed classifier. The STOP-list gate lives there."""
    return "P_self" in flags_for(t, None)


def share(g, col="sp"):
    tot = g["path_prob"].sum()
    return g.loc[g[col], "path_prob"].sum() / tot if tot else float("nan")


def main():
    d = pd.read_parquet(BEAMS)
    d["text"] = d.text.fillna("")
    d["sp"] = [self_predicates(t) for t in d.text]
    d["think"] = [bool(THINK.search(t)) for t in d.text]

    both = {f for f in d.family.unique()
            # REPAIRED: this required an arm literally named "aligned".
            # The parquet carries base/ego/superego/reinforced_superego, so the
            # condition was unsatisfiable, every family was filtered out, and
            # the script crashed on an empty frame -- which is why the figures
            # it is cited for could not be recomputed by anyone.
            if {"base"} <= set(d[d.family == f].arm)
            and len(set(d[d.family == f].arm) - {"base"}) > 0}
    d = d[d.family.isin(both)]

    # correction 2: flag families whose identity beams are mostly scaffolding
    scaff = (d[(d.pclass == "identity") & (d["mode"] != "raw")]
             .groupby("family").apply(lambda g: share(g, "think"), include_groups=False))
    reasoning = set(scaff[scaff > 0.30].index)

    print(f"families with both arms: {len(both)}")
    if reasoning:
        print(f"instrument-limited (thinking scaffolding fills the 10-token window): "
              f"{', '.join(sorted(reasoning))}")
    print(f"excluded from pooled means (motivated the control change): "
          f"{', '.join(sorted(EXCLUDE_POOLED))}\n")

    core = d[~d.family.isin(reasoning | EXCLUDE_POOLED)]

    print("=== P_self by mode x arm x prompt class (pooled, corrected pattern) ===")
    nb = core[core.arm == "base"].model_id.nunique()
    print(f"    base arms deduplicated by model: {nb} distinct base models "
          f"from {core[core.arm=='base'].family.nunique()} families")
    keep = (core[core.arm == "base"].drop_duplicates(["model_id", "mode", "prompt"])
            .index.union(core[core.arm != "base"].index))
    s = (core.loc[keep].groupby(["family", "arm", "mode", "pclass"])
         .apply(share, include_groups=False).rename("P").reset_index())
    print(s.pivot_table(index=["mode", "arm"], columns="pclass",
                        values="P", aggfunc="mean").round(3).to_string())

    print("\n=== identity prompts, per family ===")
    ident = (d[d.pclass == "identity"].groupby(["family", "mode", "arm"])
             .apply(share, include_groups=False).rename("P").reset_index())
    piv = ident.pivot_table(index="family", columns=["mode", "arm"], values="P")
    cols = [(m, a) for m in ("chatml", "chat", "chat_nosys", "raw")
            for a in ("base",) + ALIGNED_ARMS if (m, a) in piv.columns]
    print(piv[cols].round(3).to_string())
    print("\nchatml holds the FORMAT constant across families, so base-arm variance")
    print("there is about the model rather than about whose template says what.")

    print("\n=== specificity: control leakage at chat/aligned, by family ===")
    ctl = (d[(d.pclass == "control") & (d["mode"] == "chat") & is_aligned(d.arm)]
           .groupby("family").apply(share, include_groups=False)
           .squeeze().sort_values(ascending=False))
    print(ctl.round(3).head(6).to_string())
    if len(ctl) and ctl.iloc[0] > 0.10:
        print(f"NOTE: {ctl.index[0]} leaks at {ctl.iloc[0]:.3f} -- specificity is not uniform.")


if __name__ == "__main__":
    main()
