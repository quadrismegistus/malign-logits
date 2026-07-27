"""F20x generation battery: the registered analysis. Regex arm.

    uv run .venv/bin/python scripts/f20x_generation_analyse.py

Registered at docs/f20x_generation_spec.md with six amendments. This file
implements what those amendments declare and adds nothing they do not.

## Declared before running

DATA: `data/f20x_generations.parquet`, 18,720 completions, 39 families with
data, 29 distinct base models. Ten families are absent and every one is a load
or kernel failure recorded in `data/f20x_generations_failures.parquet` with a
`pass_no` column. Amendment 6/6b: the losses are STRUCTURED -- the entire
non-transformer arm, the only sparse-MoE family, the top of the size range --
so nothing here covers the registry and nothing generalises to SSM
architectures.

INSTRUMENT: `f20x_identity.flags_for`, the committed classifier. Amendment 5
rebased condition A onto it: the as-published pattern anchors on `^` and every
generation begins with a space, so it fires on 0.00 of base and 0.00 of superego
completions in this text type. Both defective patterns are still computed and
printed so the size of that defect stays inspectable rather than becoming a
claim about it.

UNIT: the distinct base model, aligned arms averaged within base. Rule 2. n=29.

WINDOWS: both, always. The 10-token prefix is the beam-comparable one; the full
60 tokens is the interpretable one. Sixty tokens scored by classifiers built for
ten is not comparable to the beam figures and the difference would be confounded
with window length.

GATING: PRIMARY IS UNGATED (Amendment 5). Excluding drifted completions would
change the question to "among completions that stayed on task", and staying on
task is plausibly part of what alignment does. The gated secondary is reported
beside it as a DIFFERENT QUESTION, with per-arm retention printed every time,
because retention is differential (roughly base 18%, superego 31%, reinforced
72%) and a reader must see the selection.

BASELINE for H1: the RE-MEASURED beam figures (Amendment 4). The published table
was withdrawn as permanently unrecomputable and must never be the comparison.

TEST: paired Wilcoxon over distinct base models, each measure stating its own n.

WHAT THIS FILE DOES NOT DO: H2a (identity entropy) and H2b (within-completion
float) both need the LLM annotator, which has not been run at scale. H2a
additionally needs a normalisation rule declared before anyone sees an entropy --
'ai', 'ai language model' and 'artificial intelligence' are three strings for one
identity, and choosing how to fold them after seeing the numbers is fishing.
Neither is attempted here.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scipy import stats

from f20x_identity import flags_for
from malign_logits.provenance import provenance, describe

GEN = "data/f20x_generations.parquet"
OUT = "data/f20x_generation_results.parquet"

# The re-measured beam figures, Amendment 4. Base -> aligned, terminal/mass/
# symmetric, n=22, with the number of the 8 specification cells that reached
# p<0.05. These are the ONLY citable beam numbers.
BEAM = {
    "P_self":       (0.567, 0.748, "6/8"),
    "AI-ness":      (0.235, 0.509, "8/8"),
    "human any":    (0.468, 0.199, "8/8"),
    "human role":   (0.216, 0.043, "8/8"),
    "human name":   (0.104, 0.117, "0/8 NULL"),
    "biography":    (0.148, 0.041, "8/8"),
    "names":        (0.117, 0.211, "8/8"),
    "own name":     (0.009, 0.153, "8/8"),
    "other's name": (0.108, 0.058, "0/8 NULL"),
}
MEASURES = ["P_self", "AI-ness", "human any", "human role", "human name",
            "biography", "names", "own name", "other's name"]

# Scaffolding proxy. Validated 36/36 against the LLM annotator's `format_drift`
# label; used here because the annotator has not been run at scale.
DRIFT = re.compile(r"\n[A-D][:.)]\s|\bAnswer:|\bQuestion:|\nQ:|\nA:", re.M)


def score(d, col):
    F = [set(flags_for(t, f)) for t, f in zip(d[col], d.family)]
    out = d.copy()
    for m in MEASURES:
        out[m] = [m in x for x in F]
    out["human any"] = [bool({"human role", "human name", "biography"} & x) for x in F]
    return out


def paired(d, measure):
    """Rule 2: distinct base model is the unit; aligned arms averaged within it."""
    sub = d if measure == "P_self" else d[d.P_self]
    if not len(sub):
        return None
    r = sub.groupby(["base_model_id", "arm"])[measure].mean().reset_index()
    b = r[r.arm == "base"].set_index("base_model_id")[measure].rename("b")
    a = r[r.arm != "base"].groupby("base_model_id")[measure].mean().rename("a")
    j = pd.concat([b, a], axis=1).dropna()
    if len(j) < 6:
        return dict(measure=measure, n=len(j), base=float("nan"),
                    aligned=float("nan"), delta=float("nan"), up=0, p=float("nan"))
    return dict(measure=measure, n=len(j), base=j.b.mean(), aligned=j.a.mean(),
                delta=(j.a - j.b).mean(), up=int((j.a > j.b).sum()),
                p=stats.wilcoxon(j.a, j.b).pvalue)


def table(d, title, rows_out, window, gate):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"  {'measure':<13}{'base':>8}{'aligned':>9}{'delta':>9}{'up/n':>8}"
          f"{'p':>9}   {'beam base->aligned':>20}")
    for m in MEASURES:
        r = paired(d, m)
        if r is None:
            continue
        bb, ba, cells = BEAM[m]
        same = "" if pd.isna(r["delta"]) else (
            "  agrees" if (r["delta"] > 0) == (ba > bb) else "  OPPOSITE")
        print(f"  {m:<13}{r['base']:>8.3f}{r['aligned']:>9.3f}{r['delta']:>+9.3f}"
              f"{str(r['up'])+'/'+str(r['n']):>8}{r['p']:>9.4f}   "
              f"{bb:.3f}->{ba:.3f} {cells:<9}{same}")
        rows_out.append({**r, "window": window, "gate": gate})


def main():
    prov = provenance(__file__, closure=["scripts/f20x_identity.py"])
    print(describe(prov))

    d = pd.read_parquet(GEN)
    d["drift"] = [bool(DRIFT.search(t)) for t in d.text]
    print(f"\n{len(d):,} completions | {d.family.nunique()} families | "
          f"{d.base_model_id.nunique()} distinct base models")

    rows = []
    for window, col in [("prefix", "prefix"), ("full", "text")]:
        s = score(d, col)
        table(s, f"UNGATED (primary) -- {window} window",
              rows, window, "ungated")

    print(f"\n{'='*78}\nGATED SECONDARY -- a DIFFERENT QUESTION, not a robustness check"
          f"\n{'='*78}")
    ret = d.groupby("arm").apply(lambda g: 1 - g.drift.mean(), include_groups=False)
    print("  RETENTION, and it is differential -- these arms are selected at "
          "different rates:")
    for arm, v in ret.items():
        print(f"    {arm:<22} {v:.1%} of completions retained")
    g = d[~d.drift]
    print(f"  {len(g):,} of {len(d):,} completions retained overall "
          f"({len(g)/len(d):.1%})")
    for window, col in [("prefix", "prefix"), ("full", "text")]:
        table(score(g, col), f"GATED (on-task only) -- {window} window",
              rows, window, "gated")

    print(f"\n{'='*78}\nFORMAT DRIFT AS AN OUTCOME (Amendment 5), paired unit"
          f"\n{'='*78}")
    r = d.groupby(["base_model_id", "arm"]).drift.mean().reset_index()
    b = r[r.arm == "base"].set_index("base_model_id").drift.rename("b")
    a = r[r.arm != "base"].groupby("base_model_id").drift.mean().rename("a")
    j = pd.concat([b, a], axis=1).dropna()
    p = stats.wilcoxon(j.a, j.b).pvalue
    print(f"  base {j.b.mean():.3f}  aligned {j.a.mean():.3f}  "
          f"delta {(j.a-j.b).mean():+.3f}  aligned LOWER in "
          f"{int((j.a < j.b).sum())}/{len(j)}  p={p:.4f}")
    print("  Registered consequence (Amendment 5): if this misses p<0.05 the "
          "ordering is\n  NOT ESTABLISHED, and the pooled arm rates are never "
          "quoted without this test.")
    print(f"  pooled arm rates, for contrast: "
          f"{d.groupby('arm').drift.mean().round(3).to_dict()}")

    print(f"\n{'='*78}\nINSTRUMENT CHECK: the defective patterns, kept inspectable"
          f"\n{'='*78}")
    PUB = re.compile(r"(^|[.!?,]\s+|^\s*\w{1,12}[,!]\s+)(I am|I'm|My name is|This is)\b", re.I)
    d["published"] = [bool(PUB.search(t)) for t in d.text]
    d["committed"] = [("P_self" in flags_for(t, None)) for t in d.text]
    print(f"  as-published fires {d.published.mean():.3f} | "
          f"committed fires {d.committed.mean():.3f} | "
          f"ratio {d.committed.mean()/max(d.published.mean(),1e-9):.2f}x | "
          f"disagree on {(d.published!=d.committed).mean():.1%}")
    print(d.groupby("arm")[["published", "committed"]].mean().round(3).to_string()
          .replace("\n", "\n  ").rjust(2))

    df = pd.DataFrame(rows)
    df.attrs["provenance"] = json.dumps(prov)
    df.to_parquet(OUT, compression="zstd", index=False)
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
