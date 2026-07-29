"""BOS restatement supply check -- precondition (i) of the BOS registration.

    uv run .venv/bin/python scripts/f20x_bos_supply.py [--n 200]

Does an unconditional completion adopt a referent and then describe it a SECOND
time? If opportunity is in single digits, the BOS anchor is dead for the same reason
fact-drift was ([235]: 4.8% opportunity, and the pre-measurement estimate was wrong
by a factor of nine), and this run reports that rather than coding 1,800 passages to
discover it.

ROSTER, from docket [251]: the text lives in `bos_generations.parquet`, NOT in
`jakobson.parquet` -- which carries precomputed metrics and no text column at all,
and whose `mean_drift` / `total_drift` are F36 embedding-trajectory measures, not
the coder's drift. Nine families carry both a base and an aligned arm at BOS; tulu
is base-only and is EXCLUDED here so every family contributes to both arms.

SAMPLING is stratified by family x arm so no family dominates, and the seed is
declared. The check reports per-arm as well as pooled, because the whole point is
whether the frame supplies events at all -- and if it supplies them to one arm and
not the other, that is a bigger finding than the rate.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402
from malign_logits.tasks.code_bos_opportunity import BosOpportunityTask, prepare  # noqa: E402

SRC = os.path.join(PATH_DATA, "bos_generations.parquet")
OUT = os.path.join(PATH_DATA, "f20x_bos_supply.parquet")
SEED = 20260729


def sample(n):
    d = pd.read_parquet(SRC)
    d = d[d.prompt_type == "bos"].copy()
    d["aligned"] = d.layer != "base"
    # Nine paired families; tulu is base-only at BOS and would enter one arm alone.
    paired = d.groupby("family").aligned.nunique()
    keep = paired[paired == 2].index
    dropped = sorted(set(d.family) - set(keep))
    d = d[d.family.isin(keep)]
    d = d[d.text.fillna("").str.split().str.len() >= 10]

    per = max(1, n // (len(keep) * 2))
    # Index-based selection: groupby.apply moves the grouping columns into the
    # index, and a frame whose `family` column has silently become an index level
    # is the same class of defect as a key read back from disk.
    idx = []
    for _, g in d.groupby(["family", "aligned"]):
        idx.extend(g.sample(min(per, len(g)), random_state=SEED).index)
    out = d.loc[idx].reset_index(drop=True)
    assert "family" in out.columns and len(out) <= len(d)
    print(f"{len(out)} passages | {out.family.nunique()} families x 2 arms "
          f"| ~{per}/cell | excluded (unpaired): {dropped}")
    return out


def main(n):
    if os.path.exists(OUT):
        rec = pd.read_parquet(OUT)
        print(f"reusing {len(rec)} coded rows from {OUT}")
    else:
        s = sample(n)
        task = BosOpportunityTask()
        res = task.map([prepare(t) for t in s.text], num_proc=8, desc="bos supply")
        rec = s.copy()
        rec["verdict"] = [r.verdict if r else None for r in res]
        rec["topic"] = [r.topic if r else None for r in res]
        rec["reason"] = [r.reason if r else None for r in res]
        rec["n_doublings"] = [len(r.doublings) if r else None for r in res]
        rec["doublings"] = [
            "; ".join(f"({x.entity}|{x.attribute}|{x.value_1}|{x.value_2}|"
                      f"{'ok' if x.compatible else 'CONFLICT'})" for x in r.doublings)
            if r else None for r in res]
        rec.to_parquet(OUT, compression="zstd", index=False)
        print(f"wrote {OUT}")

    rec = rec[rec.verdict.notna()]
    print(f"\n{'='*70}\nBOS RESTATEMENT SUPPLY  (n={len(rec)} coded)\n{'='*70}")
    print(rec.verdict.value_counts().to_string())

    rec["opportunity"] = rec.verdict.isin(
        ["restatement_consistent", "restatement_incompatible"])
    rec["adopted"] = rec.verdict != "no_topic_adopted"
    print(f"\n  topic adopted        {rec.adopted.mean():.3f}")
    print(f"  RESTATEMENT (opp)    {rec.opportunity.mean():.3f}   <- the precondition")
    print(f"  incompatible         {(rec.verdict == 'restatement_incompatible').mean():.3f}")

    print("\n  by arm:")
    for al in (False, True):
        s = rec[rec.aligned == al]
        print(f"    {'aligned' if al else 'base   '}  n={len(s):3d}  adopted "
              f"{s.adopted.mean():.3f}  opportunity {s.opportunity.mean():.3f}")

    print("\n  by family (opportunity rate):")
    print(rec.groupby("family").opportunity.agg(["mean", "sum", "count"])
             .round(3).to_string())

    ex = rec[rec.n_doublings.fillna(0) > 0].head(6)
    if len(ex):
        print("\n  sample tuples:")
        for r in ex.itertuples():
            print(f"    [{r.family}/{'al' if r.aligned else 'base'}] {r.doublings[:150]}")

    # A "doubling" whose two values are IDENTICAL is not a doubling. Objective,
    # so it is applied here rather than left to a reader; the degenerate classes
    # that are NOT objective (a variable ranging over an equation, a lookup
    # table, a maths sequence) still need eyes and are reported, not filtered.
    def nondegenerate(s):
        if not isinstance(s, str) or not s.strip():
            return False
        for t in (x.strip("()").split("|") for x in s.split("; ") if x.strip()):
            if len(t) >= 4 and t[2].strip().lower() != t[3].strip().lower():
                return True
        return False

    rec["opp_filtered"] = rec.doublings.map(nondegenerate) & rec.opportunity
    print(f"\n  raw opportunity              {rec.opportunity.mean():.3f}")
    print(f"  after identical-value filter {rec.opp_filtered.mean():.3f}")
    print(f"  families supplying ZERO: "
          f"{sorted(rec.groupby('family').opportunity.sum().pipe(lambda s: s[s == 0]).index)}")

    print(f"\n{'='*70}")
    print("NO AUTOMATED VERDICT. An earlier version of this script printed one on a")
    print("0.10 threshold this seat chose, and the raw rate landed at 0.101 -- ONE")
    print("passage from flipping it, on a number itself inflated by degenerate cases.")
    print("A threshold that fine, set by the seat reading the result, is not a rule.")
    print("Read the tuples. The classes to reject by eye: a variable ranging over an")
    print("equation, a lookup table, a maths sequence, a morphological variant.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    main(ap.parse_args().n)
