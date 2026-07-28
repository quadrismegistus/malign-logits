"""Smoke test for the nonce coder, exercising the parent's audit conditions.

    uv run .venv/bin/python scripts/f20x_nonce_smoke.py [--n 30]

WHAT IT CHECKS, and why each is here rather than assumed:

A. IT RUNS, AND ON REAL COMPLETIONS. Stratified across condition and arm so the
   sample is not all one cell.

B. RELIABILITY IS TESTED BY PERTURBING THE PROMPT, NOT BY RERUNNING IT. The
   parent instrument's first reliability check ran the same prompt twice at T=0,
   got kappa=1.000 on everything, and certified that the provider is
   deterministic -- which a consistently wrong annotator also passes. The
   perturbation here reverses the order in which the codes are described.

C. THE BLINDNESS CLAIM IS CHECKED, NOT ASSERTED. The coder is given the term and
   the passage only. This prints what it was actually sent, so a reader can see
   that the prompt -- and therefore the condition -- is absent.

D. DEGENERATE OUTPUT IS LOOKED FOR. An annotator that answers `stable` to
   everything scores well against itself. Code distribution is printed.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydantic import Field  # noqa: E402
from malign_logits.tasks.code_nonce import NonceCodingTask, NonceCoding, prepare  # noqa: E402


def _reversed_schema():
    """A genuinely different prompt: the code definitions in reverse order.

    THE FIRST VERSION OF THIS DID NOT PERTURB ANYTHING. It computed a reversed
    description into an unused attribute and left `schema` pointing at the
    original, so both runs sent an identical prompt at temperature 0 and agreement
    came back 1.000 on every field. That is the parent instrument's condition-C
    failure reproduced exactly -- a reliability check that certifies the provider
    is deterministic. The schema is now rebuilt, so the two prompts differ.
    """
    from pydantic import create_model
    fields = {}
    for name, f in NonceCoding.model_fields.items():
        desc = f.description or ""
        if name == "codes":
            head, *defs = desc.split("\n")
            desc = "\n".join([head] + defs[::-1])
        fields[name] = (f.annotation, Field(description=desc))
    return create_model("NonceCodingRev", **fields)


NonceCodingRev = _reversed_schema()


class Perturbed(NonceCodingTask):
    """Same fields, code definitions in reverse order. Condition B."""
    name = "f20x_nonce_coding_perturbed"
    schema = NonceCodingRev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()

    d = pd.read_parquet("data/f20x_nonce.parquet")
    d["text"] = d.text.fillna("")
    d = d[d.text.str.strip().str.len() > 0]
    # stratify: equal draw per condition x arm, so no cell dominates
    d["al"] = d.arm != "base"
    per = max(1, a.n // (d.condition.nunique() * 2))
    # sample per cell explicitly: groupby.apply folds the grouping keys into the
    # index and the columns then vanish, which cost one run before it was noticed
    samp = pd.concat([g.sample(min(per, len(g)), random_state=7)
                      for _, g in d.groupby(["condition", "al"])]).reset_index(drop=True)
    print(f"{len(samp)} completions sampled, "
          f"{samp.condition.nunique()} conditions x 2 arms\n")

    print("=== CONDITION C: exactly what the annotator is sent ===")
    r0 = samp.iloc[0]
    print(prepare(r0.word, r0.text)[:300])
    print(f"[condition={r0.condition}, arm={r0.arm} -- NEITHER appears above]\n")

    task = NonceCodingTask()
    items = [prepare(r.word, r.text) for r in samp.itertuples()]
    out = task.map(items, num_proc=8, desc="coding")

    ok = [o for o in out if o is not None]
    print(f"=== CONDITION A: {len(ok)}/{len(items)} coded, "
          f"{len(items)-len(ok)} failed ===")
    codes = pd.Series([c for o in ok for c in o.codes]).value_counts()
    print("\n=== CONDITION D: code distribution (degenerate if one dominates) ===")
    print(codes.to_string())
    nval = pd.Series([len(o.values) for o in ok]).value_counts().sort_index()
    print(f"\nvalues extracted per passage:\n{nval.to_string()}")
    print(f"drift_from_genre true in {sum(o.drift_from_genre for o in ok)}/{len(ok)}")

    print("\n=== three codings, read rather than counted ===")
    for o, r in list(zip(ok, samp.itertuples()))[:3]:
        print(f"  [{r.word}] codes={o.codes} values={o.values[:3]}")
        print(f"     note: {o.value_note[:120]}")
        print(f"     text: {r.text[:110].replace(chr(10),' | ')!r}")

    print("\n=== CONDITION B: prompt-perturbation agreement ===")
    p = Perturbed()
    out2 = p.map(items, num_proc=8, desc="perturbed")
    both = [(x, y) for x, y in zip(out, out2) if x and y]
    agree = sum(1 for x, y in both if set(x.codes) == set(y.codes))
    prim = sum(1 for x, y in both
               if ("no_value_posed" in x.codes) == ("no_value_posed" in y.codes))
    drift = sum(1 for x, y in both
                if ("quiet_drift" in x.codes) == ("quiet_drift" in y.codes))
    print(f"  exact code-set agreement : {agree}/{len(both)} = {agree/len(both):.3f}")
    print(f"  no_value_posed agreement : {prim}/{len(both)} = {prim/len(both):.3f}  <- outcome one")
    print(f"  quiet_drift agreement    : {drift}/{len(both)} = {drift/len(both):.3f}  <- primary")


if __name__ == "__main__":
    main()
