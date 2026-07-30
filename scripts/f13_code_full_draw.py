"""Code the FULL drawn item set with the frozen v5 instrument.

    uv run .venv/bin/python scripts/f13_code_full_draw.py --model M --out F

FROZEN, and these are the terms:
  instrument sha256  9636474914b37bb2010fc413598d40b5590565630eb85579cdf700dde8ba5687
    -- THAT IS THE SHA OF THE INSTRUMENT THAT PRODUCED f13_full_v5_*, NOT of the
    one this script imports today. The module has since gained `intensity` and
    `direction`, so a run now codes under a DIFFERENT instrument. That was
    documented ([914].4) but nothing in this file said it, and the line above
    reads as a live guarantee. The runner now prints the live sha, warns on a
    mismatch, and STAMPS IT INTO EVERY ROW -- so a file says which taxonomy
    coded it instead of a reader having to inspect its value set to find out.
  AXIS is read from relations[0] -- the field says "most important first", and on
    the dev set reading ANY paradigmatic value in the list flipped pure-syntagmatic
    pairs (arrest/disperse) to paradigmatic. Primary-label axis unanimity 83% vs
    75% any-value, n=12. Declared here because it is an analysis choice made on
    dev data and must not be re-chosen on test data.
  AGREEMENT is computed on the FULL SET (intersection), because v4's runner-up data
    showed 6 of 9 residual disagreements were the same two labels in opposite order.
  DEV SET excluded upstream by the draw ([707].2).
  temperature is NOT pinned on claude-sonnet-5 (the API rejects it on that family);
    task.usage.report()['dropped_params'] records it and it goes in the methods note.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from malign_logits.tasks.code_displacement_relation import (
    RELATION, DisplacementRelationTask, prepare)

SRC = "data/f13_relation_items.parquet"

# The nine-relation instrument that produced f13_full_v5_*. Recorded so a file
# can say which taxonomy it was coded under WITHOUT anyone reading its value
# set to find out -- which is what [912] had to do, by hand, to establish that
# v5 and the frozen schema shared a taxonomy. Not an assertion: OPPOSITION's
# addition changes this sha legitimately. A MISMATCH IS A NOTICE, NOT A FAILURE.
SHA_NINE = "9636474914b37bb2010fc413598d40b5590565630eb85579cdf700dde8ba5687"


def main(model, out, workers=8, limit=0):
    d = pd.read_parquet(SRC)
    if limit:
        d = d.sample(limit, random_state=20260730).reset_index(drop=True)
    task = DisplacementRelationTask()
    if model:
        task.model = model
    sha = task.instrument_sha256()
    values = list(getattr(RELATION, "__args__", ()))
    print(f"instrument sha256: {sha}")
    if sha != SHA_NINE:
        print(f"  NOTE: differs from the nine-relation instrument ({SHA_NINE[:12]}...). "
              f"{len(values)} relation values: {','.join(values) if values else 'unread'}")
        print("  Judgments from a different instrument DO NOT POOL with it, and an "
              "agreement statistic across the two would measure the schema.")
    print(f"model {task.model}   items {len(d)}   "
          f"unique item strings {d.apply(lambda r: (r.prompt, r.a, r.b), axis=1).nunique()}")
    print(f"edges {d.groupby(['base_id','aligned_id']).ngroups}   "
          f"families {d.family.nunique()}   prompts {d.prompt_name.nunique()}")
    errs = {}
    anns = task.map([prepare(r.prompt, r.a, r.b) for r in d.itertuples()],
                    num_workers=workers, verbose=True, errors=errs,
                    fail_fast=False)
    keep = []
    for r, a in zip(d.itertuples(), anns):
        if a is None:
            continue
        v = a.model_dump()
        v["relations"] = list(v["relations"])
        v["relation_primary"] = v["relations"][0]
        # STAMPED INTO EVERY ROW, not printed and lost. The runner already
        # computed the sha and only ever showed it to whoever watched the run;
        # the file it wrote could not say what coded it. That is the same
        # omission `rule_version` exists to prevent at the cell layer, one
        # level up: without it a re-code leaves two taxonomies in one table
        # with nothing to tell them apart.
        v["instrument_sha256"] = sha
        v["n_relation_values"] = len(values) or None
        v["coder_model"] = task.model
        keep.append({**{k: getattr(r, k) for k in d.columns}, **v})
    o = pd.DataFrame(keep)
    o["relations"] = o.relations.map(lambda x: "|".join(x))
    o.to_parquet(out, compression="zstd", index=False)
    print(f"\ncoded {len(o)}, failed {len(errs)} -> {out}")
    if errs:
        import collections
        sig = collections.Counter(str(v.get("error"))[:70] for v in errs.values())
        print(f"  failure rate {len(errs)/len(d):.2%}; signatures:")
        for k, n in sig.most_common(4):
            print(f"    {n:>4}  {k}")
    try:
        print(task.usage.summary_line())
        rep = task.usage.report()
        if rep.get("dropped_params"):
            print(f"DROPPED PARAMS: {rep['dropped_params']}  <- goes in the methods note")
    except Exception as e:
        print("usage unavailable:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.model, a.out, a.workers, a.limit)
