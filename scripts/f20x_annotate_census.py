"""Annotate every F20x generation. A census, not a test.

    uv run .venv/bin/python scripts/f20x_annotate_census.py [--limit N]

WHY A CENSUS AND WHY ALL OF IT. The registered analysis had no measure for the
most interesting thing in this corpus, because the interesting thing was not
conceivable before anyone looked: base models do not contradict themselves about
who they are, and aligned models do. That came from reading passages, not from a
hypothesis. This run gives it a denominator.

NO PREFILTER. `subject_continuity` is only meaningful for completions carrying
two or more self-assertions, and a regex can find those cheaply -- 3,591 of
18,720. It is not used, because the regex fires at arm-dependent rates (base
0.197, reinforced 0.146, superego 0.191) and its known failure modes are
arm-dependent too: the `^` anchor, typos like "Myname is Jack", and name answers
with no copula at all ("It's <PRESIDIO_ANONYMIZED_PERSON> (Bryan)"). Selecting
the census with the instrument the census exists to audit would import exactly
the defect it is checking for.

Annotating everything also buys an LLM replication of the whole regex analysis,
which is Amendment 2's stated purpose -- neither instrument audits itself -- and
`predicated_identity` on every completion for the H2a entropy work.

TWO SINKS, deliberately. The Task's own HashStash caches by prompt, which makes
re-runs free. The cache's `set_gen_annotation(tagger, model, prompt, temp, idx)`
is the HOUSE sink for annotations and is keyed by the generation it annotates, so
anything else in this repo can find these by asking for the completion rather
than by knowing this script exists. `f20x_generate.py` wrote a parquet and
nothing else, and the completions sat in exactly one file for a whole run. Not
repeating that here.

RESUMABLE. Writes every 250 completions. A kill loses at most that.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from malign_logits.cache import get_cache
from malign_logits.provenance import provenance, describe
from malign_logits.tasks.annotate_identity import IdentityTask, prepare

GEN = "data/f20x_generations.parquet"
OUT = "data/f20x_annotations.parquet"
RUNG = "Q: {q}\nA:"
TAGGER = "f20x_identity_v1"
CHUNK = 250
WORKERS = 8

FIELDS = ["coherence", "format_drift", "self_predicates", "predicated_identity",
          "identity_kind", "calls_self_ai", "claims_human_role",
          "gives_human_name", "gives_biography", "proper_names", "declines",
          "contentless", "redaction", "self_assertions", "subject_continuity"]


def main(limit=0):
    prov = provenance(__file__,
                      closure=["malign_logits/tasks/annotate_identity.py"])
    print(describe(prov))

    d = pd.read_parquet(GEN).reset_index(drop=True)
    if limit:
        d = d.head(limit)
    print(f"\n{len(d):,} completions | {d.family.nunique()} families | "
          f"{d.base_model_id.nunique()} distinct base models")

    done = {}
    if os.path.exists(OUT):
        prev = pd.read_parquet(OUT)
        done = {(r.model_id, r.question, r.temperature, r.idx_in_cell): True
                for r in prev.itertuples()}
        print(f"resuming: {len(prev):,} already annotated")

    # idx within cell, so the house cache key is stable and re-runs land on the
    # same slot rather than renumbering.
    d["idx_in_cell"] = d.groupby(["model_id", "question", "temperature"]).cumcount()
    todo = d[[not done.get((r.model_id, r.question, r.temperature, r.idx_in_cell))
              for r in d.itertuples()]]
    print(f"{len(todo):,} to annotate\n")

    cm = get_cache()
    task = IdentityTask()
    rows = [] if not os.path.exists(OUT) else pd.read_parquet(OUT).to_dict("records")
    ok = fail = 0

    for start in range(0, len(todo), CHUNK):
        blk = todo.iloc[start:start + CHUNK]
        prompts = [prepare(r.question, r.text) for r in blk.itertuples()]
        anns = task.map(prompts, num_workers=WORKERS, verbose=False)
        for r, a in zip(blk.itertuples(), anns):
            if a is None:
                fail += 1
                continue
            ok += 1
            v = a.model_dump()
            cm.set_gen_annotation(TAGGER, r.model_id, RUNG.format(q=r.question),
                                  v, temp=float(r.temperature),
                                  idx=int(r.idx_in_cell))
            rows.append(dict(
                family=r.family, arm=r.arm, model_id=r.model_id,
                base_model_id=r.base_model_id, prompt=r.prompt,
                question=r.question, temperature=r.temperature,
                idx_in_cell=int(r.idx_in_cell), text=r.text,
                **{k: (json.dumps(v[k]) if isinstance(v[k], list) else v[k])
                   for k in FIELDS}))
        df = pd.DataFrame(rows)
        df.attrs["provenance"] = json.dumps(prov)
        df.to_parquet(OUT, compression="zstd", index=False)
        print(f"  {min(start+CHUNK, len(todo)):>6,}/{len(todo):,}  "
              f"ok={ok:,} failed={fail:,}  -> {OUT}")

    print(f"\n  {ok:,} annotated, {fail:,} failed to parse")
    print(f"  house cache: tagger={TAGGER!r}, keyed by (model, rung, temp, idx)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args().limit)
