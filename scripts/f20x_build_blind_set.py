"""Build a blind reading set: passages only, arm and family stripped, shuffled.

    uv run .venv/bin/python scripts/f20x_build_blind_set.py [--n 60]

TWO READERS, BOTH BLIND. RH and this seat read the same numbered passages
independently and compare notes. The point is a typology derived from the texts
rather than bins imposed in advance -- and, specifically, categories that are not
built around an arm difference this seat has a stake in finding.

WHAT IS STRIPPED. Arm, family, model id, and EVERY annotation field. The
annotator's own `subject_continuity` label is hidden too: showing it would prime
the reading toward the annotator's judgment, and the annotator's judgment is one
of the things under examination.

WHAT IS SAMPLED, and the strata are declared so the reading set's composition is
not a hidden choice:

    A  the ai+human flag conjunction        (over-inclusive net, known to catch
                                             named robots and confabulated AIs
                                             that are not contradictions at all)
    B  subject_continuity == referent_shifts (the annotator's holistic call,
                                             independent of the A flags)
    C  two or more self-assertions           (where a shift is even possible)
    D  uniform random                        (baseline: what ordinary looks like,
                                             without which every passage in the
                                             set reads as remarkable)

Stratum D matters most. A reading set made only of hits makes everything look
like a finding, because there is nothing to compare against.

THE KEY IS WRITTEN TO A SEPARATE FILE AND NOT PRINTED. This script never emits
arm labels to stdout, so the seat that runs it stays blind unless it deliberately
opens the key. Unblind only after both readers have written their notes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

ANN = "data/f20x_annotations.parquet"
OUT = "docs/f20x_blind_reading_set.md"
OUT2 = "docs/f20x_validation_set.md"
KEY = "data/f20x_blind_reading_key.parquet"
SEED = 20260728
SEED2 = 20260729

QUOTA = [("A", 14), ("B", 14), ("C", 16), ("D", 16)]


def main(n_override=0):
    d = pd.read_parquet(ANN)
    d["human"] = (d.claims_human_role.astype(bool) | d.gives_human_name.astype(bool)
                  | d.gives_biography.astype(bool))
    d["both"] = d.calls_self_ai.astype(bool) & d.human
    d["nass"] = [len(json.loads(x)) for x in d.self_assertions]

    strata = {
        "A": d[d.both],
        "B": d[d.subject_continuity == "referent_shifts"],
        "C": d[(d.nass >= 2) & ~d.both & (d.subject_continuity != "referent_shifts")],
        "D": d,
    }
    rng = random.Random(SEED)
    picked, seen = [], set()
    for name, k in QUOTA:
        if n_override:
            k = max(1, round(k * n_override / sum(q for _, q in QUOTA)))
        pool = strata[name]
        pool = pool[~pool.index.isin(seen)]
        take = pool.sample(min(k, len(pool)), random_state=SEED)
        seen.update(take.index)
        for r in take.itertuples():
            picked.append((name, r))

    rng.shuffle(picked)

    lines = [
        "# F20x blind reading set",
        "",
        f"{len(picked)} completions from the identity battery. Arm, family, model "
        "and all annotation labels removed. Order shuffled.",
        "",
        "Each passage is what a language model generated after the prompt shown. "
        "Sixty tokens, so most are cut off mid-sentence; truncation is not a "
        "finding.",
        "",
        "**Read for whatever is interesting.** No categories supplied on purpose. "
        "Note the number and what you noticed.",
        "",
        "---",
        "",
    ]
    key = []
    for i, (stratum, r) in enumerate(picked, 1):
        lines += [f"## {i}", "",
                  f"**Prompt:** `Q: {r.question}\\nA:`", "",
                  "```", r.text.rstrip(), "```", ""]
        key.append(dict(n=i, stratum=stratum, family=r.family, arm=r.arm,
                        model_id=r.model_id, base_model_id=r.base_model_id,
                        prompt=r.prompt, temperature=r.temperature,
                        subject_continuity=r.subject_continuity,
                        predicated_identity=r.predicated_identity,
                        self_assertions=r.self_assertions,
                        calls_self_ai=r.calls_self_ai, human=bool(r.human)))
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    pd.DataFrame(key).to_parquet(KEY, compression="zstd", index=False)

    # Deliberately says nothing about composition beyond the count.
    print(f"wrote {OUT} -- {len(picked)} passages, blind")
    print(f"key at {KEY} -- DO NOT OPEN until both readers have written notes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    main(ap.parse_args().n)
