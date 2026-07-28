"""Blind validation set #2: passages neither reader has seen.

    uv run .venv/bin/python scripts/f20x_build_validation_set.py [--n 30]

WHY A SECOND SET. The first 60 is a DEVELOPMENT set and is spent. Every category
in the coding scheme was derived from it, and twelve of its passages are now
few-shot examples inside the coder. Any agreement number computed on it measures
how well the prompt transcribes the scheme, not whether the scheme travels.

This set excludes every passage in set #1 and is drawn with a different seed.

ENRICHED ON PURPOSE, and the enrichment uses only the census annotations that
already exist -- nothing here consults the coder being validated. Sampling at
random would hand two human readers twenty passages that are obviously `stable`
and produce an agreement rate dominated by easy negatives. The strata:

    40%  census flags a non-trivial handling (subject_continuity not
         not_applicable, or 2+ self-assertions)
    20%  calls_self_ai, where person/machine bothness can occur
    20%  format drift into a genre, where contradiction_from_genre is live
    20%  uniform random, so the set contains ordinary passages and a reader can
         still calibrate what unremarkable looks like

An agreement rate measured here is agreement ON CODEABLE MATERIAL. It is not the
corpus base rate and must never be reported as one.

KEY WRITTEN, NOT PRINTED. Same discipline as set #1.
"""
from __future__ import annotations

import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

ANN = "data/f20x_annotations.parquet"
SET1 = "docs/f20x_blind_reading_set.md"
OUT = "docs/f20x_validation_set.md"
KEY = "data/f20x_validation_key.parquet"
SEED = 20260729


def main(n=30):
    d = pd.read_parquet(ANN)
    seen = set()
    if os.path.exists(SET1):
        import re
        seen = {t.strip() for t in re.findall(r"```\n(.*?)\n```", open(SET1).read(), re.S)}
    d = d[~d.text.str.strip().isin(seen)]
    d["nass"] = [len(json.loads(x)) for x in d.self_assertions]

    strata = [
        ("handling", d[(d.subject_continuity != "not_applicable") | (d.nass >= 2)], 0.40),
        ("ai",       d[d.calls_self_ai.astype(bool)],                              0.20),
        ("genre",    d[d.format_drift != "none"],                                  0.20),
        ("random",   d,                                                            0.20),
    ]
    picked, used = [], set()
    for name, pool, frac in strata:
        k = max(1, round(n * frac))
        pool = pool[~pool.index.isin(used)]
        take = pool.sample(min(k, len(pool)), random_state=SEED)
        used.update(take.index)
        picked += [(name, r) for r in take.itertuples()]
    random.Random(SEED).shuffle(picked)
    picked = picked[:n]

    L = ["# F20x validation set", "",
         f"{len(picked)} completions. **None appears in the first reading set.** "
         "Arm, family, model and all annotation labels removed; order shuffled.",
         "",
         "Code each against `docs/f20x_identity_coding_scheme.md`. Write the "
         "code names, or `stable`. Several codes per passage are fine and "
         "expected.",
         "",
         "```",
         "bothness  marked_contradiction  quiet_drift  mania  dissolution",
         "name_arbitrary  number_shift  origin_displaced  split_trace  stable",
         "```",
         "",
         "Sixty tokens each, so most stop mid-sentence. Truncation is not a code.",
         "", "---", ""]
    key = []
    for i, (stratum, r) in enumerate(picked, 1):
        L += [f"## {i}", "", f"**Prompt:** `Q: {r.question}\\nA:`", "",
              "```", r.text.rstrip(), "```", "", "**Codes:** ", "", "---", ""]
        key.append(dict(n=i, stratum=stratum, family=r.family, arm=r.arm,
                        base_model_id=r.base_model_id, prompt=r.prompt,
                        temperature=r.temperature, text=r.text))
    open(OUT, "w").write("\n".join(L))
    pd.DataFrame(key).to_parquet(KEY, compression="zstd", index=False)
    print(f"wrote {OUT} -- {len(picked)} passages, none from set #1")
    print(f"key at {KEY} -- unopened until both readers have coded")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30)
    main(ap.parse_args().n)
