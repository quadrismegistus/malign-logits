"""Freeze the full-grid spec for the true_word_probs re-run.

    uv run .venv/bin/python scripts/build_grid_spec.py --dry-run
    uv run .venv/bin/python scripts/build_grid_spec.py --exclude-finding F19
    uv run .venv/bin/python scripts/build_grid_spec.py --out /tmp/grid.json

WHAT THIS REPLACES. The current roster is a UNION OF EXPERIMENTAL HISTORIES, not
a design: per-model prompt counts run 3 to 569 and record how many past
experiments touched a model, nothing about the model. Only the 73-prompt core is
rectangular. Any comparison wider than that silently compares different prompt
sets across models, which no amount of care at analysis time can repair.

A grid fixes it by construction: every model gets every prompt, so support is
identical everywhere and coverage stops being a confound.

IT ALSO RETIRES THE TWO-RULE HAZARD. The existing store was produced before the
CJK punctuation fix, the dictionary trie, and the script-transition rule. The
boundary rule is NOT in the cache key, so old-rule and new-rule cells are
indistinguishable once both are present -- the defect that made `beam_words`
unusable when two beam widths coexisted across 70+ models. Running the whole
grid under one rule means there is never a mixture to detect.

THE PROMPT UNIVERSE IS THE CENSUS, NOT THE STORE. `prompt_inventory.csv` sees
only what true_word_probs scored (604). `prompt_census_all.csv` is the union
over 13 stashes (725), and the categorisation adds 43 F21 pairs no stash has
ever scored. The empty string is EXCLUDED: it is the cache key for ingested
human corpora that have no model prompt, not a prompt.

--exclude-finding EXISTS FOR ONE REAL DECISION. F19's 109 prompts are
surprisal-corpus passages (dreams, fiction, abstracts), not battery stimuli.
Word probabilities on them are meaningful but answer a different question, and
they are 17% of the run. In or out is a scope call, so it is a flag rather than
a silent filter -- and whichever way it goes, the excluded set is COUNTED and
PRINTED, never dropped quietly.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402

CENSUS = os.path.join(PATH_DATA, "prompt_census_all.csv")
CATS = os.path.join(PATH_DATA, "prompt_categorisation.json")
TWP = os.path.join(PATH_DATA, "twp_cloud")
OUT = os.path.join(PATH_DATA, "grid_spec.json")


def main(a):
    cats = json.load(open(CATS))["prompts"]
    by_prompt = {e["prompt"]: e for e in cats if e["prompt"] != ""}

    # models: every model that has ever produced a true_word_probs file
    have = defaultdict(set)
    for f in sorted(glob.glob(os.path.join(TWP, "*.jsonl"))):
        mid = os.path.basename(f)[:-6].replace("__", "/")
        for line in open(f):
            try:
                have[mid].add(json.loads(line)["prompt"])
            except Exception:
                pass
    models = sorted(have)

    universe = sorted(by_prompt)
    excl = set(a.exclude_finding or [])
    keep = [p for p in universe if by_prompt[p].get("finding") not in excl]
    dropped = [p for p in universe if by_prompt[p].get("finding") in excl]

    grid = len(models) * len(keep)
    todo = sum(len(set(keep) - have[m]) for m in models)
    fin = Counter(by_prompt[p].get("finding") for p in keep)

    print(f"models            {len(models)}")
    print(f"prompt universe   {len(universe)}   (census union + categorised, "
          f"empty string excluded)")
    if excl:
        # THE HOUSE RULE: what a filter removes is counted and named, never
        # silently absent -- a grid that quietly omits a stratum reads as
        # complete coverage of a smaller world.
        dc = Counter(by_prompt[p].get("finding") for p in dropped)
        print(f"EXCLUDED          {len(dropped)}   {dict(dc)}   "
              f"({len(models)*len(dropped):,} cells not run)")
    print(f"prompts in grid   {len(keep)}   {dict(fin)}")
    print(f"grid cells        {grid:,}")
    print(f"already present   {grid - todo:,}")
    print(f"TO RUN            {todo:,}   ~{todo/4/3600:.1f} h at 4 p/s")

    if a.dry_run:
        print("\nDRY RUN -- no spec written")
        return

    spec = [{"model": m, "prompts": sorted(set(keep) - have[m])} for m in models]
    spec = [e for e in spec if e["prompts"]]
    # ascending by work, so a cancellation costs the least-finished model
    spec.sort(key=lambda e: len(e["prompts"]))
    json.dump(spec, open(a.out, "w"))
    print(f"\nwrote {a.out}: {len(spec)} models, "
          f"{sum(len(e['prompts']) for e in spec):,} cells")
    print("BOUNDARY RULE: CJK punctuation + dictionary trie + script transition. "
          "The rule is NOT in the cache key, so this must OVERWRITE.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-finding", nargs="*", default=[])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
