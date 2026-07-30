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

POPULATIONS OF RECORD (registrar [787].1, superseding earlier figures):

    GRID      767 STRINGS   -- frozen at docket [781], unchanged by any of this
    ANALYSIS  754 ACTIVE rows, one-to-many join
    RETIRED    69 rows, excluded from every join BY STATUS

**An analysis that reads the categorisation file without a status filter is
wrong by construction.** The earlier figure of 772 is superseded. None of this
touches the grid: scoring is per STRING and the reconciliation moved only rows
and fields, which is the property that let the run and the reconciliation
proceed in parallel.

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
    # THE UNIVERSE IS A UNION OF STRINGS, NOT A READ OF THE CATEGORISATION FILE.
    # Built this way so the spec is INDEPENDENT of the pending reconciliation:
    # retiring duplicate rows cannot remove a prompt (both rows carry the same
    # string), and unkeying groups or deleting pair_role/group_role are field
    # operations. Verified: categorisation 767, census 724, store 604, UNION 767
    # -- the census and store are subsets, and the 43 extra are the F21 pairs no
    # stash has scored, which are the additions we want.
    #
    # The earlier version was `{e["prompt"]: e for e in cats}` -- a dict
    # comprehension that silently dropped 56 rows by last-write-wins, of which 5
    # were genuine dual membership. Deduplication is now explicit and counted.
    # STATUS FILTER, IN THE CODE. Amendment 1a's filter was applied BY HAND to
    # grid_spec.json and never landed here, so every rebuild silently reinstated
    # retired prompts -- undoing a freeze amendment with nobody deciding to.
    # Measured when found: 109 retired-only strings, 60 already back in the spec,
    # 49 more that a rebuild would have added.
    #
    # A string RETIRED in one row but ACTIVE in another is KEPT: 46 of 67 retired
    # strings are dual-membership, and dropping on any-retired would delete
    # prompts that a live design still uses.
    cats = json.load(open(CATS))["prompts"]
    # AMENDMENT 1a HAD TWO HALVES AND ONLY ONE REACHED THE CODE. The status
    # filter landed here; the BOS/label exclusion did not, so the rebuild put all
    # four per-family BOS strings back. The round-trip check caught ONE of them
    # (deepseek's, which fails to encode) and could not see the other three --
    # they round-trip perfectly on their own tokenizer. They are wrong for a
    # SEMANTIC reason: a per-family BOS scored on 103 models is that family's
    # token fed to models where it is a literal string, shattering into
    # characters (<|begin_of_text|> is 9 tokens on Amber). No encoding test can
    # detect that, which is why the exclusion has to be declared, not derived.
    EXCLUDE_LITERAL = {
        "<s>", "<|endoftext|>", "<|begin_of_text|>", "\uff5cbegin\u2581of\u2581sentence\uff5c",
        "<\uff5cbegin\u2581of\u2581sentence\uff5c>", "bos",
    }
    retired = {e["prompt"] for e in cats if e.get("status") == "RETIRED"}
    active = {e["prompt"] for e in cats if e.get("status") == "ACTIVE"}
    drop_retired = (retired - active) | EXCLUDE_LITERAL
    rows_by_prompt = defaultdict(list)
    for e in cats:
        if e["prompt"]:
            rows_by_prompt[e["prompt"]].append(e)
    universe_extra = set()
    if os.path.exists(CENSUS):
        universe_extra |= {r["prompt"] for r in csv.DictReader(open(CENSUS))
                           if r["prompt"]}
    by_prompt = {p: rs[0] for p, rs in rows_by_prompt.items()}
    dup = sum(len(rs) - 1 for rs in rows_by_prompt.values())
    print(f"categorisation    {len(cats)} rows -> {len(by_prompt)} distinct "
          f"({dup} duplicate rows collapsed; scoring is per STRING)")

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

    universe = sorted((set(by_prompt) | universe_extra |
                       {p for v in have.values() for p in v}) - drop_retired)
    print(f"excluded: {len(retired - active)} retired-only + "
          f"{len(EXCLUDE_LITERAL)} BOS/label literals "
          f"(kept {len(retired & active)} retired-but-active-elsewhere)")
    excl = set(a.exclude_finding or [])
    fof = lambda p: (by_prompt.get(p) or {}).get("finding")
    keep = [p for p in universe if fof(p) not in excl]
    dropped = [p for p in universe if fof(p) in excl]

    grid = len(models) * len(keep)
    todo = sum(len(set(keep) - have[m]) for m in models)
    fin = Counter(fof(p) for p in keep)

    print(f"models            {len(models)}")
    print(f"prompt universe   {len(universe)}   (census union + categorised, "
          f"empty string excluded)")
    if excl:
        # THE HOUSE RULE: what a filter removes is counted and named, never
        # silently absent -- a grid that quietly omits a stratum reads as
        # complete coverage of a smaller world.
        dc = Counter(fof(p) for p in dropped)
        print(f"EXCLUDED          {len(dropped)}   {dict(dc)}   "
              f"({len(models)*len(dropped):,} cells not run)")
    print(f"prompts in grid   {len(keep)}   {dict(fin)}")
    print(f"grid cells        {grid:,}")
    print(f"already present   {grid - todo:,}")
    print(f"TO RUN            {todo:,}   ~{todo/4/3600:.1f} h at 4 p/s")

    if a.dry_run:
        print("\nDRY RUN -- no spec written")
        return

    # STAMP THE SOURCE. The categorisation file moved three times in thirty
    # minutes while these rebuilds ran, so two specs built minutes apart differ
    # for reasons no one can reconstruct from the spec alone. A spec that does
    # not name the categorisation it was built from is undatable, which is the
    # live-store lesson applied to a file instead of a stash.
    import subprocess
    cat_sha = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", CATS],
        capture_output=True, text=True).stdout.strip() or "UNCOMMITTED"
    dirty = subprocess.run(["git", "status", "--porcelain", CATS],
                           capture_output=True, text=True).stdout.strip()
    print(f"categorisation: {cat_sha[:12]}{' +UNCOMMITTED CHANGES' if dirty else ''}")

    spec = [{"model": m, "prompts": sorted(set(keep) - have[m])} for m in models]
    spec = [e for e in spec if e["prompts"]]
    # ascending by work, so a cancellation costs the least-finished model
    spec.sort(key=lambda e: len(e["prompts"]))
    json.dump({"_meta": {"categorisation_sha": cat_sha,
                         "categorisation_dirty": bool(dirty),
                         "prompts": len(keep), "models": len(models),
                         "cells_to_run": sum(len(e["prompts"]) for e in spec)},
               "spec": spec}, open(a.out, "w"))
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
