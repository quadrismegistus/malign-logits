"""Merge cloud JSONL into the canonical `true_word_probs` stash.

    uv run .venv/bin/python scripts/twp_ingest.py --dry-run
    uv run .venv/bin/python scripts/twp_ingest.py
    uv run .venv/bin/python scripts/twp_ingest.py --src data/twp_cloud --force

THE CLOUD FILES ARE A TRANSPORT FORMAT, NOT THE STORE. `twp_cloud.py` writes
one flushed JSON line per prompt because that is what survives a kill and what
rsync can pull incrementally. The canonical store is HashStash through
CacheManager, where the pinned open is enforced and `theta` is in the key. This
script is the one-way bridge, and it is idempotent: re-running after another
rsync ingests only what is new.

IT VALIDATES BEFORE IT WRITES, which is the whole point of a separate step.
Every line is checked against the invariant the algorithm exists to satisfy --
`sum(P(words)) + residual == 1.0` -- and a line that fails is COUNTED AND
SKIPPED, never written. A defective record in a transport file is an accident;
a defective record in the canonical store is a result nobody can trust
afterwards, because at that point the file it came from is gone.

The rejection classes are reported separately rather than as one "bad lines"
count, because they mean different things:

  truncated   the last line of a file being written right now. EXPECTED and
              harmless -- the next sync completes it. Not a defect.
  conserve    sum + residual is off by more than TOL. A REAL DEFECT: it means
              mass went missing in expansion. The two such defects found during
              the build (unrecorded depth-1 tail, whitespace-only prefixes
              terminating into no word) were both invisible in the word lists
              and visible only here.
  dup         the same (model, prompt) twice in one file. Possible after a
              resume: `done_prompts` reads back completed lines, so a line
              truncated mid-write is re-done and the partial one is skipped by
              the JSON parse -- but a line that was COMPLETE when re-done would
              appear twice. Last one wins; the count is printed because a
              nonzero value means the resume logic let something through.

`open` RESIDUAL IS SURFACED, NOT REJECTED. It is mass still unterminated at
MAX_DEPTH -- the defect channel and the free CJK detector, since Chinese has no
whitespace so its mass lands there. A high `open` is a finding about that model
and prompt, not a reason to drop the row, so it is reported as a distribution
and the worst offenders are named.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402
from malign_logits.cache import get_cache  # noqa: E402

SRC = os.path.join(PATH_DATA, "twp_cloud")
TOL = 1e-4          # conservation is exact to ~2e-07 in practice; 1e-4 is loose
OPEN_LOUD = 0.01    # an `open` residual above this is worth naming


def records(path):
    """Yield (lineno, obj) for parseable lines; a truncated tail is reported."""
    trunc = 0
    with open(path) as f:
        for n, ln in enumerate(f, 1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield n, json.loads(ln)
            except json.JSONDecodeError:
                trunc += 1      # only ever the final line of a live file
    if trunc:
        yield -1, {"__truncated__": trunc}


def main(a):
    files = sorted(glob.glob(os.path.join(a.src, "*.jsonl")))
    if not files:
        print(f"no jsonl under {a.src}")
        return
    cm = get_cache()
    tot = Counter()
    mix = Counter()
    loud, per_model = [], []

    for path in files:
        model = os.path.basename(path)[:-6].replace("__", "/")
        seen, stats = {}, Counter()
        for n, rec in records(path):
            if "__truncated__" in rec:
                stats["truncated"] += rec["__truncated__"]
                continue
            if rec.get("model") != model:
                # filename and payload must agree or the key is a lie
                stats["model_mismatch"] += 1
                continue
            p = rec["prompt"]
            if p in seen:
                stats["dup"] += 1
            seen[p] = rec

        for p, rec in seen.items():
            res = rec["residual"]
            got = sum(r["p"] for r in rec["rows"]) + res["total"]
            if abs(got - 1.0) > TOL:
                stats["conserve"] += 1
                loud.append((model, p[:38], f"conservation {got:.6f}"))
                continue
            if res.get("open", 0.0) > OPEN_LOUD:
                loud.append((model, p[:38], f"open {res['open']:.4f}"))
            theta = rec.get("theta", 0.001)
            if not a.force and cm.has_true_word_probs(model, p, theta=theta):
                stats["already"] += 1
                continue
            stats["write"] += 1
            mix[rec.get("rule_version", 1)] += 1
            if not a.dry_run:
                cm.set_true_word_probs(model, p, {
                    "rows": rec["rows"], "residual": res,
                    "batches": rec.get("batches"),
                    # CARRIED INTO THE STORE, not left in the transport file.
                    # The boundary rule is not in the cache key, so without this
                    # a partial re-run leaves the stash holding two rules with
                    # nothing to tell them apart. v1 (absent field) is run 1.
                    "rule_version": rec.get("rule_version", 1),
                    "rule_commits": rec.get("rule_commits"),
                    "dict_sha": rec.get("dict_sha"),
                }, theta=theta)

        per_model.append((model, stats))
        tot.update(stats)

    # CELLS ALREADY IN THE STORE PREDATE THIS FIELD, SO THEY ARE v1 BY
    # DEFINITION -- countable without reading the stash. Counting only what THIS
    # pass writes would leave the warning silent in exactly the case it exists
    # for: a store that already holds v1 and is now receiving v2.
    present = dict(mix)
    if tot["already"]:
        present[1] = present.get(1, 0) + tot["already"]
    if len(present) > 1:
        print(f"!! STORE HOLDS {len(present)} BOUNDARY-RULE VERSIONS: "
              f"{dict(sorted(present.items()))}")
        print("!! v1 predates the CJK fixes (ASCII punctuation only). Chinese "
              "cells resolve 3-16% of mass there against 80-90% for English, "
              "and English-prompt cells can contain glued cross-script units.")
        print("!! DO NOT COMPARE v1 AND v2 ON ANY CJK OR MIXED-SCRIPT CELL.\n")
    elif present:
        print(f"boundary rule: all cells v{list(present)[0]}\n")
    w = max(len(m) for m, _ in per_model)
    print(f"{'model':<{w}}{'write':>8}{'already':>9}{'dup':>6}"
          f"{'conserve':>10}{'trunc':>7}")
    for model, s in per_model:
        print(f"{model:<{w}}{s['write']:>8,}{s['already']:>9,}{s['dup']:>6}"
              f"{s['conserve']:>10}{s['truncated']:>7}")
    print(f"\n{'TOTAL':<{w}}{tot['write']:>8,}{tot['already']:>9,}{tot['dup']:>6}"
          f"{tot['conserve']:>10}{tot['truncated']:>7}")
    if tot["model_mismatch"]:
        print(f"MODEL MISMATCH {tot['model_mismatch']} -- filename disagrees "
              f"with payload; keys would be wrong. INVESTIGATE.")

    if loud:
        print(f"\nFLAGGED {len(loud)} (conservation failures skipped; "
              f"open>{OPEN_LOUD} written and named):")
        for m, p, why in loud[:20]:
            print(f"  {m:<38}{p:<40}{why}")
        if len(loud) > 20:
            print(f"  ... {len(loud)-20} more")
    else:
        print(f"\nno row failed conservation and none carried open>{OPEN_LOUD}")

    if a.dry_run:
        print("\nDRY RUN -- nothing written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rewrite keys already present")
    main(ap.parse_args())
