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


def store_versions(cm):
    """{rule_version: count} over the whole store. Full payload scan, ~1.6s/14k.

    THE VERSION IS IN THE PAYLOAD, NOT THE KEY -- deliberately, per RH's ruling
    of 2026-07-30. The key is {model, prompt, theta, mode}, so the store CANNOT
    REPRESENT two boundary rules: a second version silently loses every
    collision to whatever is already there. Keying the version would make
    mixture safe to hold, and nothing here wants to hold it. The store holds
    ONE rule and a mixture is an error, so this exists to find one rather than
    to tolerate it.
    """
    # rule_version lives in the VALUE, not the key -- which is precisely why
    # two boundary rules can collide here and why this gate exists at all.
    return Counter(cm.value_count_by("true_word_probs", "rule_version",
                                     default=1))


def version_gate(store_vs, incoming_vs, force_mix=False):
    """(ok, message). Refuses a write that would leave two rules in the store.

    ABORT BEFORE WRITING, not a warning after. The previous form reported the
    mixture once it existed, which is a description of damage rather than a
    guard against it -- and the damage is unrecoverable by inspection, because
    `has_true_word_probs()` answers the same for both versions and the OLDER
    cell wins every collision. A default ingest into a v1 store would have kept
    12,855 stale cells and discarded their replacements silently.
    """
    su, iu = set(store_vs), set(incoming_vs)
    if len(iu) > 1:
        return False, (f"THE INCOMING FILES THEMSELVES HOLD {len(iu)} RULE "
                       f"VERSIONS {dict(incoming_vs)}. Nothing to ingest into: "
                       f"fix the source, do not filter it here.")
    if not su or su == iu:
        return True, ""
    if force_mix:
        return True, (f"--force-mix: writing {iu} into a store holding {su}. "
                      f"THE STORE IS NOW MIXED AND THE KEY CANNOT TELL THEM "
                      f"APART.")
    return False, (
        f"REFUSING TO MIX BOUNDARY RULES.\n"
        f"  store holds    {dict(store_vs)}\n"
        f"  incoming is    {dict(incoming_vs)}\n"
        f"The key is {{model, prompt, theta, mode}} -- rule_version is NOT in it,\n"
        f"so these cannot coexist and the older cell wins every collision.\n"
        f"v3 changed what a WORD is (contractions, mojibake, CJK), so a v1 base\n"
        f"against a v3 superego books an instrument change as alignment movement.\n"
        f"Wipe data/raw/cache/true_word_probs and ingest whole, or --force-mix\n"
        f"if you genuinely intend a store nothing can disentangle.")


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

    # ---- THE GATE, BEFORE A SINGLE WRITE ----
    incoming = Counter()
    for path in files:
        for n, rec in records(path):
            if "__truncated__" not in rec:
                incoming[rec.get("rule_version", 1)] += 1
    sv = store_versions(cm)
    ok, msg = version_gate(sv, incoming, a.force_mix)
    print(f"store  {dict(sv) or 'EMPTY'}\nfiles  {dict(incoming)}")
    if not ok:
        print(f"\n{msg}")
        return 1
    if msg:
        print(f"\n!! {msg}")

    tot = Counter()
    resident = Counter()   # rule_version of cells ALREADY in the store
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
                # READ the resident cell's version; do not assume it. See below.
                try:
                    resident[cm.get_true_word_probs(model, p, theta=theta)
                             .get("rule_version", 1)] += 1
                except Exception:
                    resident[None] += 1
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

    # RESIDENT VERSIONS ARE READ, NOT ASSUMED. This block used to count every
    # already-present cell as v1 -- "countable without reading the stash" --
    # because when it was written the store predated the rule_version field. That
    # assumption expired: the store is now uniformly v3, and the shortcut
    # manufactured a two-version warning on a single-version store, reporting
    # {1: 144, 3: 1795} where the truth was {3: 1939}.
    #
    # A FALSE ALARM ON A GATE IS WORSE THAN NO GATE: it either blocks a correct
    # ingest or teaches the operator to click past the one warning that matters.
    # The cost of reading is one stash hit per already-present cell, paid only on
    # re-ingest.
    present = dict(mix)
    for rv, n in resident.items():
        present[rv] = present.get(rv, 0) + n
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
        return 0

    # ---- VERIFY THE INVARIANT HELD, rather than assume the gate did its job ----
    after = store_versions(cm)
    if len(after) > 1:
        print(f"\n!! POST-WRITE CHECK FAILED: store now holds {dict(after)}. "
              f"The gate passed and the invariant broke anyway -- INVESTIGATE "
              f"before anything reads this store.")
        return 1
    print(f"\nstore is single-version: {dict(after) or 'EMPTY'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rewrite keys already present")
    ap.add_argument("--force-mix", action="store_true",
                    help="write a version the store does not already hold. "
                         "Leaves a store nothing can disentangle; the key has "
                         "no rule_version field.")
    raise SystemExit(main(ap.parse_args()) or 0)
