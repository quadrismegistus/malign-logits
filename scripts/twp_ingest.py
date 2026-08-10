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


def _value_sig(rows):
    """Order-independent signature of a cell's word probabilities.

    **SORTED, because row ORDER is not part of the observation.** Two runs of
    the same battery on the same cell can emit the same words in a different
    sequence; comparing the raw list would then report every multi-source cell
    as disagreeing and bury the 364 that actually do.

    **ROUNDED to 12 places** so float formatting through JSON cannot manufacture
    a difference. That is far tighter than any real disagreement here -- the
    ones measured differ in WORD COUNT (194/197/201 on one cell), not in the
    last bits of a probability.

    Compares only (word, p). `t1` is derived from the word by the tokenizer and
    cannot differ where the word does not.
    """
    return tuple(sorted((r.get("word"), round(float(r.get("p", 0.0)), 12))
                        for r in rows))


def store_versions(cm):
    """{rule_version: count} over the whole store.

    THIS DOCSTRING PREVIOUSLY STATED A PREMISE THE CODEBASE THEN FALSIFIED, and
    the correction is booked as a rule rather than a fix: *the prose was true
    the day it was written and a later commit made it false, and the commit that
    falsifies a stated premise must update it in the same delta* -- not leave it
    to be discovered by the first user a gate wrongly refuses.

    What it said: "the key is {model, prompt, theta, mode}, so the store CANNOT
    REPRESENT two boundary rules; a second version silently loses every
    collision." Every clause of that is now false. `rule_version` and `dict_sha`
    ARE key fields; two rules coexist on one (model, prompt) and both remain
    retrievable; nothing loses a collision.

    So this is no longer a DAMAGE DETECTOR. It reports what the store holds, and
    the guard below is a POLICY guard: one rule per store is what we INTEND, not
    what the key enforces.
    """
    #: reads the KEY when the rule is keyed, the payload when it is not -- the
    #: branch is inside CacheManager and this call is correct either way.
    from malign_logits.cache_schema import schema_for
    if "rule_version" in schema_for("true_word_probs").fields:
        return Counter(rv for rv, _ds in cm._twp_rules())
    return Counter(cm.value_count_by("true_word_probs", "rule_version",
                                     default=1))


def version_gate(store_vs, incoming_vs, allow_second_rule=False):
    """(ok, message). Refuses a write that would leave two rules in the store.

    A POLICY GUARD, not a damage detector -- and the distinction is the whole
    point of the rewrite. It was written when the key could not represent two
    rules, so a mixture was UNRECOVERABLE: `has_true_word_probs()` answered the
    same for both versions and the older cell won every collision. A default
    ingest into a v1 store would have kept 12,855 stale cells and discarded
    their replacements silently.

    THE KEY NOW REPRESENTS THE RULE, so none of that is true: two rules coexist,
    both retrievable, nothing overwritten. The refusal STAYS anyway, because
    one rule per store is what this project INTENDS -- a v1 base compared against
    a v3 superego books an instrument change as alignment movement, and that
    hazard is about ANALYSIS, not about storage. What changed is that a
    deliberate second rule is now a supported state rather than a corruption.
    """
    su, iu = set(store_vs), set(incoming_vs)
    if len(iu) > 1:
        return False, (f"THE INCOMING FILES THEMSELVES HOLD {len(iu)} RULE "
                       f"VERSIONS {dict(incoming_vs)}. Nothing to ingest into: "
                       f"fix the source, do not filter it here.")
    if not su or su == iu:
        return True, ""
    if allow_second_rule:
        return True, (f"--allow-second-rule: writing {iu} into a store holding "
                      f"{su}. The key distinguishes them and both stay "
                      f"retrievable; a read that names NO rule will now refuse "
                      f"as ambiguous rather than pick one silently.")
    return False, (
        f"REFUSING TO ADD A SECOND BOUNDARY RULE.\n"
        f"  store holds    {dict(store_vs)}\n"
        f"  incoming is    {dict(incoming_vs)}\n"
        f"This store is INTENDED to hold one rule. The key can represent both\n"
        f"(rule_version and dict_sha are key fields), so this is a policy\n"
        f"refusal, not a corruption warning -- nothing would be overwritten.\n"
        f"The hazard is ANALYSIS, not storage: v3 changed what a WORD is\n"
        f"(contractions, mojibake, CJK), so a v1 base against a v3 superego\n"
        f"books an instrument change as alignment movement.\n"
        f"Ingest whole after a clear, or pass --allow-second-rule to hold both\n"
        f"deliberately -- reads must then name their rule.")


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
    ok, msg = version_gate(sv, incoming, a.allow_second_rule)
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
            #: SKIP ROWS ARE ROWS AND THEY HAVE NO DISTRIBUTION. `SkipPrompt`
            #: writes {"skipped": reason, "rows": [], "residual": None} so the
            #: shard stays its own ledger -- and this loop then did
            #: `res["total"]` on None and killed the ingest at 246,491 of
            #: 266,049 cells. Written through, counted, and NOT conservation-
            #: checked: a prompt that could not be scored has no mass to
            #: conserve, and asserting on it would fail every one.
            if rec.get("skipped") is not None:
                #: A SKIP CELL IS NOT WRITTEN, and the reason is structural
                #: rather than convenient. Two facts, both from the artifact:
                #:
                #:   the row has no distribution -- rows=[], residual=None
                #:   the row carries rule_version but NOT dict_sha, so the
                #:   rule-keyed schema REFUSES it rather than invent one
                #:
                #: The second is the design working: `set_true_word_probs`
                #: will not guess provenance. Supplying dict_sha from a
                #: sibling row would be exactly that guess.
                #:
                #: So the store holds SCORED cells only, and the SHARD REMAINS
                #: THE LEDGER for what could not be scored -- which is where
                #: the skip was recorded in the first place, by the process
                #: that observed it.
                stats["skipped"] += 1
                continue
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
                #: **"ALREADY INGESTED" AND "A DIFFERENT OBSERVATION WAS
                #: DISCARDED" PRINTED IDENTICALLY, AND I READ THEM AS THE FIRST
                #: WHILE RUNNING EIGHT DIRECTORIES.** 2026-08-10, docket
                #: [5302]/[5303]. This is first-write-wins, so a cell already
                #: resident is SKIPPED -- which is pure idempotence when the two
                #: agree and a silent choice between observations when they do
                #: not. The same cell scored in two boxes carries identical
                #: theta, rule_version and dict_sha and STILL differs (194/197/
                #: 201 words on one), because those are two observations rather
                #: than two versions.
                #:
                #: A message that cannot distinguish two states is not a lapse
                #: in the reader's attention. So compare and count them apart:
                #: `identical` is routine, `DIFFERENT` never is.
                try:
                    cur = cm.get_true_word_probs(model, p, theta=theta) or {}
                    resident[cur.get("rule_version", 1)] += 1
                    if _value_sig(cur.get("rows") or []) == _value_sig(rec["rows"]):
                        stats["skip_same"] += 1
                    else:
                        stats["skip_diff"] += 1
                        loud.append((model, p[:38],
                                     "SKIPPED, DIFFERENT VALUE: resident %d words, "
                                     "this file %d -- first-write-wins discarded THIS one"
                                     % (len(cur.get("rows") or []), len(rec["rows"]))))
                except Exception:
                    resident[None] += 1
                    stats["skip_unknown"] += 1
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
                    # **PROVENANCE, CARRIED RATHER THAN DROPPED (2026-08-07).**
                    # The jsonl has stamped torch and transformers versions all
                    # along and this ingest threw both away, so no cell in the
                    # store could say what computed it. `device` is new at the
                    # producer today; the two versions were always available and
                    # merely not carried -- the same defect merge_fc_jsonl had
                    # for beam_fc, found the same way, by needing the answer and
                    # not having it.
                    #
                    # A cell missing all three predates this change. That is
                    # informative rather than empty: it is grid-v3-era and
                    # believed CUDA. Readers should map absence, NOT backfill
                    # it -- device was never recorded and the raw jsonl is gone,
                    # so a written value would be an assertion, not a record.
                    "device": rec.get("device"),
                    "torch_version": rec.get("torch_version"),
                    "transformers_version": rec.get("transformers_version"),
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
    #: `already` is SPLIT because the two halves are different events: `same` is
    #: idempotence and `DIFF` is first-write-wins silently choosing between two
    #: observations of one cell. Reported per model as well as in total, since a
    #: single model carrying every disagreement is the signature of a duplicated
    #: shard rather than of a noisy store -- which is exactly what the census
    #: turned out to be (364 cells, one model, two boxes).
    print(f"{'model':<{w}}{'write':>8}{'same':>8}{'DIFF':>7}{'dup':>6}"
          f"{'conserve':>10}{'trunc':>7}")
    for model, s in per_model:
        print(f"{model:<{w}}{s['write']:>8,}{s['skip_same']:>8,}"
              f"{s['skip_diff']:>7,}{s['dup']:>6}"
              f"{s['conserve']:>10}{s['truncated']:>7}")
    print(f"\n{'TOTAL':<{w}}{tot['write']:>8,}{tot['skip_same']:>8,}"
          f"{tot['skip_diff']:>7,}{tot['dup']:>6}"
          f"{tot['conserve']:>10}{tot['truncated']:>7}")
    #: **NEVER ROUTINE, SO IT GETS ITS OWN LINE.** Folded into `already` this
    #: was invisible for eight consecutive directory ingests.
    if tot["skip_diff"]:
        print(f"\n!! {tot['skip_diff']:,} CELLS SKIPPED WITH A DIFFERENT VALUE "
              f"ALREADY RESIDENT.")
        print("!! These are NOT idempotent skips. The same (model, prompt) at "
              "the same theta, rule_version and dict_sha")
        print("!! holds different word probabilities in this file and in the "
              "store -- two OBSERVATIONS, not two versions.")
        print("!! First-write-wins kept the resident one, so ingest ORDER "
              "decided which. If that order was a shell")
        print("!! glob rather than a declared rule, re-run in the declared "
              "order with --force. Named above.")
    if tot["skip_unknown"]:
        print(f"!! {tot['skip_unknown']:,} resident cells could not be read "
              f"back for comparison; treated as neither same nor different.")
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
    #: dest is pinned so the flag can be renamed without a half-rename
    #: breaking the read: the previous edit renamed the FLAG and left
    #: `a.force_mix` at the call site, which argparse would have turned into an
    #: AttributeError on the one path nobody exercises.
    ap.add_argument("--allow-second-rule", dest="allow_second_rule",
                    action="store_true",
                    help="deliberately hold a second boundary rule. The key "
                         "distinguishes them and both stay retrievable -- but "
                         "reads that name no rule will then REFUSE as "
                         "ambiguous rather than silently pick one.")
    raise SystemExit(main(ap.parse_args()) or 0)
