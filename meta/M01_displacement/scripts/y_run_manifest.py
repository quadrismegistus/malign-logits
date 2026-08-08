#!/usr/bin/env python
"""Annotate the frozen manifest. Resumable, hash-verified, append-only.

    python y_run_manifest.py --limit 50      # final smoke against the real path
    python y_run_manifest.py                 # the run
    python y_run_manifest.py --status        # what is done, spend nothing

THREE PROPERTIES, EACH BECAUSE OF A SPECIFIC WAY THIS GOES WRONG.

RESUMABLE. 62,681 items is hours. A job that dies at 80% and cannot resume
either loses the spend or gets restarted with a "skip the first N" flag written
under time pressure, which is how a run ends up with a gap nobody can name.
Output is append-only JSONL keyed by the manifest id; on restart, ids already
present are skipped. Killing and relaunching is the supported path, not a
recovery procedure.

HASH-VERIFIED. Every manifest row carries a sha256 of its passage. The runner
recomputes it from the corpus and REFUSES on mismatch. A manifest is only a pin
if something checks it; otherwise it is a list of coordinates that used to mean
something.

FIDELITY RECORDED, NEVER RAISED. `roundtrip()` runs on every row. Measured on
200 gate items: 35% exact, 50% whitespace-only, 13% under 1% drift, 0.5%
severe. The task instruction says "CHANGE NO CHARACTER" and the validator
deliberately does not enforce it -- so the only way the 0.5% is ever visible is
if the band is written down at annotation time. It parses, it carries every
field, and the tag rates computed on it are internally consistent; only a span
mapped back to the source would ever reveal it.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

MANIFEST = os.path.join(CAMP, "registrations", "y_annotation_manifest.jsonl")
OUT = os.path.join(CAMP, "results", "y_confirmatory_coded.jsonl")


def read_manifest():
    head, rows = None, []
    for line in open(MANIFEST):
        d = json.loads(line)
        if d.get("_manifest"):
            head = d
        else:
            rows.append(d)
    return head, rows


def key(pair, role, prompt_id, word, seq_i):
    """THE CELL COORDINATE, AND `role` IS PART OF IT.

    The first version of this key omitted role. A pair holds a base record AND
    an aligned record at the same (prompt_id, word, seq_i), so whichever was
    read second overwrote the first and half the manifest resolved to the wrong
    arm's passage. Caught on the first invocation by the sha256 check, which
    refused 20 of 40 rows. Without the hash it would have annotated base
    passages under aligned labels and produced a clean, plausible, entirely
    inverted arm contrast -- with nothing anywhere in the output looking wrong.
    """
    return (pair, role, prompt_id, word, seq_i)


def load_texts(need):
    """Passage text for every cell coordinate the manifest names."""
    files = [f for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*", "*.jsonl")))
             if "FAILED" not in f]
    got = {}
    for f in files:
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "sequences" not in r:
                continue
            for i, s in enumerate(r["sequences"]):
                k = key(r.get("pair"), r.get("role"), r.get("prompt_id"), r.get("word"), i)
                if k in need:
                    got[k] = s.get("text") or ""
    return got


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32,
                    help="parallel threads. library default is 4; 32 measured safe "
                         "against deepseek. plain ThreadPoolExecutor, no rate limiter.")
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)

    head, man = read_manifest()
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)["mid"])
            except Exception:
                pass
    todo = [r for r in man if r["mid"] not in done]
    print("manifest %s  rows %d  sha256 %s" % (os.path.basename(MANIFEST), len(man),
                                               (head or {}).get("sha256", "?")[:16]))
    print("already coded %d   remaining %d" % (len(done), len(todo)))
    if a.status:
        if done:
            band = collections.Counter()
            for line in open(OUT):
                try:
                    band[json.loads(line).get("rt_band")] += 1
                except Exception:
                    pass
            print("  round-trip so far: %s" % dict(band.most_common()))
        return 0
    if not todo:
        print("nothing to do."); return 0
    if a.limit:
        todo = todo[:a.limit]
        print("LIMIT: %d items this invocation" % len(todo))

    need = {key(r["pair"], r["role"], r["prompt_id"], r["word"], r["seq_i"]) for r in todo}
    texts = load_texts(need)
    missing = [r["mid"] for r in todo
               if key(r["pair"], r["role"], r["prompt_id"], r["word"], r["seq_i"]) not in texts]
    if missing:
        print("REFUSED: %d manifest rows have no passage in the corpus (%s ...)"
              % (len(missing), ", ".join(missing[:4])))
        return 1
    #: THE PIN, CHECKED. A manifest hash nothing verifies is decoration.
    bad = []
    for r in todo:
        t = texts[key(r["pair"], r["role"], r["prompt_id"], r["word"], r["seq_i"])]
        if hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] != r["sha256"]:
            bad.append(r["mid"])
    if bad:
        print("REFUSED: %d passages do not match their manifest hash (%s ...)"
              % (len(bad), ", ".join(bad[:4])))
        print("The corpus changed under the manifest. Rebuild deliberately or fix the corpus.")
        return 1
    print("hash check: %d/%d passages match the manifest" % (len(todo), len(todo)))

    from malign_logits.tasks.code_y_superego_v3 import (
        prepare, roundtrip, SuperegoV3Task, COMPOSITES)
    PR = json.load(open(os.path.join(CAMP, "registrations", "registration_y_slots.json")))
    stem = {p["prompt_id"]: (p.get("text") or p.get("prompt") or "") for p in PR["prompts"]}

    srcs = [texts[key(r["pair"], r["role"], r["prompt_id"], r["word"], r["seq_i"])]
            for r in todo]
    items = [prepare(stem.get(r["prompt_id"], ""), r["word"], s) for r, s in zip(todo, srcs)]

    task = SuperegoV3Task()
    errors = {}
    n_ok = 0

    #: CHUNKED, AND THE FIRST VERSION OF THIS WAS WRONG. It called map() once
    #: over all 62,641 items and wrote afterwards, so results lived in memory
    #: until the very end -- a kill at 90% wrote NOTHING and the resume found
    #: the same work outstanding. The docstring above claimed "killing and
    #: relaunching is the supported path" while the code made it a way to lose
    #: two hours. Caught by killing it at 4 minutes and reading the row count.
    #:
    #: A kill now costs at most one chunk of MODEL calls, and even those are
    #: usually free on restart: the library caches by (prompt, model, params),
    #: so re-running an item computed before the kill is a cache hit, not a
    #: second charge. The chunk size trades write frequency against the
    #: progress bar restarting; 2000 is ~4 minutes at the observed 8 it/s.
    CHUNK = 2000
    for start in range(0, len(todo), CHUNK):
        part = todo[start:start + CHUNK]
        psrc = srcs[start:start + CHUNK]
        pit = items[start:start + CHUNK]
        res = task.map(pit, model=a.model, num_workers=a.workers, errors=errors)
        #: zip() TRUNCATES TO THE SHORTEST, SILENTLY. If map ever returned fewer
        #: results than items -- a dedup, a partial failure path, a future
        #: change -- the zip below would write a short chunk and the missing
        #: mids would sit outstanding with nothing anywhere saying so. The
        #: resume would eventually re-run them, so this self-heals and is
        #: therefore exactly the kind of defect that never gets noticed.
        #:
        #: malign [4990].3 hit the same shape from the other side: a guard that
        #: dropped beams arm-asymmetrically, and a canary that zipped the two
        #: arrays and so could not see what the guard had done. Asserting the
        #: length is the cheap half of that lesson.
        if len(res) != len(pit):
            raise RuntimeError(
                "map returned %d results for %d items at chunk %d. zip would "
                "truncate and write a short chunk silently."
                % (len(res), len(pit), start))
        with open(OUT, "a", encoding="utf-8") as fh:
            for r, src, out in zip(part, psrc, res):
                row = dict(r)
                row["coder"] = a.model
                row["parsed"] = out is not None
                if out is not None:
                    n_ok += 1
                    row.update(json.loads(out.model_dump_json()))
                    #: ALSO THE ALIGNMENT CHECK. roundtrip compares this row's
                    #: `tagged` against the source it was zipped with, so a
                    #: misaligned zip would read SEVERE on nearly every row
                    #: rather than the observed whitespace/exact split.
                    row.update(roundtrip(src, out.tagged or ""))
                    row["tag_field_mismatches"] = out.tag_field_mismatches()
                    for name, fn in COMPOSITES.items():
                        row[name] = bool(fn(row))
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print("[chunk %d-%d] written. parsed %d/%d so far, errors %d"
              % (start, start + len(part), n_ok, start + len(part), len(errors)))

    print("\nparsed %d/%d   errors %d" % (n_ok, len(todo), len(errors)))
    try:
        print("usage: %s" % task.usage.summary_line())
    except Exception:
        pass
    band = collections.Counter()
    for line in open(OUT):
        try:
            band[json.loads(line).get("rt_band")] += 1
        except Exception:
            pass
    print("round-trip bands over everything written so far: %s" % dict(band.most_common()))
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
