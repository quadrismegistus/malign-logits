#!/usr/bin/env python3
"""Run the F38 narratology rubric over a blind passage sample.

Wraps NarratologyTask (scripts/score_passage_narratology.py, the v3 rubric) with
the things an 8k-row API job needs and the ad-hoc runner lacks:

  - RESUMABLE: codes already present in the output file are skipped, so a crash
    or a kill costs at most one chunk.
  - INCREMENTAL WRITES: results are appended per chunk rather than held until
    the end.
  - FAILURE LEDGER: unparseable/failed codes are written to a sidecar
    <out>.failures.txt as a bare code list, per the delivery spec.

Blind protocol: this script sees code/opening/continuation only. It performs no
joins, infers no model identity, and must not be given the key.

Usage:
    uv run python scripts/rate_passages.py \
        data/f38_tierA_sample.csv \
        data/f38_tierA_ratings_v4flash.csv \
        deepseek/deepseek-v4-flash --workers 8
"""
import argparse, csv, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_passage_narratology import NarratologyAnnotation, NarratologyTask, make_input

FIELDS = list(NarratologyAnnotation.model_fields)


def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {r["code"] for r in csv.DictReader(f) if r.get("code")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("model")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=250,
                    help="rows per incremental write (resume granularity)")
    ap.add_argument("--limit", type=int, help="stop after N new rows (smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="bypass the llm.Task cache and re-query the API. Use for "
                         "genuine test-retest measurement; note it overwrites the "
                         "cached value for those items.")
    ap.add_argument("--codes", help="file of codes, one per line: rate only these")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.in_path)))
    if a.codes:
        want = {l.strip() for l in open(a.codes) if l.strip()}
        rows = [r for r in rows if r["code"] in want]
        print(f"restricted to {len(rows)} codes from {a.codes}", flush=True)
    done = load_done(a.out_path)
    todo = [r for r in rows if r["code"] not in done]
    print(f"{len(rows)} passages; {len(done)} already rated; {len(todo)} to do", flush=True)
    if a.limit:
        todo = todo[:a.limit]
    if not todo:
        print("nothing to do"); return

    new_file = not os.path.exists(a.out_path)
    task = NarratologyTask(model=a.model)
    failures, n_ok, t0 = [], 0, time.time()

    for i in range(0, len(todo), a.chunk):
        chunk = todo[i:i + a.chunk]
        results = task.map([make_input(r["opening"], r["continuation"]) for r in chunk],
                           num_workers=a.workers, verbose=False, force=a.force)
        with open(a.out_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["code", "model"] + FIELDS)
                new_file = False
            for r, res in zip(chunk, results):
                if res is None:
                    failures.append(r["code"]); continue
                w.writerow([r["code"], a.model] + [getattr(res, k) for k in FIELDS])
                n_ok += 1
        el = time.time() - t0
        rate = (i + len(chunk)) / el if el else 0
        print(f"  {i+len(chunk)}/{len(todo)}  ok={n_ok}  fail={len(failures)}  "
              f"{rate:.1f}/s  eta {(len(todo)-i-len(chunk))/max(rate,1e-9)/60:.0f}m", flush=True)
        if failures:
            with open(a.out_path + ".failures.txt", "w") as f:
                f.write("\n".join(failures) + "\n")

    print(f"\nDONE {n_ok}/{len(todo)} annotated -> {a.out_path}")
    if failures:
        print(f"{len(failures)} failures -> {a.out_path}.failures.txt "
              f"({100*len(failures)/len(todo):.2f}%)")


if __name__ == "__main__":
    main()
