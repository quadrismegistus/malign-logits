#!/usr/bin/env python
"""passage_rebalance.py — hand a finished box some of the long pole's work.

    scripts/passage_rebalance.py --from 1 --to 7 [--apply]

## WHY THIS EXISTS

`build_fleet` balances on CELLS, and its own docstring says so: *"a balanced
plan is balanced in CELLS and not in TIME."* Cards in this fleet differ ~3x in
throughput, so cell-balance produced a 3x spread in finishing time. box0 was
the first instance (Turing, replaced by an A100); box1 is the second, holding
the largest shard on an Ada while boxes 5-7 go idle hours earlier.

**A PAIR IS INDEPENDENT, so this is a manifest split and not a restart of the
science.** Nothing about the frozen population, the arms or the scoring
changes; only which machine does which pair.

## WHAT IT MOVES, AND WHAT IT REFUSES TO MOVE

Only pairs the donor has NOT completed, minus the one it is working on now.
A pair whose `.jsonl` exists and is non-empty on the donor is DONE and stays
done. The donor's in-flight pair is left with the donor, because taking it
would discard up to an hour of generation for nothing.

The donor is then restarted on the pairs it keeps. It loses at most its
current partial pair, which is why `--apply` is a separate flag: run it dry
first and read the split.
"""
import argparse, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MF = os.path.join(ROOT, "data", "passage_manifests")


def ssh(box, cmd, timeout=90):
    env = dict(os.environ, MALIGN_VAST_STATE=".vastai.passage%s.json" % box,
               PATH=os.path.join(ROOT, ".venv/bin") + ":" + os.environ["PATH"])
    p = subprocess.run(["malign", "cloud", "ssh", cmd], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    return p.stdout.strip()


def done_pairs(box):
    """Pairs with a NON-EMPTY output file: finished, and never moved."""
    out = ssh(box, "find /root/out -name '*.jsonl' -size +1k -printf '%f\\n' 2>/dev/null")
    return {l.strip() for l in out.splitlines() if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    src = json.load(open(os.path.join(MF, "box%s.json" % a.src)))
    fin = done_pairs(a.src)
    def fname(p):
        return "y__%s.jsonl" % p["base"].replace("/", "__")
    remaining = [p for p in src["pairs"] if fname(p) not in fin]
    if len(remaining) < 3:
        print("donor has %d pairs left; not worth splitting" % len(remaining))
        return 0
    #: leave the in-flight pair (the first remaining) with the donor
    inflight, movable = remaining[0], remaining[1:]
    half = len(movable) // 2 or 1
    move, keep = movable[-half:], [inflight] + movable[:-half]

    def cells(ps):
        return sum(len(q["cells"]) for p in ps for q in p["prompts"])
    print("donor box%s: %d pairs total, %d finished, %d remaining"
          % (a.src, len(src["pairs"]), len(fin), len(remaining)))
    print("  in-flight (stays): %s" % inflight["base"].split("/")[-1])
    print("  KEEP  %d pairs, %s arm-cells" % (len(keep), format(cells(keep), ",")))
    print("  MOVE  %d pairs, %s arm-cells -> box%s"
          % (len(move), format(cells(move), ","), a.dst))
    for p in move:
        print("        %s" % p["base"].split("/")[-1])
    if not a.apply:
        print("\ndry run. re-run with --apply to write the manifests.")
        return 0
    for tag, ps in ((a.src, keep), (a.dst, move)):
        mf = dict(src)
        mf["pairs"] = ps
        mf["units_per_model"] = cells(ps)
        mf["_rebalanced_from"] = "box%s" % a.src
        json.dump(mf, open(os.path.join(MF, "box%s.json" % tag), "w"))
        print("wrote manifest for box%s (%d pairs)" % (tag, len(ps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
