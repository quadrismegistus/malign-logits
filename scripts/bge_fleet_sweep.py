#!/usr/bin/env python3
"""bge fleet sweep. Constants pinned; live is never a floor on what exists.

    process  bge_cloud.py      outdir /workspace/bge      log /workspace/bge.log
    route    per-box host/port in data/bge_fleet/instances.json

THE TOTAL IS NOT THE BLT TOTAL, and reusing it would misreport progress for the
whole run. The BLT pass scored all 483,085 passages. bge runs under
`--mixed-policy refuse` (lacan [5955]), which excludes the 32,103 `mixed`
passages, so the denominator is:

    en 372,103 + zh 78,879 = 450,982

Against 483,085 that reads 93.4%, so a shard that has embedded everything it
will ever embed would sit at 93.4% forever and look stalled at the end -- the
failure mode where the tool, not the fleet, is wrong.

A SEPARATE FILE from blt_fleet_sweep.py on purpose. The BLT fleet is live while
this is written, and the alternative -- a --job flag on the shared tool -- both
edits a running monitor and reintroduces the free-form constant that
`probe constants from ps` was learned by getting wrong four ways.

REFUSALS ARE NOT PROGRESS AND NOT FAILURE. Under `refuse` a mixed passage is
written to `.refused.jsonl` and never embedded. It is counted and displayed
separately: folding it into `rows` would report the run complete before it is,
and treating it as an error would report a policy as a fault.
"""
import json, subprocess, sys, concurrent.futures as cf

#: en 372,103 + zh 78,879, measured on data/raw/blt_passages.jsonl.gz.
TOTAL = 450_982
BOXES = "data/bge_fleet/instances.json"

CMD = ("echo -n 'proc='; pgrep -fc 'bge_clou[d]'; "
       "echo -n '|rows='; cat /workspace/bge/bge_shard*.jsonl 2>/dev/null "
       "| grep -c n_sentences; "
       "echo -n '|ref='; cat /workspace/bge/*.refused.jsonl 2>/dev/null | wc -l; "
       "echo -n '|mb='; du -sm /workspace/bge 2>/dev/null | cut -f1; "
       "echo -n '|free='; df -BG /workspace | tail -1 | awk '{print $4}'; "
       "echo -n '|rate='; grep -oE '[0-9.]+/s' /workspace/bge.log 2>/dev/null | tail -1")


def probe(b):
    cmd = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
           "-o", "StrictHostKeyChecking=no", "-p", str(b["port"]),
           "root@%s" % b["host"], CMD]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:
        return b, None
    if r.returncode != 0:
        return b, None
    d = {}
    for part in r.stdout.strip().split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v.strip()
    return b, d


def main():
    try:
        boxes = json.load(open(BOXES))
    except FileNotFoundError:
        print("  no %s yet -- bge fleet not launched" % BOXES)
        return 0
    with cf.ThreadPoolExecutor(max(len(boxes), 1)) as ex:
        res = list(ex.map(probe, boxes))
    print("  %-6s %-10s %-6s %-9s %-8s %-7s %-6s %s"
          % ("shard", "id", "proc", "rows", "refused", "MB", "free", "rate"))
    tot = ref = live = 0
    rates = []
    for b, d in res:
        if d is None:
            print("  %-6s %-10s ALL ROUTES FAILED -- check cur_state before "
                  "concluding anything" % (b["shard"], b["id"]))
            continue
        n = int(d.get("rows") or 0)
        rf = int(d.get("ref") or 0)
        tot += n; ref += rf
        if (d.get("proc") or "0") != "0":
            live += 1
        if d.get("rate"):
            try: rates.append(float(d["rate"].rstrip("/s")))
            except ValueError: pass
        print("  %-6s %-10s %-6s %-9s %-8s %-7s %-6s %s"
              % (b["shard"], b["id"], d.get("proc"), f"{n:,}", f"{rf:,}",
                 d.get("mb"), d.get("free"), d.get("rate", "?")))
    burn = sum(b.get("dph", 0) for b in boxes)
    print("\n  EMBEDDED %s / %s  (%.1f%%)   refused %s   producing %d/%d   burn $%.2f/hr"
          % (f"{tot:,}", f"{TOTAL:,}", 100 * tot / TOTAL, f"{ref:,}", live, len(res), burn))
    if rates and tot < TOTAL:
        agg = sum(rates)
        h = (TOTAL - tot) / agg / 3600
        #: AGGREGATE h IS A LOWER BOUND ON WALL CLOCK, not an estimate of it.
        #: Work is not poolable across boxes: a shard's remainder can only be
        #: done by its own box, so completion is max-over-shards, not
        #: total/aggregate-rate. The BLT sweep's equivalent line understated by
        #: 39% because the fast pair finished an hour before the slow pair.
        print("  aggregate %.1f/s -> >=%.1f h remaining (LOWER BOUND; per-shard "
              "max is the real one), ~$%.2f more" % (agg, h, h * burn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
