#!/usr/bin/env python3
"""BLT fleet sweep. Constants pinned; live is never a floor on what exists.

    process  blt_cloud.py      outdir /workspace/blt      log /workspace/blt.log
    route    per-box host/port in data/blt_fleet/instances.json (shard 1 is
             DIRECT-only: its proxy refuses publickey, runbook 2.26)

TOTAL is 483,085 deduped passages across 4 shards.
"""
import json, os, subprocess, sys, concurrent.futures as cf

TOTAL = 483_085
BOXES = "data/blt_fleet/instances.json"
LOCAL = "data/raw/blt_fleet"
#: `pgrep -c` PRINTS 0 AND EXITS NON-ZERO when nothing matches, so
#: `pgrep -c ... || echo 0` emits "00" -- and an alive-test of
#: `proc not in ('0','')` then reads a DEAD runner as producing.
#: Caught on shard 2 at 9.8%. Count without the fallback and parse as int.
CMD = ("echo -n 'proc='; pgrep -fc 'blt_clou[d]'; "
       "echo -n '|rows='; cat /workspace/blt/blt_shard*.jsonl 2>/dev/null | wc -l; "
       "echo -n '|f32='; du -sm /workspace/blt 2>/dev/null | cut -f1; "
       "echo -n '|free='; df -BG /workspace | tail -1 | awk '{print $4}'; "
       "echo -n '|rate='; grep -oE '[0-9.]+/s' /workspace/blt.log 2>/dev/null | tail -1")

def probe(b):
    try:
        r = subprocess.run(['ssh','-n','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=no',
            '-o','BatchMode=yes','-p',str(b['port']),'root@'+b['host'],CMD],
            capture_output=True, text=True, timeout=70, stdin=subprocess.DEVNULL)
        if r.returncode != 0 and 'rows=' not in r.stdout:
            return (b, None)
        d = {k: v.strip() for k, v in
             (x.split('=',1) for x in r.stdout.replace('\n','').split('|') if '=' in x)}
        return (b, d)
    except Exception:
        return (b, None)

def main():
    #: A DESTROYED BOX IS NOT PROBED BUT ITS WORK STILL EXISTS. Filtering
    #: them out of the roster fixed the burn ($1.67/hr for two live boxes)
    #: and immediately broke the thing that matters more: SCORED fell from
    #: 476,644 to 235,983, because the tool started counting what its probe
    #: could REACH instead of what exists. That is the verse fleet defect
    #: exactly, reintroduced while fixing something else.
    #:
    #: So: probe the live boxes, and count the destroyed ones FROM DISK.
    #: Their rows are pulled, sha256-verified against the box before it was
    #: torn down, and are not going to change.
    _all = json.load(open(BOXES))
    boxes = [b for b in _all if not b.get("destroyed")]
    retired = 0
    for b in _all:
        if not b.get("destroyed"):
            continue
        d = os.path.join("data/raw/blt_fleet", str(b["id"]))
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith(".jsonl") and not f.endswith(".skipped.jsonl"):
                with open(os.path.join(d, f), "rb") as fh:
                    retired += sum(1 for _ in fh)
    with cf.ThreadPoolExecutor(len(boxes)) as ex:
        res = list(ex.map(probe, boxes))
    #: SEED THE TOTAL WITH THE RETIRED ROWS counted from disk above, so the
    #: figure is what EXISTS, not what the probe reached. Monotone by
    #: construction: a box leaving the fleet moves its rows from the live
    #: term to the retired term and the sum is unchanged.
    tot = retired; burn = 0; live = 0; att = []
    print("  %-6s %-10s %-6s %-9s %-7s %-6s %s" % ("shard","id","proc","rows","MB","free","rate"))
    for b, d in sorted(res, key=lambda x: x[0]['shard']):
        burn += b.get('dph') or 0
        if d is None:
            #: an unreachable box does not zero its contribution -- fall back to
            #: what we hold locally for it (the six-defect family, [5885])
            lp = os.path.join(LOCAL, str(b['id']))
            n = 0
            if os.path.isdir(lp):
                for f in os.listdir(lp):
                    if f.endswith('.jsonl'):
                        with open(os.path.join(lp,f),'rb') as fh: n += sum(1 for _ in fh)
            tot += n
            print("  %-6d %-10s UNREACHABLE -- not grounds to destroy; %s rows held locally"
                  % (b['shard'], b['id'], f"{n:,}"))
            att.append((b['id'],'unreachable')); continue
        n = int(d.get('rows','0') or 0); tot += n
        try: alive = int(d.get('proc','0').strip() or 0) > 0
        except ValueError: alive = False
        if alive and n > 0: live += 1
        else: att.append((b['id'], 'proc=%s rows=%d' % (d.get('proc'), n)))
        print("  %-6d %-10s %-6s %-9s %-7s %-6s %s" % (
            b['shard'], b['id'], d.get('proc'), f"{n:,}", d.get('f32'), d.get('free'), d.get('rate','?')))
    print("\n  SCORED %s / %s  (%.1f%%)   producing %d/%d   burn $%.2f/hr"
          % (f"{tot:,}", f"{TOTAL:,}", 100*tot/TOTAL, live, len(res), burn))
    rates = []
    for _, d in res:
        if d and d.get('rate'):
            try: rates.append(float(d['rate'].rstrip('/s')))
            except ValueError: pass
    if rates and tot < TOTAL:
        agg = sum(rates)
        h = (TOTAL - tot)/agg/3600
        print("  aggregate %.1f/s -> ~%.1f h remaining, ~$%.2f more" % (agg, h, h*burn))
    if att: print("  ATTENTION: %s" % att)
    return 0

if __name__ == '__main__':
    sys.exit(main())
