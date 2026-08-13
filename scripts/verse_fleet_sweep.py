#!/usr/bin/env python3
"""Verse fleet sweep. Reads LIVE vast state, never a local record of it.

Four constants this script exists to pin, each of which was wrong in an
ad-hoc probe on 13 Aug and each of which reported a HEALTHY fleet as dead:

    process   twp_cloud.py          (there is no 'verse_batch')
    outdir    /workspace/verse      (the --out flag; NOT the script default /workspace/twp)
    log       /workspace/verse.log
    route     proxy ssh_host:ssh_port, FALLING BACK to public_ipaddr:direct_port_start

The route fallback is the important one: on 13 Aug box 47628582 refused the
proxy with 'Permission denied (publickey)' while producing normally over the
direct IP.  An unreachable box and a dead box look identical from one route.
NEVER destroy on a single route's silence.
"""
import json, os, subprocess, sys, concurrent.futures as cf

DEST_LOCAL = 'data/raw/verse_fleet'
TOTAL_EXPECTED = 446_500          # 250 rungs x 1,786 distinct contexts. NOT 455,000.
PROBE = ("echo -n 'proc=';  pgrep -fc 'twp_clou[d]' || echo 0; "
         "echo -n 'files='; ls /workspace/verse/*.jsonl 2>/dev/null | wc -l; "
         "echo -n 'cells='; cat /workspace/verse/*.jsonl 2>/dev/null | wc -l; "
         "echo -n 'free=';  df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}'; "
         "echo -n 'prog=';  grep -oE '^\\[[0-9]+/[0-9]+\\]' /workspace/verse.log 2>/dev/null | tail -1; echo; "
         "echo -n 'rate=';  grep -oE '[0-9.]+ p/s' /workspace/verse.log 2>/dev/null | tail -1")

def live_instances():
    r = subprocess.run(['vastai','show','instances','--raw'],
                       capture_output=True, text=True, timeout=90)
    return json.loads(r.stdout)

def _ssh(host, port, cmd, timeout=80):
    return subprocess.run(
        ['ssh','-n','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=no',
         '-o','BatchMode=yes','-p',str(port),'root@'+host,cmd],
        capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)

def probe(i):
    """Try proxy, then direct. Report WHICH route answered."""
    routes = []
    if i.get('ssh_host'):       routes.append(('proxy',  i['ssh_host'], i['ssh_port']))
    if i.get('public_ipaddr'):  routes.append(('direct', i['public_ipaddr'].strip(),
                                               i.get('direct_port_start')))
    for name, h, p in routes:
        if not p: continue
        try:
            r = _ssh(h, p, PROBE)
            if r.returncode == 0 or 'cells=' in r.stdout:
                d = dict(x.split('=',1) for x in r.stdout.strip().split('\n') if '=' in x)
                d['_route'] = name
                return (i['id'], d, i.get('dph_total') or 0.0, None)
        except Exception:
            continue
    return (i['id'], None, i.get('dph_total') or 0.0, 'ALL ROUTES FAILED')

def main():
    inst = live_instances()
    with cf.ThreadPoolExecutor(max(1, len(inst))) as ex:
        res = list(ex.map(probe, inst))
    tot = burn = 0.0; producing = 0; attention = []
    print("  %-10s %-8s %-6s %-9s %-6s %-10s %s" %
          ("id","prog","files","cells","free","rate","route"))
    for bid, d, dph, err in sorted(res, key=lambda x: str(x[0])):
        #: only RUNNING boxes burn GPU. A stopped box still appears in
        #: `vastai show instances` with its full dph_total, which overstated
        #: burn by $0.70/hr the moment vast reclaimed one.
        if d is not None: burn += dph
        if d is None:
            #: AN UNREACHABLE BOX MUST NOT SILENTLY REDUCE THE TOTAL. It did once:
            #: 237,417 -> 208,215 when one box dropped out, and a monotone counter
            #: going DOWN is the impossibility that revealed it. Fall back to what
            #: we hold LOCALLY for that box, which is a floor on what it wrote.
            local = os.path.join(DEST_LOCAL, str(bid))
            lc = 0
            if os.path.isdir(local):
                for f in os.listdir(local):
                    if f.endswith('.jsonl'):
                        with open(os.path.join(local, f), 'rb') as fh:
                            lc += sum(1 for _ in fh)
            tot += lc
            print("  %-10s ALL ROUTES FAILED  -- do NOT destroy on this alone; "
                  "counting %s cells held locally" % (bid, f"{lc:,}"))
            attention.append((bid,'unreachable')); continue
        #: TAKE THE MAX OF LIVE AND LOCAL. A box that has been restarted by vast
        #: comes back with an EMPTY disk but answers ssh, so the live reading is 0
        #: and it silently replaces the 30,079 cells we hold for it -- ROWS WRITTEN
        #: fell 446,097 -> 417,105 the moment that happened. Live is authoritative
        #: for PROGRESS, never for a floor on what exists.
        c = int(d.get('cells','0') or 0)
        _lp = os.path.join(DEST_LOCAL, str(bid))
        if os.path.isdir(_lp):
            _lc = 0
            for _f in os.listdir(_lp):
                if _f.endswith('.jsonl'):
                    with open(os.path.join(_lp, _f), 'rb') as _fh:
                        _lc += sum(1 for _ in _fh)
            if _lc > c:
                print("  %-10s live reads %s cells but %s are held locally; using local"
                      % (bid, f"{c:,}", f"{_lc:,}"))
                c = _lc
        tot += c
        alive = d.get('proc','0').strip() not in ('0','')
        if alive and c > 0: producing += 1
        else: attention.append((bid, 'proc=%s cells=%d' % (d.get('proc'), c)))
        print("  %-10s %-8s %-6s %-9s %-6s %-10s %s" % (
            bid, d.get('prog','?'), d.get('files'), f"{c:,}",
            d.get('free'), d.get('rate','?'), d['_route']))
    #: A DESTROYED BOX IS DELISTED, AND ITS CELLS ARE NOT LOST -- THEY ARE ON DISK.
    #: The total is built from what vast LISTS, so tearing down a finished box made
    #: WRITTEN fall 372,818 -> 320,870, and inflated the ETA to match. Same shape as
    #: the unreachable case one door along, and staged teardown would repeat it on
    #: every destroy. Add the locally-held cells of every box-id directory that is
    #: no longer live.
    live_ids = {str(i['id']) for i in inst}
    if os.path.isdir(DEST_LOCAL):
        for d in sorted(os.listdir(DEST_LOCAL)):
            if d.startswith('_') or not d.isdigit() or d in live_ids:
                continue
            n = 0
            for f in os.listdir(os.path.join(DEST_LOCAL, d)):
                if f.endswith('.jsonl'):
                    with open(os.path.join(DEST_LOCAL, d, f), 'rb') as fh:
                        n += sum(1 for _ in fh)
            if n:
                tot += n
                print("  %-10s RETIRED, counting %s cells held locally" % (d, f"{n:,}"))

    #: TWO UNITS, AND THEY ARE NOT THE SAME. `tot` counts ROWS on disk and the
    #: runner writes 1,820 per model; TOTAL_EXPECTED is 250 rungs x 1,786 DISTINCT
    #: contexts, which is the science target. Dividing rows by distinct contexts
    #: read 99.6% while 8 of 250 rungs were still outstanding. Rows are compared
    #: to rows; RUNG COVERAGE below is the honest progress measure.
    rows_expected = 250 * 1820
    print("\n  ROWS WRITTEN %s / %s  (%.1f%%)   [science target: %s distinct cells]"
          % (f"{int(tot):,}", f"{rows_expected:,}", 100*tot/rows_expected,
             f"{TOTAL_EXPECTED:,}"))
    print("  producing %d/%d    burn $%.2f/hr" % (producing, len(res), burn))
    rates = [float(d['rate'].split()[0]) for _,d,_,_ in res
             if d and d.get('rate','').strip()]
    if rates and tot < TOTAL_EXPECTED:
        agg = sum(rates)
        hrs = (TOTAL_EXPECTED - tot)/agg/3600
        print("  aggregate %.1f p/s -> ~%.1f h remaining, ~$%.2f more needed" % (agg, hrs, hrs*burn))
    #: RENTAL WINDOWS. Box 47630611 was lost on 13 Aug not to preemption and not
    #: to a reclaimed GPU -- is_bid was False and host reliability 0.9966 -- but
    #: because its advertised availability window was 4.10 h (13:47:21 start,
    #: 17:53:28 end_date) against a job needing ~8. It stopped exactly at its
    #: end_date. The field is on the OFFER before you rent, so this is a
    #: preflight check that was never run, not a hazard.
    import time as _t
    for i in inst:
        ed = i.get('end_date')
        if not ed or i.get('cur_state') != 'running': continue
        hrs = (ed - _t.time()) / 3600
        if rates and tot < TOTAL_EXPECTED:
            need = (TOTAL_EXPECTED - tot) / sum(rates) / 3600
            if hrs < need:
                print("  WINDOW RISK %s: rental ends in %.1f h, run needs %.1f h"
                      % (i['id'], hrs, need))
                attention.append((i['id'], 'window %.1fh < need %.1fh' % (hrs, need)))

    if attention:
        print("  ATTENTION: %s" % attention)
    return 0

if __name__ == '__main__':
    sys.exit(main())
