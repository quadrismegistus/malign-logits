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
        c = int(d.get('cells','0') or 0); tot += c
        alive = d.get('proc','0').strip() not in ('0','')
        if alive and c > 0: producing += 1
        else: attention.append((bid, 'proc=%s cells=%d' % (d.get('proc'), c)))
        print("  %-10s %-8s %-6s %-9s %-6s %-10s %s" % (
            bid, d.get('prog','?'), d.get('files'), f"{c:,}",
            d.get('free'), d.get('rate','?'), d['_route']))
    pct = 100*tot/TOTAL_EXPECTED
    print("\n  WRITTEN %s / %s  (%.2f%%)" % (f"{int(tot):,}", f"{TOTAL_EXPECTED:,}", pct))
    print("  producing %d/%d    burn $%.2f/hr" % (producing, len(res), burn))
    rates = [float(d['rate'].split()[0]) for _,d,_,_ in res
             if d and d.get('rate','').strip()]
    if rates and tot < TOTAL_EXPECTED:
        agg = sum(rates)
        hrs = (TOTAL_EXPECTED - tot)/agg/3600
        print("  aggregate %.1f p/s -> ~%.1f h remaining, ~$%.2f more needed" % (agg, hrs, hrs*burn))
    if attention:
        print("  ATTENTION: %s" % attention)
    return 0

if __name__ == '__main__':
    sys.exit(main())
