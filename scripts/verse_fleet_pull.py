#!/usr/bin/env python3
"""Incremental pull of verse-fleet output. EXCLUDES the .hidden.f32 tier.

RH's ruling (12 Aug): the f32 hidden states are not needed from the box.
They are 84% of the volume -- 9.2 GB per box against 1.6 GB of .f16 and
100 MB of .jsonl -- so pulling everything would need ~290 GB for the full
run and pulling jsonl+f16 needs ~42 GB.

Routes: proxy first, then direct. See runbook 2.26 -- a proxy can refuse
publickey while the box produces normally, so a failed route is not a failed box.
"""
import json, subprocess, sys, os, concurrent.futures as cf

DEST = "data/raw/verse_fleet"

def routes(i):
    out = []
    if i.get('ssh_host'):      out.append((i['ssh_host'], i['ssh_port']))
    ip = (i.get('public_ipaddr') or '').strip()
    if ip and i.get('direct_port_start'): out.append((ip, i['direct_port_start']))
    return out

def pull(i):
    dest = os.path.join(DEST, str(i['id']))
    os.makedirs(dest, exist_ok=True)
    for h, p in routes(i):
        cmd = ['rsync', '-az', '--partial', '--info=stats2',
               '--exclude', '*.hidden.f32', '--exclude', '*.f32',
               '-e', 'ssh -p %s -o StrictHostKeyChecking=no -o BatchMode=yes '
                     '-o ConnectTimeout=15' % p,
               'root@%s:/workspace/verse/' % h, dest + '/']
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode == 0:
            n = len([f for f in os.listdir(dest) if f.endswith('.jsonl')])
            sz = sum(os.path.getsize(os.path.join(dest,f)) for f in os.listdir(dest))
            return (i['id'], 'ok', n, sz/1e9, h)
    return (i['id'], 'FAILED', 0, 0, None)

def main():
    inst = json.loads(subprocess.run(['vastai','show','instances','--raw'],
                      capture_output=True, text=True, timeout=90).stdout)
    with cf.ThreadPoolExecutor(min(8, len(inst))) as ex:
        res = list(ex.map(pull, inst))
    tot = 0.0
    for bid, st, n, gb, h in sorted(res):
        tot += gb
        print("  %-10s %-7s jsonl=%-4s %6.2f GB  %s" % (bid, st, n, gb, h or ''))
    print("  TOTAL LOCAL %.2f GB in %s (f32 tier excluded)" % (tot, DEST))
    free = os.statvfs('.').f_bavail * os.statvfs('.').f_frsize / 1e9
    print("  local free %.1f GB" % free)
    if free < 20: print("  WARNING: below the 20 GB floor")
    return 0

if __name__ == '__main__':
    sys.exit(main())
