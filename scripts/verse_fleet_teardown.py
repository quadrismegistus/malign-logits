#!/usr/bin/env python3
"""Staged teardown: verify a box byte-for-byte, then destroy it.

WHY STAGED. Shards are fixed, so the fleet finishes when the SLOWEST box does
(measured 13 Aug: 7.6 h critical path against a 5.0 h fleet average -- the
average assumes work can be redistributed and it cannot). Holding all eight to
that point costs $34.72; destroying each as it completes costs $21.50. The
expensive Ada boxes finish first; the two cheap Quadros are the long pole.

RH's standing permission is to destroy a box AFTER verifying its data is
downloaded. This script is that verification, and it defaults to REPORT ONLY --
pass --destroy to act.

Verification is byte-level and per file: remote size + sha256 against local.
A count match is NOT verification (runbook 2.26: an in-progress .jsonl looks
complete), and the local copy must hold BOTH tiers -- .jsonl and .f16.
"""
import json, subprocess, sys, os, hashlib, glob, concurrent.futures as cf

DEST   = "data/raw/verse_fleet"
SPECS  = "data/verse_fleet/specs"

def sh(host, port, cmd, timeout=180):
    return subprocess.run(
        ['ssh','-n','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=no',
         '-o','BatchMode=yes','-p',str(port),'root@'+host,cmd],
        capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)

def routes(i):
    out=[]
    if i.get('ssh_host'): out.append((i['ssh_host'], i['ssh_port']))
    ip=(i.get('public_ipaddr') or '').strip()
    if ip and i.get('direct_port_start'): out.append((ip, i['direct_port_start']))
    return out

def live(host_port_pairs, cmd):
    for h,p in host_port_pairs:
        try:
            r = sh(h,p,cmd)
            if r.returncode==0: return r.stdout
        except Exception: continue
    return None

def shard_rosters():
    out={}
    for f in sorted(glob.glob(os.path.join(SPECS,'shard*.json'))):
        models=json.load(open(f))
        names=[(m if isinstance(m,str) else (m.get('model') or m.get('id'))) for m in models]
        out[os.path.basename(f)] = set(n for n in names if n)
    return out

def safe(name):
    return name.replace('/','__')

def check(i, rosters):
    bid=i['id']; rt=routes(i)
    # 1. what the box has WRITTEN, with sizes and hashes
    out = live(rt, "cd /workspace/verse 2>/dev/null && sha256sum *.jsonl *.f16 2>/dev/null; "
                   "echo '--SIZES--'; stat -c '%s %n' *.jsonl *.f16 2>/dev/null; "
                   "echo '--PROC--'; pgrep -fc 'twp_clou[d]' || echo 0")
    if out is None:
        return (bid,'UNREACHABLE','all routes failed -- NOT grounds to destroy',0,0)
    hpart,_,rest = out.partition('--SIZES--')
    spart,_,ppart = rest.partition('--PROC--')
    rhash={}
    for l in hpart.strip().split('\n'):
        if not l.strip(): continue
        h,_,n = l.partition('  ')
        rhash[n.strip()]=h.strip()
    rsize={}
    for l in spart.strip().split('\n'):
        p=l.split(None,1)
        if len(p)==2: rsize[p[1].strip()]=int(p[0])
    proc = ppart.strip().split('\n')[0].strip() if ppart.strip() else '0'

    # 2. which shard is this box running? derive from the models present
    present = set(n[:-6] for n in rhash if n.endswith('.jsonl'))
    best,score=None,-1
    for sname,roster in rosters.items():
        s=len(present & set(safe(m) for m in roster))
        if s>score: best,score=sname,s
    roster=rosters[best]; expected=set(safe(m) for m in roster)
    missing = expected - present

    # 3. complete only if every model in the shard has a jsonl AND the runner exited
    complete = (not missing) and proc in ('0','')

    # 4. byte-level verify local against remote
    ddir=os.path.join(DEST,str(bid)); bad=[]
    for n,h in rhash.items():
        lp=os.path.join(ddir,n)
        if not os.path.exists(lp): bad.append(n+' MISSING LOCALLY'); continue
        if os.path.getsize(lp)!=rsize.get(n,-1):
            bad.append('%s SIZE %d != %d' % (n,os.path.getsize(lp),rsize.get(n,-1))); continue
        lh=hashlib.sha256(open(lp,'rb').read()).hexdigest()
        if lh!=h: bad.append(n+' HASH MISMATCH')
    verified = (not bad) and len(rhash)>0
    st = ('READY' if complete and verified else
          'incomplete' if not complete else 'UNVERIFIED')
    detail = ('shard=%s %d/%d models, proc=%s, %d files verified'
              % (best,len(present),len(expected),proc,len(rhash)))
    if missing and len(missing)<=3: detail += ', missing '+','.join(sorted(missing)[:3])
    elif missing: detail += ', %d models still to run' % len(missing)
    if bad: detail += ' | FAILED: ' + '; '.join(bad[:3])
    return (bid, st, detail, len(rhash), len(bad))

def main(argv):
    do_destroy = '--destroy' in argv
    inst = json.loads(subprocess.run(['vastai','show','instances','--raw'],
                      capture_output=True,text=True,timeout=90).stdout)
    rosters = shard_rosters()
    with cf.ThreadPoolExecutor(min(8,len(inst))) as ex:
        res = list(ex.map(lambda i: check(i,rosters), inst))
    ready=[]
    for bid,st,detail,nf,nbad in sorted(res, key=lambda x:str(x[0])):
        print("  %-10s %-11s %s" % (bid,st,detail))
        if st=='READY': ready.append(bid)
    print("\n  READY TO DESTROY: %s" % (ready or 'none'))
    if ready and do_destroy:
        for bid in ready:
            r=subprocess.run(['vastai','destroy','instance',str(bid)],
                             capture_output=True,text=True,timeout=60)
            print("  destroyed %s: %s" % (bid, r.stdout.strip() or r.stderr.strip()))
    elif ready:
        print("  (report only -- pass --destroy to act)")
    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv[1:]))
