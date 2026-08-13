#!/usr/bin/env python3
"""verse_rebalance.py -- hand a finished box some of the long pole's work.

    scripts/verse_rebalance.py --from 47628506 --to 47628645 [--apply]
    scripts/verse_rebalance.py --auto [--apply]      # every idle box, best donor

## WHY THIS EXISTS

The verse shards are balanced in MODELS, not in TIME, and the cards differ ~1.7x
in throughput. Measured 13 Aug: 12.4 min/model on the Ada boxes against 21.7 on
the Quadros -- and the two Quadros also drew the two 32-model shards. Result is
a 4-hour idle tail: six boxes finish ~20:35, shard1 finishes ~00:38.

Adapted from `passage_rebalance.py` (12 Aug), which cannot be reused directly:
that one resolves boxes through MALIGN_VAST_STATE files, reads
`data/passage_manifests`, writes to /root/out, and its unit is a PAIR. This
fleet has shard specs, /workspace/verse, and a unit of one MODEL/rung.

## WHY A RESTART IS SAFE HERE, verified in twp_cloud.py before this was written

  * `--purge` purges model WEIGHTS from the HF cache (line 361), never output.
  * The runner resumes BY READBACK at PROMPT grain: `done_prompts(path)` reads
    the existing .jsonl and re-offers only what is missing.
  * The .f16 row counter resumes FROM THE FILE'S OWN SIZE (line 393-398), never
    from a remembered count -- so appending after a restart cannot mis-index.

A donor therefore loses nothing by being restarted, not even its in-flight model.

## WHAT IT MOVES, AND WHAT IT REFUSES TO MOVE

Only models with NO output file on the donor at all. A model with a .jsonl --
complete or partial -- stays with the donor, because the donor can resume it
per-prompt and the recipient would start it from zero. This is stricter than
the passage version's `-size +1k` test and needs no size threshold.

Never moves a model to a box that already has output for it: one model, one box,
or two half-files claim the same rung.
"""
import argparse, json, os, subprocess, sys, glob

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS  = os.path.join(ROOT, "data", "verse_fleet", "specs")
DEST_LOCAL = os.path.join(ROOT, "data", "raw", "verse_fleet")
ROWS_COMPLETE = 1820   # measured: median 1820, p10 1820 over 132 pulled files
REMOTE = "/workspace/malign-logits"

def vast_instances():
    r = subprocess.run(['vastai','show','instances','--raw'],
                       capture_output=True, text=True, timeout=90)
    return json.loads(r.stdout)

def routes(i):
    out=[]
    if i.get('ssh_host'): out.append((i['ssh_host'], i['ssh_port']))
    ip=(i.get('public_ipaddr') or '').strip()
    if ip and i.get('direct_port_start'): out.append((ip, i['direct_port_start']))
    return out

def sh(i, cmd, timeout=120):
    for h,p in routes(i):
        try:
            r=subprocess.run(['ssh','-n','-o','ConnectTimeout=15',
                '-o','StrictHostKeyChecking=no','-o','BatchMode=yes','-p',str(p),
                'root@'+h,cmd], capture_output=True,text=True,timeout=timeout,
                stdin=subprocess.DEVNULL)
            if r.returncode==0: return r.stdout
        except Exception: continue
    return None

def scp(i, local, remote):
    for h,p in routes(i):
        try:
            r=subprocess.run(['scp','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=no',
                '-o','BatchMode=yes','-P',str(p),local,'root@%s:%s'%(h,remote)],
                capture_output=True,text=True,timeout=180)
            if r.returncode==0: return True
        except Exception: continue
    return False

def safe(m): return m.replace('/','__')

def roster(path):
    ms=json.load(open(path))
    return [(m if isinstance(m,str) else (m.get('model') or m.get('id'))) for m in ms]


def roster_entries(path):
    """model name -> the FULL spec entry.

    twp_cloud.py reads entry["model"] and entry["prompts"]; a list of bare model
    NAMES makes it die on `TypeError: string indices must be integers` the
    instant it starts. The recovery wrote names and the freed box sat idle.
    """
    out={}
    for m in json.load(open(path)):
        if isinstance(m, dict):
            k = m.get('model') or m.get('id')
            if k: out[k] = m
    return out

def rosters():
    return {os.path.basename(f): roster(f) for f in sorted(glob.glob(os.path.join(SPECS,'shard*.json')))}

def box_state(i, rs):
    """(shard_name, models_with_output, running) derived from the BOX, not a record."""
    out = sh(i, "ls /workspace/verse/*.jsonl 2>/dev/null | xargs -n1 basename 2>/dev/null; "
                "echo ---; pgrep -fc 'twp_clou[d]' || echo 0")
    if out is None: return None
    body,_,proc = out.partition('---')
    have = set(l.strip()[:-6] for l in body.strip().split('\n') if l.strip())
    sname = max(rs, key=lambda s: len(have & set(safe(m) for m in rs[s]))) if rs else None
    return (sname, have, proc.strip().split('\n')[0].strip() not in ('0',''))

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--from', dest='src')
    ap.add_argument('--to',   dest='dst')
    ap.add_argument('--auto', action='store_true',
                    help='pair every idle box with the donor holding most unstarted work')
    ap.add_argument('--recover', action='store_true',
                    help='reassign a DEAD box\'s unfinished models from LOCAL state')
    ap.add_argument('--apply', action='store_true')
    a=ap.parse_args(argv)

    inst={str(i['id']): i for i in vast_instances()}
    rs=rosters()
    state={bid: box_state(i,rs) for bid,i in inst.items()}
    for bid,s in state.items():
        if s is None: print("  %s UNREACHABLE -- skipped (not a fault, see runbook 2.26)" % bid)

    def unstarted(bid):
        s=state.get(bid)
        if not s: return []
        sname,have,_=s
        return [m for m in rs[sname] if safe(m) not in have]

    # ---- dead-donor recovery -------------------------------------------------
    # A box can be STOPPED by vast (GPU reclaimed) rather than merely unreachable:
    # actual_status='running' beside cur_state='stopped', direct route refused,
    # `vastai start` answering "resources currently unavailable, state change
    # queued". Its shard then has no live donor, so the normal path cannot move
    # the work. What it DID write is already pulled, so the local tree is the
    # record of what survives and the roster minus that is what must be re-run.
    if a.recover:
        live = set(inst)
        #: A PRESENT FILE IS NOT A FINISHED MODEL. Box 47630611 died mid-write and
        #: left `pythia-6.9b@step8000.jsonl` with 959 rows; every complete model in
        #: this corpus has exactly ROWS_COMPLETE (median 1820, p10 1820 over 132
        #: files). Counting presence would have retired that rung silently -- the
        #: same two-states-one-appearance shape the byte-level teardown exists for,
        #: and it cannot run here because the remote is gone.
        held = {}
        if os.path.isdir(DEST_LOCAL):
            for d in os.listdir(DEST_LOCAL):
                if d.startswith('_'): continue
                done = set()
                for f in os.listdir(os.path.join(DEST_LOCAL, d)):
                    if not f.endswith('.jsonl'): continue
                    with open(os.path.join(DEST_LOCAL, d, f), 'rb') as fh:
                        n = sum(1 for _ in fh)
                    if n >= ROWS_COMPLETE: done.add(f[:-6])
                    elif d not in live:
                        #: only say this for a box that is actually GONE. On a live
                        #: box a short file is the model being written right now and
                        #: nothing is re-running it -- announcing otherwise is prose
                        #: asserting an action that is not happening.
                        print("     TRUNCATED on dead box %s: %s has %d rows -- will re-run"
                              % (d, f, n))
                held[d] = done
        #: EXCLUDE WORK ALREADY IN FLIGHT. `held` is what is on DISK, so a model
        #: being generated RIGHT NOW on another box still looks owed -- and the
        #: second idle box was handed the identical 15, both burning to produce
        #: the same rungs. Ask every live box what spec it is running.
        inflight = set()
        for bid_, i_ in inst.items():
            out_ = live(routes(i_),
                        "cat %s/data/recover.json 2>/dev/null" % REMOTE)
            if not out_: continue
            try:
                for e_ in json.loads(out_):
                    m_ = e_.get('model') if isinstance(e_, dict) else e_
                    if m_: inflight.add(safe(m_))
            except Exception:
                pass
        if inflight:
            print("  %d model(s) already in flight on a live box; excluded" % len(inflight))

        orphaned = {}
        for bid, models in held.items():
            if bid in live and state.get(bid): continue      # box still with us
            sname = max(rs, key=lambda sn: len(models & set(safe(m) for m in rs[sn])))
            missing = [m for m in rs[sname]
                       if safe(m) not in models and safe(m) not in inflight]
            if missing: orphaned[bid] = (sname, missing)
        if not orphaned:
            print("  no orphaned shard work. Nothing to recover."); return 0
        for bid,(sname,missing) in orphaned.items():
            print("  DEAD BOX %s ran %s: %d of %d models held locally, %d TO RE-RUN"
                  % (bid, sname.replace('.json',''), len(rs[sname])-len(missing),
                     len(rs[sname]), len(missing)))
            for m in missing[:5]: print("      %s" % m)
            if len(missing)>5: print("      ... and %d more" % (len(missing)-5))
        idle = [b for b,st_ in state.items() if st_ and not st_[2]]
        if not idle:
            print("  no idle box to take it yet -- re-run when one frees."); return 0
        allm = [m for _,ms in orphaned.values() for m in ms]
        per = max(1, len(allm)//len(idle))
        for n,b in enumerate(idle):
            chunk = allm[n*per:(n+1)*per] if n < len(idle)-1 else allm[n*per:]
            if not chunk: continue
            print("  -> %d models to %s" % (len(chunk), b))
            if not a.apply: continue
            entries=[]
            for sname_ in sorted(glob.glob(os.path.join(SPECS,'shard*.json'))):
                ent = roster_entries(sname_)
                for m in chunk:
                    if m in ent and not any(e.get('model')==m for e in entries):
                        entries.append(ent[m])
            missing_e=[m for m in chunk if not any(e.get('model')==m for e in entries)]
            if missing_e:
                print("     REFUSING: no spec entry for %s" % missing_e[:3]); continue
            mp = os.path.join(SPECS, 'recover_to_%s.json' % b)
            json.dump(entries, open(mp,'w'))
            print("     wrote %d full entries (%d prompts each)"
                  % (len(entries), len(entries[0].get('prompts',[]))))
            if not scp(inst[b], mp, REMOTE+'/data/recover.json'):
                print("     scp FAILED"); continue
            run=("cd %s && tmux kill-session -t verse 2>/dev/null; "
                 "tmux new-session -d -s verse 'cd %s && python3 scripts/twp_cloud.py "
                 "--models data/recover.json --out /workspace/verse --purge "
                 "--gpu-budget-gb 45 --dict %s/data/dict/jieba_dict_big.txt "
                 ">> /workspace/verse.log 2>&1'") % (REMOTE, REMOTE, REMOTE)
            sh(inst[b], run)
            print("     launched")
        if not a.apply: print("  dry run. re-run with --apply.")
        return 0

    pairs=[]
    if a.auto:
        idle=[b for b,s in state.items() if s and not s[2]]
        if not idle:
            print("  no idle box yet. Nothing to rebalance."); return 0
        for b in idle:
            donors=sorted((bid for bid in state if state[bid] and state[bid][2]),
                          key=lambda d: -len(unstarted(d)))
            if not donors: break
            d=donors[0]
            u=unstarted(d)
            if len(u)<3:
                print("  best donor %s has %d unstarted; not worth splitting" % (d,len(u))); break
            pairs.append((d,b))
    else:
        if not (a.src and a.dst): ap.error('need --from and --to, or --auto')
        pairs=[(a.src,a.dst)]

    for src,dst in pairs:
        if src not in state or not state[src]: print("  donor %s unreachable" % src); continue
        if dst not in state or not state[dst]: print("  recipient %s unreachable" % dst); continue
        sname,have,running = state[src]
        u = unstarted(src)
        if len(u) < 2:
            print("  donor %s has %d unstarted model(s); not worth splitting" % (src,len(u))); continue
        half = len(u)//2 or 1
        move, keep_un = u[-half:], u[:-half]
        keep = [m for m in rs[sname] if safe(m) in have] + keep_un
        # never send a model the recipient already has output for
        dhave = state[dst][1]
        clash = [m for m in move if safe(m) in dhave]
        if clash:
            print("  REFUSING: recipient %s already has output for %s" % (dst, clash[:2])); continue

        print("\n  donor %s (%s): %d models, %d with output, %d unstarted"
              % (src, sname.replace('.json',''), len(rs[sname]), len(have), len(u)))
        print("    KEEP %d  (incl. every model with any output -- it resumes per-prompt)" % len(keep))
        print("    MOVE %d -> %s" % (len(move), dst))
        for m in move[:6]: print("         %s" % m)
        if len(move)>6: print("         ... and %d more" % (len(move)-6))

        if not a.apply:
            print("  dry run. re-run with --apply to write specs, restart donor, launch recipient.")
            continue

        kp=os.path.join(SPECS,'keep_%s.json'%src); mp=os.path.join(SPECS,'move_%s_to_%s.json'%(src,dst))
        json.dump(keep, open(kp,'w')); json.dump(move, open(mp,'w'))
        ok1=scp(inst[src], kp, REMOTE+'/data/keep.json')
        ok2=scp(inst[dst], mp, REMOTE+'/data/move.json')
        if not (ok1 and ok2): print("  scp FAILED -- nothing restarted"); continue
        run=("cd %s && tmux kill-session -t verse 2>/dev/null; "
             "tmux new-session -d -s verse 'cd %s && python3 scripts/twp_cloud.py "
             "--models data/%s --out /workspace/verse --purge --gpu-budget-gb 45 "
             "--dict %s/data/dict/jieba_dict_big.txt >> /workspace/verse.log 2>&1'")
        r1=sh(inst[src], run % (REMOTE,REMOTE,'keep.json',REMOTE))
        r2=sh(inst[dst], run % (REMOTE,REMOTE,'move.json',REMOTE))
        #: A tmux command returning 0 says the SESSION was created, not that the
        #: runner is running -- the same exists-vs-ran gap the fleet probe hit.
        #: Confirm a live twp_cloud process on both before calling this done.
        import time; time.sleep(20)
        for tag,bid in (('donor',src),('recipient',dst)):
            alive = sh(inst[bid], "pgrep -fc 'twp_clou[d]' || echo 0")
            n = (alive or '0').strip().split('\n')[0]
            if n in ('0',''):
                tail = sh(inst[bid], "tail -5 /workspace/verse.log 2>/dev/null") or ''
                print("    !! %s %s HAS NO RUNNER after restart -- log tail:\n%s"
                      % (tag,bid,tail))
            else:
                print("    %s %s: runner alive (proc=%s)" % (tag,bid,n))
    return 0

if __name__=='__main__':
    sys.exit(main())
