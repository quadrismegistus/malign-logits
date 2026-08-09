#!/usr/bin/env python
"""f11_l2_topup.py — give idle boxes the pairs nobody is doing. Loop.

**ORPHANS ARE THE FLEET'S DEFAULT FAILURE, NOT AN EDGE CASE.** Work is assigned
to a box by name; when that box is destroyed its pairs are assigned to a machine
that no longer exists and nothing notices. Six pairs were orphaned this way in
one afternoon -- n6, n9 and n10 were each holding 2-3 when they were destroyed
for being unreachable.

So the assignment is recomputed from OBSERVED STATE every pass, never stored:

    complete  = both arms have a 3,940-row file in the corpus
    claimed   = appears in the --pairs of a RUNNING process on a live box
    orphan    = live pair, not complete, not claimed  -> hand to an idle box

A box counts as idle when `pgrep` finds no runner, regardless of what the API
says about the rental. "Instance running" is the rental, not the work.
"""
import json,glob,os,subprocess,sys,time

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
DEAD=('mpt-7b','gpt-sw3','zamba2','croissantllm','deepseek-llm','teuken',
      'jais','baichuan','internlm2','pharia')
CORP='/Volumes/chambers/malign-l2/gen'

def ssh(h,p,cmd,t=25):
    S=["ssh","-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null",
       "-o","LogLevel=ERROR","-o","ConnectTimeout=8","-p",str(p),"root@%s"%h,cmd]
    return subprocess.run(S,capture_output=True,text=True,timeout=t).stdout

def state():
    for f in sorted(glob.glob(os.path.join(ROOT,'.vastai.l2*.json'))):
        try: d=json.load(open(f))
        except Exception: continue
        if d.get('ssh_host'): yield f,d

def main():
    q=json.load(open(os.path.join(ROOT,'data','f11_l2_queue_fp16.json')))
    live={r['idx']:r for r in q
          if not any(k in (r['base']+r['aligned']).lower() for k in DEAD)}
    while True:
      try:
        done={os.path.basename(f)[:-len('.gen.jsonl')].replace('__','/')
              for f in glob.glob(os.path.join(CORP,'*.gen.jsonl'))
              if sum(1 for _ in open(f))==3940}
        complete={i for i,r in live.items()
                  if r['base'] in done and r['aligned'] in done}
        claimed=set(); idle=[]
        for f,d in state():
            try:
                out=ssh(d['ssh_host'],d['ssh_port'],
                        "ps -eo args | grep '[f]11_l2_cloud' | head -1")
            except Exception:
                continue                      # unreachable: not idle, not claimed
            if '--pairs' in out:
                claimed |= {int(x) for x in out.split('--pairs')[1].split()[0].split(',')
                            if x.strip().isdigit()}
            else:
                idle.append((f,d))
        orphan=sorted(set(live)-complete-claimed)
        if orphan and idle:
            per=max(1,-(-len(orphan)//len(idle)))
            for (f,d),k in zip(idle,range(0,len(orphan),per)):
                give=orphan[k:k+per]
                if not give: break
                idxs=",".join(map(str,give))
                #: **THE ASSIGNMENT SSH MUST BE GUARDED TOO.** The status call
                #: was wrapped and this one was not, so one 25 s timeout killed
                #: the scheduler -- AFTER the remote command had already
                #: started. A supervisor that dies on a slow ssh is worse than
                #: no supervisor, because its last log line reads like success.
                try:
                    ssh(d['ssh_host'],d['ssh_port'],
                        "PY=$(command -v python || echo /usr/local/bin/python); "
                        "cd /workspace/malign-logits && mkdir -p /workspace/f11_l2 && "
                        "nohup $PY scripts/f11_l2_cloud.py --pairs %s --dtype float16 "
                        "--chunk 48 --out /workspace/f11_l2 >> /workspace/run.log 2>&1 &"%idxs,
                        t=40)
                    print("%s  %-10s <- %s" % (time.strftime('%H:%M:%S'),
                          os.path.basename(f)[8:-5], idxs), flush=True)
                except Exception as e:
                    #: the nohup may well have started anyway; the next pass
                    #: reads `claimed` from ps and will not double-assign
                    print("%s  %-10s assign timed out (%s) -- next pass re-reads ps"
                          % (time.strftime('%H:%M:%S'),
                             os.path.basename(f)[8:-5], type(e).__name__), flush=True)
        else:
            print("%s  complete=%d claimed=%d orphan=%d idle=%d"
                  % (time.strftime('%H:%M:%S'),len(complete),len(claimed),
                     len(orphan),len(idle)), flush=True)
      except Exception as e:
        print('%s  pass failed: %r' % (time.strftime('%H:%M:%S'), e), flush=True)
      time.sleep(180)

if __name__=="__main__": sys.exit(main())
