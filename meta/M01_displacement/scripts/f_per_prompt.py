"""F-P: registration F's rate question asked PER PAIR instead of pooled inside
the checkpoint. Approved as a NEW QUESTION by registrar [5584]; F and G's
registrations are untouched and this is add-beside.

    uv run python meta/M01_displacement/scripts/f_per_prompt.py

WHY. F pools its 684 pairs inside each base checkpoint and returns one rate per
checkpoint, then signs those. Findings K showed that pooling across prompts does
not merely attenuate -- it can INVERT a sign, because scenes have opposite
directions and the average is a cancellation. **So F's null may be an absence or
it may be a cancellation, and F as written cannot tell the difference.**

THE SITE RULE IS THE FROZEN ONE, RECOMPUTED FROM THE TABLE, NOT REIMPLEMENTED
LOOSELY. m05_sites @ b8fd9a52cd5c794b: a cell FIRES when the top word changes
AND the aligned top word sits within AVAIL_MAX=19 of the base ordering. Both are
derivable from `movement` -- argmax of p_base, argmax of p_aligned, and the rank
of the latter in the base ordering -- so no distribution is re-read and the rule
is not re-typed as a threshold on something else.

WHAT IT CANNOT DO. This is NOT F. F's unit is the base checkpoint, its corpus is
frozen, and its statistic is a one-sided sign test over checkpoints. This asks
whether the per-PAIR rate differences that F averages away are homogeneous. A
null here would say F's null is an absence; a wide two-signed spread would say
it is a cancellation and that F's unit hid it.
"""
import collections, json, math, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import k_analysis as A
import k_population as KP
from within_pair import m01_pairs
AVAIL_MAX=19

def main():
    pm,DOM=m01_pairs()
    edges=KP.reps("en"); esc=lambda s:s.replace("\\","\\\\").replace("'","\\'")
    texts={t for v in pm.values() for t in v.values()}
    tl="','".join(esc(t) for t in texts)
    ep=" OR ".join("(m.base='%s' AND m.aligned='%s')"%(esc(b),esc(a)) for b,a in edges)
    #: top word per arm, and the base-ordering rank of the aligned top word --
    #: the two quantities `classify` needs, computed in SQL over the stored
    #: probabilities rather than by re-reading any distribution.
    rows=A.q("""
      SELECT base, aligned, prompt,
             argMax(word, p_base)    AS wb,
             argMax(word, p_aligned) AS wa
      FROM %s.movement
      WHERE rule='canonical' AND (%s) AND prompt IN ('%s')
      GROUP BY base, aligned, prompt"""%(A.DB,ep.replace("m.",""),tl))
    top={(r["base"],r["aligned"],r["prompt"]):(r["wb"],r["wa"]) for r in rows}
    rk=A.q("""
      SELECT base, aligned, prompt, word, row_number() OVER
             (PARTITION BY base, aligned, prompt ORDER BY p_base DESC) AS r
      FROM %s.movement WHERE rule='canonical' AND (%s) AND prompt IN ('%s')"""
      %(A.DB,ep.replace("m.",""),tl))
    rank={}
    for r in rk: rank[(r["base"],r["aligned"],r["prompt"],r["word"])]=int(r["r"])-1
    def fires(b,a,p):
        t=top.get((b,a,p))
        if not t: return None
        wb,wa=t
        if wb is None or wa is None or wb==wa: return False
        return rank.get((b,a,p,wa),10**9)<=AVAIL_MAX
    per={}
    for pid,mem in pm.items():
        fm=fu=n=0
        for b,a in edges:
            x=fires(b,a,mem.get("MARKED")); z=fires(b,a,mem.get("UNMARKED"))
            if x is None or z is None: continue
            n+=1; fm+=x; fu+=z
        if n>=20: per[pid]=(fm/n-fu/n,n,DOM.get(pid))
    d=np.array([v[0] for v in per.values()])
    print("F-P: %d of %d pairs with >=20 lineages\n"%(len(per),len(pm)))
    print("  per-pair rate difference, MARKED minus UNMARKED")
    print("     mean   %+.4f      F's pooled unit-level median was +0.0132"%d.mean())
    print("     median %+.4f"%np.median(d))
    print("     sd     %.4f"%d.std(ddof=1))
    print("     range  %+.3f to %+.3f"%(d.min(),d.max()))
    print("     positive %d / negative %d / zero %d"%((d>0).sum(),(d<0).sum(),(d==0).sum()))
    print("\n  IS IT A CANCELLATION?")
    print("     mean |difference|  %.4f"%np.abs(d).mean())
    print("     |mean| / mean|.|   %.3f   (near 0 = the pairs cancel; near 1 = they agree)"
          %(abs(d.mean())/np.abs(d).mean()))
    big=(np.abs(d)>=0.20).sum()
    print("     pairs with |difference| >= 0.20 : %d (%.0f%%)"%(big,100*big/len(d)))
    print("\n  BY DOMAIN")
    bd=collections.defaultdict(list)
    for pid,(v,n,dm) in per.items(): bd[dm].append(v)
    for dm,v in sorted(bd.items(),key=lambda kv:-len(kv[1])):
        if len(v)<20: continue
        v=np.array(v)
        print("     %-14s n %3d  mean %+.4f  mean|.| %.4f  %3d+/%3d-"
              %(dm,len(v),v.mean(),np.abs(v).mean(),(v>0).sum(),(v<0).sum()))
    json.dump({k:list(v) for k,v in per.items()},
              open(os.path.join(ROOT,"meta/M01_displacement/results/f_per_prompt.json"),"w"))
if __name__=="__main__": main()
