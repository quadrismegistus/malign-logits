"""Plan K on the M01 TRANSGRESSIVE/NEUTRAL MINIMAL PAIRS.

    uv run python meta/M01_displacement/scripts/k_pairs.py

The per-prompt run showed the pooled analysis was not merely attenuated but
INVERTED on three scales -- transgressiveness pooled +0.012 against a
site-conditional -0.055, falling at 10 of 13 sites. That run used thirteen
prompts RH named, which is a chosen population with no control.

This is the same question on the declared one: 684 M01 minimal pairs, each a
MARKED prompt and its UNMARKED twin. The twin is the control the thirteen
prompts lacked, and the pair is the unit -- F and G's own design.

THE STATISTIC IS A WITHIN-PAIR DIFFERENCE. For each pair and each scale,
partial(MARKED) - partial(UNMARKED), then a sign test over pairs. A scale that
predicts movement equally at both twins contributes nothing; only a difference
BETWEEN the transgressive site and its matched neutral one counts. That is what
makes this a test of site-specificity rather than of charge in general.
"""
import json, math, os, sys, collections
from math import comb
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import k_analysis as A
from k_frequency import fpm
import k_population as KP
from within_pair import m01_pairs
K=os.path.join(ROOT,"meta/M01_displacement/results/k")
MIN_N=40

def main():
    pairs_map,_=m01_pairs()
    norm=json.load(open(os.path.join(K,"normalisation_en.json")))
    rate=json.load(open(os.path.join(K,"ratings_en.json")))["ratings"]
    t2u=norm["token_to_unit"]
    edges=KP.reps("en"); esc=lambda s: s.replace("\\","\\\\").replace("'","\\'")
    ep=" OR ".join("(m.base='%s' AND m.aligned='%s')"%(esc(b),esc(a)) for b,a in edges)
    texts={t for v in pairs_map.values() for t in v.values()}
    print("M01 pairs %d | %d prompts | %d lineage pairs"%(len(pairs_map),len(texts),len(edges)))
    tl="','".join(esc(t) for t in texts)
    rows=A.q("""SELECT prompt, word, countIf(cls='rise')-countIf(cls='fall') AS net,
                avg(p_base) AS pbase FROM (
        SELECT m.prompt AS prompt, m.word AS word, m.cls AS cls, m.p_base AS p_base,
          row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt ORDER BY m.p_base DESC) rb,
          row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt ORDER BY m.p_aligned DESC) ra
        FROM %s.movement m WHERE m.rule='canonical' AND (%s)
          AND m.prompt IN ('%s'))
      WHERE rb<=50 OR ra<=50 GROUP BY prompt, word"""%(A.DB,ep,tl))
    print("rows: %d"%len(rows))
    byp=collections.defaultdict(list)
    for r in rows: byp[r["prompt"]].append(r)

    def partials(prompt):
        u=[]
        for r in byp.get(prompt,[]):
            k=t2u.get(r["word"])
            if k and k in rate and r["pbase"]>0 and fpm(k,"en","coca_fic") is not None:
                u.append((k,r["net"],r["pbase"]))
        if len(u)<MIN_N: return None
        y=A.ranks([x[1] for x in u])
        lp=A.ranks([math.log10(x[2]) for x in u])
        lf=A.ranks([math.log10(fpm(x[0],"en","coca_fic")) for x in u])
        yr=A.resid(y,[lp,lf])
        return [A.pearson(A.resid(A.ranks([rate[x[0]][s] for x in u]),[lp,lf]),yr) for s in A.SCALES]

    D=collections.defaultdict(list)
    used=0
    for pid,mem in pairs_map.items():
        pm,pu=partials(mem.get("MARKED")),partials(mem.get("UNMARKED"))
        if pm is None or pu is None: continue
        used+=1
        for i,s in enumerate(A.SCALES): D[s].append(pm[i]-pu[i])
    def sf(k,n): return sum(comb(n,i)*.5**n for i in range(k,n+1))
    def two(k,n): return min(1.0,2*min(sf(k,n),sf(n-k,n)+comb(n,k)*.5**n))
    print("\nPAIRS USED: %d of %d (both twins with >=%d rated words)\n"%(used,len(pairs_map),MIN_N))
    print("  WITHIN-PAIR DIFFERENCE  partial(MARKED) - partial(UNMARKED)")
    print("  %-18s %10s %10s %9s %10s"%("scale","median","mean","neg/pos","sign p"))
    for s in A.SCALES:
        v=D[s]; neg=sum(1 for x in v if x<0)
        med=float(np.median(v)); mean=float(np.mean(v))
        print("  %-18s %+10.4f %+10.4f %5d/%-4d %10.5f  %s"
              %(s,med,mean,neg,len(v)-neg,two(max(neg,len(v)-neg),len(v)),
                "MORE negative at the transgressive twin" if med<0 else "more positive at the transgressive twin"))
if __name__=="__main__": main()
