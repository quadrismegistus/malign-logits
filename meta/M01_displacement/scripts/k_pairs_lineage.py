"""Plan K on the M01 minimal pairs, WITH THE LINEAGE AS THE UNIT.

    uv run python meta/M01_displacement/scripts/k_pairs_lineage.py

RH, 2026-08-12: "in most of our analyses we have n of m lineages as whether
something is significant?" Yes -- and the first version of this test did not.

`k_pairs.py` takes the PAIR as the unit: 684 pairs, each partial computed on
movement pooled across all 46 lineage representatives. That puts lineage
variation INSIDE the measurement instead of making it the exchangeable object,
and a sign test over 684 pairs then treats 684 correlated numbers as
independent draws. It is the same pseudo-replication the campaign's unit
discipline exists to prevent -- F and G take the base checkpoint, U the rung
within family, T the edge, and every one of them reports n of m.

Here the unit is the LINEAGE. For each of the 46 representatives, the statistic
is the MEDIAN over the 684 pairs of

    partial(MARKED) - partial(UNMARKED)

computed on that lineage's movement alone. Then a sign test over lineages. n
falls from 684 to 46 and the test gets much harder, which is the point: a result
that holds at the pair unit and dies at the lineage unit was a result about
pairs, not about alignment.
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
    pm,DOM=m01_pairs()
    norm=json.load(open(os.path.join(K,"normalisation_en.json")))
    rate=json.load(open(os.path.join(K,"ratings_en.json")))["ratings"]
    t2u=norm["token_to_unit"]
    edges=KP.reps("en"); esc=lambda s: s.replace("\\","\\\\").replace("'","\\'")
    texts={t for v in pm.values() for t in v.values()}
    tl="','".join(esc(t) for t in texts)
    print("M01 pairs %d | prompts %d | LINEAGES %d"%(len(pm),len(texts),len(edges)))
    #: frequency and rating are properties of the word, so precompute once
    FQ={}; RT={}
    for u in rate:
        f=fpm(u,"en","coca_fic")
        if f and f>0: FQ[u]=math.log10(f); RT[u]=[rate[u][s] for s in A.SCALES]
    per_lin={}; CELL={}
    for li,(b,a) in enumerate(edges,1):
        rows=A.q("""SELECT prompt, word, countIf(cls='rise')-countIf(cls='fall') AS net,
                    avg(p_base) AS pbase FROM (
            SELECT m.prompt AS prompt, m.word AS word, m.cls AS cls, m.p_base AS p_base,
              row_number() OVER (PARTITION BY m.prompt ORDER BY m.p_base DESC) rb,
              row_number() OVER (PARTITION BY m.prompt ORDER BY m.p_aligned DESC) ra
            FROM %s.movement m WHERE m.rule='canonical'
              AND m.base='%s' AND m.aligned='%s' AND m.prompt IN ('%s'))
          WHERE rb<=50 OR ra<=50 GROUP BY prompt, word"""%(A.DB,esc(b),esc(a),tl))
        byp=collections.defaultdict(list)
        for r in rows: byp[r["prompt"]].append(r)
        def part(p):
            u=[(t2u[r["word"]],r["net"],r["pbase"]) for r in byp.get(p,[])
               if t2u.get(r["word"]) in FQ and r["pbase"]>0]
            if len(u)<MIN_N: return None
            y=A.ranks([x[1] for x in u])
            lp=A.ranks([math.log10(x[2]) for x in u]); lf=A.ranks([FQ[x[0]] for x in u])
            yr=A.resid(y,[lp,lf])
            return [A.pearson(A.resid(A.ranks([RT[x[0]][i] for x in u]),[lp,lf]),yr)
                    for i in range(len(A.SCALES))]
        diffs={}
        for pid,mem in pm.items():
            x,z=part(mem.get("MARKED")),part(mem.get("UNMARKED"))
            if x and z: diffs[pid]=[p-q for p,q in zip(x,z)]
        if diffs:
            #: THE FULL (lineage x pair) MATRIX IS KEPT, not just its median.
            #: Medianing here would make the domain cut a second full run, and
            #: it would also hide how many pairs each lineage rests on.
            CELL[b]=diffs
            per_lin[b]=[float(np.median([d[i] for d in diffs.values()])) for i in range(len(A.SCALES))]
            print("  %2d/%d %-42s %4d pairs"%(li,len(edges),b.split("/")[-1][:42],len(diffs)),flush=True)
    def sf(k,n): return sum(comb(n,i)*.5**n for i in range(k,n+1))
    def two(k,n): return min(1.0,2*min(sf(k,n),sf(n-k,n)+comb(n,k)*.5**n))
    n=len(per_lin)
    print("\nLINEAGE AS UNIT: n=%d\n"%n)
    print("  %-18s %11s %9s %10s   %s"%("scale","median","neg/pos","sign p","pair-unit p"))
    PAIRP={"vulgarity":.157,"register_level":.339,"transgressiveness":.731,"charge":.619,
           "valence":.468,"bodily_harm":.00027,"concreteness":.00001}
    for i,s in enumerate(A.SCALES):
        v=[per_lin[k][i] for k in per_lin]; neg=sum(1 for x in v if x<0)
        print("  %-18s %+11.4f %5d/%-4d %10.5f   %10.5f"
              %(s,float(np.median(v)),neg,n-neg,two(max(neg,n-neg),n),PAIRP[s]))
    json.dump(per_lin,open(os.path.join(K,"pairs_by_lineage.json"),"w"))
    json.dump({"scales":list(A.SCALES),"domain":DOM,"cell":CELL},
              open(os.path.join(K,"pairs_lineage_cell.json"),"w"))
    #: ---- BY DOMAIN, LINEAGE STILL THE UNIT ----
    #: S finding 3's actual structure: violence against property against taboo.
    #: A harm calculus should bite hardest where harm is, and this is the cut
    #: that says so or does not.
    doms=collections.Counter(DOM.get(pid) for pid in next(iter(CELL.values())))
    print("\nBY DOMAIN, lineage still the unit (n=%d). Bonferroni over 7 scales: %.4f"
          %(len(CELL),0.05/7))
    for dm,cnt in doms.most_common():
        if not dm or cnt<15: continue
        print("\n  DOMAIN %s -- %d pairs" % (dm,cnt))
        for i,s2 in enumerate(A.SCALES):
            v=[float(np.median([d[i] for pid,d in CELL[b].items() if DOM.get(pid)==dm]))
               for b in CELL if any(DOM.get(pid)==dm for pid in CELL[b])]
            if len(v)<10: continue
            neg=sum(1 for x in v if x<0); pv=two(max(neg,len(v)-neg),len(v))
            star="  <-- survives correction" if pv<0.05/7 else ""
            print("     %-18s %+9.4f  %2d/%-2d  p %.5f%s"%(s2,float(np.median(v)),neg,len(v)-neg,pv,star))
if __name__=="__main__": main()
