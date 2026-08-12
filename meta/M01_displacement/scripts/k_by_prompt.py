"""Which PROMPTS does each scale predict alignment shift at, and how many?

    uv run python meta/M01_displacement/scripts/k_by_prompt.py en

One test per (prompt, scale). Inside a prompt the unit is the LINEAGE: each of
the 46 representatives gives its own partial correlation between the scale and
net movement at that prompt, controlling base probability and frequency, and the
46 values are tested by sign-flip permutation. Magnitudes are preserved -- a
lineage with a large partial contributes its size, not a bit.

WHY PER PROMPT. The pooled run said transgressiveness +0.012 and concreteness
-0.166; conditioning on 13 sites INVERTED both. A count over prompts asks the
question the pooled number cannot: not "how big on average" but "at how many
sites, and which ones".

THE COUNT IS READ AGAINST CHANCE, NOT ADMIRED. At alpha .05, 5% of prompts fire
by construction. A scale is interesting only if it fires at meaningfully more,
and the expected-by-chance count is printed beside every observed one. FDR
(Benjamini-Hochberg) is reported too, because thousands of tests at .05 is a
counting exercise otherwise.
"""
import json, math, os, sys, collections
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import k_analysis as A
from k_frequency import fpm
import k_population as KP
K=os.path.join(ROOT,"meta/M01_displacement/results/k")
DRAWS=2000; MIN_WORDS=40; MIN_LIN=15
rng=np.random.default_rng(20260812)

def main(lang):
    meas="coca_fic" if lang=="en" else "SUBTLEX_CH"
    norm=json.load(open(os.path.join(K,"normalisation_%s.json"%lang)))
    rate=json.load(open(os.path.join(K,"ratings_%s.json"%lang)))["ratings"]
    t2u=norm["token_to_unit"]
    FQ={u:math.log10(f) for u in rate for f in [fpm(u,lang,meas)] if f and f>0}
    RT={u:[rate[u][s] for s in A.SCALES] for u in FQ}
    edges=KP.reps(lang); esc=lambda s: s.replace("\\","\\\\").replace("'","\\'")
    ep=" OR ".join("(m.base='%s' AND m.aligned='%s')"%(esc(b),esc(a)) for b,a in edges)
    rows=A.q("""SELECT prompt, base, word, countIf(cls='rise')-countIf(cls='fall') AS net,
                avg(p_base) AS pbase FROM (
        SELECT m.prompt AS prompt, m.base AS base, m.word AS word, m.cls AS cls,
               m.p_base AS p_base,
          row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt ORDER BY m.p_base DESC) rb,
          row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt ORDER BY m.p_aligned DESC) ra
        FROM %s.movement m
        INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                    WHERE status='ACTIVE' AND language='%s') p ON m.prompt=p.prompt
        WHERE m.rule='canonical' AND (%s))
      WHERE rb<=50 OR ra<=50 GROUP BY prompt, base, word"""%(A.DB,A.DB,lang,ep))
    print("[%s] %d (prompt,lineage,word) rows"%(lang,len(rows)),flush=True)
    cells=collections.defaultdict(list)
    for r in rows:
        u=t2u.get(r["word"])
        if u in FQ and r["pbase"]>0: cells[(r["prompt"],r["base"])].append((u,r["net"],r["pbase"]))
    byp=collections.defaultdict(dict)
    for (p,b),v in cells.items():
        if len(v)>=MIN_WORDS: byp[p][b]=v
    byp={p:v for p,v in byp.items() if len(v)>=MIN_LIN}
    print("[%s] %d prompts with >=%d words on >=%d lineages"%(lang,len(byp),MIN_WORDS,MIN_LIN),flush=True)

    res={}
    for pi,(p,lins) in enumerate(byp.items()):
        M=[]
        for b,v in lins.items():
            y=A.ranks([x[1] for x in v]); lp=A.ranks([math.log10(x[2]) for x in v])
            lf=A.ranks([FQ[x[0]] for x in v]); yr=A.resid(y,[lp,lf])
            M.append([A.pearson(A.resid(A.ranks([RT[x[0]][i] for x in v]),[lp,lf]),yr)
                      for i in range(len(A.SCALES))])
        M=np.array(M); n=len(M)
        sd=M.std(axis=0,ddof=1)/np.sqrt(n)
        obs=np.where(sd>0,M.mean(axis=0)/np.where(sd>0,sd,1),0.0)
        S=rng.choice([-1.,1.],size=(DRAWS,n))
        nt=np.where(sd>0,(S@M)/n/np.where(sd>0,sd,1),0.0)
        res[p]={"n":n,"t":obs.tolist(),"med":np.median(M,axis=0).tolist(),
                "p":(((np.abs(nt)>=np.abs(obs)).sum(axis=0)+1)/(DRAWS+1)).tolist()}
        if pi%400==0: print("   %d/%d"%(pi,len(byp)),flush=True)
    json.dump(res,open(os.path.join(K,"by_prompt_%s.json"%lang),"w"))
    N=len(res)
    print("\n[%s] AT HOW MANY OF %d PROMPTS DOES EACH SCALE PREDICT?\n"%(lang,N))
    print("  %-18s %10s %10s %11s %9s"%("scale","p<=.05","expected","FDR .05","neg share"))
    for i,s in enumerate(A.SCALES):
        ps=np.array([res[p]["p"][i] for p in res]); ts=np.array([res[p]["t"][i] for p in res])
        k=int((ps<=.05).sum())
        o=np.sort(ps); m=len(o)
        thr=max([o[j] for j in range(m) if o[j]<=(j+1)/m*0.05], default=0)
        fdr=int((ps<=thr).sum())
        negsh=float((ts[ps<=.05]<0).mean()) if k else float("nan")
        print("  %-18s %10d %10.0f %11d %9.2f"%(s,k,0.05*N,fdr,negsh))
    print("\n  'neg share' is the fraction of FIRING prompts where the scale predicts")
    print("  FALLING. Near 1.0 means one direction everywhere; near 0.5 means the")
    print("  sites disagree and any pooled number is a cancellation.")
    return res

if __name__=="__main__": main(sys.argv[1] if len(sys.argv)>1 else "en")
