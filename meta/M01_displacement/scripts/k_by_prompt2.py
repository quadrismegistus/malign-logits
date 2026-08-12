"""Per-prompt test, rebuilt with the WORD-permutation null, and its negative
control built in rather than optional.

    uv run python meta/M01_displacement/scripts/k_by_prompt2.py en

WHAT THE FIRST VERSION GOT WRONG. It made the LINEAGE the unit inside each
prompt and sign-flipped over 46 lineages. But all 46 see THE SAME WORDS with THE
SAME RATINGS at that prompt: if the rating vector happens to correlate with
movement in that word sample, every lineage shows it, and the permutation counts
one shared draw as 46 independent ones. The effective n is the word sample, not
the lineage count -- pseudo-replication one level below where the campaign's
unit discipline guards.

It failed its negative control decisively. With ratings shuffled across words so
no real link could exist, it fired at 35-42% of prompts against an expected 5%,
and for five of seven scales the SHUFFLED count exceeded the observed one.

THE FIX. The unit inside a prompt is the WORD. Lineages are averaged into the
outcome, where they belong -- they are replicate measurements of one quantity,
not independent draws. The null permutes RATINGS ACROSS WORDS within the prompt,
which breaks the rating-movement link and nothing else.

THE CONTROL IS NOT OPTIONAL AND RUNS EVERY TIME. A globally shuffled rating map
goes through the identical pipeline in the same invocation, and its firing count
prints beside the real one. A per-prompt count without its shuffled twin is not
interpretable, and the previous version proved that the hard way.
"""
import json, math, os, sys, collections
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import k_analysis as A
from k_frequency import fpm
import k_population as KP
K=os.path.join(ROOT,"meta/M01_displacement/results/k")
DRAWS=4000; MIN_WORDS=60
rng=np.random.default_rng(20260812)

def partial_and_p(words, net, pb, RT, FQ, draws=DRAWS):
    """Partial rho per scale, p by permuting ratings ACROSS WORDS."""
    n=len(words)
    y=A.ranks(net); lp=A.ranks([math.log10(x) for x in pb]); lf=A.ranks([FQ[w] for w in words])
    yr=A.resid(y,[lp,lf]); ys=np.sqrt((yr*yr).sum())
    if ys==0: return None,None
    X=np.array([A.resid(A.ranks([RT[w][i] for w in words]),[lp,lf]) for i in range(len(A.SCALES))])
    xs=np.sqrt((X*X).sum(axis=1)); xs[xs==0]=1
    obs=(X@yr)/(xs*ys)
    #: PERMUTE THE RATING LABELS ACROSS WORDS. The residualisation is redone on
    #: each draw because the controls are word properties too -- permuting after
    #: residualising would leave the control structure attached to the wrong
    #: words and give a null that is too narrow.
    idx=np.argsort(rng.random((draws,n)),axis=1)
    hits=np.zeros(len(A.SCALES))
    for d in range(draws):
        Xp=X[:,idx[d]]
        r=(Xp@yr)/(np.sqrt((Xp*Xp).sum(axis=1))*ys)
        hits+=(np.abs(r)>=np.abs(obs))
    return obs,(hits+1)/(draws+1)

def main(lang):
    meas="coca_fic" if lang=="en" else "SUBTLEX_CH"
    norm=json.load(open(os.path.join(K,"normalisation_%s.json"%lang)))
    rate=json.load(open(os.path.join(K,"ratings_%s.json"%lang)))["ratings"]
    t2u=norm["token_to_unit"]
    FQ={u:math.log10(f) for u in rate for f in [fpm(u,lang,meas)] if f and f>0}
    RT={u:[rate[u][s] for s in A.SCALES] for u in FQ}
    ks=sorted(FQ); pv=rng.permutation(len(ks))
    SH={ks[i]:RT[ks[pv[i]]] for i in range(len(ks))}      #: the built-in control
    edges=KP.reps(lang); esc=lambda s: s.replace("\\","\\\\").replace("'","\\'")
    ep=" OR ".join("(m.base='%s' AND m.aligned='%s')"%(esc(b),esc(a)) for b,a in edges)
    rows=A.q("""SELECT prompt, word, countIf(cls='rise')-countIf(cls='fall') AS net,
                avg(p_base) AS pbase FROM (
        SELECT m.prompt AS prompt, m.word AS word, m.cls AS cls, m.p_base AS p_base,
          row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt ORDER BY m.p_base DESC) rb,
          row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt ORDER BY m.p_aligned DESC) ra
        FROM %s.movement m
        INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                    WHERE status='ACTIVE' AND language='%s') p ON m.prompt=p.prompt
        WHERE m.rule='canonical' AND (%s))
      WHERE rb<=50 OR ra<=50 GROUP BY prompt, word"""%(A.DB,A.DB,lang,ep))
    byp=collections.defaultdict(list)
    for r in rows:
        u=t2u.get(r["word"])
        if u in FQ and r["pbase"]>0: byp[r["prompt"]].append((u,r["net"],r["pbase"]))
    byp={p:v for p,v in byp.items() if len(v)>=MIN_WORDS}
    print("[%s] %d prompts with >=%d rated words (unit inside a prompt = the WORD)"
          %(lang,len(byp),MIN_WORDS),flush=True)
    out={}; shp=[]
    for i,(p,v) in enumerate(byp.items()):
        w=[x[0] for x in v]; nt=[x[1] for x in v]; pb=[x[2] for x in v]
        o,pp=partial_and_p(w,nt,pb,RT,FQ)
        if o is None: continue
        o2,p2=partial_and_p(w,nt,pb,SH,FQ)
        out[p]={"n":len(v),"rho":o.tolist(),"p":pp.tolist()}
        shp.append(p2)
        if i%300==0: print("   %d/%d"%(i,len(byp)),flush=True)
    json.dump(out,open(os.path.join(K,"by_prompt2_%s.json"%lang),"w"))
    P=np.array([out[p]["p"] for p in out]); S=np.array(shp); N=len(P)
    print("\n[%s] %d prompts. WORD-permutation null, with the shuffled control beside it.\n"%(lang,N))
    print("  %-18s %9s %9s %9s %10s"%("scale","fires","SHUFFLED","expected","share fall"))
    R=np.array([out[p]["rho"] for p in out])
    for i,s in enumerate(A.SCALES):
        k=int((P[:,i]<=.05).sum()); sh=int((S[:,i]<=.05).sum())
        neg=float((R[P[:,i]<=.05,i]<0).mean()) if k else float("nan")
        flag="" if k>2*max(sh,1) else "   <- NOT above its own control"
        print("  %-18s %9d %9d %9.0f %10.2f%s"%(s,k,sh,0.05*N,neg,flag))
    print("\n  mean p: real %.3f  shuffled %.3f  (0.50 = calibrated)"%(P.mean(),S.mean()))
if __name__=="__main__": main(sys.argv[1] if len(sys.argv)>1 else "en")
