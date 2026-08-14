"""Three DECLARED conjunctions, and only three.

    uv run python meta/M01_displacement/scripts/k_conjunctions.py en

[5592] found that transgressiveness predicts falling only among high-charge
words. Seven scales give 21 possible pairs and running them all is fishing, so
these three are declared in advance, each motivated by an existing claim:

  1. concreteness x charge      THE DISCRIMINATING TEST. If charge gates
     concreteness too, charge is a GENERAL precondition -- plausibly about how
     much room a word has to move -- and transgressiveness is a special case.
     If it gates only transgressiveness, the conjunction is specific. [5592]
     cannot currently tell these apart and they are different claims.

  2. transgressiveness x n_arousal   THE GATE'S IDENTITY. Warriner's human
     arousal replaces the coder's charge. The two calibrate at only 0.54, so
     this asks whether the gate is a real affective dimension or a property of
     one model's scale. Survives -> the finding strengthens. Fails -> [5592]'s
     stated limit becomes its headline.

  3. bodily_harm x charge       THE PARALLEL. S finding 3's harm calculus is the
     other live claim in this space; is it gated the same way?

METHOD, identical to the corrected k_charge_control: split each prompt's
vocabulary at its own median of the GATE, on REAL gate values in both
conditions; residualise on base probability and register-matched frequency
inside each half; permute the tested scale WITHIN each half; test the
hi-minus-lo difference against its own shuffled difference.

Those three choices are not stylistic -- each was a bug in the first version of
the charge analysis, and each moved the answer.
"""
import collections, json, math, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import k_analysis as A
from k_frequency import fpm
import k_population as KP
K=os.path.join(ROOT,"meta/M01_displacement/results/k")
MIN_WORDS=60; MIN_HALF=25
rng=np.random.default_rng(20260812)

PAIRS=[("concreteness","charge","does charge gate concreteness too?"),
       ("transgressiveness","n_arousal","is the gate a HUMAN affective dimension?"),
       ("bodily_harm","charge","is the harm calculus gated the same way?")]

def corr(x,y):
    a=np.sqrt((x*x).sum()); b=np.sqrt((y*y).sum())
    return float((x@y)/(a*b)) if a>0 and b>0 else 0.0

def main(lang="en"):
    from malign_logits.fields import _norms
    N=_norms(); N=N[0] if isinstance(N,tuple) else N
    meas="coca_fic" if lang=="en" else "SUBTLEX_CH"
    norm=json.load(open(os.path.join(K,"normalisation_%s.json"%lang)))
    rate=json.load(open(os.path.join(K,"ratings_%s.json"%lang)))["ratings"]
    t2u=norm["token_to_unit"]
    FQ={u:math.log10(f) for u in rate for f in [fpm(u,lang,meas)] if f and f>0}
    def val(u,s):
        if s.startswith("n_"):
            return N.get(u.strip().lower(),{}).get(s[2:])
        return rate[u][s]
    edges=KP.reps(lang); esc=lambda s:s.replace("\\","\\\\").replace("'","\\'")
    ep=" OR ".join("(m.base='%s' AND m.aligned='%s')"%(esc(b),esc(a)) for b,a in edges)
    rows=A.q("""SELECT prompt, word, countIf(cls='rise')-countIf(cls='fall') AS net,
                avg(p_base) AS pb FROM (
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
        if u in FQ and r["pb"]>0: byp[r["prompt"]].append((u,r["net"],r["pb"]))
    byp={p:v for p,v in byp.items() if len(v)>=MIN_WORDS}
    print("[%s] %d prompts\n"%(lang,len(byp)),flush=True)
    print("  %-22s %-12s %9s %9s %9s %8s %7s"
          %("tested scale","gate","@ hi gate","@ lo gate","hi-lo","z","n"))
    for scale,gate,why in PAIRS:
        HI=[];LO=[];HS=[];LS=[]
        for p,v in byp.items():
            w=[x[0] for x in v]
            g=[val(u,gate) for u in w]; t=[val(u,scale) for u in w]
            ok=[i for i in range(len(w)) if g[i] is not None and t[i] is not None]
            if len(ok)<MIN_WORDS: continue
            gr=A.ranks([g[i] for i in ok]); med=np.median(gr)
            for ix,store,ss in ((np.where(gr>med)[0],HI,HS),(np.where(gr<=med)[0],LO,LS)):
                if len(ix)<MIN_HALF: break
                jj=[ok[i] for i in ix]
                ll=[A.ranks([math.log10(v[j][2]) for j in jj]),
                    A.ranks([FQ[w[j]] for j in jj])]
                yy=A.resid(A.ranks([v[j][1] for j in jj]),ll)
                tv=[val(w[j],scale) for j in jj]
                store.append(corr(A.resid(A.ranks(tv),ll),yy))
                sh=list(tv); rng.shuffle(sh)
                ss.append(corr(A.resid(A.ranks(sh),ll),yy))
        n=min(len(HI),len(LO),len(HS),len(LS))
        if n<50:
            print("  %-22s %-12s  too few prompts (%d)"%(scale,gate,n)); continue
        HI,LO,HS,LS=(np.array(z[:n]) for z in (HI,LO,HS,LS))
        d=HI-LO; ds=HS-LS; dd=d-ds
        z=dd.mean()/(dd.std(ddof=1)/np.sqrt(n))
        print("  %-22s %-12s %+9.4f %+9.4f %+9.4f %8.1f %7d   %s"
              %(scale,gate,HI.mean(),LO.mean(),d.mean(),z,n,
                "GATED" if abs(z)>4 and abs(HI.mean())>2*abs(LO.mean()) else "not gated"))
        print("      %-64s shuffled hi %+.4f lo %+.4f"%(why,HS.mean(),LS.mean()))
if __name__=="__main__": main(sys.argv[1] if len(sys.argv)>1 else "en")
