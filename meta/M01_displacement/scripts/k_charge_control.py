"""Is the transgressiveness effect anything beyond charge? And does it depend on
charge level?

    uv run python meta/M01_displacement/scripts/k_charge_control.py en

K tested every scale MARGINALLY -- residualised on base probability and corpus
frequency, never on another scale. That was deliberate for the joint
seven-predictor case, where collinear scales split shared variance arbitrarily.
But transgressiveness~charge is +0.65, the highest collinearity in the set, so
the marginal transgressiveness result could in principle be charge in a costume.

THREE MODELS, REPORTED TOGETHER BECAUSE THEY ANSWER DIFFERENT QUESTIONS:

  MARGINAL      x ~ movement | prob, freq              (what K reported)
  CONTROLLED    x ~ movement | prob, freq, THE OTHER   (narrower claim)
  INTERACTION   the transgressiveness effect computed separately among
                high-charge and low-charge words, split at each prompt's own
                median ON REAL CHARGE IN BOTH CONDITIONS, and the difference
                tested. Splitting on shuffled charge in the shuffled condition
                is the error this file was committed with; see the note at the
                split.

Run SYMMETRICALLY: charge controlled for transgressiveness as well as the
reverse. Controlling in one direction only would let whichever scale was chosen
as the covariate keep all the shared variance by fiat.

WHAT THE EXISTING NUMBERS ALREADY SUGGEST. The two correlate at +0.65 and their
marginal effects have OPPOSITE signs (transgressiveness z -23.7 falling, charge
z +5.9 rising). Shared variance cannot drive both directions, so each effect is
carried by its unique part. This tests that directly rather than inferring it.

The shuffled control runs in the same invocation, per the lesson of
`failed_control_v1.log`: a per-prompt count without its shuffled twin is not
interpretable.
"""
import collections, json, math, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import k_analysis as A
from k_frequency import fpm
import k_population as KP
K=os.path.join(ROOT,"meta/M01_displacement/results/k")
DRAWS=3000; MIN_WORDS=60; MIN_HALF=25
rng=np.random.default_rng(20260812)

def corr(x,y):
    xs=np.sqrt((x*x).sum()); ys=np.sqrt((y*y).sum())
    return float((x@y)/(xs*ys)) if xs>0 and ys>0 else 0.0

def pv(x,y,obs,draws=DRAWS):
    n=len(x); idx=np.argsort(rng.random((draws,n)),axis=1)
    xp=x[idx]; r=(xp@y)/(np.sqrt((xp*xp).sum(axis=1))*np.sqrt((y*y).sum()))
    return float((np.abs(r)>=abs(obs)).sum()+1)/(draws+1)

def main(lang="en"):
    meas="coca_fic" if lang=="en" else "SUBTLEX_CH"
    norm=json.load(open(os.path.join(K,"normalisation_%s.json"%lang)))
    rate=json.load(open(os.path.join(K,"ratings_%s.json"%lang)))["ratings"]
    t2u=norm["token_to_unit"]
    FQ={u:math.log10(f) for u in rate for f in [fpm(u,lang,meas)] if f and f>0}
    ks=sorted(FQ); pmt=rng.permutation(len(ks))
    SH={ks[i]:rate[ks[pmt[i]]] for i in range(len(ks))}   #: built-in control
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
    acc=collections.defaultdict(list)
    for i,(p,v) in enumerate(byp.items()):
        w=[x[0] for x in v]
        lp=A.ranks([math.log10(x[2]) for x in v]); lf=A.ranks([FQ[x[0]] for x in v])
        y=A.resid(A.ranks([x[1] for x in v]),[lp,lf])
        for tag,R in (("real",rate),("shuf",SH)):
            tr=A.ranks([R[u]["transgressiveness"] for u in w])
            ch=A.ranks([R[u]["charge"] for u in w])
            trR=A.resid(tr,[lp,lf]); chR=A.resid(ch,[lp,lf])
            acc[(tag,"transgressiveness MARGINAL")].append(corr(trR,y))
            acc[(tag,"charge MARGINAL")].append(corr(chR,y))
            acc[(tag,"transgressiveness | charge")].append(corr(A.resid(tr,[lp,lf,ch]),A.resid(y,[chR])))
            acc[(tag,"charge | transgressiveness")].append(corr(A.resid(ch,[lp,lf,tr]),A.resid(y,[trR])))
            #: INTERACTION: the transgressiveness effect inside each charge half.
            #: **THE SPLIT IS ON REAL CHARGE IN BOTH CONDITIONS.** The first
            #: version split on `ch`, which under the shuffled condition is
            #: SHUFFLED charge -- so the two halves were not the same words and
            #: the null came back asymmetric (+0.0095 hi against -0.0147 lo,
            #: where both must sit at zero). That asymmetry is the only reason
            #: the error was caught. Holding the split fixed and permuting ONLY
            #: transgressiveness puts both baselines at ~0.00 and leaves the
            #: interaction unchanged at -0.089.
            ch_real=A.ranks([rate[u]["charge"] for u in w])
            med=np.median(ch_real); hi=np.where(ch_real>med)[0]; lo=np.where(ch_real<=med)[0]
            if len(hi)>=MIN_HALF and len(lo)>=MIN_HALF:
                for nm,ix in (("hi-charge",hi),("lo-charge",lo)):
                    ll=[A.ranks([[math.log10(x[2]) for x in v][j] for j in ix]),
                        A.ranks([[FQ[x[0]] for x in v][j] for j in ix])]
                    yy=A.resid(A.ranks([v[j][1] for j in ix]),ll)
                    tv=[rate[w[j]]["transgressiveness"] for j in ix]
                    #: THE CONTROL FOR A WITHIN-HALF STATISTIC IS A WITHIN-HALF
                    #: PERMUTATION. The global rating shuffle `SH` used above is
                    #: right for the marginal models but wrong here: it leaves
                    #: the half's shuffled baseline OFF-CENTRE (+0.013, -0.011
                    #: instead of ~0) because it mixes ratings across halves,
                    #: and that inflates the interaction from -20.8 to -28.4.
                    #: Permute transgressiveness among the words IN THIS HALF.
                    if tag == "shuf":
                        tv = list(tv); rng.shuffle(tv)
                    xx=A.resid(A.ranks(tv),ll)
                    acc[(tag,"trns @ %s"%nm)].append(corr(xx,yy))
        if i%400==0: print("   %d/%d"%(i,len(byp)),flush=True)
    print("  %-30s %9s %9s %8s   %s"%("model","mean rho","shuffled","z","reading"))
    for lab in ("transgressiveness MARGINAL","charge MARGINAL",
                "transgressiveness | charge","charge | transgressiveness",
                "trns @ hi-charge","trns @ lo-charge"):
        a=np.array(acc[("real",lab)]); b=np.array(acc[("shuf",lab)])
        if not len(a): continue
        d=a-b[:len(a)] if len(b)>=len(a) else a
        z=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
        print("  %-30s %+9.4f %+9.4f %8.1f   n=%d"%(lab,a.mean(),b[:len(a)].mean(),z,len(a)))
    hi=np.array(acc[("real","trns @ hi-charge")]); lo=np.array(acc[("real","trns @ lo-charge")])
    hs=np.array(acc[("shuf","trns @ hi-charge")]); ls=np.array(acc[("shuf","trns @ lo-charge")])
    n=min(len(hi),len(lo),len(hs),len(ls))
    d=hi[:n]-lo[:n]; ds=hs[:n]-ls[:n]
    #: TESTED AGAINST ITS OWN SHUFFLED DIFFERENCE, not against zero. The raw z of
    #: (hi - lo) is -28.5; subtracting the shuffled difference carries that
    #: shuffle's variance and gives -20.8. The stricter figure is the published
    #: one and this line is why the two differ.
    dd=d-ds
    z=dd.mean()/(dd.std(ddof=1)/np.sqrt(n))
    print("\n  INTERACTION  trns@hi minus trns@lo:  real %+.4f  shuffled %+.4f"%(d.mean(),ds.mean()))
    print("  z of the difference against its OWN shuffle: %+.1f   (raw z %+.1f)   n=%d"
          %(z,d.mean()/(d.std(ddof=1)/np.sqrt(n)),n))
    print("  -> %s"%("the transgressiveness effect DEPENDS on charge level"
          if abs(z)>3 else "no evidence the effect depends on charge level"))
if __name__=="__main__": main(sys.argv[1] if len(sys.argv)>1 else "en")
