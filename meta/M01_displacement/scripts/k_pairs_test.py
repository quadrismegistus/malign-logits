"""Plan K's lineage-unit test, done with G's instrument and a correction that
respects the scales' correlation.

    uv run python meta/M01_displacement/scripts/k_pairs_test.py

TWO FIXES, BOTH ON RH'S CHALLENGE AND BOTH WITH CAMPAIGN PRECEDENT.

1. SIGN-FLIP PERMUTATION, NOT THE SIGN TEST. Registration G section 4 exists for
   this: the sign test "reduced 663 paired measurements per unit to one bit each
   and then could not see what was there". Its arithmetic at n=34 -- sign test
   needs standardised d 0.599, sign-flip permutation 0.426, 29% smaller. The
   permutation keeps each lineage's MAGNITUDE while still treating the lineage
   as the exchangeable object, so it fixes the bluntness without touching the
   pseudo-replication guard.

2. MAX-STATISTIC PERMUTATION, NOT BONFERRONI. Bonferroni assumes independent
   tests. These seven scales are not: transgressiveness~charge is +0.65,
   valence~bodily_harm -0.47. Correcting as though they were seven independent
   tests is over-conservative with no justification, and T section 16 makes the
   same objection to correcting "over a set containing near-duplicates".

   THE SIGN FLIPS ARE SHARED ACROSS SCALES WITHIN A DRAW. A lineage's seven
   values are one correlated vector, so flipping each scale independently would
   build a max-null narrower than the truth and hand back a correction that is
   too lenient -- the opposite error from Bonferroni and a worse one, because it
   looks like rigour.
"""
import json, os, sys, collections
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
K=os.path.join(ROOT,"meta/M01_displacement/results/k")
D=json.load(open(os.path.join(K,"pairs_lineage_cell.json")))
SC=D["scales"]; DOM=D["domain"]; CELL=D["cell"]
DRAWS=50000; rng=np.random.default_rng(20260812)

def stat(v): return float(np.mean(v))/ (np.std(v,ddof=1)/np.sqrt(len(v))) if np.std(v,ddof=1)>0 else 0.0

def test(M, label, n_pairs=None):
    """M: (lineages x scales). Sign-flip permutation, shared flips, max-stat."""
    n=M.shape[0]
    if n<10: print("  %-26s n=%d, too few lineages"%(label,n)); return
    obs=np.array([stat(M[:,i]) for i in range(M.shape[1])])
    S=rng.choice([-1.0,1.0],size=(DRAWS,n))          #: SHARED across scales
    null=np.einsum("dn,ns->ds",S,M)/n                 #: permuted means
    sd=M.std(axis=0,ddof=1)/np.sqrt(n)
    nullt=np.where(sd>0,null/np.where(sd>0,sd,1),0.0)
    per=(np.abs(nullt)>=np.abs(obs)).sum(axis=0)+1
    p_uncorr=per/(DRAWS+1)
    maxnull=np.abs(nullt).max(axis=1)
    p_max=np.array([( (maxnull>=abs(o)).sum()+1)/(DRAWS+1) for o in obs])
    print("\n  %s   lineages=%d%s"%(label,n,"  pairs/lineage=%d"%n_pairs if n_pairs else ""))
    print("    %-18s %9s %9s %10s %10s"%("scale","median","t","p","p max-stat"))
    for i,s in enumerate(SC):
        star="  *" if p_max[i]<=0.05 else ""
        print("    %-18s %+9.4f %+9.2f %10.5f %10.5f%s"
              %(s,float(np.median(M[:,i])),obs[i],p_uncorr[i],p_max[i],star))

lin=sorted(CELL)
allpids=sorted({p for b in lin for p in CELL[b]})
M=np.array([[np.median([CELL[b][p][i] for p in CELL[b]]) for i in range(len(SC))] for b in lin])
test(M,"ALL PAIRS",len(allpids))

#: MDE, simulated, so the nulls are interpretable rather than bare
def mde(n,draws=4000):
    lo,hi=0.0,2.0
    for _ in range(30):
        d=(lo+hi)/2
        x=rng.normal(d,1.0,size=(draws,n))
        t=x.mean(axis=1)/(x.std(axis=1,ddof=1)/np.sqrt(n))
        S2=rng.choice([-1.,1.],size=(2000,n))
        crit=np.quantile(np.abs((S2@x[:200].T/n)/(x[:200].std(axis=1,ddof=1)/np.sqrt(n))),0.95)
        if (np.abs(t)>=crit).mean()>=0.80: hi=d
        else: lo=d
    return hi
print("\n  MDE: sign-flip permutation at n=%d detects standardised d >= %.3f at 80%% power"%(len(lin),mde(len(lin))))
print("       (the SIGN TEST at this n needed P(neg) >= 0.718, i.e. 31 of 46)")

cnt=collections.Counter(DOM.get(p) for p in allpids)
for dm,c in cnt.most_common():
    if not dm or c<20: continue
    rows=[]
    for b in lin:
        v=[CELL[b][p] for p in CELL[b] if DOM.get(p)==dm]
        if len(v)>=10: rows.append([np.median([x[i] for x in v]) for i in range(len(SC))])
    if len(rows)>=10: test(np.array(rows),"DOMAIN = %s"%dm,c)
