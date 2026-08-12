#!/usr/bin/env python
"""Is the ALIGNED excess profile really different from the BASE one?

    uv run python both_excess_permutation.py

RH's objection, and it is the right one: restricting the arm-difference test to
fields that survived in BASE only asks whether BASE's fields declined. It cannot
speak to the fields that surfaced in ALIGNED (possession, volition, duration),
because selecting on the aligned column and then testing aligned-minus-base is
circular -- the selection guarantees the difference it then measures.

TWO TESTS, AND ONLY THE SECOND ESCAPES THE CIRCLE.

  A. BASE-SELECTED DIFFERENCE. Fields surviving Holm in the base arm, tested for
     aligned-minus-base. The selection rule is stateable without looking at the
     aligned column, so it is honest -- and it answers only RH's narrow reading.

  B. PERMUTATION ON THE ARM LABEL. No field is selected at all. Under the null
     that alignment does not change the excess profile, the two arms of a group
     are EXCHANGEABLE, so swapping their labels within a group generates the null
     distribution of any whole-profile statistic. Two statistics:

        n_survivors(arm)          how many fields clear Holm in that arm
        total |mean excess|       summed over every field, one arm

     The observed aligned-minus-base is compared against the permuted spread.
     One test, no selection, no multiplicity, and it answers the actual question:
     is the aligned profile different, not is any particular field different.

WHY EXCHANGEABILITY HOLDS. The unit is the (group, arm) excess; under the null
the arm label carries no information about the profile, so a within-group swap
is a relabelling of equals. It does NOT assume the two arms have equal noise --
which matters, because 'aligned simply has lower within-group variance' is a
live alternative to 'aligned has a different profile', and this design cannot
separate them. Stated, not solved: see the variance check printed at the end.
"""
import collections, json, math, os, random, statistics as st

HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP))
SRC=os.path.join(CAMP,"results","both_vs_controls_fields.json")
OUT=os.path.join(CAMP,"results","both_excess_permutation.json")
MIN_GROUPS=8; NPERM=2000; SEED=4946


def sign_test(v):
    v=[x for x in v if x!=0.0]; n,k=len(v),sum(1 for x in v if x>0)
    if not n: return float("nan")
    t=min(k,n-k)
    return min(1.0,2*sum(math.comb(n,i) for i in range(t+1))/2**n)


def holm_survivors(ps):
    s=sorted(ps.items(),key=lambda x:x[1]); m=len(s); run=0.0; out=[]
    for i,(k,p) in enumerate(s):
        run=max(run,min(1.0,(m-i)*p))
        if run<0.05: out.append(k)
    return out


def profile(exc_by_arm, arm_of):
    """{field: [excess per group]} for the arm assignment given."""
    prof=collections.defaultdict(list)
    for (gid,which),vals in exc_by_arm.items():
        if arm_of(gid,which)!="aligned": continue
        for k,v in vals.items(): prof[k].append(v)
    return prof


def stats(prof):
    ps={k:sign_test(v) for k,v in prof.items() if len(v)>=MIN_GROUPS}
    ps={k:p for k,p in ps.items() if p==p}
    surv=holm_survivors(ps)
    tot=sum(abs(st.mean(v)) for k,v in prof.items() if len(v)>=MIN_GROUPS)
    return len(surv),tot,surv


def main():
    d=json.load(open(SRC))
    cells={tuple(k.split("|")):v for k,v in d["cells"].items()}
    groups=sorted({g for (_,g,_) in cells})
    def rates(arm,gid,role,kind):
        c=cells.get((arm,gid,role))
        if not c or not c["n_content"]: return None
        n=c["n_content"]; return {k:v/n for k,v in c[kind].items()}
    exc={}
    for arm in ("base","aligned"):
        for gid in groups:
            acc={}
            for kind in ("fields","words"):
                B=rates(arm,gid,"BOTH",kind); A=rates(arm,gid,"POLE_A",kind)
                Bb=rates(arm,gid,"POLE_B",kind)
                if not (B and A and Bb): continue
                for k in set(B)|set(A)|set(Bb):
                    acc[k]=B.get(k,0.0)-0.5*(A.get(k,0.0)+Bb.get(k,0.0))
            if acc: exc[(gid,arm)]=acc
    usable=[g for g in groups if (g,"base") in exc and (g,"aligned") in exc]
    print("%d groups with BOTH arms complete\n"%len(usable))
    #: THE KEY SET MUST BE THE UNION ACROSS GROUPS AND MUST BE THE SAME ON BOTH
    #: SIDES. The first version took the observed key list from ONE group
    #: (`exc[(usable[0], arm)]`) while the permutation loop took the union --
    #: so observed and null were computed over different field sets and their
    #: comparison meant nothing. Same class as every population defect this
    #: campaign has met: the statistic was fine, the population was not.
    ALLK = set().union(*[set(exc[(g, a)]) for g in usable for a in ("base", "aligned")])
    def prof_for(arm):
        return collections.defaultdict(list, {
            k: [exc[(g, arm)][k] for g in usable if k in exc[(g, arm)]] for k in ALLK})
    nb,tb,sb=stats(prof_for("base"))
    na,ta,sa=stats(prof_for("aligned"))
    print("field keys in the union: %d"%len(ALLK))
    print("OBSERVED")
    print("   base     survivors %3d   total |mean excess| %.5f"%(nb,tb))
    print("   aligned  survivors %3d   total |mean excess| %.5f"%(na,ta))
    print("   difference          %+3d                       %+.5f"%(na-nb,ta-tb))
    rng=random.Random(SEED); dn=[]; dt=[]
    for _ in range(NPERM):
        flip={g:rng.random()<0.5 for g in usable}
        A={}; B={}
        for g in usable:
            a,b=("aligned","base") if not flip[g] else ("base","aligned")
            A[g]=exc[(g,a)]; B[g]=exc[(g,b)]
        pa=collections.defaultdict(list); pb=collections.defaultdict(list)
        for g in usable:
            for k in ALLK:
                if k in A[g]: pa[k].append(A[g][k])
                if k in B[g]: pb[k].append(B[g][k])
        x=stats(pa); y=stats(pb)
        dn.append(x[0]-y[0]); dt.append(x[1]-y[1])
    def pval(obs,null):
        return (1+sum(1 for x in null if abs(x)>=abs(obs)))/(1+len(null))
    print("\nPERMUTATION NULL, %d draws, arm label swapped within group"%NPERM)
    print("   survivors diff   observed %+4d   null mean %+.1f sd %.1f   p = %.4f"
          %(na-nb,st.mean(dn),st.pstdev(dn),pval(na-nb,dn)))
    print("   total|excess|    observed %+.5f   null mean %+.5f sd %.5f   p = %.4f"
          %(ta-tb,st.mean(dt),st.pstdev(dt),pval(ta-tb,dt)))
    vb=st.mean([st.pstdev([exc[(g,"base")][k] for g in usable if k in exc[(g,"base")]])
                for k in list(exc[(usable[0],"base")])[:400]])
    va=st.mean([st.pstdev([exc[(g,"aligned")][k] for g in usable if k in exc[(g,"aligned")]])
                for k in list(exc[(usable[0],"aligned")])[:400]])
    print("\nTHE ALTERNATIVE THIS CANNOT RULE OUT")
    print("   mean within-field sd across groups:  base %.5f   aligned %.5f  (%.2fx)"
          %(vb,va,va/vb if vb else float('nan')))
    print("   if aligned is QUIETER, more fields clear at the same effect size.")
    json.dump({"observed":{"base_survivors":nb,"aligned_survivors":na,
                           "base_total":tb,"aligned_total":ta},
               "perm":{"n":NPERM,"seed":SEED,
                       "p_survivors":pval(na-nb,dn),"p_total":pval(ta-tb,dt)},
               "within_field_sd":{"base":vb,"aligned":va},
               "base_survivors":sb,"aligned_survivors":sa},open(OUT,"w"),indent=1)
    print("\n-> %s"%os.path.relpath(OUT,ROOT))


if __name__=="__main__":
    main()
