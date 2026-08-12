#!/usr/bin/env python
"""Split-sample: select fields on half the lineages, test them on the other half.

    uv run python both_split_sample.py

WHY. Both single-sample routes are biased and in OPPOSITE directions, because
`excess_aligned = excess_base + difference` and every selection shares a term
with the test:

    select on ALIGNED, test aligned-base   -> selection contains +difference,
                                              biased TOWARD finding change
    select on BASE,    test aligned-base   -> selection contains -difference,
                                              biased toward finding DECLINE
                                              (regression to the mean)

Splitting by LINEAGE breaks it: the fields are chosen from one set of models and
tested on a disjoint set, so the noise in selection is independent of the noise
in the test. The 23 GROUPS are identical in both halves -- only the models
differ -- so the test unit does not change and the excess still means what it
meant.

BOTH DIRECTIONS ARE RUN AND BOTH ARE REPORTED. Selecting on s0 and testing s1
is one experiment; s1 -> s0 is another. If they disagree the honest reading is
that neither half has the power, and reporting only the agreeable one is the
whole reason this design exists.

THREE QUESTIONS, EACH WITH ITS OWN SELECTION:

    Q1a  fields selected for BASE excess     -> tested for BASE excess
    Q1b  fields selected for ALIGNED excess  -> tested for ALIGNED excess
    Q2   fields selected for ARM DIFFERENCE  -> tested for ARM DIFFERENCE

Q1a/Q1b are honest replications: does the excess reproduce on unseen models?
Q2 is the one RH's objection was about and the only one that can speak to
whether alignment changes the profile.
"""
import collections, json, math, os, statistics as st

HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP))
SRC=os.path.join(CAMP,"results","both_vs_controls_fields_split.json")
OUT=os.path.join(CAMP,"results","both_split_sample.json")
MIN_GROUPS=8
SELECT_P=0.05          #: uncorrected, on the SELECTION half only -- it is a
                       #: filter, not a claim; the claim is the test half.


def sign_test(v):
    v=[x for x in v if x!=0.0]; n,k=len(v),sum(1 for x in v if x>0)
    if not n: return 0,0,float("nan")
    t=min(k,n-k)
    return n,k,min(1.0,2*sum(math.comb(n,i) for i in range(t+1))/2**n)


def holm(items):
    """FWER. Kept for comparison only -- see `bh` for why it is not the default."""
    s=sorted(items,key=lambda x:x[1]); m=len(s); out={}; run=0.0
    for i,(k,p,*_) in enumerate(s):
        run=max(run,min(1.0,(m-i)*p)); out[k]=run
    return out


def bh(items):
    """Benjamini-Hochberg FDR, and it is the right correction for THIS question.

    Holm controls the FAMILY-WISE error rate -- the chance of ANY false
    positive -- which is what you want when one wrong entry discredits the set.
    RH's question is exploratory screening: is there anything else here. The
    quantity that matters is then the expected PROPORTION of the returned list
    that is false, which is FDR.

    Two further reasons. (1) These tests are positively dependent -- fields
    overlap, words co-occur, and the three USAS views are one lexicon -- which
    is exactly the regime where BH is valid and Holm is merely conservative.
    (2) The split-sample has ALREADY filtered the hypothesis space on
    independent data, so FWER control on top of it penalises twice for one
    problem.

    Returns adjusted q-values by the standard step-up with monotonicity
    enforced from the largest p downward.
    """
    s=sorted(items,key=lambda x:x[1]); m=len(s); out={}; run=1.0
    for i in range(m-1,-1,-1):
        k,p=s[i][0],s[i][1]
        run=min(run,min(1.0,p*m/(i+1))); out[k]=run
    return out


def main():
    d=json.load(open(SRC))
    cells={tuple(k.split("|")):v for k,v in d["cells"].items()}
    splits=sorted({s for (s,_,_,_) in cells}); groups=sorted({g for (_,_,g,_) in cells})
    print("%s passages | splits %s | %d groups\n"%(format(d["_meta"]["passages"],","),
                                                   splits,len(groups)))
    def rates(sp,arm,gid,role,kind):
        c=cells.get((sp,arm,gid,role))
        if not c or not c["n_content"]: return None
        n=c["n_content"]; return {k:v/n for k,v in c[kind].items()}
    exc=collections.defaultdict(dict)      # (split,arm) -> {field: {gid: excess}}
    for sp in splits:
        for arm in ("base","aligned"):
            acc=collections.defaultdict(dict)
            for gid in groups:
                for kind in ("fields","words"):
                    B=rates(sp,arm,gid,"BOTH",kind); A=rates(sp,arm,gid,"POLE_A",kind)
                    Bb=rates(sp,arm,gid,"POLE_B",kind)
                    if not (B and A and Bb): continue
                    for k in set(B)|set(A)|set(Bb):
                        acc[k][gid]=B.get(k,0.0)-0.5*(A.get(k,0.0)+Bb.get(k,0.0))
            exc[(sp,arm)]=acc
    def series(sp,which,k):
        if which in ("base","aligned"): return exc[(sp,which)].get(k,{})
        b=exc[(sp,"base")].get(k,{}); a=exc[(sp,"aligned")].get(k,{})
        return {g:a[g]-b[g] for g in set(a)&set(b)}
    res={}
    for label,which in (("Q1a  BASE excess","base"),
                        ("Q1b  ALIGNED excess","aligned"),
                        ("Q2   ARM DIFFERENCE (aligned - base)","diff")):
        print("="*78); print(label); print("="*78)
        res[label]={}
        for sel,tst in ((splits[0],splits[1]),(splits[1],splits[0])):
            keys=set(exc[(sel,"base")])|set(exc[(sel,"aligned")])
            chosen=[]
            for k in keys:
                v=list(series(sel,which,k).values())
                if len(v)<MIN_GROUPS: continue
                n,u,p=sign_test(v)
                if n>=MIN_GROUPS and p<SELECT_P: chosen.append(k)
            items=[]
            for k in chosen:
                v=list(series(tst,which,k).values())
                if len(v)<MIN_GROUPS: continue
                n,u,p=sign_test(v)
                if n>=MIN_GROUPS: items.append((k,p,u,n,st.mean(v)))
            q=bh(items) if items else {}
            h=holm(items) if items else {}
            surv=[x for x in items if q[x[0]]<0.10]
            print("\n  select on %s (%d fields at p<%.2f) -> test on %s"
                  %(sel,len(chosen),SELECT_P,tst))
            print("     %d testable | %d p<0.05 raw | BH q<0.05: %d | q<0.10: %d | Holm: %d"
                  %(len(items),sum(1 for x in items if x[1]<0.05),
                    sum(1 for x in items if q[x[0]]<0.05),len(surv),
                    sum(1 for x in items if h[x[0]]<0.05)))
            if surv:
                print("     %-44s %6s %10s %9s %9s"%("key","up/n","mean","BH q","holm"))
            for k,p,u,n,m in sorted(surv,key=lambda x:x[1]):
                print("        %-44s %2d/%-2d %+10.5f %9.4g %9.4g"%(k,u,n,m,q[k],h[k]))
            res[label]["%s->%s"%(sel,tst)]={"n_selected":len(chosen),"n_tested":len(items),
                "n_uncorrected":sum(1 for x in items if x[1]<0.05),
                "survivors":[{"key":k,"up":u,"n":n,"mean":m,"bh_q":q[k],"holm":h[k]}
                             for k,p,u,n,m in surv]}
        print()
    json.dump({"_meta":d["_meta"],"select_p":SELECT_P,"min_groups":MIN_GROUPS,
               "result":res},open(OUT,"w"),indent=1)
    print("-> %s"%os.path.relpath(OUT,ROOT))


if __name__=="__main__":
    main()
