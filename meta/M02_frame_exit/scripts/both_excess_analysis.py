#!/usr/bin/env python
"""The excess: what a BOTH generation carries that neither of its poles does.

    uv run python both_excess_analysis.py

Reads `both_vs_controls_fields.json`. For every (arm, group):

    excess(x) = rate_BOTH(x) - 0.5 * (rate_POLE_A(x) + rate_POLE_B(x))

rates are per content token, so a longer BOTH passage cannot manufacture excess.

THE UNIT IS THE GROUP. A group contributes one excess per arm; the test is a
sign test over groups. 46 model pairs are not 46 observations of a group -- the
arms are averaged within group before testing, and the groups are what vary.

TWO SEPARATE QUESTIONS, AND THEY ARE NOT THE SAME QUESTION:

    1. IS THERE AN EXCESS AT ALL?      base arm alone against zero.
       A contradiction prompt producing something neither pole produces is a
       fact about contradiction, not about alignment.

    2. DOES ALIGNMENT CHANGE IT?       aligned excess minus base excess.
       This is the M02 question and it is paired within group.

**Q1 IS RUN ON BOTH ARMS SEPARATELY AND THE FIRST VERSION OF THIS FILE RAN IT
ONLY ON BASE.** RH caught it. "Alignment changes the excess" being null says
NOTHING about whether the aligned arm has an excess -- a null difference between
two large effects and a null difference between two zeros print identically.
The two Q1 columns are what distinguish them and one of them was missing.

**THE THREE USAS VIEWS ARE ONE LEXICON.** `meta`, `usas_fine` and `usas` are
reported separately and never pooled; a field clearing in all three is one
result, not three.

MULTIPLICITY IS REAL HERE. Hundreds of fields and thousands of words are tested;
Holm is applied WITHIN each source and the uncorrected count is printed beside
it so the reader can see how much of the table is noise.
"""
import collections, json, math, os, statistics as st

HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP))
SRC=os.path.join(CAMP,"results","both_vs_controls_fields.json")
OUT=os.path.join(CAMP,"results","both_excess.json")
MIN_GROUPS=8


def sign_test(v):
    v=[x for x in v if x!=0.0]; n,k=len(v),sum(1 for x in v if x>0)
    if not n: return 0,0,float("nan")
    t=min(k,n-k)
    return n,k,min(1.0,2*sum(math.comb(n,i) for i in range(t+1))/2**n)


def holm(items):
    """[(key, p, ...)] -> {key: adjusted p}. Within-source, never across."""
    s=sorted(items,key=lambda x:x[1]); m=len(s); out={}; run=0.0
    for i,(k,p,*_) in enumerate(s):
        run=max(run,min(1.0,(m-i)*p)); out[k]=run
    return out


def main():
    d=json.load(open(SRC))
    cells={tuple(k.split("|")):v for k,v in d["cells"].items()}
    groups=sorted({g for (_,g,_) in cells})
    print("%s passages, %d groups, %d cells"%(format(d["_meta"]["passages"],","),
                                              len(groups),len(cells)))
    def rates(arm,gid,role,kind):
        c=cells.get((arm,gid,role))
        if not c or not c["n_content"]: return None
        n=c["n_content"]
        return {k:v/n for k,v in c[kind].items()}
    #: excess per (arm, group)
    exc={"base":collections.defaultdict(dict),"aligned":collections.defaultdict(dict)}
    for arm in ("base","aligned"):
        for gid in groups:
            for kind in ("fields","words"):
                B=rates(arm,gid,"BOTH",kind); A=rates(arm,gid,"POLE_A",kind)
                Bb=rates(arm,gid,"POLE_B",kind)
                if not (B and A and Bb): continue
                keys=set(B)|set(A)|set(Bb)
                for k in keys:
                    e=B.get(k,0.0)-0.5*(A.get(k,0.0)+Bb.get(k,0.0))
                    exc[arm][k][gid]=e
    res={}
    for label,getter in (("Q1a excess in BASE, against zero",
                          lambda k: list(exc["base"].get(k,{}).values())),
                         ("Q1b excess in ALIGNED, against zero",
                          lambda k: list(exc["aligned"].get(k,{}).values())),
                         ("Q2 alignment CHANGES the excess, paired by group",
                          lambda k: [exc["aligned"][k][g]-exc["base"][k][g]
                                     for g in exc["base"].get(k,{})
                                     if g in exc["aligned"].get(k,{})])):
        print("\n"+"="*78); print(label); print("="*78)
        bysrc=collections.defaultdict(list)
        for k in set(exc["base"])|set(exc["aligned"]):
            src=k.split(":")[0] if ":" in k else "word"
            v=getter(k)
            if len(v)<MIN_GROUPS: continue
            n,u,p=sign_test(v)
            if n<MIN_GROUPS: continue
            bysrc[src].append((k,p,u,n,st.mean(v)))
        for src in sorted(bysrc):
            items=bysrc[src]; adj=holm(items)
            sig=[x for x in items if adj[x[0]]<0.05]
            print("\n  %-12s %d tested, %d p<0.05 uncorrected, %d SURVIVING HOLM"
                  %(src,len(items),sum(1 for x in items if x[1]<0.05),len(sig)))
            for k,p,u,n,m in sorted(sig,key=lambda x:x[1])[:8]:
                print("     %-46s %2d/%-2d up  mean %+.5f  holm %.4g"
                      %(k,u,n,m,adj[k]))
        res[label]={s:[{"key":k,"p":p,"up":u,"n":n,"mean":m,
                        "holm":holm(bysrc[s])[k]} for k,p,u,n,m in bysrc[s]
                       if holm(bysrc[s])[k]<0.05] for s in bysrc}
    json.dump({"_meta":d["_meta"],"min_groups":MIN_GROUPS,"result":res},
              open(OUT,"w"),indent=1)
    print("\n-> %s"%os.path.relpath(OUT,ROOT))


if __name__=="__main__":
    main()
