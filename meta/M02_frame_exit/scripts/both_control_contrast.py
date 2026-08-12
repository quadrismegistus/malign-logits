#!/usr/bin/env python
"""BOTH against its CONSONANT-PAIR controls. The correct contrast.

    uv run python both_control_contrast.py

RH asked for CONTROL_A / CONTROL_B. Three earlier versions of this analysis used
POLE_A / POLE_B and the substitution went unnoticed for the whole session.

    BOTH       He was beautiful and disgusting and she wanted to   two OPPOSED
    POLE_A     He was beautiful and she wanted to                  ONE adjective
    CONTROL_A  He was beautiful and radiant and she wanted to      two CONSONANT

**The poles vary FORM as well as content** -- one adjective against two, 7.5
tokens against 9.6 -- so `BOTH - mean(POLE)` confounds contradiction with prompt
length and syntactic weight. The controls hold the form fixed and vary only
whether the adjective pair is opposed. Everything form-sensitive that looked
like a contradiction effect against the poles DIES against the controls:

    base cosmos_weather   17/17 p 1.5e-05  ->  11/17 p 0.33
    base emotion_affect   15/17 p 0.0024   ->  11/17 p 0.33
    base `options`         2/17 p 0.0024   ->   7/17 p 0.63
    base `answer`          2/17 p 0.0024   ->   8/17 p 1.00

## TWO FAMILIES, TWO CORRECTIONS, AND THE DIFFERENCE IS NOT SPECIAL PLEADING

EXPLORATORY: every field and every word. BH across the whole sweep. At 17 groups
against ~17k words this is empty and was always going to be -- reported so the
emptiness is on the record rather than implied.

CONFIRMATORY: the SECOND_ORDER family imported from `z_second_order.py`, which
was written for `findings/second_order_naming.md` before this analysis existed.
BH WITHIN THE FAMILY. Correcting a pre-specified set against 25,129 exploratory
keys penalises a confirmatory test for the existence of an unrelated sweep in
the same script.

## WHAT IT FINDS

Confirmatory, arm difference, correct control: **7 of 16 survive BH within the
family, and ALL SIXTEEN have a positive mean.** This is STRONGER than the
finding it corroborates, which uses a POLE control -- the effect cannot be
attributed to prompt length or to having two adjectives, only to opposition.

Exploratory: 9 fields in the ALIGNED arm, 0 in base, 0 words anywhere, 0 on the
arm difference.
"""
import collections, json, math, os, statistics as st, sys

HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,HERE)
from z_second_order import SECOND_ORDER          # noqa: E402  the pre-specified family
SRC=os.path.join(CAMP,"results","both_vs_controls_fields_v2.json")
OUT=os.path.join(CAMP,"results","both_control_contrast.json")
MIN=8
CTRL=("CONTROL_A","CONTROL_B")
#: single-word members of SECOND_ORDER plus their inflections. FIXED BEFORE
#: LOOKING -- the stems come from the committed regex table, not from this data.
FAMILY=["torn","conflict","conflicts","conflicted","conflicting","contradiction",
        "contradictions","contradictory","contradict","paradox","paradoxical",
        "simultaneous","simultaneously","ambivalence","ambivalent","warring"]


def sign_test(v):
    v=[x for x in v if x!=0.0]; n,k=len(v),sum(1 for x in v if x>0)
    if not n: return 0,0,float("nan")
    t=min(k,n-k)
    return n,k,min(1.0,2*sum(math.comb(n,i) for i in range(t+1))/2**n)


def bh(items):
    s=sorted(items,key=lambda x:x[1]); m=len(s); out={}; run=1.0
    for i in range(m-1,-1,-1): out[s[i][0]]=run=min(run,min(1.0,s[i][1]*m/(i+1)))
    return out


def main():
    C={tuple(k.split("|")):v for k,v in json.load(open(SRC))["cells"].items()}
    groups=sorted({g for (_,_,g,_) in C})
    def rate(arm,g,role,key,kind):
        tot=0;n=0
        for sp in ("s0","s1"):
            c=C.get((sp,arm,g,role))
            if c: tot+=c[kind].get(key,0); n+=c["n_content"]
        return tot/n if n else None
    def exc(arm,key,kind):
        o={}
        for g in groups:
            B=rate(arm,g,"BOTH",key,kind)
            A=rate(arm,g,CTRL[0],key,kind); Bb=rate(arm,g,CTRL[1],key,kind)
            if None not in (B,A,Bb): o[g]=B-0.5*(A+Bb)
        return o
    res={}
    print("groups: %d   control: %s\n"%(len(groups),"/".join(CTRL)))
    print("="*72); print("CONFIRMATORY -- pre-specified SECOND_ORDER family (%d regexes)"
                         %len(SECOND_ORDER)); print("="*72)
    items=[]
    for w in FAMILY:
        b=exc("base",w,"words"); a=exc("aligned",w,"words")
        dd=[a[g]-b[g] for g in b if g in a]
        if len(dd)>=MIN:
            n,u,p=sign_test(dd); items.append((w,p,u,n,st.mean(dd)))
    q=bh(items)
    print("  %-18s %8s %12s %10s"%("word","up/n","mean","BH q (family)"))
    for w,p,u,n,m in sorted(items,key=lambda x:x[1]):
        print("  %-18s %5d/%-2d %+12.6f %10.4g%s"%(w,u,n,m,q[w]," *" if q[w]<0.05 else ""))
    print("\n  %d of %d survive; %d of %d have a POSITIVE mean"
          %(sum(1 for w in q if q[w]<0.05),len(items),
            sum(1 for x in items if x[4]>0),len(items)))
    res["confirmatory"]=[{"word":w,"up":u,"n":n,"mean":m,"bh_q":q[w]} for w,p,u,n,m in items]
    print("\n"+"="*72); print("EXPLORATORY -- every field and word, BH across the sweep")
    print("="*72)
    allk={"fields":set(),"words":set()}
    for v in C.values():
        for kind in allk: allk[kind].update(v[kind])
    res["exploratory"]={}
    for arm_label,getter in (("base",lambda k,d: list(exc("base",k,d).values())),
                             ("aligned",lambda k,d: list(exc("aligned",k,d).values())),
                             ("arm_difference",lambda k,d: [exc("aligned",k,d)[g]-exc("base",k,d)[g]
                                  for g in exc("base",k,d) if g in exc("aligned",k,d)])):
        res["exploratory"][arm_label]={}
        for kind in ("fields","words"):
            it=[]
            for key in allk[kind]:
                v=getter(key,kind)
                if len(v)<MIN: continue
                n,u,p=sign_test(v)
                if n>=MIN: it.append((key,p,u,n,st.mean(v)))
            if not it: continue
            qq=bh(it); s=sorted([x for x in it if qq[x[0]]<0.05],key=lambda x:x[1])
            print("  %-16s %-7s %6d tested | %4d raw p<.05 | %2d BH"
                  %(arm_label,kind,len(it),sum(1 for x in it if x[1]<0.05),len(s)))
            for k,p,u,n,m in s[:12]:
                print("       %-44s %2d/%-2d %+.5f q=%.4g"%(k,u,n,m,qq[k]))
            res["exploratory"][arm_label][kind]=[{"key":k,"up":u,"n":n,"mean":m,"bh_q":qq[k]}
                                                 for k,p,u,n,m in s]
    json.dump({"_meta":{"control":list(CTRL),"groups":len(groups),"min_groups":MIN,
                        "family_source":"z_second_order.SECOND_ORDER, pre-specified",
                        "source":os.path.relpath(SRC,ROOT)},"result":res},
              open(OUT,"w"),indent=1)
    print("\n-> %s"%os.path.relpath(OUT,ROOT))


if __name__=="__main__":
    main()
