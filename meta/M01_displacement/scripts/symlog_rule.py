#!/usr/bin/env python
"""SYMLOG: a symmetric riser/faller rule, tried against CANONICAL on one battery.

    uv run python symlog_rule.py

WHY. CANONICAL uses DIFFERENT CRITERIA IN THE TWO DIRECTIONS, which is the defect
under most of 2026-08-11's damage:

    faller  iff  P >= 0.003  and  Q < 0.5 * P            a RATIO, no absolute floor
    riser   iff  max(P,Q) > 0.003  and  Q - P > 0.003    an ABSOLUTE gain
                 and  Q > null                           and only risers face this

A word going 0.0040 -> 0.0019 is a faller (it lost 0.0021). A word going
0.0019 -> 0.0040 is NOT a riser (it gained 0.0021 < 0.003). Same movement,
opposite direction, opposite verdict. That single inconsistency manufactures
fall-dominance, and it is inherited by every population contrast built on the
classes.

THE RULE. Movement is a QUANTITY, not a class:

    m = log2( (Q + eps) / (P + eps) )        eps = THETA/10
    eligible  iff  max(P,Q) >= THETA         one floor, arm-neutral
    faller    iff  eligible and m <= -TAU
    riser     iff  eligible and m >= +TAU

Same criterion in both signs. Scale-free, so it stops being a frequency ranking.
The smoothing keeps words absent from one arm finite instead of infinite; a word
absent from BASE still cannot fall, but that is the world, not the rule.

NOT A REPLACEMENT FOR CANONICAL. 41 committed findings are computed on CANONICAL
and swapping the rule underneath them converts a clean corpus into an
unauditable one. This is for new work, and as a DIAGNOSTIC: which existing
results are rule-dependent?
"""
import collections, csv, json, os, sys
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,ROOT)
import math
from malign_logits.step import Step
from malign_logits.checkpoint import Checkpoint
from malign_logits.movement import CANONICAL

THETA=0.003; EPS=THETA/10.0; TAU=1.0; DELTA=0.003
BATTERY=os.path.join(ROOT,"data","beam_sample_105_plus_anger.csv")
PAIRS=os.path.join(ROOT,"data","lineage_representative_pairs.txt")
OUT=os.path.join(CAMP,"results","symlog_vs_canonical.json")
FUNC=set("the a an and or but of to in on at for with from by he she it they "
         "i we you him her them his hers its their my your our me us was were "
         "is are be been being had has have do does did not no so then now "
         "there here that this these those as if when while all some any "
         "one two".split())

def symabs(P,Q):
    """The OTHER symmetric rule: same ABSOLUTE threshold in both directions.

    SYMLOG and SYMABS correct CANONICAL in OPPOSITE directions, which is the
    point of running both. CANONICAL is a ratio on the faller side and an
    absolute gain on the riser side; symmetrising in ratio space tightens
    risers, symmetrising in absolute space tightens fallers. Neither is more
    "correct" a priori -- so the question that matters is which conclusions
    survive BOTH.
    """
    out={}
    for k in set(P)|set(Q):
        if k=="__residual__": continue
        p,q=P.get(k,0.0),Q.get(k,0.0)
        if max(p,q)<THETA: continue
        d=q-p
        if d<=-DELTA: out[k]="faller"
        elif d>=DELTA: out[k]="riser"
    return out


def symlog(P,Q):
    out={}
    for k in set(P)|set(Q):
        if k=="__residual__": continue
        p,q=P.get(k,0.0),Q.get(k,0.0)
        if max(p,q)<THETA: continue
        m=math.log2((q+EPS)/(p+EPS))
        if m<=-TAU: out[k]=("faller",m)
        elif m>=TAU: out[k]=("riser",m)
    return out

def main():
    prompts=[r["prompt"].strip() for r in csv.DictReader(open(BATTERY))]
    pairs=[l.strip().split(">") for l in open(PAIRS) if l.strip()]
    C=collections.Counter(); S=collections.Counter()
    Ccls=collections.defaultdict(lambda:[0,0]); Scls=collections.defaultdict(lambda:[0,0])
    B=collections.Counter(); Bcls=collections.defaultdict(lambda:[0,0])
    caps=collections.defaultdict(lambda:[0,0])     # CAP forms under SYMLOG
    cells=0
    for b,a in pairs:
        try: stp=Step(Checkpoint(b),Checkpoint(a))
        except Exception: continue
        for pr in prompts:
            try: mv=stp.cell(pr).movement(CANONICAL)
            except Exception: continue
            if mv is None or not mv.pre or not mv.post: continue
            cells+=1
            for w in mv.fallers:
                C["fall"]+=1; Ccls["FUNC" if w.lower() in FUNC else "CONTENT"][1]+=1
            for w in mv.risers:
                C["rise"]+=1; Ccls["FUNC" if w.lower() in FUNC else "CONTENT"][0]+=1
            for w,(role,m) in symlog(mv.pre,mv.post).items():
                S["fall" if role=="faller" else "rise"]+=1
                c="FUNC" if w.lower() in FUNC else "CONTENT"
                Scls[c][1 if role=="faller" else 0]+=1
                if w[:1].isupper(): caps[w][1 if role=="faller" else 0]+=1
            for w,role in symabs(mv.pre,mv.post).items():
                B["fall" if role=="faller" else "rise"]+=1
                Bcls["FUNC" if w.lower() in FUNC else "CONTENT"][1 if role=="faller" else 0]+=1
        print("  %-44s cells %d"%(b.split('/')[-1],cells),flush=True)
    print("\n%d cells   THETA %.4g  EPS %.4g  TAU %.2f\n"%(cells,THETA,EPS,TAU))
    for nm,c,cc in (("CANONICAL",C,Ccls),("SYMLOG ratio",S,Scls),("SYMABS delta",B,Bcls)):
        t=c["rise"]+c["fall"]
        print("%-10s rises %9s  falls %9s   FALL SHARE %.1f%%"
              %(nm,format(c["rise"],","),format(c["fall"],","),100*c["fall"]/t if t else 0))
        for k in ("FUNC","CONTENT"):
            r,f=cc[k]
            print("             %-8s rise %8s fall %8s  net %+8d  fall share %.1f%%"
                  %(k,format(r,","),format(f,","),r-f,100*f/(r+f) if r+f else 0))
    capf=sum(1 for w,(r,f) in caps.items() if f>0)
    print("\nCAPITALISED FORMS UNDER SYMLOG: %d distinct, %d ever classed a FALLER"
          %(len(caps),capf))
    json.dump({"cells":cells,"theta":THETA,"eps":EPS,"tau":TAU,
               "canonical":dict(C),"symlog":dict(S),
               "symabs":dict(B),"canonical_class":{k:v for k,v in Ccls.items()},
               "symabs_class":{k:v for k,v in Bcls.items()},
               "symlog_class":{k:v for k,v in Scls.items()},
               "cap_forms":len(caps),"cap_ever_faller":capf},open(OUT,"w"),indent=1)
    print("-> %s"%os.path.relpath(OUT,ROOT))

if __name__=="__main__": main()
