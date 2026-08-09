"""IS IT PICKING A POLE AND DISAGREEING ABOUT WHICH ONE?

RH's reading: if the motion runs along the axis in either direction, each model
resolves the contradiction and they differ about the direction, so the average
cancels. That predicts BIMODALITY -- a clump of models strongly positive and a
clump strongly negative -- and it is testable.

THE MEASURES NEST, which is what makes base and aligned comparable:

    position(arm) = centroid(arm) . a        mass-weighted place on the axis
    Delta . a     = position(aligned) - position(base)
    cos(Delta, a) = (Delta . a) / |Delta|

so the base arm, the aligned arm, and the shift are all on ONE scale, and the
pole words themselves anchor it: a distribution sitting entirely on `love` scores
cos(V(love), a).
"""
import collections, pickle, json, os, sys, difflib, statistics as st
ROOT="/Users/rj416/github/malign-logits"; sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
import numpy as np
from scipy import stats as sst
from malign_logits.cache import CacheManager
SCR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results")
D=pickle.load(open(SCR+"/dp.pkl","rb"))
DP,VOC,E,AX,by=D["DP"],D["VOC"],D["E"],D["AX"],D["by"]
IX={w:i for i,w in enumerate(VOC)}
Q={q["group"]:q for q in json.load(open(ROOT+"/data/f11_quintuplets.json"))["quintuplets"]}
GS=sorted(DP); cm=CacheManager(); pairs=json.load(open(ROOT+"/data/base_aligned_pairs.json"))
POS=collections.defaultdict(list)
for pr in pairs:
    for g in GS:
        cand=set(by[g]); d={}
        for arm,mid in (("base",pr["base"]),("aligned",pr["aligned"])):
            v=cm.get_true_word_probs(mid,Q[g]["both"])
            if not v or not v.get("rows"): continue
            c={x["word"]:x["p"] for x in v["rows"] if x["word"] in cand}
            if not c: continue
            tot=sum(c.values())
            cen=sum((p/tot)*E[IX[w]] for w,p in c.items()); d[arm]=float(cen@AX[g])
        if len(d)==2: POS[g].append((pr["family"],d["base"],d["aligned"]))
def hist(v,lo=-0.25,hi=0.25,n=25):
    b=np.clip(((np.array(v)-lo)/(hi-lo)*n).astype(int),0,n-1)
    c=np.bincount(b,minlength=n); mx=max(c.max(),1)
    return "".join(" .:-=+*#%@"[min(9,int(9*x/mx))] for x in c)
print("MASS-WEIGHTED POSITION ON THE POLE AXIS, 46 model pairs per group")
print("   scale: %+.2f .......... 0 .......... %+.2f    (bar = 25 bins)"%(-0.25,0.25))
print("   %-16s %-6s %25s %8s %8s %7s %7s"%("group","arm","histogram","mean","sd","p10","p90"))
allb=[];alla=[]
for g in GS:
    b=[x[1] for x in POS[g]]; a=[x[2] for x in POS[g]]; allb+=b; alla+=a
    ta,tb=None,None
    for lab,v in (("base",b),("aligned",a)):
        print("   %-16s %-6s %25s %+8.4f %8.4f %+7.3f %+7.3f"
              %(g if lab=="base" else "",lab,hist(v),st.mean(v),st.pstdev(v),np.percentile(v,10),np.percentile(v,90)))
print("\n   POOLED  base mean %+.4f sd %.4f   |   aligned mean %+.4f sd %.4f"
      %(st.mean(allb),st.pstdev(allb),st.mean(alla),st.pstdev(alla)))
print("\n" + "="*104)
print("WHERE A GENUINE POLE-PICK WOULD SIT: the pole words' own positions")
print("="*104)
print("   %-16s %-22s %8s %8s   %s"%("group","pole_a / pole_b term","pos(a)","pos(b)","observed range of the 46 models"))
def terms(q):
    A,B=q["pole_a"].split(),q["pole_b"].split(); sm=difflib.SequenceMatcher(None,A,B); da=[];db=[]
    for t,i1,i2,j1,j2 in sm.get_opcodes():
        if t!="equal": da+=A[i1:i2]; db+=B[j1:j2]
    return " ".join(da)," ".join(db)
for g in GS:
    ta,tb=terms(Q[g])
    pa=float(E[IX[ta]]@AX[g]) if ta in IX else float("nan")
    pb=float(E[IX[tb]]@AX[g]) if tb in IX else float("nan")
    v=[x[2] for x in POS[g]]
    print("   %-16s %-22s %+8.3f %+8.3f   %+.3f to %+.3f"%(g,"%s / %s"%(ta,tb),pa,pb,min(v),max(v)))
print("\n" + "="*104)
print("BIMODALITY OF THE SHIFT: is Delta.a two clumps or one?")
print("="*104)
print("   %-16s %25s %8s %8s %8s %8s"%("group","histogram of Delta.a","mean","sd","dip p","BC"))
for g in GS:
    d=[x[2]-x[1] for x in POS[g]]
    sk=sst.skew(d); ku=sst.kurtosis(d,fisher=False)
    bc=(sk**2+1)/ku if ku>0 else float("nan")   # >0.555 suggests bimodal
    try:
        from diptest import diptest
        _,dp=diptest(np.array(d))
    except Exception:
        dp=float("nan")
    print("   %-16s %25s %+8.4f %8.4f %8s %8.3f"
          %(g,hist(d,-0.1,0.1,25),st.mean(d),st.pstdev(d),"%.3f"%dp if dp==dp else "n/a",bc))
alld=[x[2]-x[1] for g in GS for x in POS[g]]
print("\n   pooled Delta.a: mean %+.5f  sd %.5f   %.0f%% positive"%(st.mean(alld),st.pstdev(alld),100*np.mean(np.array(alld)>0)))
print("   BC > 0.555 indicates bimodality; uniform = 1.0, normal = 0.333")
