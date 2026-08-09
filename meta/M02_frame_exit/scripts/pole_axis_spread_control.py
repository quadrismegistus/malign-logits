"""Does the between-model SPREAD increase belong to the contradiction's own axis?

Alignment roughly doubles the between-model sd of the mass-weighted position, in
13 of 13 groups. That could be entirely PEAKEDNESS: effective support falls from
55 words to 44, and a peakier distribution's centroid sits further from the
vocabulary mean in EVERY direction, so its position varies more on any axis.

The control is the one that settled the straddle test: run the identical
calculation on the 12 FOREIGN axes. If the own-axis ratio matches the foreign
ratio, the doubling is sharpening and says nothing about the opposition.
"""
import collections, pickle, json, os, sys, statistics as st
ROOT="/Users/rj416/github/malign-logits"; sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
import numpy as np
from scipy import stats as sst
from malign_logits.cache import CacheManager
SCR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results")
D=pickle.load(open(os.path.join(SCR,"dp.pkl"),"rb"))
VOC,E,AX,by=D["VOC"],D["E"],D["AX"],D["by"]
IX={w:i for i,w in enumerate(VOC)}
Q={q["group"]:q for q in json.load(open(ROOT+"/data/f11_quintuplets.json"))["quintuplets"]}
GS=sorted(AX); A=np.stack([AX[g] for g in GS])
cm=CacheManager(); pairs=json.load(open(ROOT+"/data/base_aligned_pairs.json"))
P=collections.defaultdict(lambda: {"base":[], "aligned":[]})
for pr in pairs:
    for g in GS:
        cand=set(by[g]); d={}
        for arm,mid in (("base",pr["base"]),("aligned",pr["aligned"])):
            v=cm.get_true_word_probs(mid,Q[g]["both"])
            if not v or not v.get("rows"): continue
            c={x["word"]:x["p"] for x in v["rows"] if x["word"] in cand}
            if len(c)<3: continue
            ws=list(c); p=np.array([c[w] for w in ws]); p/=p.sum()
            d[arm]=(p@E[[IX[w] for w in ws]])@A.T          # position on ALL 13 axes
        if len(d)==2:
            P[g]["base"].append(d["base"]); P[g]["aligned"].append(d["aligned"])
print("BETWEEN-MODEL SD OF POSITION, own axis against the 12 foreign axes")
print("   %-16s %8s %8s %7s   %8s %8s %7s   %8s"
      %("group","own b","own a","ratio","frgn b","frgn a","ratio","own/frgn"))
ro,rf=[],[]
for gi,g in enumerate(GS):
    B=np.stack(P[g]["base"]); Aa=np.stack(P[g]["aligned"])      # (46, 13)
    sb=B.std(0,ddof=1); sa=Aa.std(0,ddof=1); r=sa/sb
    fo=[j for j in range(len(GS)) if j!=gi]
    ro.append(r[gi]); rf.append(float(np.mean(r[fo])))
    print("   %-16s %8.4f %8.4f %7.2f   %8.4f %8.4f %7.2f   %8.2f"
          %(g,sb[gi],sa[gi],r[gi],sb[fo].mean(),sa[fo].mean(),np.mean(r[fo]),r[gi]/np.mean(r[fo])))
ro=np.array(ro); rf=np.array(rf)
t,p=sst.ttest_rel(ro,rf)
print("\n   own-axis ratio      mean %.3f   (13 of 13 above 1.0: %s)"%(ro.mean(),str(bool((ro>1).all()))))
print("   foreign-axis ratio  mean %.3f   (13 of 13 above 1.0: %s)"%(rf.mean(),str(bool((rf>1).all()))))
print("   paired t on own - foreign: %+.4f   t=%+.2f  p=%.4f"%((ro-rf).mean(),t,p))
print("   own exceeds foreign in %d of %d groups"%(int((ro>rf).sum()),len(GS)))
print("\n   -> equal ratios = the doubling is uniform sharpening, not about the opposition.")
