"""THE ACTUAL F11 TEST AT THE NEXT-WORD GRAIN.

Oedipalization is not a claim about where the mean sits. It is a claim about the
SHAPE of the distribution over the axis: inclusive disjunction ("either ... or
... or") keeps mass on both sides of the opposition, exclusive disjunction
("either/or") collapses it onto one side. So the two quantities are

    position   = E_p[s]                s(w) = V(w) . a
    axisvar    = Var_p[s]              how much the distribution STRADDLES

and they separate the three hypotheses:

    Oedipalization      |position| UP,    axisvar DOWN
    frame exit          |position| -> 0,  axisvar DOWN
    superposition held  both flat

THE ENTROPY CONFOUND, and its control. Alignment lowers entropy, which shrinks
Var_p along EVERY direction, so a bare axisvar drop proves nothing. The control
is the SHARE: axisvar / totalvar, where totalvar = E_p|V - centroid|^2 is the
same shrinkage measured in all 1024 directions at once. A share that falls is
collapse onto the axis specifically; a share that holds is uniform sharpening.
Reported beside the 12 FOREIGN axes, which give the share's chance level.
"""
import collections, pickle, json, os, sys, statistics as st
ROOT="/Users/rj416/github/malign-logits"; sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
import numpy as np
from scipy import stats as sst
from malign_logits.cache import CacheManager
SCR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results")
D=pickle.load(open(SCR+"/dp.pkl","rb"))
VOC,E,AX,by=D["VOC"],D["E"],D["AX"],D["by"]
IX={w:i for i,w in enumerate(VOC)}
Q={q["group"]:q for q in json.load(open(ROOT+"/data/f11_quintuplets.json"))["quintuplets"]}
GS=sorted(AX); A=np.stack([AX[g] for g in GS])
cm=CacheManager(); pairs=json.load(open(ROOT+"/data/base_aligned_pairs.json"))
R=[]
for pr in pairs:
    for gi,g in enumerate(GS):
        cand=set(by[g]); d={}
        for arm,mid in (("base",pr["base"]),("aligned",pr["aligned"])):
            v=cm.get_true_word_probs(mid,Q[g]["both"])
            if not v or not v.get("rows"): continue
            c={x["word"]:x["p"] for x in v["rows"] if x["word"] in cand}
            if len(c)<3: continue
            ws=list(c); p=np.array([c[w] for w in ws]); p/=p.sum()
            Ew=E[[IX[w] for w in ws]]
            cen=p@Ew
            totv=float(p@((Ew-cen)**2).sum(1))
            S=Ew@A.T                                     # (|ws|, 13)
            pos=p@S; av=p@((S-pos)**2)
            eff=float(np.exp(-(p*np.log(np.maximum(p,1e-300))).sum()))
            side=float(p[S[:,gi]>0].sum())
            d[arm]=dict(pos=pos[gi],av=av[gi],share=av[gi]/totv,eff=eff,side=side,
                        fpos=np.mean([abs(pos[j]) for j in range(len(GS)) if j!=gi]),
                        fav=np.mean([av[j] for j in range(len(GS)) if j!=gi]),
                        fshare=np.mean([av[j]/totv for j in range(len(GS)) if j!=gi]))
        if len(d)==2: R.append((pr["family"],g,d))
fs=sorted({r[0] for r in R})
def cl(v):
    M=np.zeros((len(fs),len(R)))
    for j,r in enumerate(R): M[fs.index(r[0]),j]=1.
    M/=M.sum(1,keepdims=True); return M@np.asarray(v)
def t(k,f=lambda x:x):
    b=np.array([f(r[2]["base"][k]) for r in R]); a=np.array([f(r[2]["aligned"][k]) for r in R])
    tt,pp=sst.ttest_1samp(cl(a-b),0); return b.mean(),a.mean(),(a-b).mean(),tt,pp
print("%d cells, %d families, %d groups\n"%(len(R),len(fs),len(GS)))
print("%-38s %10s %10s %10s %8s %9s"%("quantity","base","aligned","delta","t","p"))
print("-"*92)
for lab,k,f in (("effective support exp(H)","eff",lambda x:x),
                ("|position| on own axis","pos",abs),
                ("  |position| on foreign axes","fpos",lambda x:x),
                ("|mass split - 0.5|","side",lambda x:abs(x-.5)),
                ("axisvar, own  (STRADDLE)","av",lambda x:x),
                ("  axisvar, foreign","fav",lambda x:x),
                ("total variance (all 1024 dims)","av",None)):
    if f is None:
        b=np.array([r[2]["base"]["av"]/r[2]["base"]["share"] for r in R])
        a=np.array([r[2]["aligned"]["av"]/r[2]["aligned"]["share"] for r in R])
        tt,pp=sst.ttest_1samp(cl(a-b),0); row=(b.mean(),a.mean(),(a-b).mean(),tt,pp)
    else: row=t(k,f)
    print("%-38s %10.5f %10.5f %+10.5f %8.2f %9.4f"%(lab,*row))
print("-"*92)
for lab,k in (("SHARE own-axis var / total var","share"),("  share, foreign axes","fshare")):
    print("%-38s %10.5f %10.5f %+10.5f %8.2f %9.4f"%(lab,*t(k)))
print("\nSHARE is the confound-controlled test. Own-axis share falling faster than")
print("foreign-axis share = collapse onto THIS opposition. Equal = uniform sharpening.\n")
so=t("share"); sf=t("fshare")
print("   own delta   %+.6f   (p=%.4f)"%(so[2],so[4]))
print("   foreign     %+.6f   (p=%.4f)"%(sf[2],sf[4]))
dif=np.array([r[2]["aligned"]["share"]-r[2]["base"]["share"]-(r[2]["aligned"]["fshare"]-r[2]["base"]["fshare"]) for r in R])
tt,pp=sst.ttest_1samp(cl(dif),0)
print("   difference  %+.6f   t=%.2f  p=%.4f  <- Oedipalization would be NEGATIVE"%(dif.mean(),tt,pp))
print("\nPER GROUP: |position| and own-axis straddle share")
print("   %-16s %9s %9s %9s   %9s %9s %9s"%("group","|pos| b","|pos| a","delta","share b","share a","delta"))
for g in GS:
    x=[r for r in R if r[1]==g]
    pb=st.mean([abs(r[2]["base"]["pos"]) for r in x]); pa=st.mean([abs(r[2]["aligned"]["pos"]) for r in x])
    sb=st.mean([r[2]["base"]["share"] for r in x]); sa=st.mean([r[2]["aligned"]["share"] for r in x])
    print("   %-16s %9.4f %9.4f %+9.4f   %9.5f %9.5f %+9.5f"%(g,pb,pa,pa-pb,sb,sa,sa-sb))
