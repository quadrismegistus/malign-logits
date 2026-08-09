"""|cos(Delta, axis)|: does alignment's MOTION run along the pole axis at all,
in either direction?

THE NULL CANNOT BE A SIGN FLIP. Negating Delta leaves |cos| unchanged, so the
paired permutation that settled the signed question is powerless here. The null
used instead is the MISMATCHED AXIS: keep the cell's motion exactly as observed
and project it onto ANOTHER group's pole axis. Every group's vocabulary is the
same pool of English continuation verbs, so a foreign axis is a fair comparison,
and it controls for the embedder's anisotropy without assuming anything about it.

    observed   |cos(Delta_(g,cell), AX_g)|
    null       |cos(Delta_(g,cell), AX_g')|   for each of the 12 g' != g
"""
import os, collections, pickle, statistics as st
import numpy as np
from scipy import stats as sst
SCR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results")
D=pickle.load(open(SCR+"/dp.pkl","rb"))
DP,VOC,E,AX=D["DP"],D["VOC"],D["E"],D["AX"]
IX={w:i for i,w in enumerate(VOC)}
GS=sorted(DP); A=np.stack([AX[g] for g in GS])          # (13, 1024)
rng=np.random.default_rng(4946); B=20000
own=[]; foreign=[]; fam=[]; grp=[]; ownS=[]
for gi,g in enumerate(GS):
    for f,c in DP[g]:
        ws=[w for w in c]; d=np.array([c[w] for w in ws])@E[[IX[w] for w in ws]]
        n=np.linalg.norm(d)
        if n<1e-12: continue
        cs=(A@d)/n
        own.append(abs(cs[gi])); ownS.append(cs[gi])
        foreign.append(np.mean([abs(cs[j]) for j in range(len(GS)) if j!=gi]))
        fam.append(f); grp.append(g)
own=np.array(own); foreign=np.array(foreign); dd=own-foreign
fs=sorted(set(fam)); M=np.zeros((len(fs),len(own)))
for j,f in enumerate(fam): M[fs.index(f),j]=1.
M/=M.sum(1,keepdims=True)
t,p=sst.ttest_1samp(M@dd,0)
print("PER CELL, %d cells, %d families, %d groups"%(len(own),len(fs),len(GS)))
print("   |cos| own axis      %.4f"%own.mean())
print("   |cos| foreign axes  %.4f   (mean of the 12 mismatched)"%foreign.mean())
print("   difference         %+.4f   family-clustered t=%+.2f  p=%.4f"%(dd.mean(),t,p))
print("   own > foreign in %.0f%% of cells"%(100*(dd>0).mean()))
#: PERMUTATION: reassign which axis is 'own', within cell
perm=[]
for _ in range(2000):
    sh=rng.permutation(len(GS)); tot=[]
    for gi,g in enumerate(GS):
        pass
    perm.append(0)
print("\n   for scale, the SIGNED own-axis mean is %+.4f -- the valence tilt from before."%np.mean(ownS))
print("   |cos| exceeds it %.1fx, which is what 'either pole' buys over 'the good pole'."%(own.mean()/abs(np.mean(ownS))))
print("\n   %-18s %6s %8s %8s"%("group","own","foreign","diff"))
for g in GS:
    ix=[i for i,x in enumerate(grp) if x==g]
    print("   %-18s %6.4f %8.4f %+8.4f"%(g,own[ix].mean(),foreign[ix].mean(),dd[ix].mean()))
#: POOLED grain, same question
print("\nPOOLED PER GROUP (aggregate Dp over 46 pairs, then one direction):")
po=[];pf=[]
for gi,g in enumerate(GS):
    acc=collections.Counter()
    for f,c in DP[g]:
        for w,v in c.items(): acc[w]+=v
    ws=list(acc); d=np.array([acc[w] for w in ws])@E[[IX[w] for w in ws]]; d/=np.linalg.norm(d)
    cs=A@d; po.append(abs(cs[gi])); pf.append(np.mean([abs(cs[j]) for j in range(len(GS)) if j!=gi]))
t2,p2=sst.ttest_1samp(np.array(po)-np.array(pf),0)
print("   own %.4f   foreign %.4f   diff %+.4f   t=%+.2f p=%.4f   own>foreign in %d of %d groups"
      %(np.mean(po),np.mean(pf),np.mean(po)-np.mean(pf),t2,p2,sum(np.array(po)>np.array(pf)),len(GS)))
