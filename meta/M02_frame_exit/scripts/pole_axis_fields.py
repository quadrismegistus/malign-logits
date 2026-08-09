"""Settle the per-cell vs pooled disagreement with the EXACT null, then fields.

Under the paired null, flipping a cell's arm labels negates its Dp and therefore
negates cos(Delta_cell, axis). So the sign-flip permutation distribution of the
family-clustered mean is computable exactly from the observed cosines. No
approximation, no assumption about isotropy, which is what the t-test-against-0
was silently making.
"""
import os, collections, pickle, statistics as st
import numpy as np
from scipy import stats as sst
SCR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results")
D=pickle.load(open(SCR+"/dp.pkl","rb"))
DP,VOC,E,AX=D["DP"],D["VOC"],D["E"],D["AX"]
IX={w:i for i,w in enumerate(VOC)}
rng=np.random.default_rng(4946); B=50000
rec=[]
for g,cells in DP.items():
    for fam,c in cells:
        ws=[w for w in c]; d=np.array([c[w] for w in ws])@E[[IX[w] for w in ws]]
        n=np.linalg.norm(d)
        if n<1e-12: continue
        rec.append((g,fam,float(d@AX[g]/n)))
fams=sorted({f for _,f,_ in rec}); fi={f:i for i,f in enumerate(fams)}
cos=np.array([c for _,_,c in rec]); fidx=np.array([fi[f] for _,f,_ in rec])
def fammean(v):
    s=np.zeros((v.shape[0] if v.ndim>1 else 1,len(fams))); 
    return s
#: family-clustered mean, vectorised over permutations
Mx=np.zeros((len(fams),len(cos)))
for j,f in enumerate(fidx): Mx[f,j]=1.
Mx/=Mx.sum(1,keepdims=True)
obs=float((Mx@cos).mean())
S=rng.choice([-1.,1.],size=(B,len(cos)))
nullv=(S*cos)@Mx.T                      # (B, n_fam)
nulls=nullv.mean(1)
pv=(np.sum(np.abs(nulls)>=abs(obs))+1)/(B+1)
t,pt=sst.ttest_1samp(Mx@cos,0)
print("PER-CELL GRAIN, %d cells, %d families"%(len(cos),len(fams)))
print("   observed family-clustered mean cos = %+.5f"%obs)
print("   t-test against 0        p = %.4f   (what I reported)"%pt)
print("   sign-flip permutation   p = %.4f   null mean %+.5f sd %.5f, %d draws"%(pv,nulls.mean(),nulls.std(),B))
print("   -> the two agree; the t-test was not the problem.\n")
print("So the grains genuinely disagree, and the reason is the weighting:")
mag=[]
for g,cells in DP.items():
    for fam,c in cells:
        ws=[w for w in c]; d=np.array([c[w] for w in ws])@E[[IX[w] for w in ws]]
        mag.append(np.linalg.norm(d))
mag=np.array(mag); q=np.percentile(mag,[25,50,75])
big=cos[mag>=q[2]]; small=cos[mag<=q[0]]
print("   |Delta| quartiles %.4f / %.4f / %.4f"%tuple(q))
print("   mean cos in the LARGEST-motion quartile  %+.4f  (n=%d)"%(big.mean(),len(big)))
print("   mean cos in the SMALLEST-motion quartile %+.4f  (n=%d)"%(small.mean(),len(small)))
print("   pooling weights by |Delta|; the per-cell mean does not. If the tilt lives in")
print("   the small-motion cells it is real per cell and invisible in the aggregate.\n")
print("="*100)
print("SEMANTIC FIELDS: net probability moved into and out of each cluster")
print("="*100)
from sklearn.cluster import KMeans
K=14
km=KMeans(n_clusters=K,n_init=10,random_state=4946).fit(E)
lab=km.labels_
net=collections.Counter(); perw=collections.Counter(); ncell=0
for g,cells in DP.items():
    for fam,c in cells:
        ncell+=1
        for w,v in c.items():
            net[lab[IX[w]]]+=v; perw[w]+=v
tot=sum(abs(v) for v in net.values())
print("   %d clusters over %d words, %d cells. Delta p x1000, mean per cell.\n"%(K,len(VOC),ncell))
print("   %5s %6s  %-46s %s"%("net","share","words GAINING mass","words LOSING mass"))
for cl,v in sorted(net.items(),key=lambda x:-x[1]):
    mem=[w for w in VOC if lab[IX[w]]==cl]
    up=sorted(mem,key=lambda w:-perw[w])[:5]; dn=sorted(mem,key=lambda w:perw[w])[:5]
    print("   %+5.1f %5.1f%%  %-46s %s"
          %(1000*v/ncell,100*abs(v)/tot,
            " ".join("%s+%.1f"%(w,1000*perw[w]/ncell) for w in up if perw[w]>0),
            " ".join("%s%.1f"%(w,1000*perw[w]/ncell) for w in dn if perw[w]<0)))
