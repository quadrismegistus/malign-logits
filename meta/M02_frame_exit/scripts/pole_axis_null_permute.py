"""THIRD NULL: permute the risers and fallers.

Hold the cell's vocabulary, its total mass moved, and the whole magnitude
distribution of Dp fixed; shuffle WHICH WORD gets which Dp. This breaks only the
word <-> movement pairing, which is exactly the thing under test. It is stricter
than the mismatched-axis null (which left the pairing intact) and it can test the
ABSOLUTE statistic, which the sign-flip null could not.

Permutation is over the cell's OBSERVED SUPPORT -- the words actually present in
one arm or the other for that model pair -- so the null cannot assign movement to
a word that never appeared for that model.

    cos  = (Dp . s) / sqrt(Dp' G Dp)     s = E_g a,  G = E_g E_g'

so a permutation costs O(|support|^2) and never materialises a 1024-d vector.
"""
import os, collections, pickle, statistics as st
import numpy as np
from scipy import stats as sst
SCR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results")
D=pickle.load(open(SCR+"/dp.pkl","rb"))
DP,VOC,E,AX=D["DP"],D["VOC"],D["E"],D["AX"]
IX={w:i for i,w in enumerate(VOC)}
rng=np.random.default_rng(4946); B=1000
GS=sorted(DP)
rows=[]      # (group, family, obs_cos, null_cos_mean, null_cos_sd, obs_abs, null_abs_mean, null_abs_sd)
for g in GS:
    ws=sorted({w for _,c in DP[g] for w in c}); wi=[IX[w] for w in ws]
    Eg=E[wi]; G=Eg@Eg.T; s=Eg@AX[g]; pos={w:i for i,w in enumerate(ws)}
    for f,c in DP[g]:
        sup=[pos[w] for w,v in c.items() if abs(v)>1e-12]
        if len(sup)<3: continue
        vals=np.array([c[ws[i]] for i in sup])
        Gs=G[np.ix_(sup,sup)]; ss=s[sup]
        den=np.sqrt(vals@Gs@vals)
        if den<1e-12: continue
        obs=float(vals@ss/den)
        P=np.stack([rng.permutation(vals) for _ in range(B)])
        nn=P@ss; dd=np.sqrt(((P@Gs)*P).sum(1))
        nc=nn/np.maximum(dd,1e-12)
        rows.append((g,f,obs,nc.mean(),nc.std(),abs(obs),np.abs(nc).mean(),np.abs(nc).std()))
fs=sorted({r[1] for r in rows})
def clus(v):
    M=np.zeros((len(fs),len(rows)))
    for j,r in enumerate(rows): M[fs.index(r[1]),j]=1.
    M/=M.sum(1,keepdims=True); return M@np.asarray(v)
obs=np.array([r[2] for r in rows]); nm=np.array([r[3] for r in rows]); nsd=np.array([r[4] for r in rows])
ao=np.array([r[5] for r in rows]); anm=np.array([r[6] for r in rows]); ansd=np.array([r[7] for r in rows])
print("RISER/FALLER PERMUTATION NULL   %d cells, %d families, %d groups, %d permutations each"
      %(len(rows),len(fs),len(GS),B))
for lab,o,m2,sd in (("SIGNED  cos (toward the GOOD pole)",obs,nm,nsd),
                    ("ABSOLUTE |cos| (toward EITHER pole)",ao,anm,ansd)):
    d=o-m2; t,p=sst.ttest_1samp(clus(d),0)
    z=np.mean(d/np.maximum(sd,1e-12))
    print("\n   %s"%lab)
    print("      observed   %.4f"%o.mean())
    print("      null       %.4f   (permuted risers/fallers, same magnitudes)"%m2.mean())
    print("      excess    %+.4f   family-clustered t=%+.2f  p=%.4f   mean per-cell z %+.2f"%(d.mean(),t,p,z))
    print("      observed exceeds its own null in %.0f%% of cells"%(100*(d>0).mean()))
print("\n   %-18s %8s %8s %8s   %8s %8s %8s"%("group","cos","null","excess","|cos|","null","excess"))
for g in GS:
    ix=[i for i,r in enumerate(rows) if r[0]==g]
    print("   %-18s %+8.4f %+8.4f %+8.4f   %8.4f %8.4f %+8.4f"
          %(g,obs[ix].mean(),nm[ix].mean(),(obs-nm)[ix].mean(),ao[ix].mean(),anm[ix].mean(),(ao-anm)[ix].mean()))
print("\nPOOLED PER GROUP (aggregate Dp over the 46 pairs, then permute):")
po=[]
for g in GS:
    ws=sorted({w for _,c in DP[g] for w in c}); wi=[IX[w] for w in ws]
    Eg=E[wi]; G=Eg@Eg.T; s=Eg@AX[g]
    acc=collections.Counter()
    for f,c in DP[g]:
        for w,v in c.items(): acc[w]+=v
    vals=np.array([acc[w] for w in ws])
    den=np.sqrt(vals@G@vals); obs1=float(vals@s/den)
    P=np.stack([rng.permutation(vals) for _ in range(4000)])
    nc=(P@s)/np.sqrt(((P@G)*P).sum(1))
    pv=(np.sum(np.abs(nc)>=abs(obs1))+1)/4001
    po.append((g,obs1,(obs1-nc.mean())/nc.std(),pv,abs(obs1),(abs(obs1)-np.abs(nc).mean())/np.abs(nc).std()))
    print("   %-18s cos %+7.3f  z %+6.2f  p %.4f   |cos| %6.3f  z %+6.2f"%po[-1])
print("\n   %d of %d groups with |cos| above its permutation null"%(sum(1 for x in po if x[5]>0),len(po)))
print("   mean signed z %+.2f   mean |cos| z %+.2f"%(st.mean([x[2] for x in po]),st.mean([x[5] for x in po])))
