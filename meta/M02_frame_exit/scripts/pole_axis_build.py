"""POOLED MASS DIRECTION: centroid(risers) - centroid(fallers), weighted by the
probability each word gains or loses, pooled over the 46 pairs before the
direction is taken.

    cos( centroid(risers) - centroid(fallers),  V(pole_a) - V(pole_b) )

IDENTITY WORTH STATING. Both arms renormalize to 1 over the content candidate
set, so sum(Dp) = 0, so sum_risers Dp = sum_fallers |Dp| = R and

    centroid(risers) - centroid(fallers) = (1/R) * sum_w Dp(w) V(w)

The centroid difference and the centroid displacement are the SAME DIRECTION.
Pooling before the direction is what is new here, not the centroids.

THE NULL IS A SIGN-FLIP PERMUTATION, which is the right one for a paired design
and is the one the previous run was missing: each pair's arm labels are flipped
independently, which negates that pair's Dp and nothing else.

Each cell is renormalized so every model pair weighs the same. Otherwise a
low-entropy model contributes ten times a high-entropy one and the "roster"
direction is three models' direction.
"""
import collections, json, os, sys, difflib, pickle, statistics as st
ROOT="/Users/rj416/github/malign-logits"; sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
import numpy as np
from scipy import stats as sst
from sentence_transformers import SentenceTransformer
from malign_logits.cache import CacheManager
from malign_logits import fields
SCR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results")
rng=np.random.default_rng(4946); B=20000
CAT={"f11_gender","f11_gender_he","f11_gender_she","f11_parent","f11_species"}
AMBIG={"f11_class","f11_loyal"}; DUP={"f11_holy_b"}
def terms(q):
    A,Bx=q["pole_a"].split(),q["pole_b"].split(); sm=difflib.SequenceMatcher(None,A,Bx); da=[];db=[]
    for t,i1,i2,j1,j2 in sm.get_opcodes():
        if t!="equal": da+=A[i1:i2]; db+=Bx[j1:j2]
    return " ".join(da)," ".join(db)
F=json.load(open(ROOT+"/data/f11_k2_units.json"))
Q={q["group"]:q for q in json.load(open(ROOT+"/data/f11_quintuplets.json"))["quintuplets"]}
surv=[r for r in F["units"] if r["survives"] and Q[r["group"]]["language"]=="en"
      and fields.is_content_word(r["surface"])]
by=collections.defaultdict(list)
for r in surv: by[r["group"]].append(r["surface"])
GROUPS=[g for g in sorted(by) if g not in CAT|AMBIG|DUP]
VOC=sorted({s for g in GROUPS for s in by[g]})
m=SentenceTransformer("BAAI/bge-m3")
E=m.encode(VOC,normalize_embeddings=True,batch_size=128,show_progress_bar=False)
IX={w:i for i,w in enumerate(VOC)}; AX={}
#: ENCODE EACH POLE TERM ALONE, AND ASSERT IT AGAINST THE VOCABULARY BATCH.
#: A TWO-ELEMENT encode() silently returns a different vector for the shorter
#: member of the pair. Measured, deterministic over five repeats:
#:     create  cos(single, in-2-batch) 0.602      guilty 0.561
#:     pain                            0.580      feared 0.824
#: which corrupted 4 of 13 axes (create, guilt, sensation, trust) in the first
#: version of this analysis. Single-element and 128-batch encodings agree to
#: 1.000000, so either is safe; the pair is not. The assert is the point -- the
#: defect was invisible until a pole word failed to sit at +-sqrt((1-cos_ab)/2).
def emb1(t):
    v=m.encode([t],normalize_embeddings=True,show_progress_bar=False)[0]
    if t in IX:
        c=float(v@E[IX[t]])
        assert c>0.999, "batch-dependent encoding of %r: cos=%.4f" % (t,c)
    return v
for g in GROUPS:
    ta,tb=terms(Q[g]); d=emb1(ta)-emb1(tb); AX[g]=d/np.linalg.norm(d)
#: and the invariant the defect violated, checked for every axis that can be
for g in GROUPS:
    ta,tb=terms(Q[g])
    if ta in IX and tb in IX:
        pa=float(E[IX[ta]]@AX[g]); pb=float(E[IX[tb]]@AX[g])
        assert abs(pa+pb)<1e-3, "axis %s not symmetric on its poles: %+.4f / %+.4f"%(g,pa,pb)
cm=CacheManager(); pairs=json.load(open(ROOT+"/data/base_aligned_pairs.json"))
DP=collections.defaultdict(list)   # group -> list of (family, dict word->Dp)
for pr in pairs:
    for g in GROUPS:
        cand=set(by[g]); d={}
        for arm,mid in (("base",pr["base"]),("aligned",pr["aligned"])):
            v=cm.get_true_word_probs(mid,Q[g]["both"])
            if not v or not v.get("rows"): continue
            c={x["word"]:x["p"] for x in v["rows"] if x["word"] in cand}
            if not c: continue
            tot=sum(c.values()); d[arm]={w:p/tot for w,p in c.items()}
        if len(d)==2:
            ws=set(d["base"])|set(d["aligned"])
            DP[g].append((pr["family"],{w:d["aligned"].get(w,0.)-d["base"].get(w,0.) for w in ws}))
pickle.dump({"DP":{g:DP[g] for g in GROUPS},"VOC":VOC,"E":E,"AX":AX,"by":dict(by)},
            open(SCR+"/dp.pkl","wb"))
print("=" * 100)
print("POOLED MASS DIRECTION   cos( centroid(risers) - centroid(fallers), V(pole_a)-V(pole_b) )")
print("   positive = alignment moves probability mass toward the GOOD pole.  %d sign-flip permutations"%B)
print("=" * 100)
print("   %-18s %5s %7s %7s %8s   %-24s %-24s"%("group","n","cos","perm z","perm p","top mass FALLERS","top mass RISERS"))
rows=[]
for g in GROUPS:
    cells=DP[g]; n=len(cells)
    ws=sorted({w for _,c in cells for w in c}); wi=[IX[w] for w in ws]
    M=np.zeros((n,len(ws)))
    for i,(_,c) in enumerate(cells):
        for j,w in enumerate(ws): M[i,j]=c.get(w,0.)
    Eg=E[wi]                        # (|ws|, 1024)
    P=M@Eg                          # (n, 1024)  each cell's Dp . E
    obs=P.mean(0); c=float(obs@AX[g]/np.linalg.norm(obs))
    S=rng.choice([-1.,1.],size=(B,n))
    Pm=(S@P)/n                      # (B, 1024)
    nl=(Pm@AX[g])/np.linalg.norm(Pm,axis=1)
    z=(c-nl.mean())/nl.std(); pv=(np.sum(np.abs(nl)>=abs(c))+1)/(B+1)
    dp=M.mean(0)
    fa=sorted(zip(ws,dp),key=lambda x:x[1])[:3]; ri=sorted(zip(ws,dp),key=lambda x:-x[1])[:3]
    rows.append((g,c,z,pv))
    print("   %-18s %5d %+7.3f %+7.2f %8.4f   %-24s %-24s"
          %(g,n,c,z,pv," ".join("%s%.3f"%(w,v) for w,v in fa)," ".join("%s+%.3f"%(w,v) for w,v in ri)))
cs=[r[1] for r in rows]; t,p=sst.ttest_1samp(cs,0)
sig=[r[0] for r in rows if r[3]<.05]
print("\n   %d groups: mean cos %+.4f   t=%+.2f  p=%.4f   |  %d groups individually p<.05: %s"
      %(len(cs),st.mean(cs),t,p,len(sig),", ".join(sig) or "none"))
print("   positive in %d of %d groups"%(sum(x>0 for x in cs),len(cs)))
#: ALL GROUPS POOLED into one direction, the roster-level statement
allP=[]
for g in GROUPS:
    cells=DP[g]; ws=sorted({w for _,c in cells for w in c}); wi=[IX[w] for w in ws]
    M=np.zeros((len(cells),len(ws)))
    for i,(_,c) in enumerate(cells):
        for j,w in enumerate(ws): M[i,j]=c.get(w,0.)
    allP.append(((M@E[wi]).mean(0), AX[g]))
print("\n   mean over groups of cos = %+.4f ; the same quantity computed per group and averaged"%st.mean(cs))
print("   (a single pooled direction is not defined across groups: each has its own axis)")
