"""INDEPENDENT REPLICATION ON GloVe. Same data, same tests, different embedder.

WHY. The headline of this analysis is a NULL (Oedipalization is not detectable at
the next-word grain), and the instrument was caught silently corrupting 4 of 13
axes: a TWO-ELEMENT SentenceTransformer.encode() returns a different vector for
the shorter member of the pair (create 0.602, guilty 0.561, pain 0.580, feared
0.824 against their single encodings). That was found and fixed, but a null from
an instrument with a demonstrated silent failure is worth very little on its own.

GloVe is the right second opinion for THIS job, not merely a different one.
Everything here embeds ISOLATED WORDS and does vector arithmetic on them, which
is word2vec/GloVe's design case and is out of distribution for a sentence
encoder. A lookup table also cannot have a batching bug, and it has no subword
tokenisation -- so the alphabetic clustering that contaminated the semantic-field
run (every w-word in one cluster) should simply not occur.

Reports the three tests that carry the conclusion, beside their BGE values.
"""
import collections, json, os, sys, difflib, statistics as st
ROOT="/Users/rj416/github/malign-logits"; sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
import numpy as np
from scipy import stats as sst
from malign_logits.cache import CacheManager
from malign_logits import fields
rng=np.random.default_rng(4946)
CAT={"f11_gender","f11_gender_he","f11_gender_she","f11_parent","f11_species"}
AMBIG={"f11_class","f11_loyal"}; DUP={"f11_holy_b"}
def terms(q):
    A,B=q["pole_a"].split(),q["pole_b"].split(); sm=difflib.SequenceMatcher(None,A,B); da=[];db=[]
    for t,i1,i2,j1,j2 in sm.get_opcodes():
        if t!="equal": da+=A[i1:i2]; db+=B[j1:j2]
    return " ".join(da)," ".join(db)
F=json.load(open(ROOT+"/data/f11_k2_units.json"))
Q={q["group"]:q for q in json.load(open(ROOT+"/data/f11_quintuplets.json"))["quintuplets"]}
surv=[r for r in F["units"] if r["survives"] and Q[r["group"]]["language"]=="en"
      and fields.is_content_word(r["surface"])]
by=collections.defaultdict(list)
for r in surv: by[r["group"]].append(r["surface"])
GROUPS=[g for g in sorted(by) if g not in CAT|AMBIG|DUP]
import gensim.downloader as api
KV=api.load("glove-wiki-gigaword-300")
def vec(t):
    """unit vector; multi-word terms are the normalised mean of their words"""
    ws=[w for w in t.lower().split() if w in KV]
    if not ws: return None
    v=np.mean([KV[w] for w in ws],0); n=np.linalg.norm(v)
    return v/n if n>0 else None
VOC=sorted({s for g in GROUPS for s in by[g]})
V={}; miss=[]
for w in VOC:
    v=vec(w)
    if v is None: miss.append(w)
    else: V[w]=v
print("GloVe %d-d, %d vectors. Vocabulary coverage %d/%d (%.1f%%); %d missing"
      %(KV.vector_size,len(KV.index_to_key),len(V),len(VOC),100*len(V)/len(VOC),len(miss)))
print("   missing sample: %s"%" ".join(miss[:12]))
AX={}
for g in GROUPS:
    ta,tb=terms(Q[g]); a,b=vec(ta),vec(tb)
    d=a-b; AX[g]=d/np.linalg.norm(d)
    pa=float(a@AX[g]); pb=float(b@AX[g])
    assert abs(pa+pb)<1e-4, "asymmetric axis %s: %+.4f %+.4f"%(g,pa,pb)
print("   all %d axes symmetric on their poles (the invariant BGE violated)\n"%len(AX))
GS=sorted(AX); A=np.stack([AX[g] for g in GS])
cm=CacheManager(); pairs=json.load(open(ROOT+"/data/base_aligned_pairs.json"))
R=[]
for pr in pairs:
    for gi,g in enumerate(GS):
        cand=set(by[g])&set(V); d={}
        for arm,mid in (("base",pr["base"]),("aligned",pr["aligned"])):
            v=cm.get_true_word_probs(mid,Q[g]["both"])
            if not v or not v.get("rows"): continue
            c={x["word"]:x["p"] for x in v["rows"] if x["word"] in cand}
            if len(c)<3: continue
            ws=list(c); p=np.array([c[w] for w in ws]); p/=p.sum()
            Ew=np.stack([V[w] for w in ws]); cen=p@Ew
            totv=float(p@((Ew-cen)**2).sum(1))
            S=Ew@A.T; pos=p@S; av=p@((S-pos)**2)
            d[arm]=dict(pos=pos,av=av,share=av/totv,ws=ws,p=p,Ew=Ew)
        if len(d)==2: R.append((pr["family"],gi,g,d))
fs=sorted({r[0] for r in R}); M=np.zeros((len(fs),len(R)))
for j,r in enumerate(R): M[fs.index(r[0]),j]=1.
M/=M.sum(1,keepdims=True)
def cl(v): return M@np.asarray(v)
print("%d cells, %d families, %d groups\n"%(len(R),len(fs),len(GS)))
print("TEST 1  OEDIPALIZATION (own-axis straddle share, foreign-axis controlled)")
own=np.array([r[3]["aligned"]["share"][r[1]]-r[3]["base"]["share"][r[1]] for r in R])
frg=np.array([np.mean([r[3]["aligned"]["share"][j]-r[3]["base"]["share"][j]
                       for j in range(len(GS)) if j!=r[1]]) for r in R])
for lab,v in (("own-axis share delta",own),("foreign-axis share delta",frg),
              ("DIFFERENCE (Oedipal. = neg)",own-frg)):
    t,p=sst.ttest_1samp(cl(v),0); print("   %-30s %+.6f   t=%+.2f  p=%.4f"%(lab,v.mean(),t,p))
print("        BGE gave: difference -0.000282, t=-0.39, p=0.7011\n")
print("TEST 2  DIRECTION OF THE SHIFT, cos and |cos|, two nulls")
co=[];ab=[];fo=[];pn=[];pa_=[]
for f,gi,g,d in R:
    ws=d["base"]["ws"]; allw=sorted(set(ws)|set(d["aligned"]["ws"]))
    pb={w:0. for w in allw}; paD={w:0. for w in allw}
    for w,x in zip(d["base"]["ws"],d["base"]["p"]): pb[w]=x
    for w,x in zip(d["aligned"]["ws"],d["aligned"]["p"]): paD[w]=x
    dp=np.array([paD[w]-pb[w] for w in allw]); Ew=np.stack([V[w] for w in allw])
    D=dp@Ew; n=np.linalg.norm(D)
    if n<1e-12: continue
    cs=(A@D)/n; co.append(cs[gi]); ab.append(abs(cs[gi]))
    fo.append(np.mean([abs(cs[j]) for j in range(len(GS)) if j!=gi]))
    G=Ew@Ew.T; s=Ew@AX[g]
    P=np.stack([rng.permutation(dp) for _ in range(400)])
    nc=(P@s)/np.sqrt(np.maximum(((P@G)*P).sum(1),1e-24))
    pn.append(np.abs(nc).mean()); pa_.append(nc.mean())
co=np.array(co);ab=np.array(ab);fo=np.array(fo);pn=np.array(pn);pa_=np.array(pa_)
M2=np.zeros((len(fs),len(co)))
for j,r in enumerate(R[:len(co)]): M2[fs.index(r[0]),j]=1.
M2/=np.maximum(M2.sum(1,keepdims=True),1)
def cl2(v): return M2@np.asarray(v)
print("   signed cos            observed %+.4f   perm null %+.4f   excess %+.4f  t=%+.2f p=%.4f"
      %(co.mean(),pa_.mean(),(co-pa_).mean(),*sst.ttest_1samp(cl2(co-pa_),0)))
print("        BGE: observed +0.0245, null +0.0004, excess +0.0241, t=+2.40, p=0.0204")
print("   |cos| vs FOREIGN AXIS observed %.4f   foreign   %.4f   excess %+.4f  t=%+.2f p=%.4f"
      %(ab.mean(),fo.mean(),(ab-fo).mean(),*sst.ttest_1samp(cl2(ab-fo),0)))
print("        BGE: own 0.1251, foreign 0.0957, excess +0.0294, t=+8.05, p<0.0001")
print("   |cos| vs PERMUTATION  observed %.4f   perm null %.4f   excess %+.4f  t=%+.2f p=%.4f"
      %(ab.mean(),pn.mean(),(ab-pn).mean(),*sst.ttest_1samp(cl2(ab-pn),0)))
print("        BGE: own 0.1251, null 0.0822, excess +0.0429, t=+9.28, p<0.0001")
print("\nTEST 3  SCALE: where the models sit against where a pole sits")
allp=[]
for f,gi,g,d in R: allp += [d["base"]["pos"][gi], d["aligned"]["pos"][gi]]
print("   pole word position (by construction, per group): %s"
      %" ".join("%.2f"%float(vec(terms(Q[g])[0])@AX[g]) for g in GS[:6]))
print("   observed model positions, all cells both arms: %+.3f to %+.3f"%(min(allp),max(allp)))
print("        BGE: poles at +-0.45 to +-0.48, models -0.12 to +0.15")
