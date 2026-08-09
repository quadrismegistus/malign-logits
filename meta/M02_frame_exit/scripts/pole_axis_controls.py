"""THE CONTROLS. Is any of [5195] about CONTRADICTION, or is it what alignment
does to any two-adjective continuation?

[5195] is BOTH-prompt only and says so in its own limits section. The controls
are same-side near-synonym conjunctions ("beautiful and lovely"), so they carry
the conjunction and the length and not the contradiction. The registered primary
contrast is BOTH against mean(CONTROL_A, CONTROL_B) and this is that contrast at
the L1 grain.

THE TEST THAT NEEDS NO LEXICON. The word-level table (kill -9.9 out, remain +6.6
in) is the most quotable thing in [5195] and the hardest to test, because sorting
words into "action" and "interior state" is a judgment that would be made by the
person who already believes the finding. So the primary here is instead

    cos( Delta_BOTH , Delta_CONTROL )   within a cell

the angle between the displacement alignment applies at the contradiction and
the displacement it applies at the same-side conjunction. Near 1 means ONE
transformation applied to both, and the contradiction adds nothing. Near 0 means
they are different operations. No lexicon, no coder, no judgment.

CANDIDATE COVERAGE IS REPORTED PER ROLE BEFORE ANY CONTRAST. The vocabulary
comes from the k>=2 frame; if it covers BOTH's mass better than the controls'
then every downstream difference is the vocabulary's provenance and not the
prompt's. A lexicon read off one arm measures that arm.
"""
import collections, json, os, sys, difflib, statistics as st
ROOT="/Users/rj416/github/malign-logits"; sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
import numpy as np
from scipy import stats as sst
from sentence_transformers import SentenceTransformer
from malign_logits.cache import CacheManager
from malign_logits import fields
CAT={"f11_gender","f11_gender_he","f11_gender_she","f11_parent","f11_species"}
AMBIG={"f11_class","f11_loyal"}; DUP={"f11_holy_b"}
ROLES=("both","control_a","control_b")
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
GROUPS=[g for g in sorted(by) if g not in CAT|AMBIG|DUP
        and Q[g].get("control_a") and Q[g].get("control_b")]
print("EN groups with both controls, after exclusions: %d"%len(GROUPS)); print(" ",", ".join(GROUPS))
VOC=sorted({s for g in GROUPS for s in by[g]})
m=SentenceTransformer("BAAI/bge-m3")
E=m.encode(VOC,normalize_embeddings=True,batch_size=128,show_progress_bar=False)
IX={w:i for i,w in enumerate(VOC)}; AX={}
def emb1(t):
    v=m.encode([t],normalize_embeddings=True,show_progress_bar=False)[0]
    if t in IX: assert float(v@E[IX[t]])>0.999,"batch-dependent encoding of %r"%t
    return v
for g in GROUPS:
    ta,tb=terms(Q[g]); d=emb1(ta)-emb1(tb); AX[g]=d/np.linalg.norm(d)
    if ta in IX and tb in IX:
        assert abs(float(E[IX[ta]]@AX[g])+float(E[IX[tb]]@AX[g]))<1e-3,"asymmetric axis %s"%g
print("all %d axes symmetric on their poles\n"%len(AX))
cm=CacheManager(); pairs=json.load(open(ROOT+"/data/base_aligned_pairs.json"))
CELL={}; cov=collections.defaultdict(list)
for pr in pairs:
    for g in GROUPS:
        cand=set(by[g])
        for role in ROLES:
            d={}
            for arm,mid in (("base",pr["base"]),("aligned",pr["aligned"])):
                v=cm.get_true_word_probs(mid,Q[g][role])
                if not v or not v.get("rows"): continue
                allm=sum(x["p"] for x in v["rows"])
                c={x["word"]:x["p"] for x in v["rows"] if x["word"] in cand}
                if len(c)<3 or not allm: continue
                cov[role].append(sum(c.values())/allm)
                tot=sum(c.values()); d[arm]={w:p/tot for w,p in c.items()}
            if len(d)==2:
                ws=set(d["base"])|set(d["aligned"])
                CELL[(pr["family"],g,role)]={w:d["aligned"].get(w,0.)-d["base"].get(w,0.) for w in ws}
print("="*94); print("0. CANDIDATE COVERAGE PER ROLE  (share of each arm's mass inside the k>=2 vocabulary)")
print("="*94)
for r in ROLES: print("   %-10s mean %.4f   median %.4f   n=%d"%(r,st.mean(cov[r]),st.median(cov[r]),len(cov[r])))
t,p=sst.ttest_ind(cov["both"],cov["control_a"]+cov["control_b"])
print("   BOTH vs controls: t=%+.2f p=%.4f  %s"%(t,p,"<-- provenance confound if large" if p<.01 else "no provenance gap"))
def dvec(dp,g):
    ws=[w for w in dp]; return np.array([dp[w] for w in ws])@E[[IX[w] for w in ws]]
print("\n"+"="*94); print("1. THE PRIMARY: cos( Delta_BOTH , Delta_CONTROL ) within a cell")
print("   near 1 = ONE transformation, contradiction adds nothing.  near 0 = different operations.")
print("="*94)
rows=[]
for (fam,g,role),dp in CELL.items():
    if role!="both": continue
    db=dvec(dp,g); nb=np.linalg.norm(db)
    for cr in ("control_a","control_b"):
        dc=CELL.get((fam,g,cr))
        if dc is None: continue
        dcv=dvec(dc,g); nc=np.linalg.norm(dcv)
        if nb<1e-12 or nc<1e-12: continue
        rows.append((fam,g,cr,float(db@dcv/(nb*nc)),nb,nc))
cs=np.array([r[3] for r in rows])
fam=collections.defaultdict(list)
for r in rows: fam[r[0]].append(r[3])
fm=[st.mean(fam[f]) for f in fam]
print("   %d cell-pairs, %d families:  mean cos %.4f  (median %.4f)"%(len(rows),len(fm),cs.mean(),np.median(cs)))
print("   family-clustered mean %.4f   95%% CI %.4f to %.4f"
      %(st.mean(fm),*sst.t.interval(.95,len(fm)-1,st.mean(fm),sst.sem(fm))))
#: NULL: the same BOTH displacement against a DIFFERENT GROUP's control displacement
nl=[]
bykey={(f,g,r):v for (f,g,r),v in CELL.items()}
for (f,g,role),dp in CELL.items():
    if role!="both": continue
    db=dvec(dp,g); nb=np.linalg.norm(db)
    for (f2,g2,r2),dp2 in CELL.items():
        if r2=="control_a" and f2==f and g2!=g:
            dc=dvec(dp2,g2); nc=np.linalg.norm(dc)
            if nb>1e-12 and nc>1e-12: nl.append(float(db@dc/(nb*nc)))
            break
print("   NULL, BOTH against ANOTHER GROUP's control (same model): mean cos %.4f  n=%d"%(np.mean(nl),len(nl)))
print("   |Delta| BOTH %.4f   controls %.4f"%(np.mean([r[4] for r in rows]),np.mean([r[5] for r in rows])))
print("\n"+"="*94); print("2. THE WORD TABLE, PER ROLE  (mean dp x1000 per cell)")
print("="*94)
agg={r:collections.Counter() for r in ROLES}; n={r:0 for r in ROLES}
for (f,g,role),dp in CELL.items():
    n[role]+=1
    for w,v in dp.items(): agg[role][w]+=v
top=sorted(agg["both"], key=lambda w:-abs(agg["both"][w]/max(n["both"],1)))[:20]
print("   %-12s %10s %10s %10s   %s"%("word","BOTH","control_a","control_b","contradiction-specific?"))
for w in top:
    b=1000*agg["both"][w]/n["both"]; a=1000*agg["control_a"][w]/max(n["control_a"],1); c=1000*agg["control_b"][w]/max(n["control_b"],1)
    mc=(a+c)/2
    tag="SPECIFIC" if abs(b)>2*abs(mc) and b*mc>=0 or (b*mc<0) else "general"
    print("   %-12s %+10.1f %+10.1f %+10.1f   %s"%(w,b,a,c,tag))
print("\n   n cells: %s"%{r:n[r] for r in ROLES})
