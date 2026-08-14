"""Does alignment pick a pole, and does it pick the POSITIVE one? RH's question.

Two things the earlier per-role run could not answer:

1. It used the BARE-WORD embedding, which [5189] showed is contaminated by
   lexical overlap with the prompt. The CONTEXTUAL delta -- embed(prompt+word)
   minus embed(prompt) -- is the clean pole measure.

2. It reported signed t_axis per group but could not pool, because which prompt
   is "a" is arbitrary between groups. It is NOT arbitrary in valence: 32 of the
   41 active groups put the POSITIVE pole in pole_a (beautiful, free, rich,
   create, desire, faithful, innocent, holy, loved, loyal, pleasure, trust). The
   9 CATEGORY-pole groups (gender x6, parent x2, species) have no positive pole
   and are HELD OUT rather than folded in -- their sign means nothing.

So: positive t_axis = toward the positively-valenced pole, comparable across the
32. f11_reason/_zh stay beside as the declared negative control.
"""
import collections, json, os, sys, statistics as st
HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
CATEGORY={"f11_gender","f11_gender_he","f11_gender_she","f11_parent","f11_species",
          "f11_gender_zh","f11_gender_he_zh","f11_gender_she_zh","f11_parent_zh"}
NEGCTRL={"f11_reason","f11_reason_zh"}
ROLES=("pole_a","pole_b","both","both_matched","control_a","control_b")

def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from malign_logits.cache import CacheManager
    F=json.load(open(os.path.join(ROOT,"data","f11_k2_units.json")))
    surv=[r for r in F["units"] if r["survives"]]
    Q={q["group"]:q for q in json.load(open(os.path.join(ROOT,"data","f11_quintuplets.json")))["quintuplets"]}
    groups=sorted({r["group"] for r in surv})
    by=collections.defaultdict(list)
    for r in surv: by[r["group"]].append(r["surface"])
    print("groups %d (category held out: %d, negative control: %d)"
          %(len(groups),len([g for g in groups if g in CATEGORY]),
            len([g for g in groups if g in NEGCTRL])))
    m=SentenceTransformer("BAAI/bge-m3")
    #: CONTEXTUAL delta, computed once per (surface, group) in the BOTH frame
    geo={}
    for gi,g in enumerate(groups):
        S=sorted(set(by[g])); ctx=Q[g]["both"]
        E=m.encode([ctx]+[ctx+" "+s for s in S],normalize_embeddings=True,
                   batch_size=128,show_progress_bar=False)
        P=E[0]; D=E[1:]-P
        D=D/(np.linalg.norm(D,axis=1,keepdims=True)+1e-9)
        ab=m.encode([Q[g]["pole_a"],Q[g]["pole_b"]],normalize_embeddings=True)
        ax=(ab[0]-ab[1]); ax=ax/np.linalg.norm(ax)
        for s,d in zip(S,D@ax): geo[(s,g)]=float(d)
        if gi%10==0: print("  ...%d/%d groups"%(gi,len(groups)),flush=True)
    cm=CacheManager(); pairs=json.load(open(os.path.join(ROOT,"data","base_aligned_pairs.json")))
    cells=[]
    for pr in pairs:
        for g in groups:
            q=Q[g]
            for role in ROLES:
                t=q.get(role)
                if not t: continue
                d={}
                for arm,mid in (("base",pr["base"]),("aligned",pr["aligned"])):
                    v=cm.get_true_word_probs(mid,t)
                    if not v or not v.get("rows"): continue
                    tot=w=0.0
                    for x in v["rows"]:
                        k=(x["word"],g)
                        if k in geo: tot+=x["p"]; w+=x["p"]*geo[k]
                    if tot>0: d[arm]=w/tot
                if len(d)==2:
                    cells.append({"family":pr["family"],"group":g,"role":role,
                                  "lang":q["language"],"base":d["base"],"aligned":d["aligned"],
                                  "cat":g in CATEGORY,"neg":g in NEGCTRL})
    print("\ncells: %d"%len(cells))
    V=[c for c in cells if not c["cat"] and not c["neg"]]
    print("\n=== DOES ALIGNMENT MOVE TOWARD THE POSITIVE POLE? (32 valenced groups) ===")
    print("  %-14s %6s %10s %10s %10s %8s"%("role","n","base","aligned","delta","% pos"))
    for role in ROLES:
        sub=[c for c in V if c["role"]==role]
        if not sub: continue
        b=st.mean(c["base"] for c in sub); a=st.mean(c["aligned"] for c in sub)
        dd=[c["aligned"]-c["base"] for c in sub]
        print("  %-14s %6d %+10.4f %+10.4f %+10.4f %7.0f%%"
              %(role,len(sub),b,a,st.mean(dd),100*sum(x>0 for x in dd)/len(dd)))
    print("\n  HELD OUT")
    for lab,sel in (("category poles",lambda c:c["cat"]),("negative control",lambda c:c["neg"])):
        sub=[c for c in cells if sel(c) and c["role"]=="both"]
        if sub:
            dd=[c["aligned"]-c["base"] for c in sub]
            print("    %-18s both cell n=%4d  delta %+.4f  %.0f%% positive"
                  %(lab,len(sub),st.mean(dd),100*sum(x>0 for x in dd)/len(dd)))
    json.dump(cells,open(os.path.join(CAMP,"results","l1_valence_pole.json"),"w"))
    print("\nwrote results/l1_valence_pole.json")
    return 0

if __name__=="__main__": sys.exit(main())
