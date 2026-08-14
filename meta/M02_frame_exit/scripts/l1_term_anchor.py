"""Word-level anchors, ENGLISH ONLY. RH's design.

    pole_a    = embed("loved")
    pole_b    = embed("hated")
    both_term = embed("loved and hated")

    for each continuation w:  cos(w, loved), cos(w, hated), cos(w, "loved and hated")

Why word-level and not sentence-level. The three-anchor sentence version put
every completed utterance within 0.05 of the BOTH prompt's own position: adding
one word to a nine-word sentence barely moves its embedding, so the prompt
dominated and the pole structure compressed. Bare terms have no prompt to
dominate them.

ENGLISH ONLY, deliberately. Chinese pole terms cannot be extracted by diffing --
character-level diff returns the negation particle alone for guilt (无/有) and
an empty string for reason. A wrong anchor is worse than no anchor.
"""
import collections, json, os, sys, difflib, statistics as st
HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
CATEGORY={"f11_gender","f11_gender_he","f11_gender_she","f11_parent","f11_species"}

def terms(q):
    A,B=q["pole_a"].split(),q["pole_b"].split()
    sm=difflib.SequenceMatcher(None,A,B); da=[];db=[]
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag!="equal": da+=A[i1:i2]; db+=B[j1:j2]
    return " ".join(da)," ".join(db)

def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from malign_logits.cache import CacheManager
    from malign_logits import fields
    F=json.load(open(os.path.join(ROOT,"data","f11_k2_units.json")))
    Q={q["group"]:q for q in json.load(open(os.path.join(ROOT,"data","f11_quintuplets.json")))["quintuplets"]}
    #: CONTENT WORDS ONLY. Function words sit near zero on the pole axis and
    #: carry real mass (17.5% in English), so they drag every cell toward the
    #: middle and compress the between-role spread. CLAWS is valid HERE because
    #: this run is English-only -- it returns True for every Chinese surface,
    #: which is why the field was given to the LLM for the bilingual arm.
    surv=[r for r in F["units"] if r["survives"] and Q[r["group"]]["language"]=="en"]
    n0=len(surv)
    surv=[r for r in surv if fields.is_content_word(r["surface"])]
    print("content-word filter: %d -> %d units (%.1f%% kept)"%(n0,len(surv),100*len(surv)/n0))
    by=collections.defaultdict(list)
    for r in surv: by[r["group"]].append(r["surface"])
    m=SentenceTransformer("BAAI/bge-m3")
    print("english groups %d, units %d"%(len(by),len(surv)))
    for g in sorted(by)[:6]: print("   %-18s %r / %r"%((g,)+terms(Q[g])))
    tw={}
    for g in sorted(by):
        ta,tb=terms(Q[g]); S=sorted(set(by[g]))
        E=m.encode([ta,tb,ta+" and "+tb]+S,normalize_embeddings=True,
                   batch_size=128,show_progress_bar=False)
        A,B,AB=E[0],E[1],E[2]
        for s,v in zip(S,E[3:]):
            tw[(s,g)]=(float(v@A),float(v@B),float(v@AB))
    #: where do the TERM anchors sit relative to each other?
    print("\n  anchor geometry: cos(a,b) and where 'a and b' sits")
    ex=[]
    for g in sorted(by)[:8]:
        ta,tb=terms(Q[g]); E=m.encode([ta,tb,ta+" and "+tb],normalize_embeddings=True)
        ax=E[0]-E[1]; t=float((E[2]-E[1])@ax/(ax@ax))
        ex.append((g,float(E[0]@E[1]),t))
    for g,c,t in ex: print("    %-18s cos(a,b) %.3f   t('a and b') %.3f"%(g,c,t))
    cm=CacheManager(); pairs=json.load(open(os.path.join(ROOT,"data","base_aligned_pairs.json")))
    cells=[]
    for pr in pairs:
        for g in sorted(by):
            q=Q[g]
            for role in ("both","pole_a","pole_b","control_a","control_b"):
                t=q.get(role)
                if not t: continue
                d={}
                for arm,mid in (("base",pr["base"]),("aligned",pr["aligned"])):
                    v=cm.get_true_word_probs(mid,t)
                    if not v or not v.get("rows"): continue
                    tot=pa=pb=pab=0.0
                    for x in v["rows"]:
                        k=(x["word"],g)
                        if k not in tw: continue
                        A_,B_,AB_=tw[k]; p=x["p"]
                        tot+=p; pa+=p*A_; pb+=p*B_; pab+=p*AB_
                    if tot>0: d[arm]=(pa/tot,pb/tot,pab/tot)
                if len(d)==2:
                    cells.append({"group":g,"role":role,"cat":g in CATEGORY,
                                  "b":d["base"],"a":d["aligned"]})
    V=[c for c in cells if not c["cat"]]
    print("\ncells %d (valenced %d)"%(len(cells),len(V)))
    print("\n  %-12s %6s %17s %17s %17s"%("role","n","cos to pole_a","cos to pole_b","cos to 'a and b'"))
    print("  %-12s %6s %8s %8s %8s %8s %8s %8s"%("","","base","algn","base","algn","base","algn"))
    for role in ("pole_a","control_a","both","control_b","pole_b"):
        sub=[c for c in V if c["role"]==role]
        if not sub: continue
        f=lambda i,k: st.mean(c[k][i] for c in sub)
        print("  %-12s %6d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f"
              %(role,len(sub),f(0,"b"),f(0,"a"),f(1,"b"),f(1,"a"),f(2,"b"),f(2,"a")))
    print("\n  a-minus-b (which pole the mass leans to)")
    for role in ("pole_a","control_a","both","control_b","pole_b"):
        sub=[c for c in V if c["role"]==role]
        if not sub: continue
        db=[c["b"][0]-c["b"][1] for c in sub]; da=[c["a"][0]-c["a"][1] for c in sub]
        dd=[y-x for x,y in zip(db,da)]
        print("    %-12s base %+.4f  aligned %+.4f  delta %+.4f  %.0f%% positive"
              %(role,st.mean(db),st.mean(da),st.mean(dd),100*sum(x>0 for x in dd)/len(dd)))
    json.dump(cells,open(os.path.join(CAMP,"results","l1_term_anchor.json"),"w"))
    return 0

if __name__=="__main__": sys.exit(main())
