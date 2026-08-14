"""RH's design: three prompt anchors, and where a COMPLETED sentence falls.

    A  = embed(pole_a prompt)      t = 1
    B  = embed(pole_b prompt)      t = 0
    AB = embed(both prompt)        VERIFIED at t = 0.445 (all 41 groups in [0,1])
                                   controls: a 0.967, b 0.030

    X  = embed(both_prompt + " " + word)      where does the completed sentence sit?

This is L3's geometry at the surface: not a word-delta projected onto an axis,
but a finished utterance placed among three reference utterances. It keeps the
BETWEEN structure that the delta version discarded.

WHY THE DELTA VERSION WAS THE WRONG OBJECT. A mean of signed positions is close
to blind to the move alignment actually makes. On llama/f11_love the base puts
kill at 0.155 and aligned at 0.096, while `be` goes 0.145 -> 0.237 -- mass moving
from a pole-committed word ONTO A NEUTRAL ONE. That is displacement, and a
signed mean barely registers it (+0.0148 -> +0.0160). So COMMITTED MASS is
reported beside the mean: the share of mass on completions that land outside the
middle of the axis.
"""
import collections, json, os, sys, statistics as st
HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
CATEGORY={"f11_gender","f11_gender_he","f11_gender_she","f11_parent","f11_species",
          "f11_gender_zh","f11_gender_he_zh","f11_gender_she_zh","f11_parent_zh"}

def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from malign_logits.cache import CacheManager
    F=json.load(open(os.path.join(ROOT,"data","f11_k2_units.json")))
    surv=[r for r in F["units"] if r["survives"]]
    Q={q["group"]:q for q in json.load(open(os.path.join(ROOT,"data","f11_quintuplets.json")))["quintuplets"]}
    by=collections.defaultdict(list)
    for r in surv: by[r["group"]].append(r["surface"])
    m=SentenceTransformer("BAAI/bge-m3")
    tw={}
    for i,g in enumerate(sorted(by)):
        S=sorted(set(by[g])); q=Q[g]
        E=m.encode([q["pole_a"],q["pole_b"],q["both"]]+[q["both"]+" "+s for s in S],
                   normalize_embeddings=True,batch_size=128,show_progress_bar=False)
        A,B=E[0],E[1]; ax=A-B; n2=float(ax@ax)
        for s,x in zip(S,E[3:]): tw[(s,g)]=float((x-B)@ax/n2)
        if i%10==0: print("  ...%d groups"%i,flush=True)
    cm=CacheManager(); pairs=json.load(open(os.path.join(ROOT,"data","base_aligned_pairs.json")))
    #: COMMITTED = the completion lands outside the middle third of the axis
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
                    tot=w=com=0.0
                    for x in v["rows"]:
                        k=(x["word"],g)
                        if k not in tw: continue
                        p=x["p"]; tv=tw[k]; tot+=p; w+=p*tv
                        if tv>0.667 or tv<0.333: com+=p
                    if tot>0: d[arm]=(w/tot,com/tot)
                if len(d)==2:
                    cells.append({"group":g,"role":role,"cat":g in CATEGORY,
                                  "b_t":d["base"][0],"a_t":d["aligned"][0],
                                  "b_c":d["base"][1],"a_c":d["aligned"][1]})
    V=[c for c in cells if not c["cat"]]
    print("\ncells %d (valenced %d)"%(len(cells),len(V)))
    print("\n  %-12s %6s %19s %19s"%("role","n","mean t (0=b, 1=a)","COMMITTED mass share"))
    print("  %-12s %6s %8s %8s   %8s %8s %7s"%("","","base","aligned","base","aligned","%down"))
    for role in ("pole_a","control_a","both","control_b","pole_b"):
        sub=[c for c in V if c["role"]==role]
        if not sub: continue
        dn=100*sum(1 for c in sub if c["a_c"]<c["b_c"])/len(sub)
        print("  %-12s %6d %8.3f %8.3f   %8.4f %8.4f %6.0f%%"
              %(role,len(sub),st.mean(c["b_t"] for c in sub),st.mean(c["a_t"] for c in sub),
                st.mean(c["b_c"] for c in sub),st.mean(c["a_c"] for c in sub),dn))
    json.dump(cells,open(os.path.join(CAMP,"results","l1_three_anchor.json"),"w"))
    print("\nwrote results/l1_three_anchor.json")
    return 0

if __name__=="__main__": sys.exit(main())
