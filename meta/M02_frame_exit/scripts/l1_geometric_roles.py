"""The geometry PER ROLE, not pooled. RH's catch.

l1_geometric_full.py summed mass over all six prompt roles into one number per
(pair, group, arm) -- so the pole_a cell (mass toward a) was averaged with the
pole_b cell (mass toward b) and they CANCEL. Everything came out near zero and
the near-zero was partly the pooling, not the models. The quintuplet design
exists to keep those cells apart; pooling them answers nothing.

Per (pair, group, ROLE, arm):
    t_axis  mass-weighted SIGNED cos(s, a-b)   where the cell's mass sits on the
                                               pole axis. Positive = toward a.
    t_sum   mass-weighted cos(s, a+b)          scene proximity.

The 2D question: do base and aligned sit in different quadrants of
(pole commitment, scene proximity) for the SAME cell?
"""
import collections, json, os, sys, statistics as st
HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
ROLES=("pole_a","pole_b","both","both_matched","control_a","control_b")

def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from malign_logits.cache import CacheManager
    F=json.load(open(os.path.join(ROOT,"data","f11_k2_units.json")))
    surv=[r for r in F["units"] if r["survives"]]
    Q={q["group"]:q for q in json.load(open(os.path.join(ROOT,"data","f11_quintuplets.json")))["quintuplets"]}
    groups=sorted({r["group"] for r in surv}); surfaces=sorted({r["surface"] for r in surv})
    m=SentenceTransformer("BAAI/bge-m3")
    E=m.encode([Q[g]["pole_a"] for g in groups]+[Q[g]["pole_b"] for g in groups]+surfaces,
               normalize_embeddings=True,batch_size=128,show_progress_bar=False)
    A={g:E[i] for i,g in enumerate(groups)}; B={g:E[len(groups)+i] for i,g in enumerate(groups)}
    S={s:E[2*len(groups)+i] for i,s in enumerate(surfaces)}
    geo={}
    for r in surv:
        g,s=r["group"],r["surface"]; ax=A[g]-B[g]; sm=A[g]+B[g]; v=S[s]
        geo[(s,g)]=(float(v@ax/np.linalg.norm(ax)), float(v@sm/np.linalg.norm(sm)))
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
                    tot=wa=ws=0.0
                    for x in v["rows"]:
                        k=(x["word"],g)
                        if k not in geo: continue
                        ta,ts=geo[k]; p=x["p"]; tot+=p; wa+=p*ta; ws+=p*ts
                    if tot>0: d[arm]=(wa/tot,ws/tot)
                if len(d)==2:
                    cells.append({"family":pr["family"],"group":g,"lang":q["language"],"role":role,
                                  "b_axis":d["base"][0],"a_axis":d["aligned"][0],
                                  "b_sum":d["base"][1],"a_sum":d["aligned"][1]})
    print("cells (pair x group x role, both arms): %d"%len(cells))
    print("\n=== DOES EITHER ARM PICK A POLE? signed t_axis by ROLE ===")
    print("  %-14s %6s %10s %10s %10s"%("role","n","base","aligned","delta"))
    for role in ROLES:
        sub=[c for c in cells if c["role"]==role]
        if not sub: continue
        b=st.mean(c["b_axis"] for c in sub); a=st.mean(c["a_axis"] for c in sub)
        print("  %-14s %6d %+10.4f %+10.4f %+10.4f"%(role,len(sub),b,a,a-b))
    print("\n=== IN-FRAME vs OFF-FRAME PROXY: scene proximity by ROLE ===")
    print("  %-14s %6s %10s %10s %10s  %s"%("role","n","base","aligned","delta","% cells aligned lower"))
    for role in ROLES:
        sub=[c for c in cells if c["role"]==role]
        if not sub: continue
        b=st.mean(c["b_sum"] for c in sub); a=st.mean(c["a_sum"] for c in sub)
        lo=100*sum(1 for c in sub if c["a_sum"]<c["b_sum"])/len(sub)
        print("  %-14s %6d %10.4f %10.4f %+10.4f  %.0f%%"%(role,len(sub),b,a,a-b,lo))
    print("\n=== 2D: DO THE ARMS SIT IN DIFFERENT QUADRANTS? ===")
    print("  quadrants cut at each ROLE's own base median, so the question is")
    print("  displacement relative to the cell type, not to a global origin.")
    for role in ROLES:
        sub=[c for c in cells if c["role"]==role]
        if len(sub)<20: continue
        mx=st.median(c["b_axis"] for c in sub); my=st.median(c["b_sum"] for c in sub)
        qb=lambda c,k: (("+" if c[k+"_axis"]>mx else "-"), ("+" if c[k+"_sum"]>my else "-"))
        same=sum(1 for c in sub if qb(c,"b")==qb(c,"a"))
        print("  %-14s n=%4d   same quadrant %5.1f%%   (chance ~25%% if unrelated)"
              %(role,len(sub),100*same/len(sub)))
    json.dump(cells,open(os.path.join(CAMP,"results","l1_geometric_roles.json"),"w"))
    print("\nwrote results/l1_geometric_roles.json")
    return 0

if __name__=="__main__": sys.exit(main())
