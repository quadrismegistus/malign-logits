"""The L1 measure done geometrically over the WHOLE k>=2 frame. Minutes, no LLM.

RH's point: at AUC 0.70 a per-unit label is unreliable, but the L1 measure is a
MASS-WEIGHTED AGGREGATE CONTRASTED BETWEEN ARMS, and a classifier wrong in the
same way on both arms has its bias cancel in the difference. An absolute
off-frame rate would stay biased and n would not fix it; base-minus-aligned is a
different quantity.

    t_axis = cos(s, a-b)   the pole axis. AUC 0.995 POLE1 vs POLE2.
    t_sum  = cos(s, a+b)   scene proximity. AUC 0.75 IN vs OFF -- weak per unit.

CANCELLATION IS AN ASSUMPTION AND IT IS NOT SAFE HERE. Composition differs by
arm: OLMo's aligned arm puts 74% of its mass on blank templates, and
BLANK-TEMPLATE is where the geometry is weakest (AUC 0.662) and pure structure
is perfect. So the arms are not scored by an equally-wrong instrument, and the
structural flag is carried alongside rather than folded in.
"""
import collections, glob, json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")

def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from malign_logits.cache import CacheManager
    F=json.load(open(os.path.join(ROOT,"data","f11_k2_units.json")))
    surv=[r for r in F["units"] if r["survives"]]
    Q={q["group"]:q for q in json.load(open(os.path.join(ROOT,"data","f11_quintuplets.json")))["quintuplets"]}
    groups=sorted({r["group"] for r in surv})
    surfaces=sorted({r["surface"] for r in surv})
    print("k>=2 frame: %d units, %d surfaces, %d groups"%(len(surv),len(surfaces),len(groups)))
    m=SentenceTransformer("BAAI/bge-m3")
    E=m.encode([Q[g]["pole_a"] for g in groups]+[Q[g]["pole_b"] for g in groups]+surfaces,
               normalize_embeddings=True,batch_size=128,show_progress_bar=True)
    A={g:E[i] for i,g in enumerate(groups)}
    B={g:E[len(groups)+i] for i,g in enumerate(groups)}
    S={s:E[2*len(groups)+i] for i,s in enumerate(surfaces)}
    geo={}
    for r in surv:
        g,s=r["group"],r["surface"]; a,b=A[g],B[g]; v=S[s]
        ax=a-b; sm=a+b
        geo[(s,g)]=(float(v@ax/np.linalg.norm(ax)), float(v@sm/np.linalg.norm(sm)))
    print("scored %d units"%len(geo))

    #: mass, per arm, from the twp store
    pairs=json.load(open(os.path.join(ROOT,"data","base_aligned_pairs.json")))
    cm=CacheManager()
    ROLES=("pole_a","pole_b","both","control_a","control_b","both_matched")
    struct=lambda s: bool(s) and (set(s)<=set("_＿") or s.isdigit() or any(not c.isalnum() for c in s))
    rows=[]
    for pr in pairs:
        for g in groups:
            q=Q[g]
            cell={}
            for arm,mid in (("base",pr["base"]),("aligned",pr["aligned"])):
                tot=0.0; wa=0.0; wsigned=0.0; ws=0.0; blank=0.0
                for role in ROLES:
                    t=q.get(role)
                    if not t: continue
                    v=cm.get_true_word_probs(mid,t)
                    if not v or not v.get("rows"): continue
                    for x in v["rows"]:
                        k=(x["word"],g)
                        if k not in geo: continue
                        p=x["p"]; ta,ts=geo[k]
                        tot+=p; wa+=p*abs(ta); wsigned+=p*ta; ws+=p*ts
                        if struct(x["word"]): blank+=p
                if tot>0: cell[arm]=(tot,wa/tot,ws/tot,blank/tot,wsigned/tot)
            if len(cell)==2:
                rows.append({"family":pr["family"],"group":g,"lang":q["language"],
                             "base_polarity":cell["base"][1],"aligned_polarity":cell["aligned"][1],
                             "base_scene":cell["base"][2],"aligned_scene":cell["aligned"][2],
                             "base_struct":cell["base"][3],"aligned_struct":cell["aligned"][3],
                             "base_signed":cell["base"][4],"aligned_signed":cell["aligned"][4]})
    import statistics as st
    print("\n(pair, group) cells with both arms: %d"%len(rows))
    print("\n  MASS-WEIGHTED, base vs aligned, mean over cells")
    for lab,kb,ka in (("|t_axis|  pole polarity","base_polarity","aligned_polarity"),
                      ("t_sum     scene proximity","base_scene","aligned_scene"),
                      ("structural blank/punct share","base_struct","aligned_struct")):
        b=[r[kb] for r in rows]; a=[r[ka] for r in rows]
        d=[x-y for x,y in zip(a,b)]
        print("    %-30s base %.4f  aligned %.4f  delta %+.4f  (n=%d, %d%% positive)"
              %(lab,st.mean(b),st.mean(a),st.mean(d),len(d),round(100*sum(x>0 for x in d)/len(d))))
    for L in ("en","zh"):
        sub=[r for r in rows if r["lang"]==L]
        if not sub: continue
        d=[r["aligned_scene"]-r["base_scene"] for r in sub]
        ds=[r["aligned_struct"]-r["base_struct"] for r in sub]
        print("    %s  n=%4d   scene delta %+.4f   structural delta %+.4f"%(L,len(sub),st.mean(d),st.mean(ds)))
    #: SIGNED, PER GROUP. |t_axis| cannot see a pole SWITCH -- a model moving
    #: mass from POLE1 to POLE2 leaves it unchanged. Signed t_axis can, but its
    #: sign convention is arbitrary BETWEEN groups (which prompt is "a" is a
    #: file-order fact), so it is only interpretable WITHIN one. RH's fix:
    #: report by prompt. Most groups are valenced with a = positive
    #: (loved/hated, free/captive); the nine CATEGORY-pole groups are not
    #: (man/woman, mother/father) and their sign means nothing even within.
    print("\n  SIGNED t_axis BY GROUP, mass-weighted (positive = toward pole_a)")
    print("  %-22s %-4s %9s %9s %9s  %s"%("group","lang","base","aligned","delta","pole_a / pole_b"))
    byg=collections.defaultdict(list)
    for r in rows: byg[r["group"]].append(r)
    CAT={"f11_gender","f11_gender_he","f11_gender_she","f11_parent","f11_species",
         "f11_gender_zh","f11_gender_he_zh","f11_gender_she_zh","f11_parent_zh"}
    out=[]
    for g in sorted(byg,key=lambda x:-abs(st.mean(r["aligned_signed"]-r["base_signed"] for r in byg[x]))):
        v=byg[g]; b=st.mean(r["base_signed"] for r in v); a=st.mean(r["aligned_signed"] for r in v)
        out.append((g,a-b))
        pa=Q[g]["pole_a"].split()[-4:]; pb=Q[g]["pole_b"].split()[-4:]
        print("  %-22s %-4s %+9.4f %+9.4f %+9.4f  %s%s"%(g,Q[g]["language"],b,a,a-b,
              "CATEGORY POLE " if g in CAT else "", " ".join(pa[:2])+" / "+" ".join(pb[:2])))
    big=[x for x in out if abs(x[1])>0.005]
    print("\n  groups with |delta| > 0.005: %d of %d"%(len(big),len(out)))
    json.dump(rows,open(os.path.join(CAMP,"results","l1_geometric_full.json"),"w"))
    print("\nwrote results/l1_geometric_full.json")
    return 0

if __name__=="__main__": sys.exit(main())
