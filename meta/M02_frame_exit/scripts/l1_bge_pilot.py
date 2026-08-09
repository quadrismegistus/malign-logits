"""Can an embedder replace the LLM coder on the POLE AXIS? RH's proposal.

    uv run python -u l1_bge_pilot.py

THE CASE FOR TRYING. The LLM coder is deterministic at temperature 0 but not
stable to CONTEXT: reordering a batch of 50 moves 6% of classifications and
changing batch size moves 14%. Batching is what made 55,055 units affordable and
it is also the instability. A projection has no batch, no order and no context.

It also inverts the cost. Embedding is per SURFACE; projection is per PAIR and
free. So 16,716 surfaces embed once and all 55,055 units fall out as arithmetic
-- the pair-relativity explosion that took this arm from 3,284 to 55,055 stops
being a cost.

THE MEASURES, per (surface, pair), with a = embed(prompt_a), b = embed(prompt_b):

    t_axis   cos(s, a - b)     signed position on the pole axis. RH's "a - b".
    prox_a   cos(s, a)         proximity to each pole separately
    prox_b   cos(s, b)
    t_sum    cos(s, a + b)     RH's "a + b": what the two poles SHARE, which is
                               the scene. If in-frame is reachable by embedding
                               at all, this is the coordinate that reaches it --
                               the one thing RH doubted could be done.

THE NULL IS NOT OPTIONAL. Cosines have no absolute meaning, so random surface
pairs give the baseline distribution every number above is read against.

VALIDATION IS AGAINST THE LLM PILOT'S OWN LABELS, restricted to units where BOTH
vendors agreed -- the subset where the categorical instrument is at its most
reliable. If the geometry separates those classes it can carry the pole axis; if
it separates POLE from IN-FRAME but not IN-FRAME from OFF-FRAME, that is the
hybrid (embeddings for the axis, a binary LLM call for the frame).
"""
import collections, json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
MODEL="BAAI/bge-m3"

def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    R=[json.loads(l) for l in open(os.path.join(CAMP,"results","l1_pilot_coded.jsonl"))
       if json.loads(l)["kind"]=="stratified"]
    Q={q["group"]:q for q in json.load(open(os.path.join(ROOT,"data","f11_quintuplets.json")))["quintuplets"]}
    #: agreement subset: both vendors, same variant, same class
    idx=collections.defaultdict(dict)
    for r in R:
        if r["variant"]=="zeroshot": idx[(r["group"],r["s"])][r["vendor"]]=r["cls"]
    agreed={k:list(v.values())[0] for k,v in idx.items() if len(v)==2 and len(set(v.values()))==1}
    print("zeroshot units: %d, both-vendor agreed: %d (%.0f%%)"
          %(len(idx),len(agreed),100*len(agreed)/max(len(idx),1)))
    print("class counts: %s"%dict(collections.Counter(agreed.values())))

    m=SentenceTransformer(MODEL)
    groups=sorted({g for g,_ in agreed})
    texts=[]; meta=[]
    for g in groups:
        texts += [Q[g]["pole_a"], Q[g]["pole_b"]]
    surfaces=sorted({s for _,s in agreed})
    E=m.encode(texts+surfaces, normalize_embeddings=True, show_progress_bar=False,
               batch_size=64)
    pole={g:(E[2*i], E[2*i+1]) for i,g in enumerate(groups)}
    se={s:E[len(texts)+i] for i,s in enumerate(surfaces)}

    rows=[]
    for (g,s),cls in agreed.items():
        a,b=pole[g]; v=se[s]
        ax=a-b; sm=a+b
        rows.append({"group":g,"s":s,"cls":cls,
                     "t_axis":float(v@ax/ (np.linalg.norm(ax)+1e-9)),
                     "prox_a":float(v@a), "prox_b":float(v@b),
                     "t_sum":float(v@sm/(np.linalg.norm(sm)+1e-9))})
    #: NULL: random surface pairs
    rng=np.random.default_rng(4946); S=np.array([se[s] for s in surfaces])
    i=rng.integers(0,len(S),4000); j=rng.integers(0,len(S),4000)
    null=(S[i]*S[j]).sum(1)
    print("\nNULL, random surface-surface cosine: mean %.3f  sd %.3f  |95%%| %.3f"
          %(null.mean(),null.std(),np.quantile(np.abs(null),0.95)))

    print("\n%-16s %5s %9s %9s %9s %9s"%("class","n","t_axis","prox_a","prox_b","t_sum"))
    for c in ("POLE1","POLE2","IN-FRAME","OFF-FRAME","BLANK-TEMPLATE"):
        sub=[r for r in rows if r["cls"]==c]
        if not sub: continue
        f=lambda k: sum(r[k] for r in sub)/len(sub)
        print("%-16s %5d %9.4f %9.4f %9.4f %9.4f"%(c,len(sub),f("t_axis"),f("prox_a"),f("prox_b"),f("t_sum")))
    p1=[r["t_axis"] for r in rows if r["cls"]=="POLE1"]
    p2=[r["t_axis"] for r in rows if r["cls"]=="POLE2"]
    inf=[r["t_axis"] for r in rows if r["cls"]=="IN-FRAME"]
    import statistics as st
    if p1 and p2:
        print("\n  POLE1 vs POLE2 on t_axis: %.4f vs %.4f  (separation %+.4f)"
              %(st.mean(p1),st.mean(p2),st.mean(p1)-st.mean(p2)))
        print("  IN-FRAME sits at %.4f, %s the two"
              %(st.mean(inf),"between" if min(st.mean(p1),st.mean(p2))<st.mean(inf)<max(st.mean(p1),st.mean(p2)) else "NOT between"))
    off=[r["t_sum"] for r in rows if r["cls"]=="OFF-FRAME"]
    ins=[r["t_sum"] for r in rows if r["cls"]=="IN-FRAME"]
    if off and ins:
        print("\n  RH's a+b coordinate, IN-FRAME %.4f vs OFF-FRAME %.4f  (gap %+.4f)"
              %(st.mean(ins),st.mean(off),st.mean(ins)-st.mean(off)))
    json.dump(rows,open(os.path.join(CAMP,"results","l1_bge_pilot.json"),"w"),ensure_ascii=False)
    return 0

if __name__=="__main__": sys.exit(main())
