"""Contextual-delta geometry against the 146 labelled units. RH's design.

    s = embed(prompt + " " + word) - embed(prompt)

The word's contribution IN CONTEXT, rather than the bare word's embedding. The
bare-word version measured lexical overlap with the prompt: `him` scored as
maximally in-scene because "him" is in the sentence, `____him` likewise, while
`provide` and `wait` -- ordinary continuations -- scored as most out-of-frame
for sharing no vocabulary. Found by reading the examples, not the AUCs.

Three coordinates:
    t_axis  cos(delta, a-b)   which pole
    t_sum   cos(delta, a+b)   scene proximity -- the one that failed before
    mag     |delta|           how much the word changes the sentence at all
"""
import collections, json, os, sys, statistics as st
HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,ROOT)

def auc(p,n):
    if not p or not n: return float("nan")
    return sum((x>y)+0.5*(x==y) for x in p for y in n)/(len(p)*len(n))

def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    R=[json.loads(l) for l in open(os.path.join(CAMP,"results","l1_frame_enriched.jsonl"))]
    idx=collections.defaultdict(dict)
    for r in R: idx[(r["group"],r["s"])][r["vendor"]]=(r["cls"],r["content"])
    lab={k:list(v.values())[0] for k,v in idx.items()
         if len(v)==2 and len(set(v.values()))==1}
    Q={q["group"]:q for q in json.load(open(os.path.join(ROOT,"data","f11_quintuplets.json")))["quintuplets"]}
    print("both-agreed labels: %d"%len(lab))
    m=SentenceTransformer("BAAI/bge-m3")
    gs=sorted({g for g,_ in lab})
    prm=m.encode([Q[g]["both"] for g in gs]+[Q[g]["pole_a"] for g in gs]+[Q[g]["pole_b"] for g in gs],
                 normalize_embeddings=True,batch_size=64,show_progress_bar=False)
    P={g:prm[i] for i,g in enumerate(gs)}
    A={g:prm[len(gs)+i] for i,g in enumerate(gs)}; B={g:prm[2*len(gs)+i] for i,g in enumerate(gs)}
    items=sorted(lab)
    ctx=m.encode([Q[g]["both"]+" "+s for g,s in items],normalize_embeddings=True,
                 batch_size=64,show_progress_bar=False)
    rows=[]
    for (g,s),c in zip(items,ctx):
        d=c-P[g]; mag=float(np.linalg.norm(d)); dn=d/(mag+1e-9)
        ax=A[g]-B[g]; sm=A[g]+B[g]
        cls,content=lab[(g,s)]
        rows.append({"g":g,"s":s,"cls":cls,"content":content,"mag":mag,
                     "t_axis":float(dn@ax/np.linalg.norm(ax)),
                     "t_sum":float(dn@sm/np.linalg.norm(sm))})
    print("\n  %-16s %5s %9s %9s %9s"%("class","n","|delta|","t_sum","|t_axis|"))
    for c in ("IN-FRAME","OFF-FRAME","BLANK-TEMPLATE"):
        sub=[r for r in rows if r["cls"]==c]
        if not sub: continue
        print("  %-16s %5d %9.4f %9.4f %9.4f"%(c,len(sub),st.mean(r["mag"] for r in sub),
              st.mean(r["t_sum"] for r in sub),st.mean(abs(r["t_axis"]) for r in sub)))
    inf=[r for r in rows if r["cls"]=="IN-FRAME"]
    neg=[r for r in rows if r["cls"]!="IN-FRAME"]
    print("\n  IN-FRAME vs (OFF + BLANK)")
    for k,lab2 in (("t_sum","scene proximity"),("mag","|delta| magnitude")):
        print("    %-22s AUC %.3f"%(lab2,auc([r[k] for r in inf],[r[k] for r in neg])))
    print("    (bare-word centred t_sum scored 0.953 on the same labels)")
    print("\n  CONTENT FIELD, does |delta| recover it?")
    cy=[r["mag"] for r in rows if r["content"]=="YES"]; cn=[r["mag"] for r in rows if r["content"]=="NO"]
    print("    content YES |delta| %.4f (n=%d)   NO %.4f (n=%d)   AUC %.3f"
          %(st.mean(cy),len(cy),st.mean(cn),len(cn),auc(cy,cn)))
    json.dump(rows,open(os.path.join(CAMP,"results","l1_ctx_geometry.json"),"w"),ensure_ascii=False)
    return 0

if __name__=="__main__": sys.exit(main())
