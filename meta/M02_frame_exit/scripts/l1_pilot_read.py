"""Read the L1 pilot: agreement, the variant effect, and batch-50 drift."""
import collections, json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
R=[json.loads(l) for l in open(os.path.join(CAMP,"results","l1_pilot_coded.jsonl"))]
print("unit-codings %d   kinds %s"%(len(R),dict(collections.Counter(r["kind"] for r in R))))
VEN=sorted({r["vendor"] for r in R})

def kappa(a,b):
    cats=sorted(set(a)|set(b)); n=len(a)
    po=sum(x==y for x,y in zip(a,b))/n
    pe=sum((a.count(c)/n)*(b.count(c)/n) for c in cats)
    return po,pe,(po-pe)/(1-pe) if pe<1 else float("nan")

print("\n=== CLASS DISTRIBUTION, stratified arm ===")
S=[r for r in R if r["kind"]=="stratified"]
for lang in ("en","zh"):
    for variant in ("zeroshot","fewshot"):
        sub=[r for r in S if r["lang"]==lang and r["variant"]==variant]
        c=collections.Counter(r["cls"] for r in sub); n=len(sub) or 1
        pole=(c["POLE1"]+c["POLE2"])/n
        print("  %-3s %-9s n=%4d  POLE %.3f  IN-FRAME %.3f  OFF %.3f  BLANK %.3f  contentNO %.3f"
              %(lang,variant,n,pole,c["IN-FRAME"]/n,c["OFF-FRAME"]/n,c["BLANK-TEMPLATE"]/n,
                sum(1 for r in sub if r["content"]=="NO")/n))

print("\n=== VENDOR AGREEMENT (the kappa), stratified ===")
for lang in ("en","zh"):
    for variant in ("zeroshot","fewshot"):
        idx=collections.defaultdict(dict)
        for r in S:
            if r["lang"]==lang and r["variant"]==variant:
                idx[(r["group"],r["s"])][r["vendor"]]=(r["cls"],r["content"])
        keys=[k for k,v in idx.items() if len(v)==2]
        if not keys: continue
        a=[idx[k][VEN[0]][0] for k in keys]; b=[idx[k][VEN[1]][0] for k in keys]
        po,pe,kp=kappa(a,b)
        ca=[idx[k][VEN[0]][1] for k in keys]; cb=[idx[k][VEN[1]][1] for k in keys]
        po2,_,kp2=kappa(ca,cb)
        print("  %-3s %-9s n=%4d  cls raw %.3f pe %.3f kappa %.3f   content raw %.3f kappa %.3f"
              %(lang,variant,len(keys),po,pe,kp,po2,kp2))

print("\n=== THE VARIANT EFFECT: does the example block move the answer? ===")
for lang in ("en","zh"):
    for ven in VEN:
        idx=collections.defaultdict(dict)
        for r in S:
            if r["lang"]==lang and r["vendor"]==ven:
                idx[(r["group"],r["s"])][r["variant"]]=r["cls"]
        keys=[k for k,v in idx.items() if len(v)==2]
        z=[idx[k]["zeroshot"] for k in keys]; f=[idx[k]["fewshot"] for k in keys]
        pz=sum(1 for x in z if x.startswith("POLE"))/max(len(z),1)
        pf=sum(1 for x in f if x.startswith("POLE"))/max(len(f),1)
        po,_,kp=kappa(z,f)
        print("  %-3s %-22s n=%4d  POLE share zeroshot %.3f  fewshot %.3f  delta %+.3f   agreement %.3f"
              %(lang,ven.split("/")[-1],len(keys),pz,pf,pf-pz,po))

print("\n=== BATCH-50 DRIFT: first half vs second half of the full-size batches ===")
I=[r for r in R if r["kind"]=="integrity"]
for ven in VEN:
    for variant in ("zeroshot","fewshot"):
        sub=[r for r in I if r["vendor"]==ven and r["variant"]==variant]
        if not sub: continue
        h1=[r for r in sub if r["pos"]<=25]; h2=[r for r in sub if r["pos"]>25]
        f=lambda g: sum(1 for r in g if r["cls"].startswith("POLE"))/max(len(g),1)
        w=lambda g: sum(len(r["why"]) for r in g)/max(len(g),1)
        print("  %-22s %-9s  POLE 1st half %.3f  2nd half %.3f  delta %+.3f   why-len %.0f -> %.0f"
              %(ven.split("/")[-1],variant,f(h1),f(h2),f(h2)-f(h1),w(h1),w(h2)))
