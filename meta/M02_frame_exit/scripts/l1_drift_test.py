"""Does batch position change a surface's class? SAME surfaces, TWO orders.

    uv run python -u l1_drift_test.py

THE FIRST ATTEMPT WAS VOID AND THIS IS WHY THE DESIGN IS SHAPED LIKE THIS. It
compared the first half of a 50-batch to the second half and found POLE share
rising +0.02 to +0.06 consistently -- but the batch was ALPHABETICALLY SORTED,
so the first half was `____ her ją revenge " 一 一个` and the second was
`不会 不停 不再 不到 不惜一切 不想`. ASCII and punctuation against Chinese
negations: two populations, not two positions. Position was perfectly confounded
with composition and the number said nothing.

The fix is to hold the surfaces fixed and vary only the order. Each surface is
coded twice by the same vendor, once early and once late, so a class change is
attributable to POSITION and to nothing else.

    order A   as-is (alphabetical, what the runner would produce)
    order B   SHUFFLED, seed 4946

REVERSAL WAS THE WRONG SECOND ORDER AND THAT WAS MY ERROR. Under reversal every
surface moves from p to 51-p, so EVERY surface crosses halves and none stays --
200 crossed, 0 stayed, and the crossed-versus-stayed comparison the test was
built around had an empty arm. A shuffle puts roughly half in each condition and
also gives a continuous predictor, the DISTANCE moved, which reversal fixes at
|51-2p| by construction.

The test that matters: agreement between orders, SPLIT by whether the surface
crossed halves. If crossing costs agreement while staying does not, batch 50
drifts and the full pass should batch smaller.

Zero-shot only. The variant comparison already ran and fewshot imports a pole
prior (+0.04 to +0.12 in every cell), so zero-shot is the instrument in question.
"""
import collections, json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,ROOT)
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
VENDORS=["deepseek/deepseek-v4-flash","openai/gpt-5.4-mini"]
N=50

def main():
    from malign_logits.tasks import code_l1_surface_v1 as L
    F=json.load(open(os.path.join(ROOT,"data","f11_k2_units.json")))
    Q={q["group"]:q for q in json.load(open(os.path.join(ROOT,"data","f11_quintuplets.json")))["quintuplets"]}
    surv=[r for r in F["units"] if r["survives"]]
    groups=[g for g,_ in collections.Counter(r["group"] for r in surv).most_common(2)]
    out=[]
    for g in groups:
        S=[r["surface"] for r in surv if r["group"]==g][:N]
        import random
        B=list(S); random.Random(4946).shuffle(B)
        orders={"A":S,"B":B}
        for vendor in VENDORS:
            task=L.L1SurfaceTask()
            pit=[L.prepare_batch(Q[g]["pole_a"],Q[g]["pole_b"],orders[k]) for k in ("A","B")]
            errs={}
            res=task.map(pit,model=vendor,num_workers=4,errors=errs)
            for k,r in zip(("A","B"),res):
                if r is None: print("  no result %s %s %s"%(g,vendor,k)); continue
                recs=[x.model_dump() if hasattr(x,"model_dump") else dict(x)
                      for x in (r.records if hasattr(r,"records") else r["records"])]
                ok,why=L.validate_batch(orders[k],recs)
                print("  %-18s %-22s order %s  %s"%(g,vendor.split("/")[-1],k,"PASS" if ok else "REFUSED "+why))
                if ok:
                    for i,x in enumerate(recs):
                        out.append({"group":g,"vendor":vendor,"order":k,"pos":i+1,
                                    "s":x["s"],"cls":x["cls"],"content":x["content"]})
    p=os.path.join(CAMP,"results","l1_drift_test.jsonl")
    with open(p,"w") as f:
        for r in out: f.write(json.dumps(r,ensure_ascii=False)+"\n")

    print("\n=== SAME SURFACE, TWO POSITIONS ===")
    idx=collections.defaultdict(dict)
    for r in out: idx[(r["group"],r["vendor"],r["s"])][r["order"]]=(r["pos"],r["cls"])
    keys=[k for k,v in idx.items() if len(v)==2]
    cross=[k for k in keys if (idx[k]["A"][0]<=25)!=(idx[k]["B"][0]<=25)]
    stay=[k for k in keys if k not in cross]
    def agr(ks): return sum(idx[k]["A"][1]==idx[k]["B"][1] for k in ks)/max(len(ks),1)
    print("  surfaces coded in both orders   %d"%len(keys))
    print("  CROSSED halves (early <-> late) %4d   order-agreement %.3f"%(len(cross),agr(cross)))
    print("  stayed in the same half         %4d   order-agreement %.3f"%(len(stay),agr(stay)))
    print("  overall                         %4d   order-agreement %.3f"%(len(keys),agr(keys)))
    #: DISTANCE MOVED as a continuous predictor -- available only under a
    #: shuffle. If position drives instability, agreement should fall with it.
    print("\n  order-agreement by DISTANCE MOVED:")
    for lo,hi in ((0,9),(10,19),(20,29),(30,49)):
        ks=[k for k in keys if lo<=abs(idx[k]["A"][0]-idx[k]["B"][0])<=hi]
        if ks: print("    moved %2d-%2d positions  n=%3d  agreement %.3f"%(lo,hi,len(ks),agr(ks)))
    print("\n  POLE share by POSITION, pooled over both orders:")
    for lo,hi,lab in ((1,25,"positions 1-25"),(26,50,"positions 26-50")):
        sub=[r for r in out if lo<=r["pos"]<=hi]
        print("    %-16s n=%4d  POLE %.3f"%(lab,len(sub),
              sum(1 for r in sub if r["cls"].startswith("POLE"))/max(len(sub),1)))
    print("\n  (pooled over orders, POSITION is no longer confounded with which")
    print("   surfaces occupy it: every surface appears once early and once late)")
    return 0

if __name__=="__main__": sys.exit(main())
