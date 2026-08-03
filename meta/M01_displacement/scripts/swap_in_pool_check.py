import sys, json, collections, hashlib
sys.path.insert(0,"meta/M01_displacement/scripts"); sys.path.insert(0,"scripts")
import within_pair as W, m01_norms as N, m01_registration_b as B, m01_concentration as CC

rows = json.load(open(W.CAT))["prompts"]
rows = list(rows.values()) if isinstance(rows,dict) else rows
contrast = {r["pair_id"]: r.get("pair_contrast") for r in rows
            if r.get("pair_role") and str(r.get("source","")).startswith("M01_PAIRS")}
pairs,_ = W.m01_pairs()
_p,models,_h,drift = CC.frozen_population()
edges,_ = CC.operation_edges(models)
norms,_f,_r = N.load_norms(verify=True)
tabs = {d: norms[("en",d,"primary")] for d in ("arousal","valence","dominance")}
texts = {t for v in pairs.values() for t in v.values()}

#: the QUALIFYING pool per member, as SURFACES -- same filter chain the
#: statistic uses, so the question is asked of the pool the statistic reads
pool = collections.defaultdict(set)
for fam,pos,step in sorted(edges):
    for t in texts:
        c = step.cell(t)
        if not c.is_present or c.language != "en": continue
        try:
            if not c.decompose(None): continue
            roles = N.cell_roles(c, "CANONICAL")
        except Exception: continue
        keep = []
        for w,wt,role in roles:
            k = N.norm_key(w,"en",fold=False)
            if N.is_function_word(k,"en"): continue
            if any(N.lookup(tabs[d],k.casefold(),"en")[0] is None
                   for d in ("arousal","valence","dominance")): continue
            keep.append((w,role))
        nf = sum(1 for _,r in keep if r=="faller")
        if nf < B.QUALIFYING_MIN or len(keep)-nf < B.QUALIFYING_MIN: continue
        for w,_ in keep: pool[t].add(w.strip().lower())

n=collections.Counter(); ex=[]
for pid,members in pairs.items():
    c=contrast.get(pid) or ""
    if "->" not in c: continue
    mw,uw=[x.strip().lower() for x in c.split("->",1)]
    for role,want in (("MARKED",mw),("UNMARKED",uw)):
        t=members.get(role); p=pool.get(t)
        if not p: n["member has no qualifying pool"]+=1; continue
        n["members checked"]+=1
        if want in p:
            n[f"{role}: SWAP IS IN THE POOL"]+=1
            if len(ex)<5: ex.append((pid,role,want))
        else:
            n[f"{role}: swap absent"]+=1
print(f"members with a qualifying pool : {n['members checked']}")
print(f"members without one            : {n['member has no qualifying pool']}")
print()
for k in ("MARKED: SWAP IS IN THE POOL","MARKED: swap absent",
          "UNMARKED: SWAP IS IN THE POOL","UNMARKED: swap absent"):
    print(f"  {k:34s} {n[k]}")
tot=n["MARKED: SWAP IS IN THE POOL"]+n["UNMARKED: SWAP IS IN THE POOL"]
print(f"\n  TOTAL members whose own swap appears in their pool: {tot}"
      f"  ({100*tot/max(n['members checked'],1):.2f}%)")
if ex: print("  examples:", ex)
