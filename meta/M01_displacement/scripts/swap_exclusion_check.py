"""EXCLUSION FOLLOW-UP [3424]: does the marked-pool excess survive removing
the swap words themselves? If yes, the tautology channel is not carrying it.

DECLARED BEFORE RUNNING: the comparison is the SAME six quantities as the
[3397] diagnostic, computed on the same pools with one change -- a member's own
swapped word is dropped from its pool. Every other filter is identical, so any
difference is attributable to the exclusion and nothing else.
"""
import sys, json, collections, statistics as st, hashlib
sys.path.insert(0,"meta/M01_displacement/scripts"); sys.path.insert(0,"scripts")
import within_pair as W, m01_norms as N, m01_registration_b as B, m01_concentration as CC

rows = json.load(open(W.CAT))["prompts"]
rows = list(rows.values()) if isinstance(rows,dict) else rows
contrast = {r["pair_id"]: r.get("pair_contrast") for r in rows
            if r.get("pair_role") and str(r.get("source","")).startswith("M01_PAIRS")}
pairs,_ = W.m01_pairs()
_p,models,_h,_d = CC.frozen_population()
edges,_ = CC.operation_edges(models)
norms,_f,_r = N.load_norms(verify=True)
tabs = {d: norms[("en",d,"primary")] for d in ("arousal","valence","dominance")}

swap_of = {}
for pid,mem in pairs.items():
    c = contrast.get(pid) or ""
    if "->" not in c: continue
    a,b = [x.strip().lower() for x in c.split("->",1)]
    swap_of[mem["MARKED"]] = a; swap_of[mem["UNMARKED"]] = b

texts = {t for v in pairs.values() for t in v.values()}
prof = collections.defaultdict(lambda: {"all":[], "excl":[]})
for fam,pos,step in sorted(edges):
    for t in texts:
        c = step.cell(t)
        if not c.is_present or c.language != "en": continue
        try:
            if not c.decompose(None): continue
            roles = N.cell_roles(c,"CANONICAL")
        except Exception: continue
        keep=[]
        for w,wt,role in roles:
            k=N.norm_key(w,"en",fold=False)
            if N.is_function_word(k,"en"): continue
            z={d:N.lookup(tabs[d],k.casefold(),"en")[0] for d in ("arousal","valence","dominance")}
            if any(v is None for v in z.values()): continue
            keep.append((w,role,abs(z["valence"])))
        nf=sum(1 for _,r,_ in keep if r=="faller")
        if nf<B.QUALIFYING_MIN or len(keep)-nf<B.QUALIFYING_MIN: continue
        sw=swap_of.get(t)
        for w,_,az in keep:
            prof[t]["all"].append(az)
            if w.strip().lower()!=sw: prof[t]["excl"].append(az)

def stats(vals):
    if not vals: return None
    return {"mean_abs_z":st.mean(vals),
            "tail_ge_1":sum(1 for v in vals if v>=1)/len(vals),
            "tail_ge_2":sum(1 for v in vals if v>=2)/len(vals)}

for mode in ("all","excl"):
    got={t:stats(p[mode]) for t,p in prof.items() if stats(p[mode])}
    n=0; agg=collections.defaultdict(list)
    for pid,mem in pairs.items():
        M,U=got.get(mem["MARKED"]),got.get(mem["UNMARKED"])
        if not M or not U: continue
        n+=1
        for k in ("mean_abs_z","tail_ge_1","tail_ge_2"):
            agg[k].append((M[k],U[k]))
    lab="WITH swap words" if mode=="all" else "SWAP WORDS EXCLUDED"
    print(f"\n{lab}   pairs {n}")
    for k in ("mean_abs_z","tail_ge_1","tail_ge_2"):
        Ms=[a for a,_ in agg[k]]; Us=[b for _,b in agg[k]]
        d=[a-b for a,b in agg[k]]
        print(f"  {k:12s} M {st.mean(Ms):.5f}  U {st.mean(Us):.5f}  "
              f"M-U {st.mean(d):+.5f}  +/- {sum(1 for x in d if x>0)}/{sum(1 for x in d if x<0)}")
