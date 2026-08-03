"""The WORD-LEVEL POOL as an artifact. [3433].1.

WHY IT EXISTS: `built["cells"]` keeps roles, z-values and weights and DISCARDS
word identity, so swap-in-pool questions can only be asked upstream. That made
my checks single-seat-by-construction. This publishes what the upstream pass
sees, so a second seat can compute against it without re-deriving the chain.

**IT MUST AGGREGATE BACK TO THE FROZEN `cells`** -- that is [3433].2's bridge
and it is what upgrades this from "trusted code" to "anchored". Per cell, the
multiset of z-values and the weights here must equal the frozen producer's.
"""
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

#: keyed (member_text, family, position) -- the SAME cell key the producer uses,
#: so the bridge has something to join on
cells = {}
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
            keep.append({"word":w,"role":role,"w":wt,
                         "valence":z["valence"],"arousal":z["arousal"],
                         "dominance":z["dominance"]})
        nf=sum(1 for r in keep if r["role"]=="faller")
        if nf<B.QUALIFYING_MIN or len(keep)-nf<B.QUALIFYING_MIN: continue
        cells[f"{t}\x1f{fam}\x1f{pos}"]=keep

swap_of={}
for pid,mem in pairs.items():
    c=contrast.get(pid) or ""
    if "->" in c:
        a,b=[x.strip().lower() for x in c.split("->",1)]
        swap_of[mem["MARKED"]]=a; swap_of[mem["UNMARKED"]]=b

payload={
 "_what":"WORD-LEVEL POOL for D2's pool-extremity work. Per qualifying cell, "
         "the words the statistic reads with their roles, weights and V/A/D z.",
 "_why":"built['cells'] discards word identity; this publishes what the "
        "upstream pass sees so a second seat need not re-derive the chain.",
 "_bridge":"[3433].2 -- per cell, the multiset of z-values and the weights here "
           "MUST equal the frozen producer's cells. That is the anchor.",
 "_cell_key":"member_text \\x1f family \\x1f position",
 "_filters":"decompose non-empty; language en; cell_roles CANONICAL; "
            "function words dropped; any missing V/A/D drops the word; "
            f">= {B.QUALIFYING_MIN} words in EACH role",
 "_roster_drift": drift,
 "n_cells":len(cells),
 "n_words":sum(len(v) for v in cells.values()),
 "swap_word_by_member":swap_of,
 "cells":cells,
}
blob=json.dumps(payload,sort_keys=True,default=float)
out="meta/M01_displacement/results/word_pool_d2.json"
open(out,"w").write(blob)
print(f"{out}\n  sha256[:16] {hashlib.sha256(blob.encode()).hexdigest()[:16]}")
print(f"  cells {len(cells):,}   words {payload['n_words']:,}   "
      f"members with a swap recorded {len(swap_of):,}")
