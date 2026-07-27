"""Site-specific discovered vocabulary: pool the top-10 words every model puts at
each prompt, then measure mass flow within that pooled set. No hand-built list."""
import collections, json
import numpy as np
from malign_logits.cache import open_stash
from malign_logits import MODEL_FAMILIES
import malign_logits.experiments as E
s=open_stash('data/raw/cache/word_probs')
idx={}
for k in s.keys():
    if isinstance(k,dict) and k.get('mode','raw')=='raw': idx[(k['model'],k['prompt'])]=k
seen={}
for k,f in MODEL_FAMILIES.items():
    b,a=f.base,getattr(f,'superego',None)
    if b and a and b not in seen: seen[b]=(k,b,a)
pairs=list(seen.values())
ip=getattr(E,'INSTITUTIONAL_PROMPTS',{}); INST=set(ip.values() if isinstance(ip,dict) else ip)
items=json.load(open('data/f37_prompt_items.json'))+json.load(open('data/f37_prompt_items_supp.json'))
CAT={i['text']:i.get('category','?') for i in items}
def cat(p):
    if p in INST: return 'institutional'
    c=CAT.get(p); return None if (not c or c=='?') else c
OLD13={"kill","fuck","die","murder","stab","cock","naked","punch","hit","shoot","slap","strangle","blood"}
TOPK=10
bycat=collections.defaultdict(lambda: collections.defaultdict(list))
vsize=[]; cover=[]
allp={p for (_,p) in idx}
for pr in sorted(allp):
    c=cat(pr)
    if c is None: continue
    vocab=set(); deltas=collections.defaultdict(list)
    used=0
    for _,b,a in pairs:
        kb,ka=idx.get((b,pr)),idx.get((a,pr))
        if kb is None or ka is None: continue
        wb,wa=s[kb],s[ka]
        if not isinstance(wb,dict) or not isinstance(wa,dict) or not wb or not wa: continue
        used+=1
        vocab|=set(sorted(wb,key=lambda w:-wb[w])[:TOPK])|set(sorted(wa,key=lambda w:-wa[w])[:TOPK])
        for w in vocab:
            deltas[w].append(wa.get(w,0.0)-wb.get(w,0.0))
    if used<12 or not vocab: continue
    vsize.append(len(vocab)); cover.append(len(vocab&OLD13)/len(vocab))
    for w,d in deltas.items():
        if len(d)>=12: bycat[c][w].append(float(np.mean(d)))
print(f"prompts: {len(vsize)}   site vocabulary size: median {np.median(vsize):.0f}, "
      f"range {min(vsize)}-{max(vsize)}")
print(f"fraction of site vocabulary covered by the in-repo 13-word list: {100*np.mean(cover):.1f}%\n")
ORDER=['institutional','neutral','sexual_explicit','violence_explicit','sexual_liminal',
       'violence_liminal','death','power','profanity','substance']
for c in ORDER:
    if c not in bycat: continue
    agg={w:float(np.mean(v)) for w,v in bycat[c].items() if len(v)>=2}
    if len(agg)<10: continue
    up=sorted(agg.items(),key=lambda kv:-kv[1])[:8]
    dn=sorted(agg.items(),key=lambda kv:kv[1])[:8]
    flow=sum(v for v in agg.values() if v>0)
    print(f"--- {c}  (vocab {len(agg)} words, mass gained by risers {100*flow:.1f}%) ---")
    print("   GAIN: "+", ".join(f"{w}+{100*v:.1f}" for w,v in up))
    print("   LOSE: "+", ".join(f"{w}{100*v:.1f}" for w,v in dn))
