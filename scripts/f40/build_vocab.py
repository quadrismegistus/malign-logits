"""Build the discovered vocabulary: every word any model puts in its top-10 at
any prompt, with cross-family movement statistics. This is the object to tag.

Fragment filter: beam decoding leaks subword pieces (thro, und, sl, CC). Require
alphabetic, length >= 3, and a nonzero English unigram frequency.
"""
import collections, json
import numpy as np
from math import comb
from wordfreq import word_frequency
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
    c=CAT.get(p); return 'unlabelled' if (not c or c=='?') else c
def ok(w):
    return w.isalpha() and len(w)>=3 and word_frequency(w.lower(),'en')>0

TOPK=10
W=collections.defaultdict(lambda: {"prompts":set(),"fams":set(),"delta":[],
                                   "up":0,"dn":0,"cats":collections.Counter()})
nraw=0
for pr in {p for (_,p) in idx}:
    c=cat(pr)
    pool=set(); cells=[]
    for _,b,a in pairs:
        kb,ka=idx.get((b,pr)),idx.get((a,pr))
        if kb is None or ka is None: continue
        wb,wa=s[kb],s[ka]
        if not isinstance(wb,dict) or not isinstance(wa,dict) or not wb or not wa: continue
        pool|=set(sorted(wb,key=lambda w:-wb[w])[:TOPK])|set(sorted(wa,key=lambda w:-wa[w])[:TOPK])
        cells.append((wb,wa))
    if len(cells)<12: continue
    nraw+=len(pool)
    for w in pool:
        if not ok(w): continue
        e=W[w]; e["prompts"].add(pr); e["cats"][c]+=1
        for i,(wb,wa) in enumerate(cells):
            d=wa.get(w,0.0)-wb.get(w,0.0)
            if d==0: continue
            e["delta"].append(d); e["fams"].add(i)
            if d>0: e["up"]+=1
            else: e["dn"]+=1
def sign_p(k,n): return min(1.0,2*sum(comb(n,i) for i in range(k,n+1))/2**n) if n else 1.0
rows=[]
for w,e in W.items():
    n=e["up"]+e["dn"]
    if n<25 or len(e["prompts"])<3: continue
    rows.append({"word":w,"n_obs":n,"n_prompts":len(e["prompts"]),"n_fam":len(e["fams"]),
                 "up_frac":round(e["up"]/n,3),"mean_delta_pct":round(100*float(np.mean(e["delta"])),3),
                 "sign_p":sign_p(max(e["up"],e["dn"]),n),
                 "top_category":e["cats"].most_common(1)[0][0],
                 "freq_zipf":round(np.log10(word_frequency(w.lower(),'en')*1e9+1),2)})
rows.sort(key=lambda r:(r["sign_p"], -abs(r["up_frac"]-.5)))
import csv
with open("data/discovered_vocabulary.csv","w",newline="") as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
print(f"pooled word slots before filtering: {nraw}")
print(f"vocabulary passing filter (>=25 obs, >=3 prompts): {len(rows)} words")
print(f"  consistent RISERS (up_frac>.5, p<.01): {sum(1 for r in rows if r['up_frac']>.5 and r['sign_p']<.01)}")
print(f"  consistent FALLERS (up_frac<.5, p<.01): {sum(1 for r in rows if r['up_frac']<.5 and r['sign_p']<.01)}")
print("\ntop 15 risers:"); [print(f"   {r['word']:14s} up={r['up_frac']:.2f} d={r['mean_delta_pct']:+.2f}% np={r['n_prompts']:2d} [{r['top_category']}]") for r in rows if r['up_frac']>.5][:15]
print("\ntop 15 fallers:"); [print(f"   {r['word']:14s} up={r['up_frac']:.2f} d={r['mean_delta_pct']:+.2f}% np={r['n_prompts']:2d} [{r['top_category']}]") for r in rows if r['up_frac']<.5][:15]
print("\nwrote data/discovered_vocabulary.csv")
