"""F06-style tagged-mass analysis with a DISCOVERED and BLIND-TAGGED vocabulary.
Rows = prompt category. Columns = word tag. Cells = mean probability mass moved
from base to aligned, summed over words of that tag, averaged over families."""
import collections, csv, json
import numpy as np
from malign_logits.cache import open_stash
from malign_logits import MODEL_FAMILIES
import malign_logits.experiments as E
TAG={r['word']:(r['primary'],r['secondary']) for r in csv.DictReader(open('/tmp/vocab_tagged.csv'))}
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
def pcat(p):
    if p in INST: return 'institutional'
    c=CAT.get(p); return None if (not c or c=='?') else c
def run(promote_secondary_for_past=False):
    tagof={}
    for w,(p1,p2) in TAG.items():
        t=p1
        if promote_secondary_for_past and p1=='OTHER' and p2=='NARRATIVE_CRAFT': t=p2
        tagof[w]=t
    cell=collections.defaultdict(lambda: collections.defaultdict(list))
    for pr in {p for (_,p) in idx}:
        c=pcat(pr)
        if c is None: continue
        per=collections.defaultdict(list)
        n=0
        for _,b,a in pairs:
            kb,ka=idx.get((b,pr)),idx.get((a,pr))
            if kb is None or ka is None: continue
            wb,wa=s[kb],s[ka]
            if not isinstance(wb,dict) or not isinstance(wa,dict) or not wb or not wa: continue
            n+=1
            acc=collections.defaultdict(float)
            for w,t in tagof.items():
                d=wa.get(w,0.0)-wb.get(w,0.0)
                if d: acc[t]+=d
            for t,v in acc.items(): per[t].append(v)
        if n<12: continue
        for t,v in per.items(): cell[c][t].append(float(np.mean(v)))
    return cell
cell=run()
TAGS=['PROCEDURAL','CONTESTATION','TRANSGRESSIVE','DEMOTIC','NARRATIVE_CRAFT','AFFECT','OTHER']
ORDER=['institutional','neutral','sexual_explicit','sexual_liminal','violence_explicit',
       'violence_liminal','death','power','profanity','substance']
print("MASS MOVED base->aligned, percentage points (mean over prompts, then families)\n")
print(f"{'prompt category':20s}"+"".join(f"{t[:9]:>11s}" for t in TAGS))
for c in ORDER:
    if c not in cell: continue
    row=f"{c:20s}"
    for t in TAGS:
        v=cell[c].get(t)
        row+=f"{100*np.mean(v):+11.2f}" if v else f"{'--':>11s}"
    print(row)
print("\nSENSITIVITY: promoting NARRATIVE_CRAFT secondary for base-form verbs (agent's flagged rule 2)")
c2=run(True)
print(f"{'prompt category':20s}{'NARR_before':>13s}{'NARR_after':>12s}{'OTHER_before':>14s}{'OTHER_after':>13s}")
for c in ORDER[:4]:
    if c not in cell: continue
    print(f"{c:20s}{100*np.mean(cell[c]['NARRATIVE_CRAFT']):+13.2f}{100*np.mean(c2[c]['NARRATIVE_CRAFT']):+12.2f}"
          f"{100*np.mean(cell[c]['OTHER']):+14.2f}{100*np.mean(c2[c]['OTHER']):+13.2f}")
