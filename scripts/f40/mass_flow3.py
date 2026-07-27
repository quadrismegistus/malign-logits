"""Sensitivity: v1 (narrow PROCEDURAL) vs v2 (widened) tagging, excess over the
proportional-drain null. Plus a v2 run dropping low-confidence assignments."""
import collections, csv, json
import numpy as np
from malign_logits.cache import open_stash
from malign_logits import MODEL_FAMILIES
import malign_logits.experiments as E
def load(p, drop_low=False):
    out={}
    for r in csv.DictReader(open(p)):
        if drop_low and r.get('confidence')=='low': continue
        out[r['word']]=r['primary']
    return out
T1=load('/tmp/vocab_tagged.csv'); T2=load('/tmp/vocab_tagged_v2.csv')
T2h=load('/tmp/vocab_tagged_v2.csv',drop_low=True)
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
TAGS=['PROCEDURAL','CONTESTATION','TRANSGRESSIVE','DEMOTIC','NARRATIVE_CRAFT','AFFECT','OTHER']
def matrix(TAG):
    bm=collections.defaultdict(lambda: collections.defaultdict(list))
    dm=collections.defaultdict(lambda: collections.defaultdict(list))
    for pr in {p for (_,p) in idx}:
        c=pcat(pr)
        if c is None: continue
        B=collections.defaultdict(list); D=collections.defaultdict(list); n=0
        for _,b,a in pairs:
            kb,ka=idx.get((b,pr)),idx.get((a,pr))
            if kb is None or ka is None: continue
            wb,wa=s[kb],s[ka]
            if not isinstance(wb,dict) or not isinstance(wa,dict) or not wb or not wa: continue
            n+=1
            bb=collections.defaultdict(float); dd=collections.defaultdict(float)
            for w,t in TAG.items():
                bb[t]+=wb.get(w,0.0); dd[t]+=wa.get(w,0.0)-wb.get(w,0.0)
            for t in TAGS: B[t].append(bb[t]); D[t].append(dd[t])
        if n<12: continue
        for t in TAGS: bm[c][t].append(float(np.mean(B[t]))); dm[c][t].append(float(np.mean(D[t])))
    out={}
    for c in bm:
        b=np.array([np.mean(bm[c][t]) for t in TAGS]); d=np.array([np.mean(dm[c][t]) for t in TAGS])
        Tt=d.sum(); sh=b/b.sum() if b.sum()>0 else b*0
        out[c]=dict(zip(TAGS,d-Tt*sh))
    return out
M1,M2,M2h=matrix(T1),matrix(T2),matrix(T2h)
ORDER=['institutional','neutral','sexual_explicit','sexual_liminal','violence_explicit',
       'violence_liminal','death','power','profanity','substance']
print("EXCESS OVER PROPORTIONAL-DRAIN NULL (pp).  v1 = narrow PROCEDURAL | v2 = widened\n")
for t in ['PROCEDURAL','DEMOTIC','TRANSGRESSIVE','OTHER']:
    print(f"--- {t} ---")
    print(f"{'prompt category':20s}{'v1':>9s}{'v2':>9s}{'v2 hi-conf':>12s}")
    for c in ORDER:
        if c not in M1: continue
        print(f"{c:20s}{100*M1[c][t]:+9.2f}{100*M2[c][t]:+9.2f}{100*M2h[c][t]:+12.2f}")
    print()
