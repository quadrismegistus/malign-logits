"""Concentration controls on the tagged mass-flow matrix.
(a) relative change per bin: delta / baseline bin mass
(b) excess over a proportional-drain null: if the total tagged-mass change were
    distributed proportionally to baseline bin shares, expected_t = T * share_t;
    report observed - expected.
Plus the OTHER/institutional gaining words."""
import collections, csv, json
import numpy as np
from malign_logits.cache import open_stash
from malign_logits import MODEL_FAMILIES
import malign_logits.experiments as E
TAG={r['word']:r['primary'] for r in csv.DictReader(open('/tmp/vocab_tagged.csv'))}
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
base_m=collections.defaultdict(lambda: collections.defaultdict(list))
delt_m=collections.defaultdict(lambda: collections.defaultdict(list))
other_gain=collections.defaultdict(list)
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
            if c=='institutional' and t=='OTHER':
                d=wa.get(w,0.0)-wb.get(w,0.0)
                if d: other_gain[w].append(d)
        for t in TAGS: B[t].append(bb[t]); D[t].append(dd[t])
    if n<12: continue
    for t in TAGS: base_m[c][t].append(float(np.mean(B[t]))); delt_m[c][t].append(float(np.mean(D[t])))
ORDER=['institutional','neutral','sexual_explicit','sexual_liminal','violence_explicit',
       'violence_liminal','death','power','profanity','substance']
print("(a) RELATIVE CHANGE per bin: 100 * delta / baseline bin mass\n")
print(f"{'prompt category':20s}"+"".join(f"{t[:9]:>11s}" for t in TAGS))
for c in ORDER:
    if c not in base_m: continue
    row=f"{c:20s}"
    for t in TAGS:
        b_=np.mean(base_m[c][t]); d_=np.mean(delt_m[c][t])
        row+=f"{100*d_/b_:+10.1f}%" if b_>1e-6 else f"{'--':>11s}"
    print(row)
print("\n(b) EXCESS OVER PROPORTIONAL-DRAIN NULL: observed - expected, pp")
print("    expected_t = T * (baseline share of bin t),  T = total tagged-mass change\n")
print(f"{'prompt category':20s}"+"".join(f"{t[:9]:>11s}" for t in TAGS)+f"{'T':>8s}")
for c in ORDER:
    if c not in base_m: continue
    b=np.array([np.mean(base_m[c][t]) for t in TAGS]); d=np.array([np.mean(delt_m[c][t]) for t in TAGS])
    T=d.sum(); sh=b/b.sum() if b.sum()>0 else b*0
    exc=d-T*sh
    print(f"{c:20s}"+"".join(f"{100*x:+11.2f}" for x in exc)+f"{100*T:+8.2f}")
print("\n(2) OTHER at institutional — top gaining words")
agg=sorted(((float(np.mean(v)),w) for w,v in other_gain.items() if len(v)>=12), reverse=True)[:12]
print("   "+", ".join(f"{w} +{100*d:.2f}" for d,w in agg))
