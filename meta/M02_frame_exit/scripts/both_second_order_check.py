#!/usr/bin/env python
"""Do the SECOND-ORDER markers show the excess my field instrument missed?

    uv run python both_second_order_check.py

ANSWER: YES, AND THE FIELD SWEEP WAS BLIND TO THEM BY CONSTRUCTION.

    ANY_SECOND_ORDER excess over own poles     base 18/21 p 0.0015
                                            aligned 21/21 p 9.5e-07
    arm difference, paired within group              20/21 p 2.1e-05

That reproduces `findings/second_order_naming.md` (2.18x, 20/22 LINEAGES,
p 0.00012) on a different unit (GROUP), a different statistic (excess over the
group's own two poles rather than an arm ratio), and a different producer. It is
CORROBORATION, not a new result, and it must not be quoted as one.

WHY `both_excess_analysis.py` MISSED IT, which is the point of keeping this file:

  1. The markers are MULTIWORD -- `at the same time`, `both at once`,
     `caught between`. A unigram counter cannot represent them at all.
  2. The single-word ones are RARE: contradict* 0.71%, paradox* 0.32%,
     torn 0.63% of passages. They sat below that analysis's occurrence floor
     before any test ran.
  3. NO LEXICON HAS A FIELD FOR IT. USAS, GI, WordNet and RID categorise what a
     passage is ABOUT. Naming a contradiction is what the passage DOES. It is a
     discourse move, not a semantic domain, and no amount of correction or
     split-sampling rescues a measure that cannot represent the construct.

**A NULL FROM A LEXICON-AND-UNIGRAM SWEEP IS A NULL ABOUT LEXICONS AND
UNIGRAMS.** `both_split_sample.py` returned zero arm-difference survivors under
BH in both split directions, and I reported it as "alignment does not change
what a contradiction prompt produces". It licensed no such thing: the construct
where alignment demonstrably DOES change it was outside the measure's reach.
RH caught this from the existing finding, not from the data.

ONE THING HERE THAT IS NOT IN THE FINDING: ANY_DEONTIC shows NO arm difference
(9/21, p 0.66) where second-order naming shows 20/21. The amplification is
specific to naming the tension rather than to moralising about it -- consistent
with the finding's own guilt-lexicon numbers (1.13x / 1.07x) and sharper.
"""
_ORIG = """Do the SECOND-ORDER markers show the excess my field instrument missed?

RH: 'regex found an unimpeachable result for contradiction language.' It did --
second_order_naming, 2.18x, 20/22 lineages, p 0.00012. My field/unigram sweep
found nothing resembling it. This asks whether that is a REFUTATION or a
COVERAGE FAILURE, by running the finding's own markers through my own design.
"""
import collections, csv, json, math, os, re, subprocess, sys, statistics as st
ROOT="/Users/rj416/github/malign-logits"; sys.path.insert(0,ROOT)
sys.path.insert(0,ROOT+"/meta/M02_frame_exit/scripts")
from z_second_order import SO, DE
CH="/opt/homebrew/bin/clickhouse"
CAT=ROOT+"/data/prompt_categorisation.json"; PAIRS=ROOT+"/data/lineage_representative_pairs.txt"

def prompt_map():
    cat=json.load(open(CAT))["prompts"]; g=collections.defaultdict(dict)
    for p in cat:
        if p.get("domain")=="contradiction" and p.get("group_id") and p.get("language")=="en":
            g[p["group_id"]][p.get("group_role")]=p
    out=collections.defaultdict(list)
    for gid,r in g.items():
        if all(k in r for k in ("POLE_A","POLE_B","BOTH")):
            for role in ("POLE_A","POLE_B","BOTH"): out[r[role]["prompt"].strip()].append((gid,role))
    return out

def sign_test(v):
    v=[x for x in v if x!=0.0]; n,k=len(v),sum(1 for x in v if x>0)
    if not n: return 0,0,float("nan")
    t=min(k,n-k); return n,k,min(1.0,2*sum(math.comb(n,i) for i in range(t+1))/2**n)

pm=prompt_map()
pairs=sorted([l.strip().split(">") for l in open(PAIRS) if l.strip()])
arm={}
for b,a in pairs: arm[b]="base"; arm[a]="aligned"
esc=lambda s:s.replace("\\","\\\\").replace("'","\\'")
q=("SELECT model, prompt, text FROM malign_logits.gen_sequences WHERE corpus='f11_l2' "
   "AND model IN (%s) FORMAT JSONEachRow"%",".join("'%s'"%esc(m) for m in arm))
pr=subprocess.Popen([CH,"client","-q",q],stdout=subprocess.PIPE,text=True,bufsize=1<<20)
acc=collections.defaultdict(lambda: collections.Counter()); npass=collections.Counter()
n=0
for line in pr.stdout:
    try: r=json.loads(line)
    except Exception: continue
    p=r["prompt"].strip()
    if r["model"] not in arm or p not in pm: continue
    t=r["text"] or ""; n+=1
    anyso=any(rx.search(t) for rx in SO.values())
    anyde=any(rx.search(t) for rx in DE.values())
    for gid,role in pm[p]:
        k=(arm[r["model"]],gid,role); npass[k]+=1
        acc[k]["ANY_SECOND_ORDER"]+=anyso; acc[k]["ANY_DEONTIC"]+=anyde
        for name,rx in SO.items():
            if rx.search(t): acc[k][name]+=1
pr.wait()
print("%s passages\n"%format(n,","))
groups=sorted({g for (_,g,_) in acc})
def rate(a,g,r,k): 
    return acc[(a,g,r)][k]/npass[(a,g,r)] if npass.get((a,g,r)) else None
print("%-26s %14s %22s %22s"%("marker","","BASE excess","ALIGNED excess"))
print("%-26s %14s %10s %11s %10s %11s"%("","BOTH rate(base)","up/n","sign p","up/n","sign p"))
DIFF={}
for key in ["ANY_SECOND_ORDER","ANY_DEONTIC"]+sorted(SO):
    row=[]; br=[]
    for a in ("base","aligned"):
        d=[]
        for g in groups:
            B,A,Bb=rate(a,g,"BOTH",key),rate(a,g,"POLE_A",key),rate(a,g,"POLE_B",key)
            if None in (B,A,Bb): continue
            d.append(B-0.5*(A+Bb))
            if a=="base": br.append(B)
        row.append(sign_test(d))
    dd=[]
    for g in groups:
        vals={}
        for a in ("base","aligned"):
            B,A,Bb=rate(a,g,"BOTH",key),rate(a,g,"POLE_A",key),rate(a,g,"POLE_B",key)
            if None in (B,A,Bb): vals=None; break
            vals[a]=B-0.5*(A+Bb)
        if vals: dd.append(vals["aligned"]-vals["base"])
    DIFF[key]=sign_test(dd)
    if not row[0][0]: continue
    print("%-26s %13.3f%% %7d/%-3d %10.4g %7d/%-3d %10.4g"
          %(key,100*st.mean(br) if br else 0,row[0][1],row[0][0],row[0][2],
            row[1][1],row[1][0],row[1][2]))

print("\nARM DIFFERENCE, aligned excess MINUS base excess, paired within group")
print("%-26s %10s %12s"%("marker","up/n","sign p"))
for k,(n,u,pv) in DIFF.items():
    if n: print("%-26s %6d/%-3d %12.4g%s"%(k,u,n,pv," *" if pv<0.05 else ""))
