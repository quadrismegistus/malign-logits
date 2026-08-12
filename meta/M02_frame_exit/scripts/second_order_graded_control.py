#!/usr/bin/env python
"""The same-side conjunction control for `findings/second_order_naming.md`.

    uv run python second_order_graded_control.py

RECOVERED, 2026-08-12, AND THAT IS THE POINT OF THIS DOCSTRING. The graded
stimulus table in that finding -- single predicate / two conjoined compatible /
two conjoined contradictory -- was computed on 2026-08-11 at 18:34 by an INLINE
HEREDOC that was never written to a file, importing marker tables and an exit
lexicon that lived only in a session scratchpad. Commit `e804b1c5` carries the
numbers in its message; nothing in the repository could produce them.

A grep for `two conjoined` across the repo hit three prose files and no code. A
grep across five earlier session logs hit nothing. The computation existed for
one evening inside a session and its dependencies were in `/private/tmp`.

RH asked "was it a heredoc script?" and it was. This file is that heredoc,
unchanged except for three path constants, with its dependencies moved into the
repo:

    scripts/second_order_markers/markers_v2.py   V1 marker set
    scripts/second_order_markers/markers_v3.py   V3_SAFE marker set
    results/second_order_exitlex.json            the exit-free filter's lexicon

**THE LOGIC IS NOT EDITED.** If the recovered numbers disagree with the
published table, that is a finding about the table, and the disagreement must
be reported rather than tuned away.

PUBLISHED, for comparison at the point of running:

    stimulus                     V1                   V3_SAFE
    single predicate             1.04x  p=1           1.15x  p=0.61
    two conjoined, compatible    1.24x  p=0.23        1.30x  p=0.24
    two conjoined, contradictory 2.22x  21/25 p=9e-4  2.26x  18/21 p=0.0015
    population                   17 en groups, 67,198 passages, exit-free
"""
import os
import json, collections, math, subprocess, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'second_order_markers'))
from markers_v2 import COMPILED as C2
from markers_v3 import COMPILED as C3
SCR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
EXIT=re.compile(r'\b(?:%s)\b'%'|'.join(re.escape(w) for w in json.load(open(SCR+'/second_order_exitlex.json'))), re.I)
SETS={'V1':C2['V1'],'V3_SAFE':C3['V3_SAFE']}
cat=json.load(open('data/prompt_categorisation.json'))['prompts']
g=collections.defaultdict(dict)
for p in cat:
    if p.get('domain')=='contradiction' and p.get('group_id'): g[p['group_id']][p.get('group_role')]=p
pm={}
GROUPS=set()
for gid,v in g.items():
    if not gid.startswith('f11_') or v.get('BOTH',{}).get('language','en')!='en': continue
    if 'CONTROL_A' not in v and 'CONTROL_B' not in v: continue   # only groups with the control
    GROUPS.add(gid)
    pm[v['BOTH']['prompt'].strip()]=(gid,'BOTH')
    for r in ('POLE_A','POLE_B'):
        if r in v: pm.setdefault(v[r]['prompt'].strip(),(gid,'POLE'))
    for r in ('CONTROL_A','CONTROL_B'):
        if r in v: pm.setdefault(v[r]['prompt'].strip(),(gid,'CONTROL'))
print('groups with all three arms: %d ; prompts mapped: %d' % (len(GROUPS), len(pm)))
pairs=[x.strip().split('>') for x in open('data/lineage_representative_pairs.txt') if x.strip()]
q="SELECT model, prompt, text FROM malign_logits.gen_sequences WHERE corpus='f11_l2' FORMAT JSONEachRow"
pr=subprocess.Popen(['/opt/homebrew/bin/clickhouse','client','-q',q],stdout=subprocess.PIPE,text=True,bufsize=1<<20)
agg=collections.defaultdict(collections.Counter)
for line in pr.stdout:
    try: r=json.loads(line)
    except Exception: continue
    hit=pm.get((r['prompt'] or '').strip())
    if not hit: continue
    t=' '.join((r['text'] or '').split()[:50])
    if EXIT.search(t): continue
    a=agg[(r['model'],hit[0],hit[1])]; a['n']+=1
    for nm,S in SETS.items():
        if any(rx.search(t) for rx in S.values()): a[nm]+=1
pr.wait()
allg=sorted(GROUPS)
def con(role,key):
    dd=[];kb=nb=ka=na=0
    for bm,am in pairs:
        gs=[gg for gg in allg if (bm,gg,role) in agg and (am,gg,role) in agg]
        if len(gs)<4: continue
        nb_=sum(agg[(bm,gg,role)]['n'] for gg in gs); na_=sum(agg[(am,gg,role)]['n'] for gg in gs)
        if nb_<50 or na_<50: continue
        kb_=sum(agg[(bm,gg,role)][key] for gg in gs); ka_=sum(agg[(am,gg,role)][key] for gg in gs)
        dd.append(ka_/na_-kb_/nb_); kb+=kb_;nb+=nb_;ka+=ka_;na+=na_
    if not kb: return None
    v=[x for x in dd if x]; n=len(v); k=sum(1 for x in v if x>0)
    return 100*kb/nb,100*ka/na,(ka/na)/(kb/nb),k,n,min(1,2*sum(math.comb(n,i) for i in range(min(k,n-k)+1))/2**n)
print('\nTHE SAME-SIDE CONJUNCTION CONTROL. Exit-free, %d groups, lineage unit.\n' % len(allg))
print('%-9s %-9s %8s %8s %8s %8s %10s'%('set','arm','base %','algn %','ratio','lins+','p'))
for nm in SETS:
    for role,lab in (('BOTH','contradict'),('CONTROL','same-side'),('POLE','single')):
        r=con(role,nm)
        if r: print('%-9s %-9s %8.3f %8.3f %7.2fx  %3d/%-3d %10.3g%s'%(nm,lab,r[0],r[1],r[2],r[3],r[4],r[5],' *' if r[5]<0.05 else ''))
    print()
print('passages: %s' % format(sum(a['n'] for a in agg.values()),','))
for role in ('BOTH','CONTROL','POLE'):
    print('   %-8s %s' % (role, format(sum(a['n'] for (m,gg,rr),a in agg.items() if rr==role),',')))
