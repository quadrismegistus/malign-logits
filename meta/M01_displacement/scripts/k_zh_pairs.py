"""Plan K on the 24 transgressive minimal pairs, IN BOTH LANGUAGES.

    uv run python meta/M01_displacement/scripts/k_zh_pairs.py

THE CORPUS IS HELD CONSTANT AND ONLY THE LANGUAGE VARIES. These are the same 24
design objects -- SETE/SETD/F36/CENSUS transgressive_swap pairs -- run once on
their Chinese strings and once on their English twins. Comparing the Chinese 24
against the English 684 would vary corpus and language together, which is the
confound the whole morning's zh_sites_unit_limited finding is about.

NOT M01's PAIRS. The M01 corpus was never translated: 1,368 M01 prompt_ids
against 356 translated ids, zero overlap on prompt_id, zero on pair_id, and no
Chinese row anywhere carries an M01 pair_id. Checked four ways 2026-08-12.

UNITS. Chinese has 17 CJK-capable lineage representatives, English 46. The MDE
is reported for each because they are not the same test, and a Chinese null at
n=17 means far less than an English one at n=46.
"""
import json, math, os, sys, collections
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import k_analysis as A
from k_frequency import fpm
import k_population as KP
from zh_site_magnitude import zh_pairs
K=os.path.join(ROOT,"meta/M01_displacement/results/k")
DRAWS=50000; rng=np.random.default_rng(20260812); MIN_N=30

def load(lang):
    n=json.load(open(os.path.join(K,"normalisation_%s.json"%lang)))
    r=json.load(open(os.path.join(K,"ratings_%s.json"%lang)))["ratings"]
    return n["token_to_unit"], r

def run(lang, pairs, meas):
    t2u,rate=load(lang)
    edges=KP.reps(lang); esc=lambda s: s.replace("\\","\\\\").replace("'","\\'")
    texts={t for v in pairs.values() for t in v.values()}
    tl="','".join(esc(t) for t in texts)
    FQ={u:math.log10(f) for u in rate for f in [fpm(u,lang,meas)] if f and f>0}
    per=[]
    for b,a in edges:
        rows=A.q("""SELECT prompt, word, countIf(cls='rise')-countIf(cls='fall') AS net,
                    avg(p_base) AS pbase FROM (
            SELECT m.prompt AS prompt, m.word AS word, m.cls AS cls, m.p_base AS p_base,
              row_number() OVER (PARTITION BY m.prompt ORDER BY m.p_base DESC) rb,
              row_number() OVER (PARTITION BY m.prompt ORDER BY m.p_aligned DESC) ra
            FROM %s.movement m WHERE m.rule='canonical'
              AND m.base='%s' AND m.aligned='%s' AND m.prompt IN ('%s'))
          WHERE rb<=50 OR ra<=50 GROUP BY prompt, word"""%(A.DB,esc(b),esc(a),tl))
        byp=collections.defaultdict(list)
        for r in rows: byp[r["prompt"]].append(r)
        def part(p):
            u=[(t2u[r["word"]],r["net"],r["pbase"]) for r in byp.get(p,[])
               if t2u.get(r["word"]) in FQ and t2u.get(r["word"]) in rate and r["pbase"]>0]
            if len(u)<MIN_N: return None
            y=A.ranks([x[1] for x in u]); lp=A.ranks([math.log10(x[2]) for x in u])
            lf=A.ranks([FQ[x[0]] for x in u]); yr=A.resid(y,[lp,lf])
            return [A.pearson(A.resid(A.ranks([rate[x[0]][s] for x in u]),[lp,lf]),yr) for s in A.SCALES]
        d=[]
        for g,mem in pairs.items():
            x,z=part(mem.get("MARKED")),part(mem.get("UNMARKED"))
            if x and z: d.append([p-q for p,q in zip(x,z)])
        if d: per.append([float(np.median([r[i] for r in d])) for i in range(len(A.SCALES))])
    M=np.array(per); n=len(M)
    if n<8: print("  %s: only %d lineages, not testable"%(lang,n)); return
    sd=M.std(axis=0,ddof=1)/np.sqrt(n)
    obs=np.where(sd>0,M.mean(axis=0)/np.where(sd>0,sd,1),0.0)
    S=rng.choice([-1.,1.],size=(DRAWS,n))
    nt=np.where(sd>0,(S@M)/n/np.where(sd>0,sd,1),0.0)
    pu=((np.abs(nt)>=np.abs(obs)).sum(axis=0)+1)/(DRAWS+1)
    mx=np.abs(nt).max(axis=1)
    pm=np.array([((mx>=abs(o)).sum()+1)/(DRAWS+1) for o in obs])
    def mde(nn,draws=3000):
        lo,hi=0.,2.5
        for _ in range(28):
            dd=(lo+hi)/2; x=rng.normal(dd,1.,size=(draws,nn))
            t=x.mean(axis=1)/(x.std(axis=1,ddof=1)/np.sqrt(nn))
            S2=rng.choice([-1.,1.],size=(1500,nn))
            crit=np.quantile(np.abs((S2@x[:150].T/nn)/(x[:150].std(axis=1,ddof=1)/np.sqrt(nn))),.95)
            if (np.abs(t)>=crit).mean()>=.80: hi=dd
            else: lo=dd
        return hi
    print("\n  %s -- %d lineages, %d pairs, freq=%s   MDE d>=%.2f"%(lang.upper(),n,len(pairs),meas,mde(n)))
    print("    %-18s %9s %8s %9s %10s"%("scale","median","t","p","p max-stat"))
    for i,s in enumerate(A.SCALES):
        print("    %-18s %+9.4f %+8.2f %9.5f %10.5f%s"
              %(s,float(np.median(M[:,i])),obs[i],pu[i],pm[i]," *" if pm[i]<=.05 else ""))

zp,meta=zh_pairs()
print("24-pair corpus: %d pairs | domains %s"
      %(len(zp),dict(collections.Counter(m["domain"] for m in meta.values()))))
run("zh",{g:{"MARKED":v["MARKED"],"UNMARKED":v["UNMARKED"]} for g,v in zp.items()},"SUBTLEX_CH")
#: the ENGLISH TWINS of the very same design objects
cat={str(r.get("prompt_id")):r for r in json.load(open(os.path.join(ROOT,"data/prompt_categorisation.json")))["prompts"]}
zt={p["prompt_id"]:p for f in ("data/chinese_translations.json","data/chinese_translations_2.json")
    for p in json.load(open(os.path.join(ROOT,f)))["prompts"]}
eng={}
for g in zp:
    mem={}
    for pid,p in zt.items():
        if p.get("group")==g and p.get("group_role") in ("MARKED","UNMARKED"):
            mem[p["group_role"]]=p["english"].strip()
    if len(mem)==2: eng[g]=mem
print("\nEnglish twins recovered for %d of %d pairs"%(len(eng),len(zp)))
run("en",eng,"coca_fic")
