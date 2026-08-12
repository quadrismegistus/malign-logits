"""Plan K, SITE-CONDITIONAL: the same partials at ONE prompt at a time.

    uv run python meta/M01_displacement/scripts/k_per_prompt.py

The pooled analysis sums a word's movement over every cell it appears in,
neutral prompts included. But displacement is SITE-SPECIFIC -- `kill -> scream`
happens at violent sites and not everywhere -- and averaging over sites is
exactly the operation that washes a site-specific effect out. F and G exist
because site matters. This conditions on the site instead.

The cost is n: a few hundred words at one prompt rather than fourteen thousand
pooled, so only a large effect can show. That is the trade and it is the right
way round -- a small effect measured on the wrong population is worth less than
a null on the right one.
"""
import json, math, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import k_analysis as A
from k_frequency import fpm
import k_population as KP
K=os.path.join(ROOT,"meta/M01_displacement/results/k")
SC=A.SCALES

def run(lang, prompt_ids):
    norm=json.load(open(os.path.join(K,"normalisation_%s.json"%lang)))
    rate=json.load(open(os.path.join(K,"ratings_%s.json"%lang)))["ratings"]
    t2u=norm["token_to_unit"]
    edges=KP.reps(lang); esc=lambda s: s.replace("\\","\\\\").replace("'","\\'")
    pairs=" OR ".join("(m.base='%s' AND m.aligned='%s')"%(esc(b),esc(a)) for b,a in edges)
    meas="coca_fic" if lang=="en" else "SUBTLEX_CH"
    print("\n%s  frequency=%s  %d lineage pairs"%(lang.upper(),meas,len(edges)))
    print("%-22s %5s %7s %7s %7s %7s %7s %7s %7s %7s"
          %("prompt","n","floor","vulg","reg","trns","chrg","val","harm","conc"))
    for pid in prompt_ids:
        rows=A.q("""SELECT word, countIf(cls='rise')-countIf(cls='fall') AS net,
                    avg(p_base) AS pbase FROM (
            SELECT m.word AS word, m.cls AS cls, m.p_base AS p_base,
              row_number() OVER (PARTITION BY m.base,m.aligned ORDER BY m.p_base DESC) rb,
              row_number() OVER (PARTITION BY m.base,m.aligned ORDER BY m.p_aligned DESC) ra
            FROM %s.movement m
            INNER JOIN (SELECT prompt FROM %s.prompt_catalogue WHERE prompt_id='%s') p
              ON m.prompt=p.prompt
            WHERE m.rule='canonical' AND (%s))
          WHERE rb<=50 OR ra<=50 GROUP BY word"""%(A.DB,A.DB,pid,pairs))
        u=[]
        for r in rows:
            k=t2u.get(r["word"])
            if k and k in rate and r["pbase"]>0 and fpm(k,lang,meas) is not None:
                u.append((k,r["net"],r["pbase"]))
        if len(u)<40:
            print("%-22s %5d   (too few rated words with a frequency entry)"%(pid,len(u))); continue
        y=A.ranks([x[1] for x in u])
        lp=A.ranks([math.log10(x[2]) for x in u]); lf=A.ranks([math.log10(fpm(x[0],lang,meas)) for x in u])
        yr=A.resid(y,[lp,lf]); floor=A.pearson(A.ranks([x[2] for x in u]),y)
        cells=[]
        for s in SC:
            xs=A.ranks([rate[x[0]][s] for x in u])
            cells.append(A.pearson(A.resid(xs,[lp,lf]),yr))
        print("%-22s %5d %+7.2f %s"%(pid,len(u),floor," ".join("%+7.3f"%c for c in cells)))

EN=["gender_anger_marked","violence_liminal_3"]+["sexual_explicit_%d"%i for i in range(1,6)]+["sexual_liminal_%d"%i for i in range(1,8)]
ZH=["sexual_explicit_%d_zh"%i for i in range(1,6)]+["sexual_liminal_%d_zh"%i for i in range(1,8)]
run("en",EN); run("zh",ZH)
