"""The tall frame: one row per (prompt, scale), both languages, coder scales
AND human norms side by side.

    uv run python meta/M01_displacement/scripts/k_frame.py en
    uv run python meta/M01_displacement/scripts/k_frame.py zh

    -> results/k/k_frame_<lang>.csv

SCALES IN THREE FAMILIES, so a coder scale can be read against a human-normed
one measuring the same construct:

    coder   vulgarity register_level transgressiveness charge valence
            bodily_harm concreteness            (7, deepseek, frozen instrument)
    coder   abs_valence                         |valence - 4|, DERIVED
    norm    n_valence n_abs_valence n_arousal n_dominance n_concreteness
            English: Warriner (V/A/D, 1-9, midpoint 5) + Brysbaert
            Chinese: the 9,877-word two-character concreteness set only

WHY abs_valence. C and E claim de-extremification -- aligned output moves toward
affective NEUTRALITY. Signed valence cannot see that: a scale that pulls both
`torture` and `joy` toward the middle correlates with neither direction. The
absolute deviation from the scale's midpoint is the quantity those registrations
are actually about, and it has never been run at the word level here.

THE CHINESE NORM SCALE RUNS 1=CONCRETE. Verified against 键盘 1.1, 桌子 1.3
against 意义 4.1, 精神 3.8 -- concrete mean 1.27, abstract mean 3.68. It is
SIGN-FLIPPED on load so that in this frame every concreteness column points the
same way. Reporting the raw figure would show the instrument failing in Chinese
when it agrees at |r| 0.81.

Each row carries its own null band from `NSHUF` shuffles of the rating labels
across words within that prompt, so `z` is comparable across scales that have
very different distributions.
"""
import csv, json, math, os, sys, collections
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import k_analysis as A
from k_frequency import fpm
import k_population as KP
K=os.path.join(ROOT,"meta/M01_displacement/results/k")
NORMS="/Users/rj416/Dropbox/Prof/Articles/TheoryMachines/norms_sources"
DRAWS=2000; NSHUF=6; MIN_WORDS=60
rng=np.random.default_rng(20260812)

#: EVERY NUMERIC NORM THE REPO HOLDS, and the EXTREMITY columns are the repo's
#: own, not hand-rolled. `fields.py:292` centres on the LEXICON MEAN rather than
#: the scale midpoint -- Warriner runs 1-9 but its mass is not centred at 5 --
#: so |x - 5| would be a different and worse quantity than the one the campaign
#: already uses. This is the artifact that already existed.
NORM_DIMS=("valence","arousal","dominance","concreteness",
           "valence_extremity","arousal_extremity","dominance_extremity",
           "concreteness_extremity")

def norms_en():
    from malign_logits.fields import _norms
    N=_norms(); N=N[0] if isinstance(N,tuple) else N
    out={}
    for w,d in N.items():
        r={"n_"+k:d[k] for k in NORM_DIMS if k in d}
        if r: out[w]=r
    return out

def norms_zh():
    import openpyxl
    p=NORMS+"/Concretenss Ratings of 9877 Two Character Chinese Words.xlsx"
    ws=openpyxl.load_workbook(p,read_only=True,data_only=True)["Concreteness Ratings"]
    rows=ws.iter_rows(values_only=True); hdr=list(next(rows))
    WI,CI=hdr.index("Word"),hdr.index("Mean of Valid Ratings")
    out={}
    for r in rows:
        if r and r[WI]:
            try: out[str(r[WI]).strip()]={"n_concreteness":-float(r[CI])}  #: SIGN-FLIPPED
            except (TypeError,ValueError): pass
    return out

def main(lang):
    meas="coca_fic" if lang=="en" else "SUBTLEX_CH"
    norm=json.load(open(os.path.join(K,"normalisation_%s.json"%lang)))
    rate=json.load(open(os.path.join(K,"ratings_%s.json"%lang)))["ratings"]
    t2u=norm["token_to_unit"]
    NM=norms_en() if lang=="en" else norms_zh()
    NCOLS=["n_"+d for d in NORM_DIMS] if lang=="en" else ["n_concreteness"]
    #: coder EXTREMITY on the same convention as the norms: distance from the
    #: LEXICON MEAN of that scale, computed over the rated vocabulary.
    CCOLS=list(A.SCALES)+[s+"_extremity" for s in ("valence","charge","concreteness")]
    ALL=CCOLS+NCOLS
    FQ={u:math.log10(f) for u in rate for f in [fpm(u,lang,meas)] if f and f>0}
    MU={s:float(np.mean([rate[u][s] for u in FQ])) for s in A.SCALES}
    print("[%s] coder lexicon means: %s"%(lang,{k:round(v,2) for k,v in MU.items()}),flush=True)
    VAL={}
    for u in FQ:
        d={s:rate[u][s] for s in A.SCALES}
        for s in ("valence","charge","concreteness"):
            d[s+"_extremity"]=abs(rate[u][s]-MU[s])
        n=NM.get(u.strip() if lang=="zh" else u.strip().lower(),{})
        d.update(n); VAL[u]=d
    meta={}
    for r in A.q("""SELECT prompt, any(prompt_id) pid, any(domain) dom, any(source) src,
                    any(finding) fnd FROM %s.prompt_catalogue
                    WHERE status='ACTIVE' AND language='%s' GROUP BY prompt"""%(A.DB,lang)):
        meta[r["prompt"]]=r
    edges=KP.reps(lang); esc=lambda s:s.replace("\\","\\\\").replace("'","\\'")
    ep=" OR ".join("(m.base='%s' AND m.aligned='%s')"%(esc(b),esc(a)) for b,a in edges)
    rows=A.q("""SELECT prompt, word, countIf(cls='rise')-countIf(cls='fall') AS net,
                avg(p_base) AS pb, avg(p_aligned) AS pa, count() AS cells FROM (
        SELECT m.prompt AS prompt, m.word AS word, m.cls AS cls, m.p_base AS p_base,
               m.p_aligned AS p_aligned,
          row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt ORDER BY m.p_base DESC) rb,
          row_number() OVER (PARTITION BY m.base,m.aligned,m.prompt ORDER BY m.p_aligned DESC) ra
        FROM %s.movement m
        INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                    WHERE status='ACTIVE' AND language='%s') p ON m.prompt=p.prompt
        WHERE m.rule='canonical' AND (%s))
      WHERE rb<=50 OR ra<=50 GROUP BY prompt, word"""%(A.DB,A.DB,lang,ep))
    byp=collections.defaultdict(list)
    for r in rows:
        u=t2u.get(r["word"])
        if u in VAL and r["pb"]>0: byp[r["prompt"]].append((u,r["net"],r["pb"],r["pa"],r["cells"]))
    byp={p:v for p,v in byp.items() if len(v)>=MIN_WORDS}
    print("[%s] %d prompts | %d scales (%d coder, %d norm)"%(lang,len(byp),len(ALL),len(CCOLS),len(NCOLS)),flush=True)
    out=[]
    for pi,(p,v) in enumerate(byp.items()):
        w=[x[0] for x in v]
        y=A.ranks([x[1] for x in v]); lp=A.ranks([math.log10(x[2]) for x in v])
        lf=A.ranks([FQ[x[0]] for x in v]); yr=A.resid(y,[lp,lf]); ys=np.sqrt((yr*yr).sum())
        md=meta.get(p,{})
        for s in ALL:
            sub=[i for i,u in enumerate(w) if s in VAL[u]]
            if len(sub)<MIN_WORDS: continue
            ww=[w[i] for i in sub]
            yy=A.resid(A.ranks([v[i][1] for i in sub]),
                       [A.ranks([math.log10(v[i][2]) for i in sub]),
                        A.ranks([FQ[w[i]] for i in sub])])
            yss=np.sqrt((yy*yy).sum())
            x=A.resid(A.ranks([VAL[u][s] for u in ww]),
                      [A.ranks([math.log10(v[i][2]) for i in sub]),
                       A.ranks([FQ[w[i]] for i in sub])])
            xs=np.sqrt((x*x).sum())
            if xs==0 or yss==0: continue
            rho=float((x@yy)/(xs*yss))
            idx=np.argsort(rng.random((DRAWS,len(sub))),axis=1)
            xp=x[idx]; rr=(xp@yy)/(np.sqrt((xp*xp).sum(axis=1))*yss)
            pv=float((np.abs(rr)>=abs(rho)).sum()+1)/(DRAWS+1)
            nul=[]
            for _ in range(NSHUF):
                q=rng.permutation(len(sub)); xq=x[q]
                nul.append(float((xq@yy)/(np.sqrt((xq*xq).sum())*yss)))
            nm,nsd=float(np.mean(nul)),float(np.std(nul,ddof=1))
            out.append({"language":lang,"prompt":p,"prompt_id":md.get("pid",""),
                "domain":md.get("dom",""),"source":md.get("src",""),"finding":md.get("fnd",""),
                "scale":s,"scale_family":"norm" if s.startswith("n_") else "coder",
                "n_words":len(sub),
                "mean_p_base":float(np.mean([v[i][2] for i in sub])),
                "mean_p_aligned":float(np.mean([v[i][3] for i in sub])),
                "mean_net":float(np.mean([v[i][1] for i in sub])),
                "rho_partial":round(rho,5),"p_perm":round(pv,5),
                "null_mean":round(nm,5),"null_sd":round(nsd,5),
                "z_vs_null":round((rho-nm)/nsd,3) if nsd>0 else "",
                "significant":int(pv<=0.05)})
        if pi%300==0: print("   %d/%d"%(pi,len(byp)),flush=True)
    f=os.path.join(K,"k_frame_%s.csv"%lang)
    with open(f,"w",newline="",encoding="utf-8") as fh:
        wtr=csv.DictWriter(fh,fieldnames=list(out[0])); wtr.writeheader(); wtr.writerows(out)
    print("\nwrote %s -- %d rows"%(os.path.relpath(f,ROOT),len(out)))
    print("\n  %-16s %7s %8s %9s %9s"%("scale","rows","sig","sig rate","mean rho"))
    for s in ALL:
        r=[o for o in out if o["scale"]==s]
        if not r: continue
        sig=sum(o["significant"] for o in r)
        print("  %-16s %7d %8d %8.1f%% %+9.4f"
              %(s,len(r),sig,100*sig/len(r),float(np.mean([o["rho_partial"] for o in r]))))
if __name__=="__main__": main(sys.argv[1] if len(sys.argv)>1 else "en")
