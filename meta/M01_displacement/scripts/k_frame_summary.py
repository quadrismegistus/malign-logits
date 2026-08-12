"""Scale-level summary of a K frame, to screen AND to disk, with the
expected-by-chance baseline beside every observed count.

    uv run python meta/M01_displacement/scripts/k_frame_summary.py zh
    -> results/k/k_summary_<lang>.csv

A significance RATE is uninterpretable on its own. At alpha .05, 5% of prompts
fire by construction, so "13.4% significant" is a number whose meaning depends
entirely on what sits beside it. Three baselines are printed:

  expected_sig    0.05 x rows -- the nominal count under a true global null
  ratio           observed / expected
  null_rho_mean   the mean partial from the per-row shuffle draws, which is the
                  EMPIRICAL chance level for that scale and can differ from zero
                  when a scale's distribution interacts with the controls
  z_scale         PAIRED over prompts: mean(rho - null_rho) divided by its own
                  standard error across prompts. Each prompt is compared to ITS
                  OWN shuffles, and the spread used is the spread of that
                  DIFFERENCE.

                  The first version divided by the spread of the per-prompt null
                  MEANS -- each already an average of six draws -- and then by
                  sqrt(rows) again, understating the variance twice. It returned
                  z = -15.8 for a scale firing BELOW chance (ratio 0.62), which
                  is the incoherence that exposed it.

z_scale is the summary statistic to read. The count-based ratio treats every
prompt as a coin flip and ignores effect size; z_scale uses the magnitudes and
tests against the scale's own shuffled distribution rather than a nominal 0.05.
"""
import csv, json, os, subprocess, sys
import numpy as np

#: DECLARED ROLL-UPS. Written down rather than inferred, because "transgressive"
#: is a judgment about which domains count and a reader must be able to see it
#: and disagree. `institutional` groups the bureaucratic batteries M03 works on;
#: `contradiction` is M02's and is neither transgressive nor neutral.
GROUPS={
 "transgressive":{"violence","sexual","taboo","property","betrayal","power",
                  "death","substance","profanity","animal"},
 "institutional":{"institutional","labor","benefits","housing","medical","civic",
                  "banking","insurance","immigration","education","class"},
 "contradiction":{"contradiction"},
 "neutral":{"neutral"},
}
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
K=os.path.join(ROOT,"meta/M01_displacement/results/k")

def pair_roles(lang):
    """prompt -> pair_role, joined from the catalogue rather than the frame.

    The frame was already generated without it and a rerun costs an hour; the
    role is a property of the prompt, so joining is exact rather than a fallback.
    """
    r=subprocess.run(["/opt/homebrew/bin/clickhouse","client","--query",
      "SELECT prompt, any(pair_role) AS role FROM malign_logits.prompt_catalogue "
      "WHERE status='ACTIVE' AND language='%s' GROUP BY prompt FORMAT JSONEachRow"%lang],
      capture_output=True,text=True).stdout
    return {json.loads(l)["prompt"]:json.loads(l)["role"] for l in r.strip().split("\n") if l.strip()}


def stat(rs):
    rho=np.array([float(r["rho_partial"]) for r in rs])
    nm=np.array([float(r["null_mean"]) for r in rs])
    sig=sum(int(r["significant"]) for r in rs); n=len(rs)
    d=rho-nm; sdd=float(d.std(ddof=1)) if n>1 else 0.0
    z=float(d.mean()/(sdd/np.sqrt(n))) if sdd>0 else float("nan")
    negs=float((rho[[bool(int(r["significant"])) for r in rs]]<0).mean()) if sig else float("nan")
    return n,sig,0.05*n,float(rho.mean()),z,negs


def breakdown(rows,lang):
    """The same statistic inside every declared slice."""
    PR=pair_roles(lang)
    for r in rows: r["role"]=PR.get(r["prompt"],"")
    dom2g={}
    for g,ds in GROUPS.items():
        for d in ds: dom2g[d]=g
    slices=[("ALL",lambda r:True)]
    for g in GROUPS:
        slices.append((g,(lambda gg: (lambda r: dom2g.get(r["domain"])==gg))(g)))
    slices.append(("transgressive MARKED",
        lambda r: dom2g.get(r["domain"])=="transgressive" and r["role"]=="MARKED"))
    slices.append(("transgressive UNMARKED",
        lambda r: dom2g.get(r["domain"])=="transgressive" and r["role"]=="UNMARKED"))
    for d in ("violence","sexual","taboo","property","betrayal","power","animal"):
        slices.append(("  > "+d,(lambda dd: (lambda r: r["domain"]==dd))(d)))
    out=[]
    for name,f in slices:
        sub=[r for r in rows if f(r)]
        if not sub: continue
        nP=len({r["prompt"] for r in sub})
        if nP<12: continue
        print("\n  %-24s %d prompts"%(name,nP))
        print("    %-24s %6s %5s %6s %9s %8s %6s"%("scale","rows","sig","exp","mean rho","z","fall"))
        per=[]
        for s2 in sorted({r["scale"] for r in sub}):
            rs=[r for r in sub if r["scale"]==s2]
            if len(rs)<12: continue
            n,sig,exp,mr,z,ng=stat(rs)
            per.append((abs(z) if z==z else 0,s2,n,sig,exp,mr,z,ng))
            out.append({"language":lang,"slice":name.strip(),"n_prompts":nP,"scale":s2,
                        "rows":n,"sig":sig,"expected_sig":round(exp,1),
                        "mean_rho":round(mr,5),"z_scale":round(z,2) if z==z else "",
                        "share_falling":round(ng,2) if ng==ng else ""})
        for _,s2,n,sig,exp,mr,z,ng in sorted(per,reverse=True)[:6]:
            print("    %-24s %6d %5d %6.1f %+9.4f %8.1f %6s"
                  %(s2,n,sig,exp,mr,z,round(ng,2) if ng==ng else "-"))
    return out


def main(lang):
    rows=list(csv.DictReader(open(os.path.join(K,"k_frame_%s.csv"%lang),encoding="utf-8")))
    by={}
    for r in rows: by.setdefault(r["scale"],[]).append(r)
    out=[]
    for s,rs in by.items():
        rho=np.array([float(r["rho_partial"]) for r in rs])
        nm=np.array([float(r["null_mean"]) for r in rs])
        nsd=np.array([float(r["null_sd"]) for r in rs if r["null_sd"]])
        sig=sum(int(r["significant"]) for r in rs)
        n=len(rs); exp=0.05*n
        negs=float((rho[[bool(int(r["significant"])) for r in rs]]<0).mean()) if sig else float("nan")
        d=rho-nm                       #: per-prompt, real minus its own null
        sdd=float(d.std(ddof=1)) if n>1 else 0.0
        z=float(d.mean()/(sdd/np.sqrt(n))) if sdd>0 else float("nan")
        out.append({"language":lang,"scale":s,
            "scale_family":rs[0]["scale_family"],"rows":n,
            "sig":sig,"expected_sig":round(exp,1),
            "ratio":round(sig/exp,2) if exp else "",
            "sig_rate":round(100*sig/n,1),
            "mean_rho":round(float(rho.mean()),5),
            "null_rho_mean":round(float(nm.mean()),5),
            "z_scale":round(z,2),
            "share_falling":round(negs,2) if sig else "",
            "median_n_words":int(np.median([int(r["n_words"]) for r in rs]))})
    out.sort(key=lambda r:-abs(r["z_scale"]) if r["z_scale"]==r["z_scale"] else 0)
    f=os.path.join(K,"k_summary_%s.csv"%lang)
    with open(f,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("%s -- %d prompts, %d scales\n"%(lang.upper(),max(r["rows"] for r in out),len(out)))
    print("  %-24s %6s %5s %6s %6s %10s %10s %8s %7s"
          %("scale","rows","sig","exp","ratio","mean rho","null rho","z_scale","fall"))
    for r in out:
        print("  %-24s %6d %5d %6.1f %6.2f %+10.4f %+10.4f %8.1f %7s"
              %(r["scale"],r["rows"],r["sig"],r["expected_sig"],r["ratio"] or 0,
                r["mean_rho"],r["null_rho_mean"],r["z_scale"],r["share_falling"]))
    print("\n  wrote %s"%os.path.relpath(f,ROOT))
    print("\n%s\nBY PROMPT CATEGORY -- top 6 scales by |z| in each slice\n%s"%("="*74,"="*74))
    bd=breakdown(rows,lang)
    fb=os.path.join(K,"k_summary_by_group_%s.csv"%lang)
    with open(fb,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(bd[0])); w.writeheader(); w.writerows(bd)
    print("\n  wrote %s -- %d rows"%(os.path.relpath(fb,ROOT),len(bd)))
    print("  z_scale is the one to read: the count ratio ignores effect size and")
    print("  assumes a nominal 5%, while z_scale tests this scale's mean partial")
    print("  against the mean of its OWN shuffles.")

if __name__=="__main__": main(sys.argv[1] if len(sys.argv)>1 else "zh")
