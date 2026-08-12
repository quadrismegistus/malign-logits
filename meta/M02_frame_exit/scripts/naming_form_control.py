#!/usr/bin/env python
"""BOTH against a FORM-MATCHED control: does the model name the contradiction?

    uv run python naming_form_control.py

Producer for `findings/naming_survives_form_control.md`.

THE CONTRAST, AND IT IS THE WHOLE POINT OF THE FILE. Each contradiction group
carries five prompts. Three matter here:

    BOTH       He was beautiful and disgusting and she wanted to   two OPPOSED
    CONTROL_A  He was beautiful and radiant and she wanted to      two CONSONANT
    CONTROL_B  He was disgusting and foul and she wanted to        two CONSONANT

The controls hold SYNTACTIC FORM fixed -- two adjectives joined by `and`, same
length, same rhythm -- and vary only whether the pair is opposed. That isolates
opposition.

**DO NOT USE POLE_A / POLE_B.** The poles carry ONE adjective (7.5 tokens
against BOTH's 9.6), so `BOTH - mean(POLE)` confounds contradiction with prompt
length and syntactic weight. Measured: against the poles, base `cosmos_weather`
reads 17/17 p 1.5e-05 and base `options` 2/17 p 0.0024; against these controls
the same two are 11/17 p 0.33 and 7/17 p 0.63. Every form-sensitive result is an
artifact of that substitution, and three earlier versions of this analysis made
it.

**PROMPT ECHO IS REMOVED BEFORE COUNTING.** Every word occurring in ANY English
prompt is stripped from the continuation first. Without it the ranking is
dominated by which adjective was in which prompt: `captive` 4.0x (it is in BOTH)
and `radiant` 0.13x, `squalid` 0.01x (they are in the CONTROLS). That is the
stimulus, not the response. The rule is blunt -- it also removes ordinary words
that happen to appear in a prompt -- and blunt in the conservative direction.

THE STATISTIC IS A RATE RATIO, POOLED, WITH PER-GROUP CONSISTENCY BESIDE IT.
No significance test and no multiple-comparison correction: at ratios of 5-7x
holding in 14-17 of 17 groups against a base arm whose best coherent word is 3x
in 8 of 17, the correction is ceremony. `groups` is the number of groups in
which BOTH's rate exceeds the controls', and it is the honest robustness column.
"""
import collections, json, os, re, subprocess, sys

HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,ROOT)
from malign_logits import fields                                  # noqa: E402
CH="/opt/homebrew/bin/clickhouse"
CAT=os.path.join(ROOT,"data","prompt_categorisation.json")
OUT=os.path.join(CAMP,"results","naming_form_control.json")
MIN_WORD=120        #: pooled occurrences across BOTH+CONTROLS to be ranked
MIN_FIELD=300


def main():
    cat=json.load(open(CAT))["prompts"]
    promptw=set()
    for p in cat:
        if p.get("language")=="en":
            promptw.update(re.findall(r"[a-z]+",p["prompt"].lower()))
    g=collections.defaultdict(dict)
    for p in cat:
        if (p.get("language")=="en" and p.get("domain")=="contradiction"
                and p.get("group_id")):
            g[p["group_id"]][p.get("group_role")]=p["prompt"].strip()
    pm=collections.defaultdict(list)
    for gid,r in g.items():
        if all(k in r for k in ("BOTH","CONTROL_A","CONTROL_B")):
            for role in ("BOTH","CONTROL_A","CONTROL_B"):
                pm[r[role]].append((gid,role))
    fp=json.load(open(os.path.join(ROOT,"data","base_aligned_pairs.json")))
    fp=fp if isinstance(fp,list) else fp.get("pairs",[])
    arm={}
    for x in fp:
        b,a=(x.get("base"),x.get("aligned")) if isinstance(x,dict) else tuple(x)
        if b and a: arm[b]="base"; arm[a]="aligned"
    esc=lambda s:s.replace("\\","\\\\").replace("'","\\'")      # noqa: E731
    q=("SELECT model, prompt, text FROM malign_logits.gen_sequences "
       "WHERE corpus='f11_l2' AND model IN (%s) FORMAT JSONEachRow"
       % ",".join("'%s'"%esc(m) for m in arm))
    pr=subprocess.Popen([CH,"client","-q",q],stdout=subprocess.PIPE,text=True,
                        bufsize=1<<20)
    W=collections.defaultdict(collections.Counter)
    F=collections.defaultdict(collections.Counter)
    N=collections.Counter(); GW=collections.defaultdict(collections.Counter)
    GF=collections.defaultdict(collections.Counter); GN=collections.Counter()
    n=0
    for line in pr.stdout:
        try: r=json.loads(line)
        except Exception: continue
        p=r["prompt"].strip()
        if r["model"] not in arm or p not in pm: continue
        n+=1
        toks=[t for t in fields.tokens(r["text"] or "") if t not in promptw]
        f=fields.count_all(" ".join(toks))
        ct=[t for t in toks if fields.is_content_word(t)]
        a=arm[r["model"]]
        for gid,role in pm[p]:
            W[(a,role)].update(ct); F[(a,role)].update(f["flat"]); N[(a,role)]+=len(ct)
            GW[(a,gid,role)].update(ct); GF[(a,gid,role)].update(f["flat"])
            GN[(a,gid,role)]+=len(ct)
        if n%20000==0: print("  ... %s"%format(n,","),flush=True)
    pr.wait()
    groups=sorted({gid for (_,gid,_) in GN})
    print("\n%s passages | %d groups | prompt vocabulary stripped (%d types)"
          % (format(n,","),len(groups),len(promptw)))

    def consist(a,key,G):
        up=s=0
        for gid in groups:
            b=G[(a,gid,"BOTH")].get(key,0); bn=GN[(a,gid,"BOTH")]
            c=sum(G[(a,gid,r)].get(key,0) for r in ("CONTROL_A","CONTROL_B"))
            cd=sum(GN[(a,gid,r)] for r in ("CONTROL_A","CONTROL_B"))
            if bn and cd: s+=1; up += (b/bn) > (c/cd)
        return up,s

    res={"_meta":{"passages":n,"groups":len(groups),"prompt_types_removed":len(promptw),
                  "control":"CONTROL_A + CONTROL_B (consonant adjective pair)",
                  "min_word":MIN_WORD,"min_field":MIN_FIELD}}
    for kind,store,gstore,floor in (("words",W,GW,MIN_WORD),("fields",F,GF,MIN_FIELD)):
        res[kind]={}
        for a in ("base","aligned"):
            B=store[(a,"BOTH")]; bn=N[(a,"BOTH")]
            K=store[(a,"CONTROL_A")]+store[(a,"CONTROL_B")]
            kn=N[(a,"CONTROL_A")]+N[(a,"CONTROL_B")]
            rows=[]
            for k in set(B)|set(K):
                if B[k]+K[k]<floor: continue
                ratio=((B[k]+0.5)/bn)/((K[k]+0.5)/kn)
                u,s=consist(a,k,gstore)
                rows.append({"key":k,"ratio":round(ratio,3),"n_both":B[k],
                             "n_ctrl":K[k],"groups_up":u,"groups":s})
            rows.sort(key=lambda r:-r["ratio"])
            res[kind][a]={"ranked":rows[:60],"depleted":rows[-15:],"n_ranked":len(rows)}
            print("\n%s %s -- top 8 of %d"%(kind.upper(),a.upper(),len(rows)))
            for r in rows[:8]:
                print("   %-22s %6.2fx  n=%-6d %2d/%d"
                      %(r["key"],r["ratio"],r["n_both"],r["groups_up"],r["groups"]))
    json.dump(res,open(OUT,"w"),indent=1)
    print("\n-> %s"%os.path.relpath(OUT,ROOT))


if __name__=="__main__":
    main()
