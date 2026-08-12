#!/usr/bin/env python
"""What is distinctive about a BOTH generation, against its OWN two poles.

    uv run python both_vs_controls_fields.py --smoke     # one model, fast
    uv run python both_vs_controls_fields.py             # full

RH's question: with CONTROL_A / CONTROL_B in hand, is there anything else in
the BOTH generations -- distinctive words, distinctive semantic fields?

THE CONTROL IS WITHIN-GROUP AND THAT IS THE WHOLE DESIGN. For each contradiction
group the corpus holds three prompts: POLE_A, POLE_B and BOTH. Comparing BOTH
against the pooled corpus would rediscover the prompt's own vocabulary. Comparing
it against the MEAN OF ITS OWN TWO POLES asks the only interesting question:
what appears when the model is given a contradiction that does NOT appear when it
is given either half separately?

    excess(x) = rate_BOTH(x) - 0.5 * (rate_A(x) + rate_B(x))

Positive excess = present in the contradiction and in neither pole alone.

UNIT IS THE (group, model) CELL, and the test is a sign test over GROUPS with a
model's groups averaged first -- 58 models are not 58 independent observations of
a group and the groups are what vary. English only; zh is 47% of these rows and
reported apart by M02 convention.

FIELDS COME FROM `fields.count_all`, so every lexicon is scored in one pass and
`coverage` travels with each. **The three USAS views are one lexicon**: meta,
usas_fine and usas are not independent and are never summed here.
"""
import argparse, collections, json, math, os, subprocess, sys, statistics as st

HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,ROOT)
from malign_logits import fields
CAT=os.path.join(ROOT,"data","prompt_categorisation.json")
CH="/opt/homebrew/bin/clickhouse"
OUT=os.path.join(CAMP,"results","both_vs_controls_fields_split.json")
PAIRS=os.path.join(ROOT,"data","lineage_representative_pairs.txt")


def prompt_map():
    """{prompt -> [(group, role)]}, English contradiction groups with all three
    roles. ONE TEXT CAN OCCUPY SEVERAL SLOTS -- five do -- so this is
    one-to-many; a dict keyed on text would starve four groups of a role and
    hand a fifth the wrong arm ([5410])."""
    cat=json.load(open(CAT))["prompts"]
    g=collections.defaultdict(dict)
    for p in cat:
        if p.get("domain")=="contradiction" and p.get("group_id") and p.get("language")=="en":
            g[p["group_id"]][p.get("group_role")]=p
    out=collections.defaultdict(list)
    n=0
    for gid,r in g.items():
        if not all(k in r for k in ("POLE_A","POLE_B","BOTH")):
            continue
        for role in ("POLE_A","POLE_B","BOTH"):
            out[r[role]["prompt"].strip()].append((gid,role)); n+=1
    return out,n


def fetch(models):
    esc=lambda s:s.replace("\\","\\\\").replace("'","\\'")
    q=("SELECT model, prompt, text FROM malign_logits.gen_sequences "
       "WHERE corpus='f11_l2'%s FORMAT JSONEachRow"
       %(" AND model IN (%s)"%",".join("'%s'"%esc(m) for m in models) if models else ""))
    pr=subprocess.Popen([CH,"client","-q",q],stdout=subprocess.PIPE,text=True,bufsize=1<<20)
    for line in pr.stdout:
        try: r=json.loads(line)
        except Exception: continue
        yield r["model"],r["prompt"].strip(),r["text"] or ""
    pr.wait()


def main(smoke):
    pmap,nslots=prompt_map()
    print("English contradiction slots with all three roles: %d (%d texts)"
          %(nslots,len(pmap)))
    pairs=sorted([l.strip().split(">") for l in open(PAIRS) if l.strip()])
    arm={}; split={}
    #: SPLIT BY LINEAGE, DECLARED BEFORE ANY FIELD IS COUNTED. Pairs sorted by
    #: base-model name, then alternating -- deterministic, no seed, no
    #: dependence on anything measured. Selection runs on one split and the
    #: test on the other, so the noise in the two is independent and neither
    #: the base-selection nor the aligned-selection circularity can operate.
    #: The GROUPS are identical in both halves; only the models differ.
    for i,(b,a) in enumerate(pairs):
        arm[b]="base"; arm[a]="aligned"
        split[b]=split[a]=i%2
    models=[m for m in arm] if not smoke else [pairs[0][0],pairs[0][1]]
    # (arm, group, role) -> [field counter, word counter, n_passages, n_content]
    acc=collections.defaultdict(lambda:[collections.Counter(),collections.Counter(),0,0])
    seen=0
    for model,prompt,text in fetch(models):
        if model not in arm or prompt not in pmap: continue
        seen+=1
        f=fields.count_all(text)
        toks=[t for t in fields.tokens(text) if fields.is_content_word(t)]
        for gid,role in pmap[prompt]:
            k=("s%d"%split[model],arm[model],gid,role)
            a=acc[k]
            for kk,v in f["flat"].items(): a[0][kk]+=v
            a[1].update(toks); a[2]+=1; a[3]+=f["n_content"]
        if seen%20000==0: print("  ... %s passages"%format(seen,","),flush=True)
    print("scored %s passages into %d (arm,group,role) cells"%(format(seen,","),len(acc)))
    json.dump({"_meta":{"slots":nslots,"passages":seen,"smoke":smoke,
                        "unit":"(split, arm, group, role); excess = BOTH - mean(POLE_A,POLE_B)",
                        "fields":"malign_logits.fields.count_all, all six sources",
                        "warning":"meta/usas_fine/usas are ONE lexicon; never summed"},
               "cells":{"|".join(k):{"fields":dict(v[0]),"words":dict(v[1].most_common(400)),
                                     "n_passages":v[2],"n_content":v[3]}
                        for k,v in acc.items()}},open(OUT,"w"))
    print("-> %s"%os.path.relpath(OUT,ROOT))


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true")
    main(ap.parse_args().smoke)
