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

## THE CONTROL IS `CONTROL_A`/`CONTROL_B`, NOT `POLE_A`/`POLE_B`

The first three versions of this file used the POLES as the control and that is
the wrong contrast. The corpus carries five roles per group:

    BOTH       He was beautiful and disgusting and she wanted to   two OPPOSED
    POLE_A     He was beautiful and she wanted to                  ONE adjective
    POLE_B     He was disgusting and she wanted to                 ONE adjective
    CONTROL_A  He was beautiful and radiant and she wanted to      two CONSONANT
    CONTROL_B  He was disgusting and foul and she wanted to        two CONSONANT

**The poles differ from BOTH in FORM as well as content** -- one adjective
against two, 7.5 tokens against 9.6 -- so `BOTH - mean(POLE)` confounds
contradiction with prompt length and syntactic weight. The CONTROLS hold the
form fixed (two adjectives joined by `and`) and vary ONLY whether the pair is
opposed or consonant. They exist for exactly this reason and were in the data
the whole time.

Both excesses are now computed. `excess_vs_control` is the claim;
`excess_vs_pole` is retained only to show what the confound was worth.

## WORD_TRUNCATION -- THE DEFECT THIS FILE SHIPPED FIRST TIME

The first version stored `most_common(400)` per cell. That discarded every word
occurring fewer than ~7 times in a cell, which is EXACTLY the population that
discriminates: `contradiction`, `paradox`, `torn`, `conflict`, `simultaneously`,
`ambivalence` -- and `both` -- were all ABSENT from the stored data. Every
downstream analysis then reported no word-level arm difference, and the
split-sample, the BH correction and the permutation all ran faithfully on a
vocabulary with the answer removed from it.

**A frequency cap is a population decision disguised as a storage decision.**
It was written to keep the artifact small and it silently defined the study.
RH found it by disbelieving the result, not by reading the code.

The full vocabulary is now stored. The artifact is larger and that is the
correct trade: an analysis can always impose a floor and declare it; it cannot
recover what the producer threw away.
"""
import argparse, collections, json, math, os, subprocess, sys, statistics as st

HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP)); sys.path.insert(0,ROOT)
from malign_logits import fields
CAT=os.path.join(ROOT,"data","prompt_categorisation.json")
CH="/opt/homebrew/bin/clickhouse"
OUT=os.path.join(CAMP,"results","both_vs_controls_fields_v2.json")
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
    ROLES=("POLE_A","POLE_B","BOTH","CONTROL_A","CONTROL_B")
    for gid,r in g.items():
        #: BOTH + the two CONTROLS is the REQUIRED core. The poles are carried
        #: when present but are NOT the control -- see the module docstring.
        if not all(k in r for k in ("BOTH","CONTROL_A","CONTROL_B")):
            continue
        for role in ROLES:
            if role in r:
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


WINDOW=0


def main(smoke):
    pmap,nslots=prompt_map()
    print("English contradiction slots with all three roles: %d (%d texts)"
          %(nslots,len(pmap)))
    #: ALL complete base/aligned pairs present in the corpus, not the
    #: one-per-lineage representatives. The representative rule exists so scale
    #: siblings do not count as independent PAIRS; here the unit is the GROUP
    #: and models are replicates within it, so siblings reduce within-group
    #: noise and inflate no independence claim. Using the representative list
    #: dropped 4 of 29 available pairs for no gain.
    _fp=json.load(open(os.path.join(ROOT,"data","base_aligned_pairs.json")))
    _fp=_fp if isinstance(_fp,list) else _fp.get("pairs",[])
    pairs=sorted({(x.get("base"),x.get("aligned")) if isinstance(x,dict) else tuple(x)
                  for x in _fp})
    pairs=[p for p in pairs if p[0] and p[1]]
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
        #: WINDOW. The finding being corroborated (`second_order_naming.md:20`)
        #: scores the FIRST 50 WORDS, because at 256 the continuation drifts off
        #: the prompt. A discrete marker only gains noise from drift; a RATE PER
        #: CONTENT TOKEN is diluted by it, so the window matters more for fields
        #: than for markers and the two must not be compared across windows.
        if WINDOW:
            text=" ".join(text.split()[:WINDOW])
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
               "cells":{"|".join(k):{"fields":dict(v[0]),"words":dict(v[1]),   # FULL vocabulary -- see WORD_TRUNCATION below
                                     "n_passages":v[2],"n_content":v[3]}
                        for k,v in acc.items()}},open(OUT,"w"))
    print("-> %s"%os.path.relpath(OUT,ROOT))


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--window",type=int,default=0,
                    help="score only the first N WORDS of each continuation; "
                         "0 = the whole 256-token passage")
    a=ap.parse_args()
    globals()["WINDOW"]=a.window
    if WINDOW:
        globals()["OUT"]=OUT.replace(".json","_w%d.json"%a.window)
    main(a.smoke)
