#!/usr/bin/env python
"""fc_explicit_read.py — RH's commission, printed as TEXT.

The commission asked what these two pairs DO on graded transgressive prompts,
not for a statistic. **The fence forbids a statistic**: not the frozen
210-prompt population, not a registered stratum, not poolable with the battery,
not comparable to the domain gradient, not a rate. So this prints beams.

Undisturbed arm only, top beam by log-prob, base beside aligned.
"""
import os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import json
import fc_analyse as F
from malign_logits.cache import get_cache

cfg = json.load(open(os.path.join(ROOT, "data", "fc_explicit_probe_mps.json")))
cat = {}
for q in cfg["pairs"]:
    for s in q["sites"]:
        cat[s["prompt"]] = s["key"]
by = F.load(get_cache(), None)
PAIRS = [(q["base"], q["aligned"]) for q in cfg["pairs"]]

for base, aligned in PAIRS:
    pid = "%s>%s" % (base, aligned)
    if pid not in by:
        print("  %s — no beams" % pid); continue
    print("=" * 78)
    print("%s   >   %s" % (base.split("/")[-1], aligned.split("/")[-1]))
    print("=" * 78)
    best = {}
    for (role, arm, w, prompt), rec in by[pid].items():
        if arm != "undisturbed":
            continue
        bm = rec.get("beams") or []
        if not bm:
            continue
        top = max(bm, key=lambda b: b.get("log_prob", -9e9))
        best.setdefault(prompt, {})[role] = top.get("text", "").strip()
    #: **ONLY THE COMMISSION'S 22.** These pairs also carry the 210 M01 beam
    #: prompts in the same stash, and an unfiltered reader prints them under
    #: the commission's heading with a "[?]" category. The fence exists to keep
    #: this set separate from the battery; a reader that silently mixes them is
    #: the fence failing at the only place anyone looks.
    for prompt in sorted(best, key=lambda p: (cat.get(p, "zz"), p)):
        if prompt not in cat:
            continue
        d = best[prompt]
        if len(d) < 2:
            continue
        print("\n  [%s] %s" % (cat.get(prompt, "?"), prompt))
        print("      base    -> %s" % d["base"][:96])
        print("      aligned -> %s" % d["aligned"][:96])
