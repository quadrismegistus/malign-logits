"""Pair census on the CANONICAL instrument — RH's spec, 2026-08-04.

Every (prompt, faller, riser) co-occurrence across the canonical operation
edges, counted per edge with the producing edges stored. NO ad-hoc movement
rule: cells come from Step.cell(), movement from Cell.movement(CANONICAL) —
the same objects and rule as Registrations D/L/M/N/O. Population mirrors N's
§3 exactly (distinct stimuli, sentinels out, zh out). Scratchpad output only.
"""
import collections, os, re, sys

REPO = os.path.expanduser("~/github/malign-logits")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.chdir(REPO)

import pandas as pd
import m01_concentration as CC
from malign_logits.movement import CANONICAL, RESIDUAL_KEY
from malign_logits.prompts import Prompts

SENTINEL = re.compile(r"^<<<[A-Z]+:")
CJK = re.compile(r"[一-鿿㐀-䶿]")

stimuli = set()
for p in Prompts().all():
    t = p if isinstance(p, str) else (getattr(p, "text", None) or str(p))
    if SENTINEL.match(t) or CJK.search(t):
        continue
    stimuli.add(t)
print(f"english stimuli: {len(stimuli):,}   (N §3 publishes 2,199)")

_p, models, _h, drift = CC.frozen_population()
edges, dropped = CC.operation_edges(models)
labels = collections.Counter(step.label for _f, _pos, step in edges)
print(f"operation_edges: {len(edges)}   (N §4 publishes 44)   drift: {bool(drift)}")
print(f"edge labels: {dict(labels)}")

pair_hits = collections.defaultdict(list)   # (prompt, a, b) -> ["fam|label", ...]
n_cells = n_absent = n_nomove = 0
for i, (fam, _pos, step) in enumerate(edges):
    tag = f"{fam}|{step.label}"
    for t in stimuli:
        c = step.cell(t)
        if not c.is_present:
            n_absent += 1
            continue
        m = c.movement(CANONICAL)
        if m is None:
            n_absent += 1
            continue
        fall = [w for w in m.fallers if w != RESIDUAL_KEY]
        rise = [w for w in m.risers if w != RESIDUAL_KEY]
        if not (fall and rise):
            n_nomove += 1
            continue
        n_cells += 1
        for a in fall:
            for b in rise:
                pair_hits[(t, a, b)].append(tag)
    print(f"  [{i+1:>2}/{len(edges)}] {fam:<22} {step.label:<14} "
          f"cells so far {n_cells:,}  pairs {len(pair_hits):,}", flush=True)

rows = []
for (t, a, b), tags in pair_hits.items():
    lab = collections.Counter(x.split("|")[1] for x in tags)
    rows.append(dict(prompt=t, a=a, b=b, n_edges=len(tags),
                     n_base_dpo=lab.get("base->dpo", 0),
                     edge_labels="|".join(sorted(set(x.split("|")[1] for x in tags))),
                     edges="|".join(sorted(tags))))
df = pd.DataFrame(rows).sort_values("n_edges", ascending=False)
out = ("/private/tmp/claude-502/-Users-rj416-Dropbox-Prof-Articles-TheoryMachines/"
       "33d62191-5812-4ce2-8fc2-ad246fd58974/scratchpad/pair_edge_census_CANONICAL.parquet")
df.to_parquet(out, index=False)

print(f"\ncells with both roles: {n_cells:,}   absent: {n_absent:,}   "
      f"one-sided: {n_nomove:,}")
print(f"distinct (prompt, a, b): {len(df):,}   instances: {int(df.n_edges.sum()):,}")
print(f"saved: {out}")
