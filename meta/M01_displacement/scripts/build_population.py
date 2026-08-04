"""Build the k>=15 annotation population from the canonical census.

RH's decisions, 2026-08-04: threshold k>=15 co-occurring edges, ALL 44
operation_edges, content pairs. Three tiers:
  REAL       k>=15 content pairs, dev-set excluded
  NEAR-MISS  per REAL pair: same prompt, same faller, the STATIONARY word
             with the highest faller-co-occurrence edge count (available and
             did not move — the comparison the design rests on)
  EXHIBIT    dev-set pairs present in the census (descriptive tier, [707].2:
             excluded from every statistic)

Stationary rule (DECLARED, for P's text): p_base >= CANONICAL.min_prob and
|delta| <= 0.0005 — candidacy floor from the canonical rule, stillness band
carried from the f13 draw where it was first declared. Movement itself is
Cell.movement(CANONICAL); stationarity is a statement about words the rule
did not move, computed beside it, never replacing it.
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

SCRATCH = ("/private/tmp/claude-502/-Users-rj416-Dropbox-Prof-Articles-"
           "TheoryMachines/33d62191-5812-4ce2-8fc2-ad246fd58974/scratchpad")
EPS = 0.0005
K = 15

import importlib.util
spec = importlib.util.spec_from_file_location(
    "draw", os.path.join(REPO, "scripts/f13_draw_relation_items.py"))
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
FUNC = D.FUNC
DEV = D.DEV


def content(w):
    return w.lower() not in FUNC and any(c.isalpha() for c in w)


census = pd.read_parquet(os.path.join(SCRATCH, "pair_edge_census_CANONICAL.parquet"))
census["both_content"] = census.a.map(content) & census.b.map(content)

real = census[(census.both_content) & (census.n_edges >= K)].copy()
real["in_dev"] = [ (r.prompt, r.a, r.b) in DEV for r in real.itertuples() ]
exhibits_from_threshold = real[real.in_dev].copy()
real = real[~real.in_dev].copy()

# exhibit tier: every dev pair present in the census AT ANY edge count
cen_dev = census[[ (r.prompt, r.a, r.b) in DEV for r in census.itertuples() ]].copy()

print(f"REAL items (k>={K}, content, dev-excluded): {len(real):,}")
print(f"dev pairs in census (exhibit tier): {len(cen_dev)} "
      f"(of which {len(exhibits_from_threshold)} clear k>={K})")

# ---- pass 2: stationary partners for exactly the (prompt, faller) pairs REAL needs
needed = collections.defaultdict(set)          # prompt -> {fallers}
for r in real.itertuples():
    needed[r.prompt].add(r.a)
print(f"(prompt, faller) keys needing decoys: "
      f"{sum(len(v) for v in needed.values()):,} across {len(needed):,} prompts")

SENTINEL = re.compile(r"^<<<[A-Z]+:")
CJK = re.compile(r"[一-鿿㐀-䶿]")
stimuli = {t for t in needed}                   # only prompts REAL touches

_p, models, _h, _d = CC.frozen_population()
edges, _drop = CC.operation_edges(models)
print(f"edges: {len(edges)}   prompts with REAL items: {len(stimuli):,}")

co = collections.Counter()                      # (prompt, a, s) -> n_edges
for i, (fam, _pos, step) in enumerate(edges):
    for t in stimuli:
        c = step.cell(t)
        if not c.is_present:
            continue
        m = c.movement(CANONICAL)
        if m is None:
            continue
        fall = [w for w in m.fallers if w in needed[t]]
        if not fall:
            continue
        P = c.pre.probs
        Q = c.post.probs
        stat = [w for w in set(P) | set(Q)
                if w != RESIDUAL_KEY
                and P.get(w, 0.0) >= CANONICAL.min_prob
                and abs(Q.get(w, 0.0) - P.get(w, 0.0)) <= EPS
                and content(w)]
        for a in fall:
            for s in stat:
                if s != a:
                    co[(t, a, s)] += 1
    if (i + 1) % 11 == 0:
        print(f"  [{i+1:>2}/{len(edges)}] stationary co-occurrences: {len(co):,}",
              flush=True)

# best decoy per (prompt, faller): highest co-occurrence count
best = {}
for (t, a, s), n in co.items():
    k = (t, a)
    if k not in best or n > best[k][1] or (n == best[k][1] and s < best[k][0]):
        best[k] = (s, n)

rows = []
for r in real.itertuples():
    rows.append(dict(item_class="REAL", prompt=r.prompt, a=r.a, b=r.b,
                     n_edges=r.n_edges, n_base_dpo=r.n_base_dpo,
                     edge_labels=r.edge_labels, edges=r.edges))
    hit = best.get((r.prompt, r.a))
    if hit:
        s, n = hit
        rows.append(dict(item_class="NEAR-MISS", prompt=r.prompt, a=r.a, b=s,
                         n_edges=n, n_base_dpo=None,
                         edge_labels="stationary", edges=""))
for r in cen_dev.itertuples():
    rows.append(dict(item_class="EXHIBIT", prompt=r.prompt, a=r.a, b=r.b,
                     n_edges=r.n_edges, n_base_dpo=r.n_base_dpo,
                     edge_labels=r.edge_labels, edges=r.edges))

items = pd.DataFrame(rows).drop_duplicates(subset=["item_class", "prompt", "a", "b"])
out = os.path.join(SCRATCH, "population_p_items.parquet")
items.to_parquet(out, index=False)

print("\n=== population summary ===")
print(items.item_class.value_counts().to_dict())
nm = items[items.item_class == "NEAR-MISS"]
print(f"REAL pairs with a matched decoy: {items[items.item_class=='REAL'].shape[0] - (len(needed and []) or 0):,} "
      f"-> decoys drawn: {len(nm):,}")
print(f"decoy stationary co-occurrence, median {nm.n_edges.median():.0f}, "
      f"min {nm.n_edges.min()}, max {nm.n_edges.max()}")
print(f"total items: {len(items):,}   x3 coders = {3*len(items):,} judgments")
print(f"saved: {out}")
