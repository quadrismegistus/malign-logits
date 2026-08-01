"""F40 AT THE LINEAGE, NOT THE BASE STRING. Malign, [2166] follow-up.

`mass_flow_boot.py` dedups on the BASE STRING (`if b not in seen`) and resamples
**39** units. `data/lineage_map_models.json` says those 39 base strings are
**34 independent pretraining lineages**: Falcon3's 1B/3B/7B/10B are one release
at four scales, OLMo-3's 7B/32B one, Qwen2.5's 0.5B/7B one. **A bootstrap that
resamples 39 units when 34 are independent draws its CI too narrow, and the
error runs toward significance.**

**NO REPRESENTATIVE IS CHOSEN.** Collapsing needs one base per multi-base
lineage, and picking one is a researcher degree of freedom on a published
number. There are exactly 2 x 2 x 4 = **16 possible choices, so all 16 are run**
and the range is reported. A conclusion that holds across all 16 does not depend
on the choice; one that does not is a conclusion about the choice.
"""
import collections, itertools, json, sys, os
import numpy as np
sys.argv = [sys.argv[0]]
src = open(os.path.join(os.path.dirname(__file__), 'mass_flow_boot.py')).read()
head = src.split('ALL = list(range(len(pairs)))')[0]
g = {'__name__': '__mf__', '__file__': os.path.join(os.path.dirname(__file__), 'mass_flow_boot.py')}
exec(compile(head, 'mass_flow_boot.py', 'exec'), g)
pairs, excess, TI, NBOOT, SEED = g['pairs'], g['excess'], g['TI'], g['NBOOT'], g['SEED']
prompts = g['prompts']

m2l = json.load(open('data/lineage_map_models.json'))["model_to_lineage"]
bylin = collections.defaultdict(list)
for i, (k, b, a) in enumerate(pairs):
    bylin[m2l.get(b, b)].append(i)
multi = {l: v for l, v in bylin.items() if len(v) > 1}
single = [v[0] for l, v in bylin.items() if len(v) == 1]
print(f"F40 units (base strings) {len(pairs)}   LINEAGES {len(bylin)}   "
      f"collapsing choices {np.prod([len(v) for v in multi.values()])}\n")
for l, v in multi.items():
    print(f"  {l}: {[pairs[i][1] for i in v]}")

KEYS = ['violence_liminal', 'sexual_explicit', 'profanity', 'violence_explicit']
CONTRASTS = [('violence_liminal', 'sexual_explicit'),
             ('violence_liminal', 'profanity')]
res = collections.defaultdict(list)
for combo in itertools.product(*[v for v in multi.values()]):
    ALL = sorted(single + list(combo))
    rng = np.random.default_rng(SEED)
    pt = {c: excess(c, ALL) for c in prompts}
    dr = collections.defaultdict(list)
    for _ in range(NBOOT):
        s = list(rng.choice(ALL, size=len(ALL), replace=True))
        for c in KEYS:
            e = excess(c, s)
            if e is not None:
                dr[c].append(e[TI])
    for a_, b_ in CONTRASTS:
        d = np.array(dr[a_]) - np.array(dr[b_])
        lo, hi = np.percentile(d, [2.5, 97.5])
        res[(a_, b_)].append((pt[a_][TI] - pt[b_][TI], lo, hi,
                              float((d >= 0).mean())))

print(f"\nALL {len(res[CONTRASTS[0]])} COLLAPSES, n={len(single)+len(multi)} lineages\n")
print(f"{'contrast':42s}{'point':>16s}{'p_boot':>18s}   holds")
for k, v in res.items():
    p = [x[0] for x in v]; pb = [x[3] for x in v]
    holds = sum(1 for x in v if (x[1] < 0) == (x[2] < 0))
    print(f"{k[0]+' - '+k[1]:42s}{min(p):+.2f}..{max(p):+.2f}   "
          f"{min(pb):.4f}..{max(pb):.4f}   {holds}/{len(v)}")
