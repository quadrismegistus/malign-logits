"""Does the concentration survive resampling? mass_flow2.py reports point
estimates against a proportional-drain null and nothing else -- no intervals, no
resampling, no correction across the ten prompt categories. Its headline claim is
a CONTRAST: transgressive-specific drain is large at violence_liminal (-1.60pp
excess) and absent at sexual_explicit (-0.06pp). A contrast between two cells of
an uncertainty-free table is not yet a finding.

THE UNIT IS THE LINEAGE, not the prompt and not the family. mass_flow2 already
dedups by base model (`if b and a and b not in seen`), which is the
pseudo-replication fix -- six Llama-3.1-8B families would otherwise vote six
times with one base. Bootstrap resamples those deduped lineages with replacement
and recomputes the whole pipeline inside each replicate: per-prompt lineage mean,
then per-category prompt mean, then the proportional-drain excess. Resampling any
later stage would treat lineage-level variation as fixed, which is the variation
that matters.

Reports the two cells and, more importantly, their DIFFERENCE -- because "one is
big and one is nought" is only a finding if the gap between them excludes zero.
"""
import collections, csv, json, sys
import numpy as np
from malign_logits.cache import open_stash
from malign_logits import MODEL_FAMILIES
import malign_logits.experiments as E

TAGFILE = sys.argv[1] if len(sys.argv) > 1 else "data/f40_vocab/vocab_tagged_v1.csv"
NBOOT, SEED = 2000, 20260727
TAGS = ['PROCEDURAL', 'CONTESTATION', 'TRANSGRESSIVE', 'DEMOTIC',
        'NARRATIVE_CRAFT', 'AFFECT', 'OTHER']
TI = TAGS.index('TRANSGRESSIVE')

TAG = {r['word']: r['primary'] for r in csv.DictReader(open(TAGFILE))}
s = open_stash('data/raw/cache/word_probs')
idx = {}
for k in s.keys():
    if isinstance(k, dict) and k.get('mode', 'raw') == 'raw':
        idx[(k['model'], k['prompt'])] = k
seen = {}
for k, f in MODEL_FAMILIES.items():
    b, a = f.base, getattr(f, 'superego', None)
    if b and a and b not in seen:
        seen[b] = (k, b, a)
pairs = list(seen.values())

ip = getattr(E, 'INSTITUTIONAL_PROMPTS', {})
INST = set(ip.values() if isinstance(ip, dict) else ip)
items = (json.load(open('data/f37_prompt_items.json'))
         + json.load(open('data/f37_prompt_items_supp.json')))
CAT = {i['text']: i.get('category', '?') for i in items}


def pcat(p):
    if p in INST:
        return 'institutional'
    c = CAT.get(p)
    return None if (not c or c == '?') else c


# Collect once: cell[(cat, prompt)][lineage] = (base_vec, delta_vec).
cell = collections.defaultdict(dict)
for pr in {p for (_, p) in idx}:
    c = pcat(pr)
    if c is None:
        continue
    for li, (_, b, a) in enumerate(pairs):
        kb, ka = idx.get((b, pr)), idx.get((a, pr))
        if kb is None or ka is None:
            continue
        wb, wa = s[kb], s[ka]
        if not isinstance(wb, dict) or not isinstance(wa, dict) or not wb or not wa:
            continue
        bb = np.zeros(len(TAGS)); dd = np.zeros(len(TAGS))
        for w, t in TAG.items():
            j = TAGS.index(t)
            bb[j] += wb.get(w, 0.0)
            dd[j] += wa.get(w, 0.0) - wb.get(w, 0.0)
        cell[(c, pr)][li] = (bb, dd)

# Keep the same >=12-lineage prompt filter as mass_flow2, applied on the FULL
# lineage set. Applying it inside replicates would let the filter itself vary.
prompts = collections.defaultdict(list)
for (c, pr), d in cell.items():
    if len(d) >= 12:
        prompts[c].append(pr)


def excess(cat, lineages):
    """Proportional-drain excess for one category under a lineage multiset."""
    B, D = [], []
    for pr in prompts[cat]:
        d = cell[(cat, pr)]
        got = [d[li] for li in lineages if li in d]
        if not got:
            continue
        B.append(np.mean([g[0] for g in got], axis=0))
        D.append(np.mean([g[1] for g in got], axis=0))
    if not B:
        return None
    b = np.mean(B, axis=0); dl = np.mean(D, axis=0)
    T = dl.sum(); sh = b / b.sum() if b.sum() > 0 else b * 0
    return (dl - T * sh) * 100


ALL = list(range(len(pairs)))
print(f"tagging: {TAGFILE}")
print(f"lineages (base-deduped): {len(pairs)}   bootstrap: {NBOOT}, seed {SEED}\n")

point = {c: excess(c, ALL) for c in prompts}
rng = np.random.default_rng(SEED)
draws = collections.defaultdict(list)
for _ in range(NBOOT):
    samp = list(rng.choice(ALL, size=len(ALL), replace=True))
    for c in prompts:
        e = excess(c, samp)
        if e is not None:
            draws[c].append(e[TI])

print(f"{'prompt category':20s}{'point':>9s}{'2.5%':>9s}{'97.5%':>9s}   excludes 0")
for c in ['violence_liminal', 'violence_explicit', 'power', 'substance',
          'sexual_liminal', 'sexual_explicit', 'profanity', 'death',
          'neutral', 'institutional']:
    if c not in point:
        continue
    a = np.array(draws[c]); lo, hi = np.percentile(a, [2.5, 97.5])
    print(f"{c:20s}{point[c][TI]:+9.2f}{lo:+9.2f}{hi:+9.2f}   "
          f"{'YES' if (lo < 0) == (hi < 0) else 'no'}")

# The claim is the CONTRAST, so test it directly on paired replicates.
print("\nTHE CONTRAST, paired within replicate (the actual claim):")
for a_, b_ in [('violence_liminal', 'sexual_explicit'),
               ('violence_liminal', 'profanity'),
               ('violence_explicit', 'sexual_explicit')]:
    if a_ not in draws or b_ not in draws:
        continue
    d = np.array(draws[a_]) - np.array(draws[b_])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {a_} - {b_}: {point[a_][TI]-point[b_][TI]:+.2f} "
          f"CI [{lo:+.2f}, {hi:+.2f}]  p_boot(>=0) = {float((d >= 0).mean()):.4f}")
