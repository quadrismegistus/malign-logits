#!/usr/bin/env python3
"""Compute bits_resistance.csv and run mixed-effects model.

Bits of resistance = log2(P_base / P_aligned) per word per model.
Symmetric filter: include if EITHER base OR aligned > 1%.

Usage:
    python scripts/compute_bits_resistance.py
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')
from malign_logits.cache import get_cache
from malign_logits import MODEL_FAMILIES
from malign_logits.experiments import DEFAULT_PROMPTS

cm = get_cache()
eps = 1e-8
THRESHOLD = 0.01

# Build data-driven vocabulary per prompt
print("Scanning all word_probs for data-driven vocabulary...")
prompt_vocabs = {}
for key, fam in MODEL_FAMILIES.items():
    checkpoints = [fam.base, fam.ego, fam.superego, fam.reinforced_superego]
    for mid in [c for c in checkpoints if c]:
        for pname, prompt in DEFAULT_PROMPTS.items():
            wp = cm.get_word_probs(mid, prompt)
            if not wp:
                continue
            for word, prob in wp.items():
                if prob >= THRESHOLD:
                    prompt_vocabs.setdefault(pname, set()).add(word)

print(f"Vocabulary: {sum(len(v) for v in prompt_vocabs.values())} word×prompt pairs")

# Compute bits_resistance for all families
rows = []
for key, fam in sorted(MODEL_FAMILIES.items()):
    checkpoints = []
    if fam.base:
        checkpoints.append(('base', fam.base))
    if fam.ego:
        checkpoints.append(('sft', fam.ego))
    if fam.superego:
        checkpoints.append(('dpo', fam.superego))
    elif fam.reinforced_superego:
        checkpoints.append(('instruct', fam.reinforced_superego))
    if fam.reinforced_superego and fam.superego:
        checkpoints.append(('rlvr', fam.reinforced_superego))

    if len(checkpoints) < 2:
        continue

    base_stage, base_mid = checkpoints[0]
    for stage, mid in checkpoints[1:]:
        for pname, prompt in DEFAULT_PROMPTS.items():
            if pname not in prompt_vocabs:
                continue
            base_wp = cm.get_word_probs(base_mid, prompt)
            aligned_wp = cm.get_word_probs(mid, prompt)
            if not base_wp or not aligned_wp:
                continue
            for word in prompt_vocabs[pname]:
                bp = base_wp.get(word, 0)
                ap = aligned_wp.get(word, 0)
                if bp < THRESHOLD and ap < THRESHOLD:
                    continue
                bits = np.log2((bp + eps) / (ap + eps))
                rows.append({
                    'family': key, 'method': stage, 'prompt': pname,
                    'word': word, 'base_prob': bp, 'aligned_prob': ap,
                    'bits_resistance': bits,
                })

df = pd.DataFrame(rows)
df.to_csv('data/bits_resistance_datadriven.csv', index=False)
print(f"Saved: {len(df)} rows, {df['family'].nunique()} families")

# Mixed-effects model
print("\n=== Mixed-effects model ===")
df_sym = df[(df['base_prob'] > THRESHOLD) | (df['aligned_prob'] > THRESHOLD)].copy()
df_sym['bits_r_clipped'] = df_sym['bits_resistance'].clip(-15, 15)

model = smf.mixedlm("bits_r_clipped ~ C(method)", df_sym, groups=df_sym["family"], re_formula="~1")
result = model.fit(reml=True)
print(result.summary().tables[0])

for param in result.fe_params.index:
    if 'method' in param:
        val = result.fe_params[param]
        se = result.bse[param]
        pval = result.pvalues[param]
        print(f"  {param}: {val:+.3f} bits (SE={se:.3f}, p={pval:.4f})")

var_fam = float(result.cov_re.iloc[0, 0])
var_res = result.scale
print(f"\nFamily: {var_fam:.3f} ({var_fam/(var_fam+var_res)*100:.1f}%)")
print(f"Residual: {var_res:.3f} ({var_res/(var_fam+var_res)*100:.1f}%)")
