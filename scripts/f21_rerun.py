"""P2: F21 political-economy re-run — coherence-controlled AlignmentAsymmetryTask.

Runs F21's OWN tagger (AlignmentAsymmetryTask) across decomposable families,
alongside the disposition tagger for coherence control.

Does proceduralization track the safety/alignment component INDEPENDENT of coherence?

Usage:
    uv run python scripts/f21_rerun.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from malign_logits import MODEL_FAMILIES, PATH_DATA
from malign_logits.cache import get_cache
from malign_logits.experiments import DEFAULT_PROMPTS
from malign_logits.tasks.score_disposition import DispositionTask, prepare_text

from largeliterarymodels.tasks.score_alignment_asymmetry import (
    AlignmentAsymmetryTask, prepare_alignment_text,
)

INST_PROMPTS = {k: v for k, v in DEFAULT_PROMPTS.items() if k.startswith('institutional_')}

FAMILIES = {
    'zephyr': MODEL_FAMILIES['zephyr'],       # No safety data
    'amber': MODEL_FAMILIES['amber'],          # PKU-SafeRLHF DPO
    'tulu': MODEL_FAMILIES['tulu'],            # CoCoNot safety SFT
    'olmo': MODEL_FAMILIES['olmo'],
    'olmo-tiny': MODEL_FAMILIES['olmo-tiny'],
    'pythia': MODEL_FAMILIES['pythia'],
}

N_PER = 5  # generations per prompt per layer


def main():
    cm = get_cache()

    # Build passages for BOTH taggers
    passages = []
    for fkey, fam in FAMILIES.items():
        for layer_key, layer_name, mid in [
            ('base', 'base', fam.base),
            ('ego', 'sft', fam.ego),
            ('superego', 'dpo', fam.superego),
        ]:
            if mid is None:
                continue
            for pkey, prompt in INST_PROMPTS.items():
                for idx in range(N_PER):
                    gen = cm.get_generation(mid, prompt, temp=1.0, idx=idx)
                    if not gen or len(gen.strip()) < 20:
                        continue
                    passages.append({
                        'family': fkey,
                        'layer': layer_name,
                        'model_id': mid,
                        'prompt_key': pkey,
                        'prompt': prompt,
                        'gen_idx': idx,
                        'text_disposition': prepare_text(gen.strip(), prompt),
                        'text_asymmetry': prepare_alignment_text(gen.strip(), prompt),
                        'raw_generation': gen.strip()[:500],
                    })

    print(f"Passages: {len(passages)}")
    print(f"By family×layer:")
    for fkey in sorted(FAMILIES.keys()):
        for layer in ['base', 'sft', 'dpo']:
            n = sum(1 for p in passages if p['family'] == fkey and p['layer'] == layer)
            if n > 0:
                print(f"  {fkey:12s} {layer:5s}: {n}")

    # ── Score with AlignmentAsymmetryTask (F21's own tagger) ──
    print(f"\nScoring with AlignmentAsymmetryTask...")
    asym_task = AlignmentAsymmetryTask()
    asym_task.model = 'deepseek/deepseek-chat'

    asym_texts = [p['text_asymmetry'] for p in passages]
    asym_results = []
    for i, (idx, result) in enumerate(asym_task.imap(asym_texts, num_workers=8)):
        if result is None:
            continue
        row = {k: v for k, v in passages[idx].items()
               if k not in ('text_disposition', 'text_asymmetry')}
        row['scored_asym'] = True
        for field in type(result).model_fields:
            val = getattr(result, field)
            if isinstance(val, list):
                row[f'asym_{field}'] = '; '.join(str(v) for v in val)
            elif isinstance(val, bool):
                row[f'asym_{field}'] = val
            else:
                row[f'asym_{field}'] = val
        asym_results.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(asym_texts)}")

    # ── Score with DispositionTask (for coherence control) ──
    print(f"\nScoring with DispositionTask (coherence)...")
    disp_task = DispositionTask()
    disp_task.model = 'deepseek/deepseek-chat'

    disp_texts = [p['text_disposition'] for p in passages]
    disp_results = {}
    for i, (idx, result) in enumerate(disp_task.imap(disp_texts, num_workers=8)):
        if result is None:
            continue
        key = (passages[idx]['family'], passages[idx]['layer'],
               passages[idx]['prompt_key'], passages[idx]['gen_idx'])
        disp_results[key] = {
            'coherence': result.coherence,
            'genre_stability': result.genre_stability,
            'de_escalation': result.de_escalation,
            'deliberation': result.deliberation,
        }
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(disp_texts)}")

    # ── Merge ──
    df = pd.DataFrame(asym_results)
    for _, row in df.iterrows():
        key = (row['family'], row['layer'], row['prompt_key'], row['gen_idx'])
        if key in disp_results:
            for k, v in disp_results[key].items():
                df.loc[df.index[_], f'disp_{k}'] = v

    # ── Results ──
    print(f"\n{'='*70}")
    print(f"P2: F21 PROCEDURALIZATION — base vs SFT vs DPO")
    print(f"{'='*70}")

    asym_dims = ['asym_institutional_deference', 'asym_agency', 'asym_assertiveness',
                 'asym_power_acknowledgment', 'asym_strategy_specificity']

    for fkey in sorted(df.family.unique()):
        print(f"\n  {fkey}:")
        print(f"    {'layer':5s} {'n':>4s}", end='')
        for d in asym_dims:
            print(f"  {d.replace('asym_','')[:6]:>6s}", end='')
        print(f"  {'coher':>5s}")

        for layer in ['base', 'sft', 'dpo']:
            sub = df[(df.family == fkey) & (df.layer == layer)]
            if sub.empty:
                continue
            print(f"    {layer:5s} {len(sub):4d}", end='')
            for d in asym_dims:
                if d in sub.columns:
                    print(f"  {sub[d].mean():6.2f}", end='')
                else:
                    print(f"    n/a", end='')
            if 'disp_coherence' in sub.columns:
                print(f"  {sub['disp_coherence'].mean():5.2f}", end='')
            print()

    out = os.path.join(PATH_DATA, "f21_rerun.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
