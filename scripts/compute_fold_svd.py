"""Compute and save full SVD data (S, Vt, diff_matrix) for fold figures.

Saves:
  data/fold_svd_{family}.npz — S (singular values), Vt (right singular vectors),
                                diff_matrix (prompt × hidden_dim), prompts (labels)

Usage: python scripts/compute_fold_svd.py [family1 family2 ...]
       Default: pythia llama olmo
"""
import sys
import gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    from malign_logits import MODEL_FAMILIES
    from malign_logits.experiments import DEFAULT_PROMPTS
    from malign_logits.trajectory import last_hidden

    families = sys.argv[1:] if len(sys.argv) > 1 else ['pythia', 'llama', 'olmo']

    for fam_key in families:
        print(f'\n=== {fam_key} ===')
        fam = MODEL_FAMILIES[fam_key]
        fold_df = pd.read_csv(f'data/fold_rank_{fam_key}.csv')
        best_L = int(fold_df['layer'].iloc[0])
        print(f'  Layer: {best_L}')

        print(f'  Loading {fam.base}...')
        tok = AutoTokenizer.from_pretrained(fam.base, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            fam.base, dtype=torch.float16, device_map='mps', trust_remote_code=True)

        print(f'  Loading {fam.superego}...')
        dpo_model = AutoModelForCausalLM.from_pretrained(
            fam.superego, dtype=torch.float16, device_map='mps', trust_remote_code=True)

        print(f'  Computing hidden state diffs for {len(DEFAULT_PROMPTS)} prompts...')
        diff_vecs = []
        prompt_labels = []
        for i, (label, prompt) in enumerate(DEFAULT_PROMPTS.items()):
            base_h = last_hidden(base_model, tok, prompt, best_L)
            dpo_h = last_hidden(dpo_model, tok, prompt, best_L)
            diff_vecs.append((dpo_h - base_h).numpy())
            prompt_labels.append(label)
            if (i + 1) % 20 == 0:
                print(f'    [{i+1}/{len(DEFAULT_PROMPTS)}]')

        diff_matrix = np.stack(diff_vecs)
        U, S, Vt = np.linalg.svd(diff_matrix, full_matrices=False)
        cumvar = np.cumsum(S**2) / np.sum(S**2)
        k_50 = int(np.searchsorted(cumvar, 0.5)) + 1

        out = f'data/fold_svd_{fam_key}.npz'
        np.savez(out, S=S, Vt=Vt, U=U, diff_matrix=diff_matrix,
                 prompt_labels=np.array(prompt_labels), layer=best_L)

        print(f'  K_50={k_50}, top1={S[0]**2/(S**2).sum()*100:.1f}%')
        print(f'  diff_matrix: {diff_matrix.shape}')
        print(f'  S: {S.shape}, Vt: {Vt.shape}')
        print(f'  Saved {out}')

        del base_model, dpo_model, tok
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print('\nDone.')
