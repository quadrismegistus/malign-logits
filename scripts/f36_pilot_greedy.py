"""Generate greedy continuations for the judge pilot.

For each site: force the model's argmax as the first token,
then greedy-decode ~15 tokens. Produces the exact object
the geometry measured (argmax/greedy), not temp-sampled text.

Usage:
    uv run python scripts/f36_pilot_greedy.py
"""

import gc
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from malign_logits import Psyche, MODEL_FAMILIES, PATH_DATA
from malign_logits.models import load_model


MAX_NEW_TOKENS = 18  # ~15-20 tokens after the forced first token


def greedy_from_token(model, tokenizer, prompt, forced_token_id, max_new=MAX_NEW_TOKENS):
    """Generate greedy continuation after forcing a specific first token."""
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Build input: prompt + forced first token
    forced = torch.tensor([[forced_token_id]], device=device)
    input_ids = torch.cat([prompt_ids, forced], dim=1)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
        )

    # Decode only the new tokens (after prompt)
    new_tokens = out[0][prompt_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text


def main():
    meta = pd.read_csv(os.path.join(PATH_DATA, "judge_pilot_metadata.csv"))

    # Group by family to minimize model loads
    families_needed = meta['family'].unique()
    results = {}

    for fkey in sorted(families_needed):
        fam = MODEL_FAMILIES[fkey]
        fam_sites = meta[meta['family'] == fkey]

        # Determine which models we need
        models_needed = set()
        for _, r in fam_sites.iterrows():
            if r['transition'] == 'base→aligned':
                models_needed.add(('base', fam.base))
                models_needed.add(('aligned', fam.superego or fam.ego))
            elif r['transition'] == 'sft→dpo':
                models_needed.add(('sft', fam.ego))
                models_needed.add(('dpo', fam.superego))

        print(f"\n{fkey}: loading {len(models_needed)} models for {len(fam_sites)} sites")

        loaded = {}
        for label, model_id in models_needed:
            print(f"  loading {model_id}...")
            model, tokenizer = load_model(model_id)
            loaded[model_id] = (model, tokenizer)

        # Get cached logits for argmax identification
        psyche = Psyche.from_family(fkey)

        for _, r in fam_sites.iterrows():
            sid = r['site_id']
            prompt = pd.read_csv(os.path.join(PATH_DATA, "judge_pilot_v3.csv"))
            prompt_row = prompt[prompt.site_id == sid]
            if prompt_row.empty:
                continue
            prompt_text = prompt_row.iloc[0]['prompt']

            if r['transition'] == 'base→aligned':
                from_id = fam.base
                to_id = fam.superego or fam.ego
            elif r['transition'] == 'sft→dpo':
                from_id = fam.ego
                to_id = fam.superego
            else:
                continue

            # Get argmax token IDs from cached logits
            from_layer = None
            to_layer = None
            if r['transition'] == 'base→aligned':
                from_layer = psyche.primary_process
                to_layer = psyche.superego or psyche.ego
            elif r['transition'] == 'sft→dpo':
                from_layer = psyche.ego
                to_layer = psyche.superego

            try:
                from_logits = from_layer.logits(prompt_text).numpy()
                to_logits = to_layer.logits(prompt_text).numpy()
            except Exception:
                continue

            f_id = int(np.argmax(softmax(from_logits.astype(np.float64))))
            a_id = int(np.argmax(softmax(to_logits.astype(np.float64))))

            from_model, from_tok = loaded[from_id]
            to_model, to_tok = loaded[to_id]

            # Generate greedy continuations
            base_cont = greedy_from_token(from_model, from_tok, prompt_text, f_id)
            aligned_cont = greedy_from_token(to_model, to_tok, prompt_text, a_id)

            f_word = from_tok.decode([f_id]).strip()
            a_word = to_tok.decode([a_id]).strip()

            results[sid] = {
                "base_argmax": f_word,
                "aligned_argmax": a_word,
                "base_greedy": base_cont.strip(),
                "aligned_greedy": aligned_cont.strip(),
            }

            print(f"  {sid}: {f_word} → {a_word}")

        # Free models
        for model_id, (model, _) in loaded.items():
            del model
        loaded.clear()
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Build final pilot
    v3 = pd.read_csv(os.path.join(PATH_DATA, "judge_pilot_v3.csv"))
    rows = []
    for _, r in v3.iterrows():
        sid = r['site_id']
        if sid in results:
            res = results[sid]
            rows.append({
                "site_id": sid,
                "prompt": r['prompt'],
                "base_argmax": res['base_argmax'],
                "aligned_argmax": res['aligned_argmax'],
                "base_continuation": res['base_greedy'],
                "aligned_continuation": res['aligned_greedy'],
                "top10_base": r['top10_base'],
                "top10_aligned": r['top10_aligned'],
            })
        else:
            rows.append({
                "site_id": sid,
                "prompt": r['prompt'],
                "base_argmax": r.get('base_argmax', ''),
                "aligned_argmax": r.get('aligned_argmax', ''),
                "base_continuation": r.get('base_continuation', ''),
                "aligned_continuation": r.get('aligned_continuation', ''),
                "top10_base": r['top10_base'],
                "top10_aligned": r['top10_aligned'],
            })

    out = pd.DataFrame(rows)
    path = os.path.join(PATH_DATA, "judge_pilot_v4.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved {len(out)} sites to {path}")

    # Show examples
    for _, r in out.head(5).iterrows():
        print(f"\n  {r.site_id}")
        print(f"    Prompt: \"{r.prompt}\"")
        print(f"    Base [{r.base_argmax}]: {r.base_continuation[:100]}")
        print(f"    Aligned [{r.aligned_argmax}]: {r.aligned_continuation[:100]}")


if __name__ == "__main__":
    main()
