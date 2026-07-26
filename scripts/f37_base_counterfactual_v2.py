"""F37 base-model counterfactual v2: decontaminated arms.

Three arms:
  A. zero-shot: instruction only, no exemplars
  B. oasst-exemplar: 2 human-written OASST exemplars
  C. alpaca-exemplar: 2 GPT-3.5 Alpaca exemplars (original, contaminated)

2000 instructions, per-text marker counts stored.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/f37_base_counterfactual_v2.py
"""
import torch
import gc
import time
import random
import os
import sys

import pandas as pd
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from f37_dataset_markers import MARKERS, count_markers, tokenize_count
from malign_logits.models import load_model

ZERO_SHOT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

OASST_EXEMPLAR = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is the difference between a virus and a bacterium?

### Response:
Bacteria are single-celled organisms that can reproduce on their own. Viruses are much smaller and cannot reproduce without a host cell. They hijack host cells to replicate.

### Instruction:
How does photosynthesis work?

### Response:
Plants convert sunlight into energy through chlorophyll in their leaves.

### Instruction:
{instruction}

### Response:
"""

ALPACA_EXEMPLAR = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Give three tips for staying healthy.

### Response:
1. Eat a balanced diet and make sure to include plenty of fruits and vegetables.
2. Exercise regularly to keep your body active and strong.
3. Get enough sleep and maintain a consistent sleep schedule.

### Instruction:
What are the three primary colors?

### Response:
The three primary colors are red, blue, and yellow.

### Instruction:
{instruction}

### Response:
"""

ARMS = {
    'zero_shot': ZERO_SHOT,
    'oasst_exemplar': OASST_EXEMPLAR,
    'alpaca_exemplar': ALPACA_EXEMPLAR,
}

MODELS = [
    ('llama_base', 'meta-llama/Llama-3.1-8B'),
    ('mistral_base', 'mistralai/Mistral-7B-v0.1'),
]

N_INSTRUCTIONS = 500


def main():
    random.seed(42)
    alpaca = load_dataset('tatsu-lab/alpaca', split='train')
    indices = random.sample(range(len(alpaca)), N_INSTRUCTIONS)
    instructions = [alpaca[i] for i in indices]

    out_path = os.path.join(DATA_DIR, 'f37_base_counterfactual_v2.csv')

    # Check existing
    done = set()
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        for _, r in existing.iterrows():
            done.add((r['model'], r['arm'], r.get('inst_idx', -1)))
        print(f"Resuming: {len(done)} rows done", flush=True)

    for model_name, model_id in MODELS:
        for arm_name, template in ARMS.items():
            # Check if this arm is done
            arm_done = sum(1 for k in done if k[0] == model_name and k[1] == arm_name)
            if arm_done >= N_INSTRUCTIONS:
                print(f"{model_name}/{arm_name}: CACHED ({arm_done})", flush=True)
                continue

            print(f"\n{model_name}/{arm_name}: generating...", flush=True)
            model, tok = load_model(model_id)
            t0 = time.time()
            batch_rows = []

            for i, inst in enumerate(instructions):
                if (model_name, arm_name, i) in done:
                    continue

                instruction = inst['instruction']
                if inst['input']:
                    instruction += '\n' + inst['input']

                prompt = template.format(instruction=instruction)
                input_ids = tok.encode(prompt, return_tensors='pt').to(model.device)
                if input_ids.shape[1] > 400:
                    input_ids = input_ids[:, -400:]

                with torch.no_grad():
                    out = model.generate(input_ids, max_new_tokens=150, temperature=0.7,
                                         do_sample=True, top_k=50, pad_token_id=tok.eos_token_id)
                text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                if '###' in text:
                    text = text[:text.index('###')].strip()
                text = text[:500]

                # Per-text marker counts
                marker_vals = {}
                n_tokens = tokenize_count(text)
                for mk, mlist in MARKERS.items():
                    marker_vals[f'mk_{mk}'] = count_markers(text, mlist)

                batch_rows.append({
                    'model': model_name, 'model_id': model_id, 'arm': arm_name,
                    'inst_idx': i, 'instruction': inst['instruction'],
                    'input': inst['input'], 'output': text, 'n_tokens': n_tokens,
                    **marker_vals,
                })

                if (i + 1) % 50 == 0:
                    # Save incrementally
                    new_df = pd.DataFrame(batch_rows)
                    header = not os.path.exists(out_path)
                    new_df.to_csv(out_path, mode='a', header=header, index=False)
                    batch_rows = []
                    print(f"  {model_name}/{arm_name}: {i+1}/{N_INSTRUCTIONS} saved ({time.time()-t0:.0f}s)", flush=True)

            # Save remaining
            if batch_rows:
                new_df = pd.DataFrame(batch_rows)
                header = not os.path.exists(out_path)
                new_df.to_csv(out_path, mode='a', header=header, index=False)

            print(f"  {model_name}/{arm_name} done ({time.time()-t0:.0f}s)", flush=True)
            del model, tok; gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()

    # Final count
    df = pd.read_csv(out_path)
    print(f"\nTotal: {len(df)} rows", flush=True)
    for (m, a), g in df.groupby(['model', 'arm']):
        print(f"  {m}/{a}: {len(g)} rows", flush=True)


if __name__ == '__main__':
    main()
