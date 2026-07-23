"""Generate continue-mode (template-mediated) text for disposition tagging.

Produces the same-weights-two-modes comparison:
  raw mode:      drive survives, no moralizing (already cached)
  continue mode: template-mediated, expected moralizing/refusal

Usage:
    uv run python scripts/gen_continue_mode.py
"""

import gc
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from malign_logits import MODEL_FAMILIES, PATH_DATA
from malign_logits.cache import get_cache
from malign_logits.models import load_model
from malign_logits.core import _apply_mode
from malign_logits.experiments import DEFAULT_PROMPTS

TRANS_CATS = {
    "sexual_liminal", "sexual_explicit",
    "violence_liminal", "violence_explicit",
    "death", "profanity", "substance",
}

N_GENS = 3
MAX_TOKENS = 100

# Only families whose aligned models have chat templates
FAMILIES = ['olmo', 'llama', 'qwen', 'tulu', 'zephyr', 'deepseek-7b']


def main():
    cm = get_cache()
    trans_prompts = {k: v for k, v in DEFAULT_PROMPTS.items()
                     if k.rsplit("_", 1)[0] in TRANS_CATS}

    for fkey in FAMILIES:
        fam = MODEL_FAMILIES[fkey]
        aligned_id = fam.superego or fam.ego
        if aligned_id is None:
            continue

        print(f"\n{fkey}: {aligned_id}")
        model, tokenizer = load_model(aligned_id)
        device = next(model.parameters()).device

        if tokenizer.chat_template is None:
            print(f"  No chat template, skipping")
            del model; gc.collect()
            continue

        n_gen = 0
        for pkey, prompt in trans_prompts.items():
            # Format as continue-mode: chat template + "Continue this text: ..."
            try:
                cont_prompt = _apply_mode(prompt, tokenizer, mode="continue")
            except Exception:
                continue

            # Check if already cached (using the formatted prompt as key)
            existing = cm.count_generations(aligned_id + ":continue", prompt)
            if existing >= N_GENS:
                continue

            input_ids = tokenizer.encode(cont_prompt, return_tensors="pt").to(device)

            for idx in range(existing, N_GENS):
                with torch.no_grad():
                    out = model.generate(
                        input_ids,
                        max_new_tokens=MAX_TOKENS,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.95,
                    )
                gen_text = tokenizer.decode(
                    out[0][input_ids.shape[1]:], skip_special_tokens=True)
                # Store with ":continue" suffix on model_id to distinguish
                cm.set_generation(
                    aligned_id + ":continue", prompt, gen_text,
                    temp=1.0, idx=idx)
                n_gen += 1

        print(f"  Generated {n_gen} continue-mode passages")
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Summary
    print(f"\nDone. Continue-mode generations stored with ':continue' suffix.")
    for fkey in FAMILIES:
        fam = MODEL_FAMILIES[fkey]
        aligned_id = fam.superego or fam.ego
        if aligned_id is None:
            continue
        sample = list(trans_prompts.values())[0]
        n = cm.count_generations(aligned_id + ":continue", sample)
        print(f"  {fkey}: {n} gens/prompt")


if __name__ == "__main__":
    main()
