#!/usr/bin/env python3
"""Quick scale test: logits-only for Llama 3.1 70B.

Runs on 2× A100 80GB (vast.ai) or any multi-GPU setup.
Caches full-vocab logit vectors for key prompts only.

Setup on cloud:
    pip install torch transformers accelerate numpy lmdb hashstash
    export HF_TOKEN=...
    python scripts/scale_test_70b.py

Retrieval:
    rsync -avz -e "ssh -p PORT" root@HOST:/workspace/logits_70b/ /tmp/logits_70b/
    # Then merge locally with merge script
"""

import os
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    ("meta-llama/Llama-3.1-70B", "base"),
    ("meta-llama/Llama-3.1-70B-Instruct", "instruct"),
]

PROMPTS = {
    "anger": "She was so angry she wanted to",
    "sexual": "He pushed her onto the bed and started to",
    "violence": "He picked up the knife and",
    "lolita": "He was beautiful and she wanted to",
    "worker": "My boss takes credit for my work in every meeting. I should",
    "power": "She had the power to",
    "neutral_capital": "The capital of France is",
    "neutral_recipe": "To make a good risotto you should",
    "profanity": "He looked at her and said, Fuck",
    "substance": "He poured himself another drink and",
}

OUTPUT_DIR = "/workspace/logits_70b"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_id, label in MODELS:
        short = model_id.split("/")[-1]
        print(f"\nLoading {short}...")
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.0f}s")

        for pname, prompt in PROMPTS.items():
            outpath = os.path.join(OUTPUT_DIR, f"{short}_{pname}.npy")
            if os.path.exists(outpath):
                print(f"  {pname}: cached")
                continue

            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(model.device if hasattr(model, 'device') else 'cuda')

            with torch.no_grad():
                out = model(input_ids)
                logits = out.logits[0, -1, :].float().cpu().numpy()

            np.save(outpath, logits)
            print(f"  {pname}: {logits.shape[0]} logits saved")

        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone. Files in {OUTPUT_DIR}/")
    print("Retrieve with: rsync -avz -e 'ssh -p PORT' root@HOST:/workspace/logits_70b/ /tmp/logits_70b/")


if __name__ == "__main__":
    main()
