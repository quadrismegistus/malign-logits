"""Pythia 6.9B checkpoint generations — local MPS version.
Generates text at each checkpoint, saves CSV for BLT scoring.
Downloads each checkpoint, generates, deletes, next.
"""
import torch
import numpy as np
import pandas as pd
import gc
import shutil
import os

CHECKPOINTS = [0, 1, 64, 512, 1000, 5000, 10000, 25000, 50000, 100000, 143000]
MODEL = "EleutherAI/pythia-6.9b"
N_GENS = 25
MAX_NEW_TOKENS = 128
BOS_PROMPT = "<|endoftext|>"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--EleutherAI--pythia-6.9b")

PROMPTS = [
    BOS_PROMPT,
    "She was so angry she wanted to",
    "The capital of France is",
    "He felt a deep sense of",
    "She loved him deeply and wanted to",
    "She hated him deeply and wanted to",
]

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    all_rows = []

    for step in CHECKPOINTS:
        step_name = f"step{step}"
        print(f"\n=== {step_name} ===", flush=True)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, revision=step_name, trust_remote_code=True,
                torch_dtype=torch.float16, device_map="mps",
            )
            model.eval()

            for prompt in PROMPTS:
                prompt_label = "bos" if prompt == BOS_PROMPT else prompt[:40]
                print(f"  Generating {N_GENS} for '{prompt_label}'...", flush=True)

                for idx in range(N_GENS):
                    inputs = tok(prompt, return_tensors="pt").to("mps")
                    with torch.no_grad():
                        out = model.generate(
                            **inputs, max_new_tokens=MAX_NEW_TOKENS,
                            temperature=1.0, do_sample=True, top_k=0,
                            pad_token_id=tok.eos_token_id,
                        )
                    text = tok.decode(out[0], skip_special_tokens=True)
                    completion = text if prompt == BOS_PROMPT else text[len(prompt):]

                    all_rows.append({
                        "model": f"{MODEL}/step{step}",
                        "prompt": prompt,
                        "text": completion.strip(),
                        "temp": 1.0,
                        "idx": idx,
                        "step": step,
                    })

                print(f"    Done ({len(all_rows)} total rows)", flush=True)

            # Save incrementally
            pd.DataFrame(all_rows).to_csv("data/pythia6.9b_checkpoint_generations.csv", index=False)

            del model; gc.collect(); torch.mps.empty_cache()

            # Clear HF cache to save disk between checkpoints
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
                print(f"  Cleared HF cache", flush=True)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()

    df = pd.DataFrame(all_rows)
    df.to_csv("data/pythia6.9b_checkpoint_generations.csv", index=False)
    print(f"\nFINISHED: {len(df)} rows", flush=True)
