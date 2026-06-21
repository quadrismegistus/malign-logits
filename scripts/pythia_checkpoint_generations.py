"""Generate text from Pythia 1B at each pretraining checkpoint, score with BLT 1B.

Traces information density development across pretraining:
- Logit entropy (already in pythia1b_battery_emergence.csv)
- Generated text bits/char under BLT 1B (this script)
- Self-surprisal (model scoring its own output)

Usage:
    python scripts/pythia_checkpoint_generations.py
"""
import torch
import numpy as np
import pandas as pd
import gc

CHECKPOINTS = [0, 1, 64, 512, 1000, 5000, 10000, 25000, 50000, 100000, 143000]
MODEL = "EleutherAI/pythia-1b"
BLT_MODEL = "itazap/blt-1b-hf"
N_GENS = 25
MAX_NEW_TOKENS = 128
BOS_PROMPT = "<|endoftext|>"

BATTERY_PROMPTS = [
    "She was so angry she wanted to",
    "The capital of France is",
    "He felt a deep sense of",
    "She loved him deeply and wanted to",
    "She hated him deeply and wanted to",
]


def generate_at_checkpoint(model, tokenizer, prompt, n, max_tokens):
    """Generate n completions from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    gens = []
    for i in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=1.0,
                do_sample=True,
                top_k=0,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if prompt in (BOS_PROMPT, "<|endoftext|>"):
            completion = text
        else:
            completion = text[len(prompt):]
        gens.append(completion.strip())
    return gens


def bits_per_char_blt(text, blt_model, blt_tokenizer):
    """Compute BLT bits/char for a passage."""
    if not text or len(text) < 10:
        return np.nan

    ids = blt_tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
    ids = ids.to(next(blt_model.parameters()).device)

    if ids.shape[1] < 2:
        return np.nan

    with torch.no_grad():
        out = blt_model(ids)
        logits = out.logits[0].float()

    log_probs = torch.log_softmax(logits, dim=-1)
    token_ids = ids[0]

    total_bits = 0.0
    total_chars = 0
    for i in range(len(token_ids) - 1):
        next_id = token_ids[i + 1]
        surprisal_nats = -log_probs[i, next_id].item()
        token_str = blt_tokenizer.decode([next_id])
        total_bits += surprisal_nats / np.log(2)
        total_chars += max(len(token_str), 1)

    return total_bits / total_chars if total_chars > 0 else np.nan


def self_surprisal(text, model, tokenizer):
    """Model scores its own text — self-surprisal in bits/char."""
    if not text or len(text) < 10:
        return np.nan

    ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
    ids = ids.to(next(model.parameters()).device)

    if ids.shape[1] < 2:
        return np.nan

    with torch.no_grad():
        out = model(ids)
        logits = out.logits[0].float()

    log_probs = torch.log_softmax(logits, dim=-1)
    token_ids = ids[0]

    total_bits = 0.0
    total_chars = 0
    for i in range(len(token_ids) - 1):
        next_id = token_ids[i + 1]
        surprisal_nats = -log_probs[i, next_id].item()
        token_str = tokenizer.decode([next_id])
        total_bits += surprisal_nats / np.log(2)
        total_chars += max(len(token_str), 1)

    return total_bits / total_chars if total_chars > 0 else np.nan


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = [BOS_PROMPT] + BATTERY_PROMPTS

    # Load BLT once, keep in memory
    print(f"Loading BLT ({BLT_MODEL})...")
    blt_tok = AutoTokenizer.from_pretrained(BLT_MODEL, trust_remote_code=True)
    blt_model = AutoModelForCausalLM.from_pretrained(
        BLT_MODEL, trust_remote_code=True, torch_dtype=torch.float32
    )
    if torch.backends.mps.is_available():
        blt_model = blt_model.to("mps")
    blt_model.eval()
    print("BLT loaded.")

    pythia_tok = AutoTokenizer.from_pretrained(MODEL)

    all_rows = []

    for step in CHECKPOINTS:
        step_name = f"step{step}"
        print(f"\n{'='*60}")
        print(f"  Checkpoint: {step_name}")
        print(f"{'='*60}")

        try:
            pythia_model = AutoModelForCausalLM.from_pretrained(
                MODEL, revision=step_name, trust_remote_code=True,
                torch_dtype=torch.float16, device_map="mps",
            )
            pythia_model.eval()

            for prompt in prompts:
                prompt_label = "bos" if prompt == BOS_PROMPT else prompt[:40]
                print(f"  Generating {N_GENS} for '{prompt_label}'...")

                gens = generate_at_checkpoint(pythia_model, pythia_tok, prompt, N_GENS, MAX_NEW_TOKENS)

                for idx, text in enumerate(gens):
                    if len(text.strip()) < 10:
                        continue

                    bpc = bits_per_char_blt(text, blt_model, blt_tok)
                    ss = self_surprisal(text, pythia_model, pythia_tok)

                    all_rows.append({
                        "step": step,
                        "prompt": prompt_label,
                        "idx": idx,
                        "blt_bits_per_char": bpc,
                        "self_surprisal_bpc": ss,
                        "text_len": len(text),
                        "text": text[:200],
                    })

                bpcs = [r["blt_bits_per_char"] for r in all_rows if r["step"] == step and r["prompt"] == prompt_label and not np.isnan(r["blt_bits_per_char"])]
                sss = [r["self_surprisal_bpc"] for r in all_rows if r["step"] == step and r["prompt"] == prompt_label and not np.isnan(r["self_surprisal_bpc"])]
                if bpcs:
                    print(f"    BLT bits/char: {np.mean(bpcs):.3f} ± {np.std(bpcs):.3f}")
                if sss:
                    print(f"    Self-surprisal: {np.mean(sss):.3f} ± {np.std(sss):.3f}")

            del pythia_model
            gc.collect()
            torch.mps.empty_cache()

        except Exception as e:
            print(f"  ERROR at {step_name}: {e}")
            import traceback; traceback.print_exc()

    # Save
    df = pd.DataFrame(all_rows)
    df.to_csv("data/pythia1b_checkpoint_generations.csv", index=False)
    print(f"\nSaved data/pythia1b_checkpoint_generations.csv ({len(df)} rows)")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary: BLT bits/char across pretraining")
    print(f"{'='*60}")
    for step in CHECKPOINTS:
        sub = df[df["step"] == step]
        if len(sub):
            bpc = sub["blt_bits_per_char"].dropna()
            ss = sub["self_surprisal_bpc"].dropna()
            print(f"  step {step:>6d}: BLT={bpc.mean():.3f} bpc  self={ss.mean():.3f} bpc  n={len(bpc)}")
