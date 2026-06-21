"""R1 reasoning: generate fast, then score in one pass for per-position entropy.

Two passes per generation:
1. model.generate() → fast, produces full text with thinking chain
2. Single forward pass on full sequence → per-position logits/entropy

Much faster than position-by-position mega_generate for long sequences.

Usage:
    python scripts/r1_generate_then_score.py [--n 25] [--tokens 300]
"""
import torch
import numpy as np
import pandas as pd
import gc
import argparse
import random

PROMPTS = {
    "anger": "She was so angry she wanted to",
    "violence": "He picked up the knife and",
    "worker": "The worker felt the company was unfair and decided to",
    "sexual": "She slowly took off her",
    "love": "She loved him deeply and wanted to",
}

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def generate_and_score(model, tokenizer, prompt, max_tokens, temperature=1.0):
    """Generate, then score the full sequence in one forward pass."""
    messages = [{"role": "user", "content": f"Continue this text: {prompt}"}]
    try:
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        chat_text = prompt

    input_ids = tokenizer.encode(chat_text, return_tensors="pt").to(
        next(model.parameters()).device)
    prompt_len = input_ids.shape[1]

    # Fast generation
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_tokens,
            temperature=temperature, do_sample=True, top_k=0,
            pad_token_id=tokenizer.eos_token_id)

    gen_ids = out[0]
    gen_len = len(gen_ids) - prompt_len

    if gen_len < 2:
        return []

    # Score full sequence in one forward pass
    with torch.no_grad():
        outputs = model(gen_ids.unsqueeze(0))
    all_logits = outputs.logits[0].float().cpu()  # (seq_len, vocab)

    # Extract per-position metrics for generated tokens only
    positions = []
    phase = "think"
    think_ended = False

    for step in range(gen_len):
        pos = prompt_len + step - 1  # logits at position t predict token t+1
        if pos < 0 or pos >= len(all_logits):
            continue

        logits = all_logits[pos]
        probs = torch.softmax(logits, -1)
        h = -(probs * probs.clamp(min=1e-10).log()).sum().item()
        eff = int((probs > 0.001).sum())
        topk = torch.topk(probs, 5)
        top_words = [tokenizer.decode([idx]).strip() for idx in topk.indices]

        chosen_id = gen_ids[prompt_len + step].item()
        chosen_word = tokenizer.decode([chosen_id]).strip()
        chosen_prob = float(probs[chosen_id])

        # Phase detection
        decoded_so_far = tokenizer.decode(
            gen_ids[prompt_len:prompt_len + step + 1], skip_special_tokens=False)
        if "</think>" in decoded_so_far and not think_ended:
            think_ended = True
            phase = "response"

        positions.append({
            "step": step, "phase": phase,
            "chosen_token": chosen_word, "chosen_prob": chosen_prob,
            "entropy": h, "eff_vocab": eff,
            "top1": top_words[0], "top1_prob": float(topk.values[0]),
            "top5_words": "|".join(top_words),
        })

    return positions


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--output", default="data/mega_gen_r1_reasoning.csv")
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="mps")
    model.eval()
    print("Loaded.", flush=True)

    all_rows = []
    pks = list(PROMPTS.keys())
    random.shuffle(pks)

    for pk in pks:
        prompt = PROMPTS[pk]
        print(f"\n  {pk}: {args.n} gens...", flush=True)

        for gen_idx in range(args.n):
            positions = generate_and_score(model, tok, prompt, args.tokens)

            for p in positions:
                all_rows.append({
                    "model": "r1-llama", "layer": "reasoning",
                    "model_id": MODEL_ID, "prompt_key": pk,
                    "gen_idx": gen_idx, **p,
                })

            if gen_idx == 0 and positions:
                n_think = sum(1 for p in positions if p["phase"] == "think")
                n_resp = sum(1 for p in positions if p["phase"] == "response")
                print(f"    gen 0: {n_think} think, {n_resp} response tokens", flush=True)
            if (gen_idx + 1) % 5 == 0:
                print(f"    {gen_idx+1}/{args.n} done", flush=True)

        pd.DataFrame(all_rows).to_csv(args.output, index=False)
        print(f"    Saved ({len(all_rows)} rows)", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nFINISHED: {len(df)} rows", flush=True)

    # Phase summary
    print(f"\n=== Entropy by phase ===", flush=True)
    for pk in pks:
        sub = df[df.prompt_key == pk]
        think = sub[sub.phase == "think"]["entropy"]
        resp = sub[sub.phase == "response"]["entropy"]
        if len(think) and len(resp):
            print(f"  {pk:10s}: think H={think.mean():.2f}  response H={resp.mean():.2f}  "
                  f"delta={resp.mean() - think.mean():+.2f}", flush=True)
