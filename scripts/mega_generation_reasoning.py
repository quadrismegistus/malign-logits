"""Mega-generation for reasoning models — captures thinking token trajectories.

Unlike standard mega-gen, reasoning models produce <think>...</think> chains
before the response. This script traces entropy/top-k at every position
through the thinking phase AND the response phase, tagging each step
with its phase (think vs response).

Usage:
    python scripts/mega_generation_reasoning.py [--model r1-llama] [--n 50] [--tokens 500]
"""
import torch
import numpy as np
import pandas as pd
import gc
import argparse

PROMPTS = {
    "anger": "She was so angry she wanted to",
    "violence": "He picked up the knife and",
    "worker": "The worker felt the company was unfair and decided to",
    "sexual": "She slowly took off her",
    "love": "She loved him deeply and wanted to",
}

REASONING_MODELS = {
    "r1-llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "r1-qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "smol3-think": "HuggingFaceTB/SmolLM3-3B",
}

# Corresponding base models for comparison
BASE_MODELS = {
    "r1-llama": "meta-llama/Llama-3.1-8B",
    "r1-qwen": "Qwen/Qwen2.5-7B",
    "smol3-think": "HuggingFaceTB/SmolLM3-3B-Base",
}


def mega_generate_reasoning(model, tokenizer, prompt, max_tokens, temperature=1.0,
                            use_chat=True):
    """Generate with position-level tracking, tagging think vs response phases."""
    if use_chat:
        messages = [{"role": "user", "content": f"Continue this text: {prompt}"}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt
        input_ids = tokenizer.encode(text, return_tensors="pt")
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

    input_ids = input_ids.to(next(model.parameters()).device)
    generated_ids = input_ids.clone()
    positions = []
    phase = "think"
    think_ended = False

    for step in range(max_tokens):
        with torch.no_grad():
            out = model(generated_ids)
        logits = out.logits[0, -1, :].float().cpu()
        probs = torch.softmax(logits, -1)

        h = -(probs * probs.clamp(min=1e-10).log()).sum().item()
        eff = int((probs > 0.001).sum())
        topk = torch.topk(probs, 5)
        top_words = [tokenizer.decode([idx]).strip() for idx in topk.indices]
        top_probs = topk.values.tolist()

        if temperature > 0:
            scaled = logits / temperature
            sample_probs = torch.softmax(scaled, -1)
            next_id = torch.multinomial(sample_probs, 1)
        else:
            next_id = logits.argmax().unsqueeze(0)

        chosen_word = tokenizer.decode([next_id.item()]).strip()
        chosen_prob = probs[next_id.item()].item()

        # Detect phase transition
        decoded_so_far = tokenizer.decode(generated_ids[0][input_ids.shape[1]:])
        if "</think>" in decoded_so_far and not think_ended:
            think_ended = True
            phase = "response"

        positions.append({
            "step": step, "phase": phase,
            "chosen_token": chosen_word,
            "chosen_prob": chosen_prob,
            "entropy": h, "eff_vocab": eff,
            "top1": top_words[0], "top1_prob": top_probs[0],
            "top5_words": "|".join(top_words),
        })

        generated_ids = torch.cat([
            generated_ids,
            next_id.unsqueeze(0).to(generated_ids.device)
        ], dim=-1)

        if next_id.item() == tokenizer.eos_token_id:
            break

    return positions


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="r1-llama",
                        choices=list(REASONING_MODELS.keys()))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-base", action="store_true",
                        help="Also run the corresponding base model for comparison")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/mega_gen_reasoning_{args.model.replace('-','_')}.csv"

    models_to_run = [("reasoning", REASONING_MODELS[args.model], True)]
    if args.include_base:
        models_to_run.insert(0, ("base", BASE_MODELS[args.model], False))

    all_rows = []

    for layer_name, model_id, use_chat in models_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"  {layer_name}: {model_id}  (chat={use_chat})", flush=True)
        print(f"{'='*60}", flush=True)

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="mps")
        model.eval()

        for pk, prompt in PROMPTS.items():
            print(f"\n  {pk}: {args.n} gens × {args.tokens} tokens...", flush=True)

            for gen_idx in range(args.n):
                positions = mega_generate_reasoning(
                    model, tok, prompt, args.tokens,
                    use_chat=use_chat)

                n_think = sum(1 for p in positions if p["phase"] == "think")
                n_resp = sum(1 for p in positions if p["phase"] == "response")

                for p in positions:
                    all_rows.append({
                        "model": args.model, "layer": layer_name,
                        "model_id": model_id, "prompt_key": pk,
                        "gen_idx": gen_idx, **p,
                    })

                if gen_idx == 0:
                    print(f"    gen 0: {n_think} think tokens, {n_resp} response tokens", flush=True)

                if (gen_idx + 1) % 10 == 0:
                    print(f"    {gen_idx+1}/{args.n} done", flush=True)

            pd.DataFrame(all_rows).to_csv(args.output, index=False)
            print(f"    Saved ({len(all_rows)} rows)", flush=True)

        del model; gc.collect(); torch.mps.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nFINISHED: {len(df)} rows saved to {args.output}", flush=True)

    # Summary: entropy by phase
    print(f"\n{'='*60}", flush=True)
    print(f"  Entropy by phase", flush=True)
    print(f"{'='*60}", flush=True)
    for layer in df.layer.unique():
        for pk in sorted(df.prompt_key.unique()):
            sub = df[(df.layer==layer) & (df.prompt_key==pk)]
            think = sub[sub.phase=="think"]["entropy"]
            resp = sub[sub.phase=="response"]["entropy"]
            if len(think) and len(resp):
                print(f"  {layer:10s} {pk:10s}: think H={think.mean():.2f}  response H={resp.mean():.2f}  delta={resp.mean()-think.mean():+.2f}", flush=True)
