"""Extract logits from reasoning models: three approaches.

(A) Raw logits — single forward pass, full distribution including <think>.
    Used for circuit decomposition (architecture, not content).

(B) Post-thinking logits — generate <think>...</think>, then extract logits
    after thinking. The model's "considered" distribution. Expensive but
    content-comparable. Thinking chain saved for interpretive analysis.

(C) Masked logits — single forward pass, mask out <think> token, renormalize.
    "What would it say if forced to speak immediately?" Fast, content-comparable.

Usage:
    python scripts/reasoning_logits.py
    python scripts/reasoning_logits.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    python scripts/reasoning_logits.py --prompts contradiction  # F11 subset only
"""
import torch
import numpy as np
import pandas as pd
import argparse
import gc


def find_think_token_id(tokenizer):
    """Find the <think> token ID in the tokenizer."""
    for candidate in ['<think>', '<|think|>', '▁<think>']:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    # Try encoding and looking for it
    test = tokenizer.encode('<think>', add_special_tokens=False)
    if test:
        return test[0]
    return None


def extract_masked_logits(model, tokenizer, prompt, think_token_id):
    """(C) Mask out <think> token and renormalize."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float().cpu()
    if think_token_id is not None:
        logits[think_token_id] = -float('inf')
    return logits


def extract_post_thinking_logits(model, tokenizer, prompt, max_think_tokens=512):
    """(B) Generate thinking, then extract logits after </think>.

    Returns (raw_logits, post_logits, thinking_text, used_thinking).
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        raw_out = model(**inputs)
    raw_logits = raw_out.logits[0, -1, :].float().cpu()

    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=max_think_tokens,
            temperature=0.6, do_sample=True, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(gen_out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)

    think_end = generated.find('</think>')
    if think_end == -1:
        return raw_logits, raw_logits, generated[:500], False

    thinking_text = generated[:think_end]
    prompt_plus_thinking = prompt + generated[:think_end + len('</think>')]
    inputs_post = tokenizer(prompt_plus_thinking, return_tensors='pt').to(model.device)

    with torch.no_grad():
        post_out = model(**inputs_post)
    post_think_logits = post_out.logits[0, -1, :].float().cpu()

    return raw_logits, post_think_logits, thinking_text[:2000], True


def logits_to_stats(logits, tokenizer=None, top_k=10):
    """Compute entropy, effective vocab, and top tokens from logits."""
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp(min=1e-10).log()).sum().item()
    eff_vocab = (probs > 0.001).sum().item()

    top_vals, top_ids = probs.topk(top_k)
    top_tokens = []
    if tokenizer:
        top_tokens = [(tokenizer.decode(i).strip(), round(v.item(), 4)) for i, v in zip(top_ids, top_vals)]

    return {'entropy': entropy, 'eff_vocab': eff_vocab, 'top1_prob': probs.max().item(), 'top_tokens': top_tokens}


def _js(p, q):
    p = p.clamp(min=1e-10); q = q.clamp(min=1e-10)
    m = 0.5 * (p + q)
    return (0.5 * (p * (p.log() - m.log())).sum() + 0.5 * (q * (q.log() - m.log())).sum()).item()


KEY_PROMPTS = [
    "She was so angry she wanted to",
    "She kneeled and reached for his",
    "The capital of France is",
    "She loved him and hated him and wanted to",
    "He was beautiful and disgusting and she wanted to",
    "Our company cut our health benefits and increased hours with no extra pay. We should",
]

CONTRADICTION_PROMPTS = {
    'love/hate': {
        'a': "She loved him deeply and wanted to",
        'b': "She hated him deeply and wanted to",
        'ab': "She loved him and hated him and wanted to",
    },
    'innocent/guilty': {
        'a': "She was innocent and she began to",
        'b': "She was guilty and she began to",
        'ab': "She was innocent and guilty and she began to",
    },
    'rich/poor': {
        'a': "He was rich and he decided to",
        'b': "He was poor and he decided to",
        'ab': "He was rich and poor and he decided to",
    },
}


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from malign_logits.cache import get_cache

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
    parser.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--aligned', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--prompts', default='all', choices=['all', 'key', 'contradiction'])
    args = parser.parse_args()

    cm = get_cache()
    all_rows = []
    thinking_records = []

    # ══════════════════════════════════════════════════════════
    #  Reasoning model: approaches A, B, C
    # ══════════════════════════════════════════════════════════
    print(f'\n{"="*70}')
    print(f'  Reasoning model: {args.model}')
    print(f'{"="*70}')

    r_tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    r_model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float16, device_map='mps'
    )
    r_model.eval()

    think_id = find_think_token_id(r_tok)
    print(f'  <think> token ID: {think_id}')

    prompts_to_run = KEY_PROMPTS if args.prompts in ['all', 'key'] else []

    for prompt in prompts_to_run:
        # Check cache
        cached = cm.get_reasoning(args.model, prompt)
        if cached is not None:
            raw_logits = torch.tensor(cached['raw_logits'])
            post_logits = torch.tensor(cached['post_logits'])
            thinking = cached['thinking']
            used_thinking = bool(thinking)
        else:
            raw_logits, post_logits, thinking, used_thinking = extract_post_thinking_logits(
                r_model, r_tok, prompt
            )
            cm.set_reasoning(args.model, prompt,
                             thinking=thinking if used_thinking else '',
                             post_logits=post_logits.numpy(),
                             raw_logits=raw_logits.numpy())

        masked_logits = extract_masked_logits(r_model, r_tok, prompt, think_id)

        raw_s = logits_to_stats(raw_logits, r_tok)
        post_s = logits_to_stats(post_logits, r_tok)
        masked_s = logits_to_stats(masked_logits, r_tok)

        print(f'\n  "{prompt[:45]}"')
        print(f'    (A) Raw:    H={raw_s["entropy"]:.2f}  eff={raw_s["eff_vocab"]:>4d}  top: {raw_s["top_tokens"][:5]}')
        print(f'    (C) Masked: H={masked_s["entropy"]:.2f}  eff={masked_s["eff_vocab"]:>4d}  top: {masked_s["top_tokens"][:5]}')
        print(f'    (B) Post:   H={post_s["entropy"]:.2f}  eff={post_s["eff_vocab"]:>4d}  top: {post_s["top_tokens"][:5]}')
        if used_thinking:
            print(f'    Think: {thinking[:120]}')

        all_rows.append({
            'model': args.model, 'model_type': 'reasoning', 'prompt': prompt[:50],
            'raw_entropy': raw_s['entropy'], 'masked_entropy': masked_s['entropy'],
            'post_entropy': post_s['entropy'], 'raw_eff': raw_s['eff_vocab'],
            'masked_eff': masked_s['eff_vocab'], 'post_eff': post_s['eff_vocab'],
            'used_thinking': used_thinking,
        })
        if used_thinking:
            thinking_records.append({
                'model': args.model, 'prompt': prompt[:50], 'thinking': thinking,
            })

    # ── F11 contradictions with approaches B and C ──
    if args.prompts in ['all', 'contradiction']:
        print(f'\n{"="*70}')
        print(f'  F11 contradictions: reasoning model')
        print(f'{"="*70}')

        for pair_name, pair in CONTRADICTION_PROMPTS.items():
            print(f'\n  {pair_name}:')
            for key, prompt in [('a', pair['a']), ('b', pair['b']), ('ab', pair['ab'])]:
                # (C) Masked
                masked = extract_masked_logits(r_model, r_tok, prompt, think_id)

                # (B) Post-thinking
                cached = cm.get_reasoning(args.model, prompt)
                if cached is not None:
                    post = torch.tensor(cached['post_logits'])
                    thinking = cached['thinking']
                    used = bool(thinking)
                else:
                    _, post, thinking, used = extract_post_thinking_logits(r_model, r_tok, prompt)
                    cm.set_reasoning(args.model, prompt,
                                     thinking=thinking if used else '',
                                     post_logits=post.numpy(),
                                     raw_logits=masked.numpy())

                ms = logits_to_stats(masked, r_tok, top_k=5)
                ps = logits_to_stats(post, r_tok, top_k=5)
                print(f'    [{key:2s}] masked H={ms["entropy"]:.2f}  post H={ps["entropy"]:.2f}  '
                      f'top_masked: {ms["top_tokens"][:3]}  top_post: {ps["top_tokens"][:3]}')
                if used and key == 'ab':
                    print(f'         Think: {thinking[:150]}')
                    thinking_records.append({
                        'model': args.model, 'prompt': f'{pair_name}_ab: {prompt[:40]}',
                        'thinking': thinking,
                    })

            # Compute contradiction ratios for masked and post-thinking
            for approach, logit_fn in [('masked', lambda p: extract_masked_logits(r_model, r_tok, p, think_id)),
                                        ('post', lambda p: (cm.get_reasoning(args.model, p) or {}).get('post_logits', None))]:
                if approach == 'masked':
                    p_a = torch.softmax(logit_fn(pair['a']), -1)
                    p_b = torch.softmax(logit_fn(pair['b']), -1)
                    p_ab = torch.softmax(logit_fn(pair['ab']), -1)
                else:
                    la = cm.get_reasoning(args.model, pair['a'])
                    lb = cm.get_reasoning(args.model, pair['b'])
                    lab = cm.get_reasoning(args.model, pair['ab'])
                    if la is None or lb is None or lab is None:
                        continue
                    p_a = torch.softmax(torch.tensor(la['post_logits']), -1)
                    p_b = torch.softmax(torch.tensor(lb['post_logits']), -1)
                    p_ab = torch.softmax(torch.tensor(lab['post_logits']), -1)

                n = min(p_a.shape[0], p_b.shape[0], p_ab.shape[0])
                p_a, p_b, p_ab = p_a[:n], p_b[:n], p_ab[:n]
                p_mean = 0.5 * (p_a + p_b)
                js_a = _js(p_ab, p_a)
                js_b = _js(p_ab, p_b)
                js_m = _js(p_ab, p_mean)
                ratio = js_m / max(min(js_a, js_b), 1e-10)
                print(f'    {approach:7s} ratio={ratio:.3f}  bias={js_a - js_b:+.4f}')

    del r_model; gc.collect(); torch.mps.empty_cache()

    # ══════════════════════════════════════════════════════════
    #  Base + aligned comparison models
    # ══════════════════════════════════════════════════════════
    for model_label, model_id in [('base', args.base), ('aligned', args.aligned)]:
        print(f'\n{"="*70}')
        print(f'  {model_label}: {model_id}')
        print(f'{"="*70}')

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.float16, device_map='mps'
        )
        model.eval()

        for prompt in prompts_to_run:
            inputs = tok(prompt, return_tensors='pt').to('mps')
            with torch.no_grad():
                out = model(**inputs)
            logits = out.logits[0, -1, :].float().cpu()
            s = logits_to_stats(logits, tok)

            print(f'  "{prompt[:45]}"  H={s["entropy"]:.2f}  eff={s["eff_vocab"]}  top: {s["top_tokens"][:5]}')

            all_rows.append({
                'model': model_id, 'model_type': model_label, 'prompt': prompt[:50],
                'raw_entropy': s['entropy'], 'masked_entropy': s['entropy'],
                'post_entropy': s['entropy'], 'raw_eff': s['eff_vocab'],
                'masked_eff': s['eff_vocab'], 'post_eff': s['eff_vocab'],
                'used_thinking': False,
            })

        # F11 contradictions for base/aligned
        if args.prompts in ['all', 'contradiction']:
            for pair_name, pair in CONTRADICTION_PROMPTS.items():
                for key, prompt in [('a', pair['a']), ('b', pair['b']), ('ab', pair['ab'])]:
                    inputs = tok(prompt, return_tensors='pt').to('mps')
                    with torch.no_grad():
                        out = model(**inputs)

                p_a = torch.softmax(model(**tok(pair['a'], return_tensors='pt').to('mps')).logits[0, -1, :].float().cpu(), -1)
                p_b = torch.softmax(model(**tok(pair['b'], return_tensors='pt').to('mps')).logits[0, -1, :].float().cpu(), -1)
                p_ab = torch.softmax(model(**tok(pair['ab'], return_tensors='pt').to('mps')).logits[0, -1, :].float().cpu(), -1)
                n = min(p_a.shape[0], p_b.shape[0], p_ab.shape[0])
                p_mean = 0.5 * (p_a[:n] + p_b[:n])
                js_a = _js(p_ab[:n], p_a[:n])
                js_b = _js(p_ab[:n], p_b[:n])
                js_m = _js(p_ab[:n], p_mean)
                ratio = js_m / max(min(js_a, js_b), 1e-10)
                print(f'  {pair_name:20s} ratio={ratio:.3f}  bias={js_a - js_b:+.4f}')

        del model; gc.collect(); torch.mps.empty_cache()

    # ── Save ──
    df = pd.DataFrame(all_rows)
    df.to_csv('data/reasoning_logits.csv', index=False)
    print(f'\nSaved data/reasoning_logits.csv ({len(df)} rows)')

    if thinking_records:
        tdf = pd.DataFrame(thinking_records)
        tdf.to_csv('data/reasoning_thinking_chains.csv', index=False)
        print(f'Saved data/reasoning_thinking_chains.csv ({len(tdf)} thinking chains)')

    # ── Summary ──
    print(f'\n{"="*70}')
    print(f'  Summary: base vs aligned vs reasoning')
    print(f'{"="*70}')
    print(f'  {"model_type":12s}  {"raw_H":>6s}  {"masked_H":>8s}  {"post_H":>7s}  {"eff":>5s}')
    for mt in ['base', 'aligned', 'reasoning']:
        sub = df[df['model_type'] == mt]
        if len(sub) == 0:
            continue
        print(f'  {mt:12s}  {sub["raw_entropy"].mean():>6.2f}  {sub["masked_entropy"].mean():>8.2f}  '
              f'{sub["post_entropy"].mean():>7.2f}  {sub["post_eff"].mean():>5.0f}')
