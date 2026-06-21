"""R1-Distill: institutional prompts (F21) + contradiction pairs (F11).

(1) F21: Raw logits on 24 institutional prompts — compare deference/proceduralisation
    between Llama base, Llama Instruct, and R1-Distill.
(2) F11: Contradiction ratios on 3 key pairs — base vs instruct vs R1-Distill.
    Both (C) plain/masked logits and (B) post-thinking logits for R1.
"""
import torch
import numpy as np
import pandas as pd
import gc

def _js(p, q):
    p = p.clamp(min=1e-10); q = q.clamp(min=1e-10)
    m = 0.5 * (p + q)
    return (0.5 * (p * (p.log() - m.log())).sum() + 0.5 * (q * (q.log() - m.log())).sum()).item()

def top_tokens(logits, tok, k=8):
    probs = torch.softmax(logits.float(), -1)
    vals, ids = probs.topk(k)
    return [(tok.decode(int(i)).strip(), round(v.item(), 4)) for i, v in zip(ids, vals)]

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from malign_logits.cache import get_cache
    from malign_logits import MODEL_FAMILIES
    from malign_logits.experiments import DEFAULT_PROMPTS
    from malign_logits.psyche import Psyche

    cm = get_cache()

    R1_ID = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    fam = MODEL_FAMILIES['llama']

    MODELS = [
        ('base', fam.base),
        ('instruct', fam.superego),
        ('r1_distill', R1_ID),
    ]

    inst_prompts = {k: v for k, v in DEFAULT_PROMPTS.items() if 'institutional' in k}

    CONTRADICTION_PAIRS = [
        {
            'name': 'love/hate',
            'prompt_a': 'She loved him deeply and wanted to',
            'prompt_b': 'She hated him deeply and wanted to',
            'prompt_ab': 'She loved him and hated him and wanted to',
        },
        {
            'name': 'innocent/guilty',
            'prompt_a': 'She was innocent and she began to',
            'prompt_b': 'She was guilty and she began to',
            'prompt_ab': 'She was innocent and guilty and she began to',
        },
        {
            'name': 'rich/poor',
            'prompt_a': 'He was rich and he decided to',
            'prompt_b': 'He was poor and he decided to',
            'prompt_ab': 'He was rich and poor and he decided to',
        },
    ]

    def get_side(pk):
        for tag in ['worker', 'tenant', 'patient', 'citizen', 'political_citizen']:
            if tag in pk:
                return 'individual'
        return 'institution'

    # ══════════════════════════════════════════════════════════
    #  F21: Institutional prompts — raw logits comparison
    # ══════════════════════════════════════════════════════════
    print(f'\n{"="*70}')
    print(f'  F21: Institutional prompt logits')
    print(f'{"="*70}')

    f21_rows = []

    for label, model_id in MODELS:
        print(f'\n  === {label} ===')
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.float16, device_map='mps'
        )
        model.eval()

        for pk, prompt in inst_prompts.items():
            side = get_side(pk)

            # Get or compute logits
            cached = cm.get_logits(model_id, prompt)
            if cached is not None:
                logits = torch.tensor(cached).float()
            else:
                inputs = tok(prompt, return_tensors='pt').to('mps')
                with torch.no_grad():
                    out = model(**inputs)
                logits = out.logits[0, -1, :].float().cpu()
                cm.set_logits(model_id, prompt, logits.numpy())

            probs = torch.softmax(logits, -1)
            h = -(probs * probs.clamp(min=1e-10).log()).sum().item()
            eff = (probs > 0.001).sum().item()

            f21_rows.append({
                'model': label, 'prompt_key': pk, 'side': side,
                'entropy': h, 'eff_vocab': eff,
            })

        del model; gc.collect(); torch.mps.empty_cache()

    f21_df = pd.DataFrame(f21_rows)

    print(f'\n  Entropy by model and side:')
    for label in ['base', 'instruct', 'r1_distill']:
        for side in ['individual', 'institution']:
            sub = f21_df[(f21_df['model'] == label) & (f21_df['side'] == side)]
            print(f'    {label:12s} {side:12s}: H={sub["entropy"].mean():.2f}  eff={sub["eff_vocab"].mean():.0f}')
        ind = f21_df[(f21_df['model'] == label) & (f21_df['side'] == 'individual')]['entropy'].mean()
        ins = f21_df[(f21_df['model'] == label) & (f21_df['side'] == 'institution')]['entropy'].mean()
        print(f'    {label:12s} gap: {ins - ind:+.2f}')

    # ══════════════════════════════════════════════════════════
    #  F11: Contradiction ratios
    # ══════════════════════════════════════════════════════════
    print(f'\n{"="*70}')
    print(f'  F11: Contradiction ratios')
    print(f'{"="*70}')

    f11_rows = []

    for label, model_id in MODELS:
        print(f'\n  === {label} ===')
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.float16, device_map='mps'
        )
        model.eval()

        for pair in CONTRADICTION_PAIRS:
            logits_dict = {}
            for key in ['prompt_a', 'prompt_b', 'prompt_ab']:
                prompt = pair[key]
                cached = cm.get_logits(model_id, prompt)
                if cached is not None:
                    logits_dict[key] = torch.tensor(cached).float()
                else:
                    inputs = tok(prompt, return_tensors='pt').to('mps')
                    with torch.no_grad():
                        out = model(**inputs)
                    logits_dict[key] = out.logits[0, -1, :].float().cpu()
                    cm.set_logits(model_id, prompt, logits_dict[key].numpy())

            n = min(l.shape[0] for l in logits_dict.values())
            p_a = torch.softmax(logits_dict['prompt_a'][:n], -1)
            p_b = torch.softmax(logits_dict['prompt_b'][:n], -1)
            p_ab = torch.softmax(logits_dict['prompt_ab'][:n], -1)
            p_mean = 0.5 * (p_a + p_b)

            js_a = _js(p_ab, p_a)
            js_b = _js(p_ab, p_b)
            js_m = _js(p_ab, p_mean)
            ratio = js_m / max(min(js_a, js_b), 1e-10)
            bias = js_a - js_b

            # Top tokens for AB prompt
            ab_top = top_tokens(logits_dict['prompt_ab'], tok)

            print(f'    {pair["name"]:20s}  ratio={ratio:.3f}  bias={bias:+.4f}  '
                  f'top: {ab_top[:5]}')

            f11_rows.append({
                'model': label, 'pair': pair['name'],
                'ratio': ratio, 'bias': bias,
                'js_to_A': js_a, 'js_to_B': js_b, 'js_to_mean': js_m,
            })

        del model; gc.collect(); torch.mps.empty_cache()

    f11_df = pd.DataFrame(f11_rows)

    # R1 post-thinking contradictions (approach B)
    print(f'\n  === R1-Distill post-thinking (approach B) ===')
    tok = AutoTokenizer.from_pretrained(R1_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        R1_ID, trust_remote_code=True, dtype=torch.float16, device_map='mps'
    )
    model.eval()

    for pair in CONTRADICTION_PAIRS:
        logits_dict = {}
        thinking_dict = {}
        for key in ['prompt_a', 'prompt_b', 'prompt_ab']:
            prompt = pair[key]
            formatted = tok.apply_chat_template(
                [{"role": "user", "content": f"Complete this sentence: {prompt}"}],
                tokenize=False, add_generation_prompt=True
            )
            cached = cm.get_reasoning(R1_ID, prompt)
            if cached is not None:
                logits_dict[key] = torch.tensor(cached['post_logits']).float()
                thinking_dict[key] = cached['thinking']
            else:
                inputs = tok(formatted, return_tensors='pt').to('mps')
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=1024, temperature=0.6,
                                        do_sample=True, top_k=50, pad_token_id=tok.eos_token_id)
                text = tok.decode(gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
                think_end = text.find('</think>')
                if think_end > 0:
                    thinking = text[:think_end].strip()
                    prompt_plus = formatted + text[:think_end + len('</think>')]
                    post_inputs = tok(prompt_plus, return_tensors='pt').to('mps')
                    with torch.no_grad():
                        post_out = model(**post_inputs)
                    post_logits = post_out.logits[0, -1, :].float().cpu()
                else:
                    thinking = ''
                    inputs_raw = tok(formatted, return_tensors='pt').to('mps')
                    with torch.no_grad():
                        raw_out = model(**inputs_raw)
                    post_logits = raw_out.logits[0, -1, :].float().cpu()

                raw_inputs = tok(prompt, return_tensors='pt').to('mps')
                with torch.no_grad():
                    raw_out2 = model(**raw_inputs)
                raw_logits = raw_out2.logits[0, -1, :].float().cpu()

                cm.set_reasoning(R1_ID, prompt,
                               thinking=thinking, post_logits=post_logits.numpy(),
                               raw_logits=raw_logits.numpy())
                logits_dict[key] = post_logits
                thinking_dict[key] = thinking

        n = min(l.shape[0] for l in logits_dict.values())
        p_a = torch.softmax(logits_dict['prompt_a'][:n], -1)
        p_b = torch.softmax(logits_dict['prompt_b'][:n], -1)
        p_ab = torch.softmax(logits_dict['prompt_ab'][:n], -1)
        p_mean = 0.5 * (p_a + p_b)

        js_a = _js(p_ab, p_a)
        js_b = _js(p_ab, p_b)
        js_m = _js(p_ab, p_mean)
        ratio = js_m / max(min(js_a, js_b), 1e-10)

        ab_top = top_tokens(logits_dict['prompt_ab'], tok)
        print(f'    {pair["name"]:20s}  ratio={ratio:.3f}  top: {ab_top[:5]}')
        if thinking_dict.get('prompt_ab'):
            clean = thinking_dict['prompt_ab'].replace('\xc4\xa0', ' ').replace('\xc4\x8a', '\n')[:200]
            print(f'      Think (AB): {clean}')

        f11_rows.append({
            'model': 'r1_post_thinking', 'pair': pair['name'],
            'ratio': ratio, 'bias': js_a - js_b,
            'js_to_A': js_a, 'js_to_B': js_b, 'js_to_mean': js_m,
        })

    del model; gc.collect(); torch.mps.empty_cache()

    # Save
    f11_df = pd.DataFrame(f11_rows)
    f21_df.to_csv('data/r1_institutional.csv', index=False)
    f11_df.to_csv('data/r1_contradictions.csv', index=False)
    print(f'\nSaved data/r1_institutional.csv ({len(f21_df)} rows)')
    print(f'Saved data/r1_contradictions.csv ({len(f11_df)} rows)')

    # Summary
    print(f'\n{"="*70}')
    print(f'  F11 Summary: contradiction ratios')
    print(f'{"="*70}')
    print(f'  {"model":20s}  {"love/hate":>10s}  {"innocent":>10s}  {"rich/poor":>10s}')
    for label in ['base', 'instruct', 'r1_distill', 'r1_post_thinking']:
        sub = f11_df[f11_df['model'] == label]
        if len(sub) == 0:
            continue
        vals = {}
        for _, r in sub.iterrows():
            vals[r['pair']] = r['ratio']
        lh = vals.get('love/hate', np.nan)
        ig = vals.get('innocent/guilty', np.nan)
        rp = vals.get('rich/poor', np.nan)
        print(f'  {label:20s}  {lh:>10.3f}  {ig:>10.3f}  {rp:>10.3f}')
