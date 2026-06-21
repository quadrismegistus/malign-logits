"""Qwen3-8B-Base vs Qwen3-8B (native reasoning): battery + institutional + contradictions."""
import torch
import numpy as np
import pandas as pd
import gc

def _js(p, q):
    p = p.clamp(min=1e-10); q = q.clamp(min=1e-10); m = 0.5*(p+q)
    return (0.5*(p*(p.log()-m.log())).sum() + 0.5*(q*(q.log()-m.log())).sum()).item()

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from malign_logits.cache import get_cache
    from malign_logits.experiments import DEFAULT_PROMPTS

    cm = get_cache()

    MODELS = [
        ('qwen3_base', 'Qwen/Qwen3-8B-Base'),
        ('qwen3_instruct', 'Qwen/Qwen3-8B'),
    ]

    prompts = list(DEFAULT_PROMPTS.values())
    prompt_keys = list(DEFAULT_PROMPTS.keys())
    inst_prompts = {k: v for k, v in DEFAULT_PROMPTS.items() if 'institutional' in k}

    CONTRADICTION_PAIRS = [
        {'name': 'love/hate', 'a': 'She loved him deeply and wanted to',
         'b': 'She hated him deeply and wanted to', 'ab': 'She loved him and hated him and wanted to'},
        {'name': 'innocent/guilty', 'a': 'She was innocent and she began to',
         'b': 'She was guilty and she began to', 'ab': 'She was innocent and guilty and she began to'},
        {'name': 'rich/poor', 'a': 'He was rich and he decided to',
         'b': 'He was poor and he decided to', 'ab': 'He was rich and poor and he decided to'},
    ]

    def get_side(pk):
        for tag in ['worker', 'tenant', 'patient', 'citizen', 'political_citizen']:
            if tag in pk: return 'individual'
        return 'institution'

    for label, model_id in MODELS:
        print(f'\n=== {label}: {model_id} ===')
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.float16, device_map='mps')
        model.eval()

        # Battery logits
        hs, effs = [], []
        for prompt in prompts:
            if not cm.has_logits(model_id, prompt):
                inputs = tok(prompt, return_tensors='pt').to('mps')
                with torch.no_grad():
                    out = model(**inputs)
                cm.set_logits(model_id, prompt, out.logits[0, -1, :].float().cpu().numpy())
            logits = torch.tensor(cm.get_logits(model_id, prompt)).float()
            probs = torch.softmax(logits, -1)
            hs.append(-(probs * probs.clamp(min=1e-10).log()).sum().item())
            effs.append((probs > 0.001).sum().item())

        # Contradiction logits
        for pair in CONTRADICTION_PAIRS:
            for key in ['a', 'b', 'ab']:
                p = pair[key]
                if not cm.has_logits(model_id, p):
                    inputs = tok(p, return_tensors='pt').to('mps')
                    with torch.no_grad():
                        out = model(**inputs)
                    cm.set_logits(model_id, p, out.logits[0, -1, :].float().cpu().numpy())

        print(f'  Output: H={np.mean(hs):.2f}  eff={np.mean(effs):.0f}')

        # Quick gen test
        test_prompt = "She was so angry she wanted to"
        inputs = tok(test_prompt, return_tensors='pt').to('mps')
        out = model.generate(**inputs, max_new_tokens=30, temperature=1.0, do_sample=True,
                            top_k=50, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0], skip_special_tokens=True)[len(test_prompt):]
        print(f'  Gen: "{gen.strip()[:80]}"')

        del model; gc.collect(); torch.mps.empty_cache()

    # Institutional gap
    print(f'\n{"="*60}')
    print(f'  Institutional entropy gap')
    print(f'{"="*60}')
    for label, model_id in MODELS:
        ind_hs, ins_hs = [], []
        for pk, prompt in inst_prompts.items():
            logits = torch.tensor(cm.get_logits(model_id, prompt)).float()
            probs = torch.softmax(logits, -1)
            h = -(probs * probs.clamp(min=1e-10).log()).sum().item()
            if get_side(pk) == 'individual': ind_hs.append(h)
            else: ins_hs.append(h)
        ind, ins = np.mean(ind_hs), np.mean(ins_hs)
        print(f'  {label:18s}: ind={ind:.2f}  inst={ins:.2f}  gap={ins-ind:+.2f}')

    # Contradictions
    print(f'\n{"="*60}')
    print(f'  Contradiction ratios')
    print(f'{"="*60}')
    for label, model_id in MODELS:
        for pair in CONTRADICTION_PAIRS:
            la = torch.tensor(cm.get_logits(model_id, pair['a'])).float()
            lb = torch.tensor(cm.get_logits(model_id, pair['b'])).float()
            lab = torch.tensor(cm.get_logits(model_id, pair['ab'])).float()
            n = min(la.shape[0], lb.shape[0], lab.shape[0])
            pa = torch.softmax(la[:n], -1); pb = torch.softmax(lb[:n], -1)
            pab = torch.softmax(lab[:n], -1); pm = 0.5*(pa+pb)
            ratio = _js(pab, pm) / max(min(_js(pab, pa), _js(pab, pb)), 1e-10)
            print(f'  {label:18s} {pair["name"]:20s} ratio={ratio:.3f}')

    # Full comparison table
    print(f'\n{"="*60}')
    print(f'  Full reasoning matrix (all Qwen variants)')
    print(f'{"="*60}')
    all_qwen = [
        ('qwen2.5_base', 'Qwen/Qwen2.5-7B'),
        ('qwen2.5_instruct', 'Qwen/Qwen2.5-7B-Instruct'),
        ('qwen2.5_r1', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'),
        ('qwen3_base', 'Qwen/Qwen3-8B-Base'),
        ('qwen3_instruct', 'Qwen/Qwen3-8B'),
    ]
    print(f'  {"model":18s}  {"H":>6s}  {"eff":>5s}  {"gap":>6s}')
    for label, model_id in all_qwen:
        batch_hs = []
        for prompt in prompts:
            l = cm.get_logits(model_id, prompt)
            if l is not None:
                p = torch.softmax(torch.tensor(l).float(), -1)
                batch_hs.append(-(p * p.clamp(min=1e-10).log()).sum().item())
        if not batch_hs:
            continue
        # Gap
        ind_hs, ins_hs = [], []
        for pk, prompt in inst_prompts.items():
            l = cm.get_logits(model_id, prompt)
            if l is None: continue
            p = torch.softmax(torch.tensor(l).float(), -1)
            h = -(p * p.clamp(min=1e-10).log()).sum().item()
            if get_side(pk) == 'individual': ind_hs.append(h)
            else: ins_hs.append(h)
        gap = np.mean(ins_hs) - np.mean(ind_hs) if ind_hs and ins_hs else float('nan')
        print(f'  {label:18s}  {np.mean(batch_hs):>6.2f}  {np.mean([int((torch.softmax(torch.tensor(cm.get_logits(model_id, p)).float(),-1)>0.001).sum()) for p in prompts[:5]]):>5.0f}  {gap:>+6.2f}')
