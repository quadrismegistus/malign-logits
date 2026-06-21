"""R1-Distill-Qwen-7B: full comparison with Qwen base + Qwen Instruct.

Same analysis as R1-Distill-Llama: battery logits, institutional gap, contradictions.
"""
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
    from malign_logits import MODEL_FAMILIES
    from malign_logits.experiments import DEFAULT_PROMPTS

    cm = get_cache()
    fam = MODEL_FAMILIES['qwen']

    MODELS = [
        ('qwen_base', fam.base),
        ('qwen_instruct', fam.superego),
        ('qwen_r1', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'),
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

    # ── Battery logits ──
    for label, model_id in MODELS:
        print(f'\n=== {label} ===')
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.float16, device_map='mps')
        model.eval()

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

        # Also cache contradiction prompts
        for pair in CONTRADICTION_PAIRS:
            for key in ['a', 'b', 'ab']:
                p = pair[key]
                if not cm.has_logits(model_id, p):
                    inputs = tok(p, return_tensors='pt').to('mps')
                    with torch.no_grad():
                        out = model(**inputs)
                    cm.set_logits(model_id, p, out.logits[0, -1, :].float().cpu().numpy())

        print(f'  Output: H={np.mean(hs):.2f}  eff={np.mean(effs):.0f}')
        del model; gc.collect(); torch.mps.empty_cache()

    # ── Institutional gap ──
    print(f'\n{"="*60}')
    print(f'  Institutional entropy gap')
    print(f'{"="*60}')
    for label, model_id in MODELS:
        for side in ['individual', 'institution']:
            side_hs = []
            for pk, prompt in inst_prompts.items():
                if get_side(pk) != side: continue
                logits = torch.tensor(cm.get_logits(model_id, prompt)).float()
                probs = torch.softmax(logits, -1)
                side_hs.append(-(probs * probs.clamp(min=1e-10).log()).sum().item())
            if side == 'individual': ind_h = np.mean(side_hs)
            else: ins_h = np.mean(side_hs)
        print(f'  {label:15s}: ind={ind_h:.2f}  inst={ins_h:.2f}  gap={ins_h-ind_h:+.2f}')

    # ── Contradictions ──
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
            print(f'  {label:15s} {pair["name"]:20s} ratio={ratio:.3f}')

    print(f'\n{"="*60}')
    print(f'  Summary: Llama family vs Qwen family')
    print(f'{"="*60}')
    print(f'  (Compare with Llama: base gap=+0.68, instruct gap=+0.85, R1 gap=+0.06)')
