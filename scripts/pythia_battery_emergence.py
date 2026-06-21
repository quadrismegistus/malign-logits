"""Trace logit-level emergence across Pythia 1B pretraining checkpoints.

Runs the 47 battery prompts + 24 institutional at 12 log-spaced checkpoints.
Tracks: entropy, effective vocab, top tokens, transgressive token mass,
institutional deference gap, contradiction ratios.

Usage:
    python scripts/pythia_battery_emergence.py
"""
import torch
import numpy as np
import pandas as pd
import gc
import re

CHECKPOINTS = [0, 1, 64, 512, 1000, 5000, 10000, 25000, 50000, 100000, 143000]

# Key tokens to track probability of
TRANSGRESSIVE_WORDS = ['kill', 'fuck', 'die', 'murder', 'shoot', 'cock', 'naked', 'sex']
PROCEDURAL_WORDS = ['consider', 'report', 'contact', 'discuss', 'negotiate', 'explain']

CONTRADICTION_PAIRS = [
    {'name': 'love/hate', 'a': 'She loved him deeply and wanted to',
     'b': 'She hated him deeply and wanted to', 'ab': 'She loved him and hated him and wanted to'},
    {'name': 'innocent/guilty', 'a': 'She was innocent and she began to',
     'b': 'She was guilty and she began to', 'ab': 'She was innocent and guilty and she began to'},
]

def _js(p, q):
    p = p.clamp(min=1e-10); q = q.clamp(min=1e-10); m = 0.5*(p+q)
    return (0.5*(p*(p.log()-m.log())).sum() + 0.5*(q*(q.log()-m.log())).sum()).item()

def get_side(pk):
    for tag in ['worker', 'tenant', 'patient', 'citizen', 'political_citizen']:
        if tag in pk: return 'individual'
    return 'institution'

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = 'EleutherAI/pythia-1b'
    tok = AutoTokenizer.from_pretrained(MODEL)

    # Get token IDs for key words
    trans_ids = {}
    for w in TRANSGRESSIVE_WORDS + PROCEDURAL_WORDS:
        ids = tok.encode(' ' + w, add_special_tokens=False)
        if len(ids) == 1:
            trans_ids[w] = ids[0]
    print(f'Tracked tokens: {len(trans_ids)}')

    # Load prompts
    from malign_logits.experiments import DEFAULT_PROMPTS
    battery = {k: v for k, v in DEFAULT_PROMPTS.items() if 'institutional' not in k}
    institutional = {k: v for k, v in DEFAULT_PROMPTS.items() if 'institutional' in k}

    all_rows = []

    for step in CHECKPOINTS:
        step_name = f'step{step}'
        print(f'\n=== {step_name} ===')

        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, revision=step_name, trust_remote_code=True,
                torch_dtype=torch.float16, device_map='mps'
            )
            model.eval()

            # ── Battery prompts ──
            for pk, prompt in list(battery.items()) + list(institutional.items()):
                inputs = tok(prompt, return_tensors='pt').to('mps')
                with torch.no_grad():
                    out = model(**inputs)
                logits = out.logits[0, -1, :].float().cpu()
                probs = torch.softmax(logits, -1)

                h = -(probs * probs.clamp(min=1e-10).log()).sum().item()
                eff = (probs > 0.001).sum().item()

                # Track key token probabilities
                trans_mass = sum(probs[trans_ids[w]].item() for w in TRANSGRESSIVE_WORDS if w in trans_ids)
                proc_mass = sum(probs[trans_ids[w]].item() for w in PROCEDURAL_WORDS if w in trans_ids)

                is_inst = 'institutional' in pk
                side = get_side(pk) if is_inst else None
                cat = 'institutional' if is_inst else (
                    'sexual' if 'sexual' in pk else
                    'violence' if 'violen' in pk else
                    'neutral' if 'neutral' in pk else 'other'
                )

                all_rows.append({
                    'step': step, 'prompt_key': pk, 'prompt': prompt[:50],
                    'category': cat, 'side': side,
                    'entropy': h, 'eff_vocab': eff,
                    'transgressive_mass': trans_mass,
                    'procedural_mass': proc_mass,
                })

            # ── Contradiction ratios ──
            for pair in CONTRADICTION_PAIRS:
                logits_dict = {}
                for key in ['a', 'b', 'ab']:
                    inp = tok(pair[key], return_tensors='pt').to('mps')
                    with torch.no_grad():
                        o = model(**inp)
                    logits_dict[key] = o.logits[0, -1, :].float().cpu()

                n = min(l.shape[0] for l in logits_dict.values())
                pa = torch.softmax(logits_dict['a'][:n], -1)
                pb = torch.softmax(logits_dict['b'][:n], -1)
                pab = torch.softmax(logits_dict['ab'][:n], -1)
                pm = 0.5 * (pa + pb)
                ratio = _js(pab, pm) / max(min(_js(pab, pa), _js(pab, pb)), 1e-10)

                all_rows.append({
                    'step': step, 'prompt_key': f'contradiction_{pair["name"]}',
                    'prompt': pair['ab'][:50],
                    'category': 'contradiction', 'side': None,
                    'entropy': ratio,  # repurpose entropy field for ratio
                    'eff_vocab': 0,
                    'transgressive_mass': 0, 'procedural_mass': 0,
                })

            n_prompts = len(battery) + len(institutional) + len(CONTRADICTION_PAIRS)
            print(f'  {n_prompts} prompts processed')

            del model; gc.collect(); torch.mps.empty_cache()

        except Exception as e:
            print(f'  ERROR: {e}')

    df = pd.DataFrame(all_rows)
    df.to_csv('data/pythia1b_battery_emergence.csv', index=False)
    print(f'\nSaved data/pythia1b_battery_emergence.csv ({len(df)} rows)')

    # ── Summary ──
    print(f'\n{"="*70}')
    print(f'  Emergence summary')
    print(f'{"="*70}')

    print(f'\nEntropy by category across training:')
    for cat in ['neutral', 'sexual', 'violence', 'institutional']:
        print(f'\n  {cat}:')
        for step in CHECKPOINTS:
            sub = df[(df['step']==step) & (df['category']==cat)]
            if len(sub):
                print(f'    step {step:>6d}: H={sub["entropy"].mean():.2f}  eff={sub["eff_vocab"].mean():.0f}  '
                      f'trans={sub["transgressive_mass"].mean():.4f}  proc={sub["procedural_mass"].mean():.4f}')

    print(f'\nInstitutional gap across training:')
    for step in CHECKPOINTS:
        ind = df[(df['step']==step) & (df['side']=='individual')]
        ins = df[(df['step']==step) & (df['side']=='institution')]
        if len(ind) and len(ins):
            gap = ins['entropy'].mean() - ind['entropy'].mean()
            print(f'  step {step:>6d}: ind={ind["entropy"].mean():.2f}  inst={ins["entropy"].mean():.2f}  gap={gap:+.2f}')

    print(f'\nContradiction ratios across training:')
    for pair in CONTRADICTION_PAIRS:
        print(f'\n  {pair["name"]}:')
        for step in CHECKPOINTS:
            sub = df[(df['step']==step) & (df['prompt_key']==f'contradiction_{pair["name"]}')]
            if len(sub):
                print(f'    step {step:>6d}: ratio={sub["entropy"].values[0]:.3f}')
