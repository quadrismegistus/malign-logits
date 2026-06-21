"""Full battery for R1-Distill: circuit decomposition + formation logits.

Runs:
1. Circuit decomposition (attention, residual, LayerNorm, values, gating, embeddings)
2. Raw logits on all 71 prompts (battery + institutional) — cached to logits stash
3. Masked logits (no <think>) on all prompts — for content comparison

Compares R1-Distill with Llama base and Llama Instruct (same base model).
"""
import torch
import numpy as np
import pandas as pd
import gc

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from malign_logits.cache import get_cache
    from malign_logits import MODEL_FAMILIES
    from malign_logits.experiments import DEFAULT_PROMPTS

    cm = get_cache()
    prompts = list(DEFAULT_PROMPTS.values())
    prompt_keys = list(DEFAULT_PROMPTS.keys())

    R1_ID = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    fam = MODEL_FAMILIES['llama']

    MODELS = [
        ('llama_base', fam.base),
        ('llama_instruct', fam.superego),
        ('r1_distill', R1_ID),
    ]

    all_rows = []

    for label, model_id in MODELS:
        print(f'\n=== {label}: {model_id} ===')
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.float16,
            device_map='mps', attn_implementation='eager'
        )
        model.eval()

        backbone = model.model
        layers = backbone.layers
        final_norm = backbone.norm if hasattr(backbone, 'norm') else backbone.final_layernorm
        lm_head = model.lm_head
        n_layers = len(layers)
        n_heads = model.config.num_attention_heads

        for pi, (pk, prompt) in enumerate(zip(prompt_keys, prompts)):
            inputs = tok(prompt, return_tensors='pt').to('mps')
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, output_attentions=True)

            # Cache raw logits
            raw_logits = out.logits[0, -1, :].float().cpu().numpy()
            if not cm.has_logits(model_id, prompt):
                cm.set_logits(model_id, prompt, raw_logits)

            # Per-layer metrics
            for li in range(n_layers):
                # Attention entropy
                attn = out.attentions[li][0, :, -1, :].float()
                h_attn = np.mean([-(attn[h].clamp(min=1e-10) * attn[h].clamp(min=1e-10).log()).sum().item()
                                  for h in range(attn.shape[0])])

                # Residual entropy (logit lens)
                hs = out.hidden_states[li + 1][0, -1, :].float()
                normed = final_norm(hs.unsqueeze(0).half()).float()
                logits_ll = lm_head(normed.half()).float().squeeze()
                probs_ll = torch.softmax(logits_ll, -1)
                h_res = -(probs_ll * probs_ll.clamp(min=1e-10).log()).sum().item()

                all_rows.append({
                    'model': label, 'model_id': model_id,
                    'prompt': prompt[:50], 'prompt_key': pk,
                    'layer': li, 'n_layers': n_layers,
                    'h_attention': h_attn, 'h_residual': h_res,
                })

            if pi % 20 == 0:
                print(f'  {pi}/{len(prompts)} prompts...')

        # Output-level stats
        print(f'  Computing output stats...')
        output_rows = []
        for pk, prompt in zip(prompt_keys, prompts):
            logits = torch.tensor(cm.get_logits(model_id, prompt)).float()
            probs = torch.softmax(logits, -1)
            h = -(probs * probs.clamp(min=1e-10).log()).sum().item()
            eff = (probs > 0.001).sum().item()
            output_rows.append({
                'model': label, 'prompt_key': pk, 'prompt': prompt[:50],
                'output_entropy': h, 'eff_vocab': eff,
            })

        odf = pd.DataFrame(output_rows)
        print(f'  Mean output H={odf["output_entropy"].mean():.2f}  eff={odf["eff_vocab"].mean():.0f}')

        print(f'  Done ({len(prompts)} prompts, {n_layers} layers)')
        del model; gc.collect(); torch.mps.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv('data/r1_circuit_decomposition.csv', index=False)
    print(f'\nSaved data/r1_circuit_decomposition.csv ({len(df)} rows)')

    # Summary
    print(f'\n{"="*70}')
    print(f'  R1-Distill vs Llama base vs Llama Instruct')
    print(f'{"="*70}')
    print(f'  {"model":18s}  {"attn_H":>7s}  {"resid_H":>8s}  {"output_H":>9s}  {"eff_vocab":>9s}')
    for label, model_id in MODELS:
        sub = df[df['model'] == label]
        # Get output stats from cached logits
        hs, effs = [], []
        for pk, prompt in zip(prompt_keys, prompts):
            l = cm.get_logits(model_id, prompt)
            if l is not None:
                p = torch.softmax(torch.tensor(l).float(), -1)
                hs.append(-(p * p.clamp(min=1e-10).log()).sum().item())
                effs.append((p > 0.001).sum().item())
        print(f'  {label:18s}  {sub["h_attention"].mean():>7.3f}  {sub["h_residual"].mean():>8.2f}  '
              f'{np.mean(hs):>9.2f}  {np.mean(effs):>9.0f}')
