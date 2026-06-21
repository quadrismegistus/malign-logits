"""Circuit decomposition: trace alignment's effect through every transformer component.

Runs all circuit analyses for a given family:
  1. Attention entropy (per head, per layer)
  2. Residual stream entropy (logit lens at each layer)
  3. LayerNorm decomposition (pre-norm vs post-norm)
  4. Value vector content (attention output projected through unembedding)
  5. MLP gate openness (SiLU gate activation)
  6. Embedding space (input + output embedding norms and clustering)

Usage:
    python scripts/decompose_circuit.py                    # OLMo only
    python scripts/decompose_circuit.py --family olmo
    python scripts/decompose_circuit.py --family all       # all families
    python scripts/decompose_circuit.py --family olmo llama amber
"""
import torch
import numpy as np
import pandas as pd
import argparse
import gc


def compute_attention_entropy(attn_weights):
    """Mean per-head entropy of last-token attention distribution."""
    last_attn = attn_weights[0, :, -1, :].float()  # [n_heads, seq_len]
    entropies = []
    for h in range(last_attn.shape[0]):
        p = last_attn[h].clamp(min=1e-10)
        entropies.append(-(p * p.log()).sum().item())
    return np.mean(entropies)


def compute_logit_lens_entropy(hidden_state, final_norm, lm_head):
    """Entropy of logit-lens projection at a given layer."""
    hs = hidden_state.float()
    normed = final_norm(hs.unsqueeze(0).half()).float()
    logits = lm_head(normed.half()).float().squeeze()
    probs = torch.softmax(logits, dim=-1)
    h = -(probs * probs.clamp(min=1e-10).log()).sum().item()
    eff = (probs > 0.001).sum().item()
    return h, eff


def compute_value_projection(model, layer_idx, attn_weights, hidden_in, final_norm, lm_head, layers=None):
    """Project attention value output through unembedding."""
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, 'num_key_value_heads', n_heads)
    head_dim = model.config.hidden_size // n_heads
    heads_per_kv = n_heads // n_kv_heads
    seq_len = hidden_in.shape[0]

    if layers is None:
        layers = model.model.layers
    v_proj = layers[layer_idx].self_attn.v_proj
    V = v_proj(hidden_in.half()).float()
    V_heads = V.view(seq_len, n_kv_heads, head_dim)

    weights = attn_weights[0, :, -1, :].float()

    concat_v = torch.zeros(1, model.config.hidden_size, device=hidden_in.device, dtype=torch.float32)
    for q_h in range(n_heads):
        kv_h = q_h // heads_per_kv
        v_out = weights[q_h] @ V_heads[:, kv_h, :]
        concat_v[0, q_h * head_dim:(q_h + 1) * head_dim] = v_out

    o_proj = layers[layer_idx].self_attn.o_proj
    attn_contribution = o_proj(concat_v.half()).float()
    logits = lm_head(final_norm(attn_contribution.half())).float().squeeze()
    probs = torch.softmax(logits, dim=-1)
    return -(probs * probs.clamp(min=1e-10).log()).sum().item()


def compute_gate_stats(gate_activation):
    """Compute MLP gate openness statistics."""
    silu_gate = torch.nn.functional.silu(gate_activation.float())
    return {
        'gate_open_frac': (silu_gate.abs() > 0.1).float().mean().item(),
        'gate_mean_mag': silu_gate.abs().mean().item(),
        'gate_sparsity': (silu_gate.abs() < 0.01).float().mean().item(),
    }


def compute_embedding_stats(model):
    """Compute embedding space statistics."""
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_w = model.model.embed_tokens.weight.float().cpu()
    elif hasattr(model, 'gpt_neox'):
        embed_w = model.gpt_neox.embed_in.weight.float().cpu()
    elif hasattr(model, 'transformer'):
        embed_w = model.transformer.wte.weight.float().cpu()
    else:
        return {'tied': None, 'embed_cos_mean': np.nan, 'lmhead_cos_mean': np.nan,
                'embed_norm_mean': np.nan, 'lmhead_norm_mean': np.nan}

    if hasattr(model, 'lm_head'):
        lm_head_w = model.lm_head.weight.float().cpu()
    elif hasattr(model, 'embed_out'):
        lm_head_w = model.embed_out.weight.float().cpu()
    else:
        lm_head_w = embed_w  # fallback
    tied = torch.equal(embed_w, lm_head_w)

    torch.manual_seed(42)
    n = embed_w.shape[0]
    idx = torch.randint(0, n, (2000,))
    cos_e = torch.nn.functional.cosine_similarity(embed_w[idx[:1000]], embed_w[idx[1000:]], dim=1)
    cos_l = torch.nn.functional.cosine_similarity(lm_head_w[idx[:1000]], lm_head_w[idx[1000:]], dim=1)

    return {
        'tied': tied,
        'embed_cos_mean': cos_e.mean().item(),
        'lmhead_cos_mean': cos_l.mean().item(),
        'embed_norm_mean': embed_w.norm(dim=1).mean().item(),
        'lmhead_norm_mean': lm_head_w.norm(dim=1).mean().item(),
    }


def get_side(pk):
    if pk is None:
        return None
    for tag in ['worker', 'tenant', 'patient', 'citizen', 'political_citizen']:
        if tag in pk:
            return 'individual'
    return 'institution'


if __name__ == '__main__':
    from malign_logits import MODEL_FAMILIES
    from malign_logits.experiments import DEFAULT_PROMPTS

    parser = argparse.ArgumentParser()
    parser.add_argument('--family', nargs='+', default=['olmo'])
    args = parser.parse_args()

    if args.family == ['all']:
        families = sorted(MODEL_FAMILIES.keys())
    else:
        families = args.family

    KEY_PROMPTS = [
        "She was so angry she wanted to",
        "She kneeled and reached for his",
        "The capital of France is",
        "She loved him and hated him and wanted to",
        "Our company cut our health benefits and increased hours with no extra pay. We should",
        "Management keeps ignoring our safety complaints at the factory. We should",
    ]
    inst_prompts = {k: v for k, v in DEFAULT_PROMPTS.items() if 'institutional' in k}

    all_prompt_list = [(None, p, 'key', None) for p in KEY_PROMPTS]
    all_prompt_list += [(pk, p, 'institutional', get_side(pk)) for pk, p in inst_prompts.items()]

    all_rows = []
    embed_rows = []

    for fam_key in families:
        fam = MODEL_FAMILIES[fam_key]

        stages = [('base', fam.base)]
        if fam.ego:
            stages.append(('sft', fam.ego))
        stages.append(('superego', fam.superego))
        if fam.reinforced_superego:
            stages.append(('rlvr', fam.reinforced_superego))

        for stage_name, model_id in stages:
            print(f'\n=== {fam_key} / {stage_name} ===')
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, dtype=torch.float16,
                device_map='mps', attn_implementation='eager'
            )
            model.eval()

            # Handle different architectures
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                # OLMo, Llama, Qwen, Tulu, Amber
                backbone = model.model
                layers = backbone.layers
                final_norm = backbone.norm if hasattr(backbone, 'norm') else backbone.final_layernorm
            elif hasattr(model, 'gpt_neox'):
                # Pythia / GPT-NeoX
                backbone = model.gpt_neox
                layers = backbone.layers
                final_norm = backbone.final_layer_norm
            elif hasattr(model, 'transformer'):
                # GPT-2 style (SmolLM)
                backbone = model.transformer
                layers = backbone.h if hasattr(backbone, 'h') else backbone.layers
                final_norm = backbone.ln_f if hasattr(backbone, 'ln_f') else backbone.final_layernorm
            else:
                print(f'  Unknown architecture: {type(model).__name__}, skipping')
                del model; gc.collect(); torch.mps.empty_cache()
                continue

            lm_head = model.lm_head if hasattr(model, 'lm_head') else model.embed_out
            n_layers = len(layers)

            # --- 6. Embedding stats (once per model) ---
            embed_stats = compute_embedding_stats(model)
            embed_stats['family'] = fam_key
            embed_stats['stage'] = stage_name
            embed_rows.append(embed_stats)

            # --- MLP gate hooks (only for gated architectures) ---
            gate_activations = {}
            hooks = []
            has_gate = hasattr(layers[0], 'mlp') and hasattr(layers[0].mlp, 'gate_proj')
            if has_gate:
                for li in range(n_layers):
                    def make_hook(layer_idx):
                        def hook_fn(module, input, output):
                            gate_activations[layer_idx] = output.detach()
                        return hook_fn
                    hooks.append(layers[li].mlp.gate_proj.register_forward_hook(make_hook(li)))

            for pk, prompt, ptype, side in all_prompt_list:
                inputs = tok(prompt, return_tensors='pt').to('mps')
                seq_len = inputs['input_ids'].shape[1]
                gate_activations.clear()

                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True, output_attentions=True)

                for layer_idx in range(n_layers):
                    row = {
                        'family': fam_key, 'stage': stage_name,
                        'prompt': prompt[:50], 'prompt_key': pk,
                        'prompt_type': ptype, 'side': side,
                        'layer': layer_idx, 'n_layers': n_layers,
                        'layer_frac': layer_idx / max(n_layers - 1, 1),
                    }

                    # 1. Attention entropy
                    row['h_attention'] = compute_attention_entropy(out.attentions[layer_idx])

                    # 2. Residual stream entropy (logit lens)
                    hs = out.hidden_states[layer_idx + 1][0, -1, :]
                    h_res, eff_res = compute_logit_lens_entropy(hs, final_norm, lm_head)
                    row['h_residual'] = h_res
                    row['eff_vocab_residual'] = eff_res

                    # 3. LayerNorm decomposition (final layer only)
                    if layer_idx == n_layers - 1:
                        last_hs = out.hidden_states[-1][0, -1, :].float()
                        logits_pre = lm_head(last_hs.unsqueeze(0).half()).float().squeeze()
                        probs_pre = torch.softmax(logits_pre, dim=-1)
                        row['h_prenorm'] = -(probs_pre * probs_pre.clamp(min=1e-10).log()).sum().item()
                        row['h_postnorm'] = h_res
                        row['h_layernorm_delta'] = row['h_postnorm'] - row['h_prenorm']
                    else:
                        row['h_prenorm'] = np.nan
                        row['h_postnorm'] = np.nan
                        row['h_layernorm_delta'] = np.nan

                    # 4. Value vector projection (only for models with accessible v_proj)
                    try:
                        layer_module = layers[layer_idx]
                        if hasattr(layer_module, 'self_attn') and hasattr(layer_module.self_attn, 'v_proj'):
                            hidden_in = out.hidden_states[layer_idx][0].float()
                            row['h_value_proj'] = compute_value_projection(
                                model, layer_idx, out.attentions[layer_idx],
                                hidden_in, final_norm, lm_head, layers=layers
                            )
                        else:
                            row['h_value_proj'] = np.nan
                    except Exception:
                        row['h_value_proj'] = np.nan

                    # 5. MLP gate stats
                    if has_gate and layer_idx in gate_activations:
                        gate = gate_activations[layer_idx][0, -1, :]
                        gate_stats = compute_gate_stats(gate)
                        row.update(gate_stats)
                    else:
                        row['gate_open_frac'] = np.nan
                        row['gate_mean_mag'] = np.nan
                        row['gate_sparsity'] = np.nan

                    all_rows.append(row)

            for h in hooks:
                h.remove()

            print(f'  Done ({len(all_prompt_list)} prompts, {n_layers} layers)')
            del model; gc.collect(); torch.mps.empty_cache()

    # Save
    df = pd.DataFrame(all_rows)
    df.to_csv('data/circuit_decomposition_full.csv', index=False)
    print(f'\nSaved data/circuit_decomposition_full.csv ({len(df)} rows)')

    edf = pd.DataFrame(embed_rows)
    edf.to_csv('data/embedding_analysis.csv', index=False)
    print(f'Saved data/embedding_analysis.csv ({len(edf)} rows)')

    # Summary
    print(f'\n{"="*70}')
    print(f'  Summary by family and stage')
    print(f'{"="*70}')
    for fam_key in families:
        print(f'\n  {fam_key}:')
        for stage in ['base', 'aligned']:
            sub = df[(df['family'] == fam_key) & (df['stage'] == stage)]
            key_sub = sub[sub['prompt_type'] == 'key']
            final = sub[sub['layer'] == sub['n_layers'].max() - 1]
            print(f'    {stage:8s}: attn_H={key_sub["h_attention"].mean():.3f}  '
                  f'resid_H={key_sub["h_residual"].mean():.2f}  '
                  f'value_H={key_sub["h_value_proj"].mean():.2f}  '
                  f'gate_open={key_sub["gate_open_frac"].mean()*100:.1f}%  '
                  f'LN_delta={final["h_layernorm_delta"].mean():+.2f}')
