"""Test contradiction intervention with nnsight."""
from nnsight import LanguageModel
import torch
import torch.nn.functional as F

print("Loading base model via nnsight...")
model = LanguageModel(
    'allenai/Olmo-3-1025-7B',
    device_map='mps',
    dtype=torch.float16,
    dispatch=True,
)
print(f"Loaded: {len(model.model.layers)} layers")

prompt_a = "She loved him deeply and wanted to"
prompt_b = "She hated him deeply and wanted to"
prompt_ab = "She loved him and hated him and wanted to"

def get_hidden_and_logits(mdl, prompt):
    states = {}
    with mdl.trace(prompt, scan=False, validate=False) as tracer:
        for i, layer in enumerate(mdl.model.layers):
            states[i] = layer.output[0, -1, :].save()
        logits = mdl.lm_head.output[0, -1, :].save()
    return {i: s.float().cpu() for i, s in states.items()}, logits.float().cpu()

print("Extracting hidden states...")
h_a, logits_a = get_hidden_and_logits(model, prompt_a)
h_b, logits_b = get_hidden_and_logits(model, prompt_b)
h_ab, logits_ab = get_hidden_and_logits(model, prompt_ab)
print(f"Hidden dim: {h_a[0].shape}")

from malign_logits.analysis import js_divergence

print(f"\nBaseline:")
print(f"  JS(A, B)  = {js_divergence(logits_a, logits_b):.4f}")
print(f"  JS(AB, A) = {js_divergence(logits_ab, logits_a):.4f}")
print(f"  JS(AB, B) = {js_divergence(logits_ab, logits_b):.4f}")

p_a_full = torch.softmax(logits_a, dim=-1)
p_b_full = torch.softmax(logits_b, dim=-1)
p_ab_full = torch.softmax(logits_ab, dim=-1)
p_mean = 0.5 * (p_a_full + p_b_full)

def _js_probs(p, q):
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)
    m = 0.5 * (p + q)
    return (0.5 * (p * (p.log() - m.log())).sum()
            + 0.5 * (q * (q.log() - m.log())).sum()).item()

js_ab_mean = _js_probs(p_ab_full, p_mean)
js_ab_a = _js_probs(p_ab_full, p_a_full)
js_ab_b = _js_probs(p_ab_full, p_b_full)
ratio = js_ab_mean / min(js_ab_a, js_ab_b)
print(f"  Superposition={js_ab_mean:.5f}  Resolution={min(js_ab_a, js_ab_b):.5f}  Ratio={ratio:.3f}")

# Cosine between AB-midpoint and the love->hate direction
directions = {}
print(f"\nCosine(AB - midpoint, B - A) at each layer:")
for i in h_a:
    diff = h_b[i] - h_a[i]
    directions[i] = F.normalize(diff, dim=-1)
    midpoint = 0.5 * (h_a[i] + h_b[i])
    ab_centered = h_ab[i] - midpoint
    cos = F.cosine_similarity(ab_centered.unsqueeze(0), directions[i].unsqueeze(0)).item()
    if i % 4 == 0 or i == 31:
        print(f"  layer {i:2d}: {cos:+.4f}")

def intervene(mdl, prompt, layer_idx, direction, alpha):
    with mdl.trace(prompt, scan=False, validate=False) as tracer:
        mdl.model.layers[layer_idx].output[0, -1, :] += alpha * direction.to(mdl.device).half()
        out_logits = mdl.lm_head.output[0, -1, :].save()
    return out_logits.float().cpu()

print(f"\nIntervention (pushing AB along love->hate axis):")
print(f"{'layer':>6s}  {'alpha':>6s}  {'JS->A':>8s}  {'JS->B':>8s}  {'shift':>8s}")
print("-" * 45)

for layer_idx in [8, 16, 24, 28, 31]:
    diff_norm = (h_b[layer_idx] - h_a[layer_idx]).norm().item()
    for alpha_frac in [-1.0, 0, 1.0]:
        alpha = alpha_frac * diff_norm
        logits_int = intervene(model, prompt_ab, layer_idx, directions[layer_idx], alpha)
        js_a = js_divergence(logits_int, logits_a)
        js_b = js_divergence(logits_int, logits_b)
        shift = js_a - js_b
        print(f"{layer_idx:>6d}  {alpha_frac:>+6.1f}  {js_a:>8.4f}  {js_b:>8.4f}  {shift:>+8.4f}")
    print()

tokenizer = model.tokenizer
best_layer = 28
diff_norm = (h_b[best_layer] - h_a[best_layer]).norm().item()
logits_pushed_hate = intervene(model, prompt_ab, best_layer, directions[best_layer], diff_norm)

p_hate = torch.softmax(logits_pushed_hate, dim=-1)

delta = p_hate - p_ab_full
top_hate = delta.topk(8)
top_love = (-delta).topk(8)

print(f"Words changed by pushing toward HATE at layer {best_layer}:")
print(f"\n  BOOSTED by hate direction:")
for v, idx in zip(top_hate.values, top_hate.indices):
    w = tokenizer.decode([idx]).strip()
    print(f"    {w:15s}  d={v.item():+.4f}  love={p_a_full[idx]:.4f}  hate={p_b_full[idx]:.4f}  AB={p_ab_full[idx]:.4f}->{p_hate[idx]:.4f}")

print(f"\n  SUPPRESSED by hate direction:")
for v, idx in zip(top_love.values, top_love.indices):
    w = tokenizer.decode([idx]).strip()
    print(f"    {w:15s}  d={-v.item():+.4f}  love={p_a_full[idx]:.4f}  hate={p_b_full[idx]:.4f}  AB={p_ab_full[idx]:.4f}->{p_hate[idx]:.4f}")
