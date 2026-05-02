"""Compare contradiction intervention across base vs SFT vs DPO."""
from nnsight import LanguageModel
import torch
import torch.nn.functional as F
from malign_logits.analysis import js_divergence

prompt_a = "She loved him deeply and wanted to"
prompt_b = "She hated him deeply and wanted to"
prompt_ab = "She loved him and hated him and wanted to"

MODELS = [
    ("BASE", "allenai/Olmo-3-1025-7B"),
    ("SFT",  "allenai/Olmo-3-7B-Instruct-SFT"),
    ("DPO",  "allenai/Olmo-3-7B-Instruct-DPO"),
]

def get_hidden_and_logits(mdl, prompt):
    states = {}
    with mdl.trace(prompt, scan=False, validate=False) as tracer:
        for i, layer in enumerate(mdl.model.layers):
            states[i] = layer.output[0, -1, :].save()
        logits = mdl.lm_head.output[0, -1, :].save()
    return {i: s.float().cpu() for i, s in states.items()}, logits.float().cpu()

def intervene(mdl, prompt, layer_idx, direction, alpha):
    with mdl.trace(prompt, scan=False, validate=False) as tracer:
        mdl.model.layers[layer_idx].output[0, -1, :] += alpha * direction.to(mdl.device).half()
        out_logits = mdl.lm_head.output[0, -1, :].save()
    return out_logits.float().cpu()

all_results = []

for label, model_id in MODELS:
    print(f"\n{'=' * 60}")
    print(f"  {label}: {model_id}")
    print(f"{'=' * 60}")

    model = LanguageModel(model_id, device_map='mps', dtype=torch.float16, dispatch=True)

    h_a, logits_a = get_hidden_and_logits(model, prompt_a)
    h_b, logits_b = get_hidden_and_logits(model, prompt_b)
    h_ab, logits_ab = get_hidden_and_logits(model, prompt_ab)

    p_a = torch.softmax(logits_a, dim=-1)
    p_b = torch.softmax(logits_b, dim=-1)
    p_ab = torch.softmax(logits_ab, dim=-1)
    p_mean = 0.5 * (p_a + p_b)

    def _js_probs(p, q):
        p, q = p.clamp(min=1e-10), q.clamp(min=1e-10)
        m = 0.5 * (p + q)
        return (0.5 * (p * (p.log() - m.log())).sum()
                + 0.5 * (q * (q.log() - m.log())).sum()).item()

    js_ab_mean = _js_probs(p_ab, p_mean)
    js_ab_a = _js_probs(p_ab, p_a)
    js_ab_b = _js_probs(p_ab, p_b)
    ratio = js_ab_mean / min(js_ab_a, js_ab_b)

    print(f"\n  Observational:")
    print(f"    JS(AB, mean) = {js_ab_mean:.5f}  (superposition)")
    print(f"    JS(AB, A)    = {js_ab_a:.5f}")
    print(f"    JS(AB, B)    = {js_ab_b:.5f}")
    print(f"    min(A,B)     = {min(js_ab_a, js_ab_b):.5f}  (resolution)")
    print(f"    Ratio        = {ratio:.3f}  {'SUPERPOSITION' if ratio < 1 else 'RESOLUTION'}")

    directions = {}
    for i in h_a:
        directions[i] = F.normalize(h_b[i] - h_a[i], dim=-1)

    print(f"\n  Intervention (alpha=+/-1.0, shift = JS->A - JS->B):")
    print(f"  {'layer':>6s}  {'push love':>10s}  {'baseline':>10s}  {'push hate':>10s}  {'range':>8s}")
    print(f"  {'-'*50}")

    for layer_idx in [8, 16, 24, 28, 31]:
        diff_norm = (h_b[layer_idx] - h_a[layer_idx]).norm().item()
        shifts = []
        for alpha_frac in [-1.0, 0, 1.0]:
            alpha = alpha_frac * diff_norm
            logits_int = intervene(model, prompt_ab, layer_idx, directions[layer_idx], alpha)
            js_a = js_divergence(logits_int, logits_a)
            js_b = js_divergence(logits_int, logits_b)
            shifts.append(js_a - js_b)
            all_results.append({
                'model': label, 'layer': layer_idx,
                'alpha': alpha_frac, 'shift': js_a - js_b,
            })
        rng = shifts[2] - shifts[0]
        print(f"  {layer_idx:>6d}  {shifts[0]:>+10.4f}  {shifts[1]:>+10.4f}  {shifts[2]:>+10.4f}  {rng:>8.4f}")

    del model
    torch.mps.empty_cache()

print(f"\n\n{'=' * 60}")
print(f"  SUMMARY: Intervention range at layer 28 (hate push - love push)")
print(f"{'=' * 60}")
for label, _, in MODELS:
    layer28 = [r for r in all_results if r['model'] == label and r['layer'] == 28]
    push_love = [r['shift'] for r in layer28 if r['alpha'] == -1.0][0]
    push_hate = [r['shift'] for r in layer28 if r['alpha'] == 1.0][0]
    rng = push_hate - push_love
    print(f"  {label:5s}: range = {rng:.4f}  (love={push_love:+.4f}, hate={push_hate:+.4f})")

print(f"\nLarger range = contradiction is more linearly decomposable")
print(f"If BASE has largest range, the primary process represents")
print(f"contradiction as clean linear superposition that alignment disrupts.")
