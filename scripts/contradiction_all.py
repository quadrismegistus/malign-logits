"""Compute contradiction ratios for all pairs including Deleuzian ones."""
import torch

if __name__ == '__main__':
    from malign_logits.psyche import Psyche

    psyche = Psyche.from_family('olmo', load=True)
    pairs = psyche.DEFAULT_CONTRADICTIONS
    layers_list = [
        ("base", psyche.primary_process),
        ("sft", psyche.ego),
        ("dpo", psyche.superego),
        ("rlvr", psyche.reinforced_superego),
    ]

    def _js(p, q):
        p = p.clamp(min=1e-10)
        q = q.clamp(min=1e-10)
        m = 0.5 * (p + q)
        return (0.5 * (p * (p.log() - m.log())).sum()
                + 0.5 * (q * (q.log() - m.log())).sum()).item()

    results = []
    for pair in pairs:
        print(f'\n{pair["name"]}:')
        for layer_name, layer in layers_list:
            logits_a = layer.logits(pair['prompt_a'])
            logits_b = layer.logits(pair['prompt_b'])
            logits_ab = layer.logits(pair['prompt_ab'])

            n = min(logits_a.shape[-1], logits_b.shape[-1], logits_ab.shape[-1])
            p_a = torch.softmax(logits_a[:n].float(), dim=-1)
            p_b = torch.softmax(logits_b[:n].float(), dim=-1)
            p_ab = torch.softmax(logits_ab[:n].float(), dim=-1)
            p_mean = 0.5 * (p_a + p_b)

            js_ab_a = _js(p_ab, p_a)
            js_ab_b = _js(p_ab, p_b)
            js_ab_mean = _js(p_ab, p_mean)
            ratio = js_ab_mean / min(js_ab_a, js_ab_b)

            results.append({
                'pair': pair['name'], 'model': layer_name,
                'js_to_A': js_ab_a, 'js_to_B': js_ab_b,
                'js_to_mean': js_ab_mean, 'ratio': ratio,
            })
            print(f'  {layer_name:5s}  ratio={ratio:.3f}  js_A={js_ab_a:.4f}  js_B={js_ab_b:.4f}')

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('data/contradiction_detail.csv', index=False)
    print('\nSaved data/contradiction_detail.csv')
