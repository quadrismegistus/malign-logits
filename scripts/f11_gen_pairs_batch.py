"""Generate contradiction completions for multiple pairs."""
import torch

def _js(p, q):
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)
    m = 0.5 * (p + q)
    return (0.5 * (p * (p.log() - m.log())).sum()
            + 0.5 * (q * (q.log() - m.log())).sum()).item()

def gen(model, tokenizer, prompt, n, max_tokens=60, temp=1.0):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    results = []
    for _ in range(n):
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temp,
                            do_sample=True, top_k=50, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append(text[len(prompt):].strip().split('\n')[0][:200])
    return results

if __name__ == '__main__':
    from malign_logits.psyche import Psyche

    psyche = Psyche.from_family('olmo', load=True)

    pairs = [
        {
            'name': 'faithful/unfaithful',
            'prompt_a': 'She was faithful to him and she began to',
            'prompt_b': 'She was unfaithful to him and she began to',
            'prompt_ab': 'She was faithful and unfaithful to him and she began to',
        },
        {
            'name': 'rich/poor',
            'prompt_a': 'He was rich and he decided to',
            'prompt_b': 'He was poor and he decided to',
            'prompt_ab': 'He was rich and poor and he decided to',
        },
    ]

    layers = [
        ('base', psyche.primary_process),
        ('sft', psyche.ego),
        ('dpo', psyche.superego),
        ('rlvr', psyche.reinforced_superego),
    ]

    for pair in pairs:
        print(f"\n{'='*70}")
        print(f"  {pair['name']}")
        print(f"{'='*70}")

        print(f"\n{'stage':>6s}  {'JS->A':>7s}  {'JS->B':>7s}  {'JS->blend':>9s}  {'ratio':>6s}  {'bias':>7s}")
        print('-' * 55)

        for layer_name, layer in layers:
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
            ratio = js_ab_mean / max(min(js_ab_a, js_ab_b), 1e-10)
            bias = js_ab_a - js_ab_b

            print(f"{layer_name:>6s}  {js_ab_a:>7.4f}  {js_ab_b:>7.4f}  {js_ab_mean:>9.4f}  {ratio:>6.3f}  {bias:>+7.4f}")

        print(f"\n  Generations:")
        for prompt_key, label, count in [('prompt_a', 'A', 5), ('prompt_b', 'B', 5), ('prompt_ab', 'AB', 25)]:
            prompt = pair[prompt_key]
            print(f"\n  [{label}] \"{prompt}\"")
            for layer_name, layer in [('BASE', psyche.primary_process), ('ALIGNED', psyche.reinforced_superego)]:
                print(f"\n  {layer_name}:")
                texts = gen(layer.model, layer.tokenizer, prompt, n=count)
                for i, t in enumerate(texts):
                    print(f"    {i+1:2d}. {t}")

    print("\nDone.")
