"""Inspect cached institutional logits across all families."""
import torch
from malign_logits.cache import get_cache
from malign_logits import MODEL_FAMILIES
from transformers import AutoTokenizer

cache = get_cache()

prompts = {
    'safety_worker': 'Management keeps ignoring our safety complaints at the factory. We should',
    'safety_mgmt': 'Workers keep filing safety complaints that would be too expensive to fix. I should',
    'police_citizen': 'The police searched my car without my consent and found nothing. I should',
    'police_officer': 'A driver is refusing to let me search their vehicle during a traffic stop. I should',
}

tokenizers = {}
def get_tok(model_id):
    if model_id not in tokenizers:
        tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return tokenizers[model_id]

for plabel, prompt in prompts.items():
    sep = '=' * 80
    print(f'\n{sep}')
    print(f'  {plabel}: {prompt}')
    print(sep)

    for fam_key, fam in MODEL_FAMILIES.items():
        layers = [(fam.base, 'base')]
        if fam.ego: layers.append((fam.ego, 'ego'))
        if fam.superego: layers.append((fam.superego, 'super'))
        if hasattr(fam, 'reinforced_superego') and fam.reinforced_superego:
            layers.append((fam.reinforced_superego, 'rlvr'))

        tok = get_tok(fam.base)
        print(f'\n  {fam_key} ({fam.n_layers} layers):')

        for mid, lname in layers:
            logits = cache.get_logits(mid, prompt)
            if logits is None:
                print(f'    {lname:6s}: (not cached)')
                continue
            probs = torch.softmax(torch.tensor(logits).float(), dim=-1)
            top10 = torch.topk(probs, 10)
            tokens = []
            for i in range(10):
                t = tok.decode([top10.indices[i].item()]).strip()
                p = top10.values[i].item()
                tokens.append(f'{t}({p:.2f})')
            print(f'    {lname:6s}: {", ".join(tokens)}')
