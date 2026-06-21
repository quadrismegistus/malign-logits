"""Compute JS contradiction ratios for new pairs across all 11 families."""
import torch
import pandas as pd
import gc

def _js(p, q):
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)
    m = 0.5 * (p + q)
    return (0.5 * (p * (p.log() - m.log())).sum()
            + 0.5 * (q * (q.log() - m.log())).sum()).item()

if __name__ == '__main__':
    from malign_logits.psyche import Psyche
    from malign_logits import MODEL_FAMILIES

    NEW_PAIRS = [
        {
            'name': 'innocent/guilty',
            'prompt_a': 'She was innocent and she began to',
            'prompt_b': 'She was guilty and she began to',
            'prompt_ab': 'She was innocent and guilty and she began to',
        },
        {
            'name': 'rational/irrational',
            'prompt_a': 'He was completely rational and chose to',
            'prompt_b': 'He was completely irrational and chose to',
            'prompt_ab': 'He was completely rational and completely irrational and chose to',
        },
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

    LAYER_TO_POS = {
        'base': (0, 'BASE'),
        'ego': (1, 'SFT'),
        'superego': (2, 'ALIGNED'),
        'reinforced_superego': (2, 'ALIGNED'),
    }

    ALL_FAMILIES = sorted(MODEL_FAMILIES.keys())
    results = []

    for fam_key in ALL_FAMILIES:
        fam = MODEL_FAMILIES[fam_key]
        n_layers = sum(1 for x in [fam.base, fam.ego, fam.superego, fam.reinforced_superego] if x)
        print(f'\n=== {fam_key} ({n_layers}L) ===')

        try:
            psyche = Psyche.from_family(fam_key, load=False)
        except Exception as e:
            print(f'  FAILED init: {e}')
            continue

        layers_list = [('base', psyche.primary_process)]
        if psyche.ego:
            layers_list.append(('ego', psyche.ego))
        layers_list.append(('superego', psyche.superego))
        if psyche.reinforced_superego:
            layers_list.append(('reinforced_superego', psyche.reinforced_superego))

        # Check if we need to load models
        needs_load = False
        for pair in NEW_PAIRS:
            for _, layer in layers_list:
                for pk in ['prompt_a', 'prompt_b', 'prompt_ab']:
                    try:
                        logits = layer.logits(pair[pk])
                        if logits is None:
                            needs_load = True
                            break
                    except:
                        needs_load = True
                        break
                if needs_load:
                    break
            if needs_load:
                break

        if needs_load:
            print(f'  Loading models...')
            try:
                psyche = Psyche.from_family(fam_key, load=True)
                layers_list = [('base', psyche.primary_process)]
                if psyche.ego:
                    layers_list.append(('ego', psyche.ego))
                layers_list.append(('superego', psyche.superego))
                if psyche.reinforced_superego:
                    layers_list.append(('reinforced_superego', psyche.reinforced_superego))
            except Exception as e:
                print(f'  FAILED load: {e}')
                continue
        else:
            print(f'  All cached')

        for pair in NEW_PAIRS:
            for layer_name, layer in layers_list:
                try:
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

                    pos, pos_name = LAYER_TO_POS[layer_name]
                    # For 4-layer families, skip superego (use reinforced_superego as ALIGNED)
                    if layer_name == 'superego' and psyche.reinforced_superego:
                        continue

                    results.append({
                        'family': fam_key, 'pair': pair['name'],
                        'pos': pos, 'pos_name': pos_name,
                        'js_to_A': js_ab_a, 'js_to_B': js_ab_b,
                        'js_to_mean': js_ab_mean, 'ratio': ratio,
                        'pole_bias': js_ab_a - js_ab_b,
                    })
                    print(f'  {pair["name"]:25s} {layer_name:>6s}  ratio={ratio:.3f}')
                except Exception as e:
                    print(f'  {pair["name"]:25s} {layer_name:>6s}  ERROR: {e}')

        if needs_load:
            del psyche
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv('data/f11_new_pairs_contradiction.csv', index=False)
    print(f'\nSaved data/f11_new_pairs_contradiction.csv ({len(df)} rows)')

    # Merge with existing meta contradiction data
    existing = pd.read_csv('data/f11_meta_contradiction.csv')
    combined = pd.concat([existing, df], ignore_index=True)
    combined.to_csv('data/f11_meta_contradiction.csv', index=False)
    print(f'Updated data/f11_meta_contradiction.csv ({len(combined)} rows total)')
