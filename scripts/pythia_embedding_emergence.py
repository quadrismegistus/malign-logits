"""Trace embedding space emergence across Pythia 410M pretraining checkpoints.

Loads only embed_tokens.weight at each checkpoint (no forward passes).
Tracks: semantic clustering, key token neighborhoods, embedding norms.

Usage:
    python scripts/pythia_embedding_emergence.py
"""
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import gc

# Key tokens to track neighborhoods for
KEY_TOKENS = {
    'violence': ['kill', 'murder', 'stab', 'shoot', 'punch', 'fight', 'attack', 'hit'],
    'sexual': ['fuck', 'sex', 'cock', 'naked', 'kiss', 'touch', 'breast'],
    'institutional': ['report', 'contact', 'consider', 'negotiate', 'file', 'consult', 'discuss'],
    'labor': ['strike', 'union', 'worker', 'employee', 'boss', 'wage', 'hire', 'fired'],
    'procedural': ['should', 'could', 'would', 'might', 'perhaps', 'consider', 'however'],
    'emotional': ['angry', 'love', 'hate', 'fear', 'happy', 'sad', 'cry', 'scream'],
}

# Log-spaced checkpoints
CHECKPOINTS = [0, 1, 8, 64, 512, 1000, 5000, 10000, 25000, 50000, 100000, 143000]

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MODEL = 'EleutherAI/pythia-1b'
    tok = AutoTokenizer.from_pretrained(MODEL)

    # Get token IDs for key words
    key_ids = {}
    for category, words in KEY_TOKENS.items():
        for word in words:
            ids = tok.encode(' ' + word, add_special_tokens=False)
            if len(ids) == 1:
                key_ids[word] = {'id': ids[0], 'category': category}

    print(f'Tracked tokens: {len(key_ids)} ({", ".join(key_ids.keys())})')

    all_rows = []

    for step in CHECKPOINTS:
        step_name = f'step{step}'
        print(f'\n=== {step_name} ===')

        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, revision=step_name, trust_remote_code=True
            )
            embed = model.gpt_neox.embed_in.weight.float().detach()
            vocab_size, hidden_dim = embed.shape
            print(f'  Embeddings: {vocab_size} × {hidden_dim}')

            # 1. Overall embedding statistics
            norms = embed.norm(dim=1)
            mean_norm = norms.mean().item()
            std_norm = norms.std().item()

            # 2. Random pairwise cosine similarity (1000 pairs)
            torch.manual_seed(42)
            idx = torch.randint(0, vocab_size, (2000,))
            cos = torch.nn.functional.cosine_similarity(embed[idx[:1000]], embed[idx[1000:]], dim=1)
            mean_cos = cos.mean().item()
            std_cos = cos.std().item()

            # 3. Key token neighborhoods
            for word, info in key_ids.items():
                tid = info['id']
                if tid >= vocab_size:
                    continue
                word_embed = embed[tid].unsqueeze(0)
                # Cosine similarity to all other tokens
                sims = torch.nn.functional.cosine_similarity(word_embed, embed, dim=1)
                # Top-10 neighbors (excluding self)
                top_vals, top_ids = sims.topk(11)
                neighbors = [(tok.decode(int(i)).strip(), v.item())
                             for i, v in zip(top_ids[1:], top_vals[1:])]

                all_rows.append({
                    'step': step, 'word': word, 'category': info['category'],
                    'self_norm': embed[tid].norm().item(),
                    'top1_neighbor': neighbors[0][0] if neighbors else '',
                    'top1_sim': neighbors[0][1] if neighbors else 0,
                    'top5_neighbors': ', '.join(n[0] for n in neighbors[:5]),
                    'mean_top5_sim': np.mean([n[1] for n in neighbors[:5]]) if neighbors else 0,
                })

            # 4. Within-category vs between-category similarity
            cat_sims = {}
            for category, words in KEY_TOKENS.items():
                cat_ids = [key_ids[w]['id'] for w in words if w in key_ids and key_ids[w]['id'] < vocab_size]
                if len(cat_ids) < 2:
                    continue
                cat_embeds = embed[cat_ids]
                within = cosine_similarity(cat_embeds.numpy()).mean()
                cat_sims[category] = within

            # 5. K-means clustering quality
            # Sample 5000 tokens, cluster into 50 groups, check if key tokens cluster by category
            sample_ids = torch.randint(0, vocab_size, (5000,))
            # Add key token IDs
            extra_ids = torch.tensor([info['id'] for info in key_ids.values() if info['id'] < vocab_size])
            all_ids = torch.cat([sample_ids, extra_ids]).unique()
            sample_embeds = embed[all_ids].numpy()

            kmeans = KMeans(n_clusters=50, random_state=42, n_init=5, max_iter=100)
            labels = kmeans.fit_predict(sample_embeds)

            # For each category, check if its tokens land in the same cluster
            for category, words in KEY_TOKENS.items():
                cat_token_ids = [key_ids[w]['id'] for w in words if w in key_ids and key_ids[w]['id'] < vocab_size]
                if len(cat_token_ids) < 2:
                    continue
                cat_positions = [int((all_ids == tid).nonzero(as_tuple=True)[0][0]) for tid in cat_token_ids
                                 if tid in all_ids]
                if len(cat_positions) < 2:
                    continue
                cat_labels = [labels[p] for p in cat_positions]
                # Clustering purity: what fraction of the category shares the most common cluster?
                from collections import Counter
                most_common_count = Counter(cat_labels).most_common(1)[0][1]
                purity = most_common_count / len(cat_labels)

                all_rows.append({
                    'step': step, 'word': f'_cluster_{category}', 'category': category,
                    'self_norm': 0,
                    'top1_neighbor': '', 'top1_sim': 0,
                    'top5_neighbors': '',
                    'mean_top5_sim': purity,  # repurpose this field for cluster purity
                })

            print(f'  norm={mean_norm:.3f}±{std_norm:.3f}, cos={mean_cos:.4f}±{std_cos:.4f}')
            print(f'  Within-cat sim: {", ".join(f"{k}={v:.3f}" for k, v in cat_sims.items())}')

            del model, embed
            gc.collect()

        except Exception as e:
            print(f'  ERROR: {e}')

    df = pd.DataFrame(all_rows)
    df.to_csv('data/pythia1b_embedding_emergence.csv', index=False)
    print(f'\nSaved data/pythia_embedding_emergence.csv ({len(df)} rows)')

    # Summary: when do neighborhoods stabilize?
    print(f'\n{"="*70}')
    print(f'  Neighborhood emergence for key tokens')
    print(f'{"="*70}')
    for word in ['kill', 'fuck', 'strike', 'report', 'love', 'hate']:
        if word not in key_ids:
            continue
        sub = df[(df['word'] == word) & (df['step'].isin(CHECKPOINTS))]
        if len(sub) == 0:
            continue
        print(f'\n  {word}:')
        for _, r in sub.iterrows():
            print(f'    step {r["step"]:>6d}: top5={r["top5_neighbors"]:60s}  sim={r["mean_top5_sim"]:.3f}')
