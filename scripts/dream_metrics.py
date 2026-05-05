"""Compute passage metrics (drift, surprisal, metonymy) on dream reports.

Reads data/dreams.csv, cleans with ftfy, filters to 100-300 words,
samples N dreams, and runs through the same pipeline as model generations.

Usage:
    python scripts/dream_metrics.py [--n 500] [--output data/dream_metrics.csv]
"""
import argparse
import pandas as pd
import ftfy
from malign_logits.embedding import compute_passage_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=500, help='Number of dreams to sample')
    parser.add_argument('--min-words', type=int, default=100)
    parser.add_argument('--max-words', type=int, default=300)
    parser.add_argument('--output', '-o', default='data/dream_metrics.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv('data/dreams.csv')
    df = df.dropna(subset=['dreams_text'])
    df['text'] = df['dreams_text'].apply(ftfy.fix_text)
    df['word_count'] = df['text'].str.split().str.len()

    filtered = df[
        (df['word_count'] >= args.min_words) &
        (df['word_count'] <= args.max_words)
    ].copy()
    print(f'Dreams after filtering ({args.min_words}-{args.max_words} words): {len(filtered)}')

    sample = filtered.sample(min(args.n, len(filtered)), random_state=args.seed)
    print(f'Sampled: {len(sample)}')

    psg_df = pd.DataFrame({
        'prompt': '',
        'model': 'dream',
        'psg': sample['text'].values,
        'family': 'dreams',
        'label': 'dream',
    })

    result = compute_passage_metrics(psg_df, min_sentences=3)
    result.to_csv(args.output, index=False)
    print(f'Saved {args.output} ({len(result)} rows)')

    print(f'\nDream metrics:')
    for col in ['total_drift', 'mean_surprisal', 'token_diameter',
                'metonymy_idx', 'directedness', 'token_metonymy_idx']:
        print(f'  {col:25s} mean={result[col].mean():.4f}  std={result[col].std():.4f}')

    # Compare to model generations
    try:
        pm = pd.read_csv('data/passage_metrics.csv')
        print(f'\nZ-scores (relative to model generations):')
        for col in ['total_drift', 'mean_surprisal', 'directedness', 'metonymy_idx']:
            m, s = pm[col].mean(), pm[col].std()
            z = (result[col].mean() - m) / s
            print(f'  {col:25s}  z = {z:+.2f}σ')
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    main()
