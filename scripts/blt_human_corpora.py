"""Run BLT byte-level surprisal on human corpora (dreams, fiction, waking, abstracts).

Reads from corpus_metrics.parquet, computes BLT bits/char, prints summary.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from malign_logits.embedding import passage_surprisal, _load_surprisal_model

REF = "itazap/blt-1b-hf"


def main():
    df = pd.read_parquet("data/corpus_metrics.parquet")
    human = df[df.family.isin(["dreams", "waking", "c20_fiction", "abstracts"])].copy()
    print(f"Human corpora: {len(human)} passages")

    model, tok = _load_surprisal_model(REF)

    bits_per_char = []
    nats_per_token = []

    for _, row in tqdm(human.iterrows(), total=len(human), desc="BLT"):
        text = row["psg"]
        prompt = row.get("prompt", "")
        if not text or len(text.strip()) < 10:
            bits_per_char.append(np.nan)
            nats_per_token.append(np.nan)
            continue

        ps = passage_surprisal(text, model=model, tokenizer=tok,
                               prompt_prefix=str(prompt) if prompt else "")

        if ps["token_surprisals"]:
            total_bits = sum(s / np.log(2) for _, s in ps["token_surprisals"])
            total_chars = sum(len(t) for t, _ in ps["token_surprisals"])
            bits_per_char.append(total_bits / total_chars if total_chars > 0 else np.nan)
            nats_per_token.append(ps["mean_surprisal"])
        else:
            bits_per_char.append(np.nan)
            nats_per_token.append(np.nan)

    human["blt_bits_per_char"] = bits_per_char
    human["blt_nats_per_token"] = nats_per_token

    human[["family", "label", "blt_bits_per_char", "blt_nats_per_token"]].to_csv(
        "data/blt_human_corpora.csv", index=False)
    print(f"\nSaved to data/blt_human_corpora.csv")

    print(f"\n{'='*50}")
    print("BLT bits/char by corpus (Shannon English ≈ 1.0)")
    print('='*50)
    summary = human.groupby("family").blt_bits_per_char.agg(["mean", "std", "count"])
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
