"""Measure BLT bits/char across Pile subsets to test whether institutional
text has lower entropy in the training data.

Causal chain: corpus entropy → model confidence → base deference (F21).

Strategy: stream from monology/pile-uncopyrighted, collect from target subsets.
Accept partial results for rare subsets. Flush output after each doc.

Usage:
    python scripts/pile_corpus_entropy.py
"""
import torch
import numpy as np
import pandas as pd
import gc
import json
import sys
import time

BLT_MODEL = "itazap/blt-1b-hf"
N_PER_SUBSET = 200
MAX_CHARS = 2000
MAX_DOCS_SCANNED = 500_000  # safety limit

TARGET_SUBSETS = {
    "FreeLaw": "institutional",
    "PubMed Central": "institutional",
    "ArXiv": "institutional",
    "Wikipedia (en)": "institutional",
    "Gutenberg (PG-19)": "individual",
    "OpenSubtitles": "individual",
    "Ubuntu IRC": "individual",
    "YoutubeSubtitles": "individual",
}


def bits_per_char_blt(text, model, tokenizer):
    if not text or len(text.strip()) < 20:
        return np.nan

    text = text[:MAX_CHARS]
    ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
    ids = ids.to(next(model.parameters()).device)

    if ids.shape[1] < 2:
        return np.nan

    with torch.no_grad():
        out = model(ids)
        logits = out.logits[0].float()

    log_probs = torch.log_softmax(logits, dim=-1)
    token_ids = ids[0]

    total_bits = 0.0
    total_chars = 0
    for i in range(len(token_ids) - 1):
        next_id = token_ids[i + 1]
        surprisal_nats = -log_probs[i, next_id].item()
        if np.isnan(surprisal_nats) or np.isinf(surprisal_nats):
            continue
        token_str = tokenizer.decode([next_id])
        total_bits += surprisal_nats / np.log(2)
        total_chars += max(len(token_str), 1)

    return total_bits / total_chars if total_chars > 0 else np.nan


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading BLT ({BLT_MODEL})...", flush=True)
    blt_tok = AutoTokenizer.from_pretrained(BLT_MODEL, trust_remote_code=True)
    blt_model = AutoModelForCausalLM.from_pretrained(
        BLT_MODEL, trust_remote_code=True, dtype=torch.float32,
    )
    if torch.backends.mps.is_available():
        blt_model = blt_model.to("mps")
    blt_model.eval()
    print("BLT loaded.\n", flush=True)

    counts = {s: 0 for s in TARGET_SUBSETS}
    all_rows = []
    total_seen = 0
    t0 = time.time()

    print("Streaming monology/pile-uncopyrighted...", flush=True)
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    for ex in ds:
        total_seen += 1

        if total_seen % 10000 == 0:
            elapsed = time.time() - t0
            scored = len(all_rows)
            print(f"  Scanned {total_seen:,} docs, scored {scored}, "
                  f"elapsed {elapsed/60:.1f}min, counts: "
                  f"{', '.join(f'{k}: {v}' for k,v in counts.items() if v > 0)}",
                  flush=True)

        if all(c >= N_PER_SUBSET for c in counts.values()):
            print("All subsets complete!", flush=True)
            break

        if total_seen >= MAX_DOCS_SCANNED:
            print(f"Reached scan limit ({MAX_DOCS_SCANNED:,})", flush=True)
            break

        meta = ex.get("meta", {})
        if isinstance(meta, str):
            meta = json.loads(meta)
        subset = meta.get("pile_set_name", "")

        if subset not in TARGET_SUBSETS:
            continue
        if counts[subset] >= N_PER_SUBSET:
            continue

        text = ex["text"]
        if len(text.strip()) < 50:
            continue

        bpc = bits_per_char_blt(text, blt_model, blt_tok)
        if np.isnan(bpc):
            continue

        counts[subset] += 1
        all_rows.append({
            "subset": subset,
            "category": TARGET_SUBSETS[subset],
            "blt_bits_per_char": bpc,
            "text_len": len(text),
            "text_preview": text[:100].replace("\n", " "),
        })

        if counts[subset] % 50 == 0:
            sub_rows = [r for r in all_rows if r["subset"] == subset]
            mean_bpc = np.mean([r["blt_bits_per_char"] for r in sub_rows])
            print(f"  >>> {subset:25s}: {counts[subset]:>3d}/{N_PER_SUBSET}  "
                  f"mean BLT={mean_bpc:.3f} bpc", flush=True)

        # Incremental save every 100 scored docs
        if len(all_rows) % 100 == 0:
            pd.DataFrame(all_rows).to_csv("data/pile_corpus_entropy.csv", index=False)

    # Final save
    df = pd.DataFrame(all_rows)
    df.to_csv("data/pile_corpus_entropy.csv", index=False)
    elapsed = time.time() - t0
    print(f"\nSaved data/pile_corpus_entropy.csv ({len(df)} rows)", flush=True)
    print(f"Scanned {total_seen:,} docs in {elapsed/60:.1f} minutes", flush=True)
    print(f"Final counts: {dict(counts)}", flush=True)

    # Summary
    if len(df) > 0:
        print(f"\n{'='*70}", flush=True)
        print(f"  BLT bits/char by Pile subset", flush=True)
        print(f"{'='*70}", flush=True)
        for subset in TARGET_SUBSETS:
            sub = df[df["subset"] == subset]
            if len(sub):
                print(f"  {subset:25s} ({TARGET_SUBSETS[subset]:12s}): "
                      f"{sub['blt_bits_per_char'].mean():.3f} ± {sub['blt_bits_per_char'].std():.3f} bpc  "
                      f"n={len(sub)}", flush=True)

        print(f"\n{'='*70}", flush=True)
        print(f"  Institutional vs Individual", flush=True)
        print(f"{'='*70}", flush=True)
        inst = df[df["category"] == "institutional"]["blt_bits_per_char"]
        indv = df[df["category"] == "individual"]["blt_bits_per_char"]
        if len(inst) > 10 and len(indv) > 10:
            print(f"  Institutional: {inst.mean():.3f} ± {inst.std():.3f} bpc  (n={len(inst)})", flush=True)
            print(f"  Individual:    {indv.mean():.3f} ± {indv.std():.3f} bpc  (n={len(indv)})", flush=True)
            print(f"  Difference:    {inst.mean() - indv.mean():+.3f} bpc", flush=True)

            from scipy import stats
            u_stat, p_val = stats.mannwhitneyu(inst, indv, alternative="two-sided")
            print(f"  Mann-Whitney U: {u_stat:.0f}, p={p_val:.2e}", flush=True)
            d = (inst.mean() - indv.mean()) / np.sqrt((inst.std()**2 + indv.std()**2) / 2)
            print(f"  Cohen's d: {d:.3f}", flush=True)
