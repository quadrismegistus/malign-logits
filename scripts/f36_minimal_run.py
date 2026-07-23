"""F36 minimal-pair battery: token survival + resistance on controlled pairs.

Computes logits for all 84 prompts across key families, then reports:
(a) Within-pair paired difference (single-swap primary)
(b) Per-family breakdown
(c) Check-1 token survival (P(f) and rank under aligned model)

Usage:
    uv run python scripts/f36_minimal_run.py --save
"""

import argparse
import gc
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import wilcoxon, mannwhitneyu

from malign_logits import Psyche, MODEL_FAMILIES, PATH_DATA
from malign_logits.models import load_model, get_base_logits
from scripts.f36_minimal_pairs import BATTERY


FAMILIES = ['olmo', 'llama', 'amber', 'qwen', 'qwen3', 'tulu', 'olmo-tiny', 'zephyr']


def run_battery():
    rows = []

    for fkey in FAMILIES:
        fam = MODEL_FAMILIES[fkey]
        psyche = Psyche.from_family(fkey)

        # Determine layers
        layers = []
        layers.append(('base', fam.base, psyche.primary_process))
        if fam.ego:
            layers.append(('sft', fam.ego, psyche.ego))
        aligned_name = 'dpo' if fam.superego else ('sft' if fam.ego else None)
        aligned_layer = psyche.superego or psyche.ego
        if aligned_layer and (aligned_name, aligned_layer.model_id) not in [(n, l.model_id) for n, _, l in layers]:
            layers.append((aligned_name, aligned_layer.model_id, aligned_layer))

        # Check if logits are cached; if not, load model
        test_prompt = BATTERY[0]['prompt']
        need_load = {}
        for lname, mid, layer in layers:
            if not layer._cache or not layer._cache.has_logits(mid, test_prompt):
                need_load[mid] = lname

        if need_load:
            print(f"  {fkey}: loading {len(need_load)} models for {len(BATTERY)} prompts...")
            for mid, lname in need_load.items():
                model, tokenizer = load_model(mid)
                for entry in BATTERY:
                    prompt = entry['prompt']
                    logits = get_base_logits(model, tokenizer, prompt)
                    # Find the matching layer and cache
                    for ln, lmid, layer in layers:
                        if lmid == mid and layer._cache:
                            layer._cache.set_logits(mid, prompt, logits.cpu().numpy())
                del model
                gc.collect()
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            print(f"    cached all logits")
        else:
            print(f"  {fkey}: all logits cached")

        # Now compute metrics from cached logits
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(fam.base, trust_remote_code=True)

        for entry in BATTERY:
            prompt = entry['prompt']

            try:
                base_logits = psyche.primary_process.logits(prompt).numpy()
            except Exception:
                continue

            base_p = softmax(base_logits.astype(np.float64))
            f_id = int(np.argmax(base_p))
            f_word = tok.decode([f_id]).strip()
            f_prob_base = float(base_p[f_id])

            for lname, mid, layer in layers:
                if lname == 'base':
                    continue
                try:
                    al_logits = layer.logits(prompt).numpy()
                except Exception:
                    continue

                n = min(len(base_logits), len(al_logits))
                al_p = softmax(al_logits[:n].astype(np.float64))
                a_id = int(np.argmax(al_p))

                if f_id >= n:
                    continue

                f_prob_al = float(al_p[f_id])
                ranking = np.argsort(-al_p)
                f_rank = int(np.where(ranking == f_id)[0][0])

                rows.append({
                    'family': fkey,
                    'layer': lname,
                    'pair': entry['pair'],
                    'prompt': prompt,
                    'transgression': entry['transgression'],
                    'trans_level': entry['trans_level'],
                    'valence': entry['valence'],
                    'swap': entry['swap'],
                    'f_word': f_word,
                    'a_word': tok.decode([a_id]).strip(),
                    'f_prob_base': f_prob_base,
                    'f_prob_aligned': f_prob_al,
                    'f_rank_aligned': f_rank,
                    'displaced': f_id != a_id,
                    'ratio': f_prob_al / f_prob_base if f_prob_base > 1e-10 else 0,
                })

    return pd.DataFrame(rows)


def print_results(df):
    # Use the main aligned layer per family (dpo where available, else sft)
    main = df.copy()
    # Keep only the "outermost" aligned layer per family
    keep = []
    for fkey in main.family.unique():
        sub = main[main.family == fkey]
        if 'dpo' in sub.layer.values:
            keep.append(sub[sub.layer == 'dpo'])
        elif 'sft' in sub.layer.values:
            keep.append(sub[sub.layer == 'sft'])
    main = pd.concat(keep)

    # Classify: is this a transgressive prompt?
    main['is_trans'] = ~main.transgression.isin(['benign', 'benign_high'])

    print("\n" + "=" * 80)
    print("CHECK 1: TOKEN SURVIVAL (minimal-pair battery)")
    print("=" * 80)

    displaced = main[main.displaced == True]
    for label, sub in [("transgressive", displaced[displaced.is_trans]),
                        ("benign", displaced[~displaced.is_trans])]:
        if sub.empty:
            continue
        print(f"\n  {label} (n={len(sub)}):")
        print(f"    P(f) ratio: mean={sub.ratio.mean():.3f}  median={sub.ratio.median():.3f}")
        print(f"    f rank:     median={sub.f_rank_aligned.median():.0f}  "
              f"top10={100*(sub.f_rank_aligned < 10).mean():.0f}%")

    # ── WITHIN-PAIR PAIRED DIFFERENCE ──
    print("\n" + "=" * 80)
    print("WITHIN-PAIR ANALYSIS (single-swap pairs, primary)")
    print("Paired difference: transgressive - benign, per pair")
    print("=" * 80)

    singles = main[main.swap == 'single'].copy()

    for fkey in sorted(main.family.unique()):
        fam_singles = singles[singles.family == fkey]
        pair_diffs_ratio = []
        pair_diffs_rank = []

        for pair in fam_singles.pair.unique():
            psub = fam_singles[fam_singles.pair == pair]
            trans_rows = psub[psub.is_trans]
            benign_rows = psub[~psub.is_trans]
            if trans_rows.empty or benign_rows.empty:
                continue

            t_ratio = trans_rows.ratio.mean()
            b_ratio = benign_rows.ratio.mean()
            pair_diffs_ratio.append(t_ratio - b_ratio)

            t_rank = trans_rows.f_rank_aligned.mean()
            b_rank = benign_rows.f_rank_aligned.mean()
            pair_diffs_rank.append(t_rank - b_rank)

        if not pair_diffs_ratio:
            continue

        dr = np.array(pair_diffs_ratio)
        dk = np.array(pair_diffs_rank)

        # Wilcoxon on paired differences
        if len(dr) >= 5:
            stat_r, p_r = wilcoxon(dr, alternative='two-sided')
            stat_k, p_k = wilcoxon(dk, alternative='two-sided')
        else:
            p_r, p_k = np.nan, np.nan

        sig_r = "*" if p_r < 0.05 else ""
        sig_k = "*" if p_k < 0.05 else ""

        print(f"\n  {fkey:12s}  n_pairs={len(dr)}")
        print(f"    ratio diff (trans-benign):  mean={dr.mean():+.4f}  "
              f"median={np.median(dr):+.4f}  p={p_r:.4f}{sig_r}")
        print(f"    rank diff  (trans-benign):  mean={dk.mean():+.1f}    "
              f"median={np.median(dk):+.1f}    p={p_k:.4f}{sig_k}")
        print(f"    interpretation: {'f SUPPRESSED more on trans' if dr.mean() < -0.05 else 'f survives equally' if abs(dr.mean()) < 0.05 else 'f survives MORE on trans'}")

    # ── PER-CATEGORY ──
    print("\n" + "=" * 80)
    print("PER-CATEGORY PAIRED DIFFERENCES (single-swap, pooled across families)")
    print("=" * 80)

    for cat in ['violence', 'sexual', 'substance', 'profanity', 'death']:
        cat_singles = singles[singles.pair.str.startswith(cat[0])]
        if cat == 'substance':
            cat_singles = singles[singles.pair.str.startswith('sub')]
        elif cat == 'profanity':
            cat_singles = singles[singles.pair.str.startswith('p') & ~singles.pair.str.startswith('p_')]

        diffs = []
        for fkey in cat_singles.family.unique():
            fsub = cat_singles[cat_singles.family == fkey]
            for pair in fsub.pair.unique():
                psub = fsub[fsub.pair == pair]
                t = psub[psub.is_trans]
                b = psub[~psub.is_trans]
                if t.empty or b.empty:
                    continue
                diffs.append(t.ratio.mean() - b.ratio.mean())

        if diffs:
            d = np.array(diffs)
            if len(d) >= 5:
                _, p = wilcoxon(d, alternative='two-sided')
            else:
                p = np.nan
            print(f"  {cat:12s}  n={len(d):3d}  mean_diff={d.mean():+.4f}  "
                  f"median={np.median(d):+.4f}  p={p:.4f}")

    # ── DISPLACEMENT RATE ──
    print("\n" + "=" * 80)
    print("DISPLACEMENT RATE: transgressive vs benign (single-swap)")
    print("=" * 80)

    for fkey in sorted(main.family.unique()):
        fsing = singles[singles.family == fkey]
        t = fsing[fsing.is_trans]
        b = fsing[~fsing.is_trans]
        if t.empty or b.empty:
            continue
        print(f"  {fkey:12s}  trans_disp={t.displaced.mean()*100:5.1f}%  "
              f"benign_disp={b.displaced.mean()*100:5.1f}%  "
              f"diff={100*(t.displaced.mean()-b.displaced.mean()):+5.1f}pp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("F36 minimal-pair battery: logit computation + analysis")
    df = run_battery()
    print_results(df)

    if args.save:
        out = os.path.join(PATH_DATA, "f36_minimal_pairs.csv")
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
