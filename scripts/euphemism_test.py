"""F01 euphemism vs. proximity test.

Adjudicates whether alignment-induced displacement is:
- Flat suppression: aligned argmax = highest base-ranked permitted token
- Metonymy: aligned argmax skips over higher-probability permitted tokens
  to land on one semantically closer to the forbidden token

Three tests:
1. Skip count: how many permitted tokens does alignment pass over?
2. Token relatedness: is the aligned argmax semantically closer to the
   forbidden token than skipped tokens (unembedding cosine)?
3. Distributional field mass: does alignment shift probability mass toward
   or away from the forbidden token's semantic field?

Usage:
    uv run python scripts/euphemism_test.py --save
    uv run python scripts/euphemism_test.py --no-embeddings  # skip tests 2-3
"""

import argparse
import gc
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import softmax

from malign_logits import Psyche, MODEL_FAMILIES, PATH_DATA
from malign_logits.experiments import DEFAULT_PROMPTS


TRANSGRESSIVE_CATS = {
    "sexual_liminal", "sexual_explicit",
    "violence_liminal", "violence_explicit",
    "death", "profanity", "substance",
}


def prompt_category(key):
    parts = key.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else key


def is_transgressive(key):
    return prompt_category(key) in TRANSGRESSIVE_CATS


def cosine_sim(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0


# ── Test 1: skip count ──────────────────────────────────────────────

def analyze_site(base_logits, aligned_logits, tokenizer, suppress_threshold=0.5):
    n = min(len(base_logits), len(aligned_logits))
    base_p = softmax(base_logits[:n].astype(np.float64))
    al_p = softmax(aligned_logits[:n].astype(np.float64))

    f_id = int(np.argmax(base_p))
    a_id = int(np.argmax(al_p))
    if f_id == a_id:
        return None

    ranking = np.argsort(-base_p)
    a_rank = int(np.where(ranking == a_id)[0][0])

    prob_ratio = np.where(base_p > 1e-10, al_p / base_p, 1.0)
    suppressed = prob_ratio < (1 - suppress_threshold)

    above = ranking[:a_rank]
    non_suppressed_above = int(np.sum(~suppressed[above]))

    flat_p = base_p.copy()
    flat_p[suppressed] = 0
    if flat_p.sum() > 0:
        flat_p /= flat_p.sum()

    simple_p = base_p.copy()
    simple_p[f_id] = 0
    simple_p /= simple_p.sum()

    skipped_ids = [int(ranking[i]) for i in range(1, a_rank)]

    return {
        "f_id": f_id,
        "f_token": tokenizer.decode([f_id]).strip(),
        "f_prob_base": float(base_p[f_id]),
        "f_prob_aligned": float(al_p[f_id]),
        "a_id": a_id,
        "a_token": tokenizer.decode([a_id]).strip(),
        "a_prob_base": float(base_p[a_id]),
        "a_prob_aligned": float(al_p[a_id]),
        "a_base_rank": a_rank,
        "skip_count": max(0, a_rank - 1),
        "n_suppressed": int(suppressed.sum()),
        "permitted_skip_count": non_suppressed_above,
        "flat_matches": (int(np.argmax(flat_p)) == a_id),
        "simple_matches": (int(np.argmax(simple_p)) == a_id),
        "simple_argmax_token": tokenizer.decode([int(np.argmax(simple_p))]).strip(),
        "skipped_ids": skipped_ids,
    }


def run_skip_analysis(families, suppress_threshold=0.5):
    from transformers import AutoTokenizer
    all_rows, baseline_rows = [], []

    for fkey in families:
        fam = MODEL_FAMILIES[fkey]
        psyche = Psyche.from_family(fkey)
        aligned = psyche.superego or psyche.ego
        if aligned is None:
            continue
        tok = AutoTokenizer.from_pretrained(fam.base)
        print(f"  {fkey}: {fam.base} → {aligned.model_id}")

        for pkey, prompt in DEFAULT_PROMPTS.items():
            cat = prompt_category(pkey)
            try:
                bl = psyche.primary_process.logits(prompt).numpy()
                al = aligned.logits(prompt).numpy()
            except Exception:
                continue

            result = analyze_site(bl, al, tok, suppress_threshold)
            if result is None:
                all_rows.append({
                    "family": fkey, "prompt_key": pkey, "prompt": prompt,
                    "category": cat, "is_transgressive": is_transgressive(pkey),
                    "displaced": False,
                })
                continue

            result.update(family=fkey, prompt_key=pkey, prompt=prompt,
                          category=cat, is_transgressive=is_transgressive(pkey),
                          displaced=True)
            all_rows.append(result)

            # Induced-suppression baseline
            n = min(len(bl), len(al))
            bp = softmax(bl[:n].astype(np.float64))
            fid = int(np.argmax(bp))
            mp = bp.copy(); mp[fid] = 0; mp /= mp.sum()
            nid = int(np.argmax(mp))
            rk = int(np.where(np.argsort(-bp) == nid)[0][0])
            baseline_rows.append({
                "family": fkey, "prompt_key": pkey, "category": cat,
                "is_transgressive": is_transgressive(pkey),
                "skip_count": max(0, rk - 1),
            })

    return pd.DataFrame(all_rows), pd.DataFrame(baseline_rows)


# ── Tests 2 & 3: semantic relatedness via unembedding ───────────────

def load_unembedding(model_id):
    import torch
    from transformers import AutoModelForCausalLM
    print(f"  Loading unembedding: {model_id}")
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True)
    W = m.lm_head.weight.detach().float().numpy()
    del m; gc.collect()
    return W


def add_semantic_tests(df, families):
    """Add token relatedness and distributional field mass tests."""
    displaced = df[(df["displaced"] == True)].copy()

    for fkey in families:
        fam = MODEL_FAMILIES[fkey]
        sub = displaced[displaced["family"] == fkey]
        if sub.empty:
            continue

        W = load_unembedding(fam.base)
        # Mean-center (removes isotropy/frequency bias)
        W_c = W - W.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(W_c, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1
        W_n = W_c / norms
        V = W.shape[0]

        psyche = Psyche.from_family(fkey)
        aligned = psyche.superego or psyche.ego
        print(f"  {fkey}: {len(sub)} displaced sites")

        for idx, row in sub.iterrows():
            f_id, a_id = int(row["f_id"]), int(row["a_id"])
            if f_id >= V or a_id >= V:
                continue

            # Test 2: token-level relatedness
            cos_a_f = cosine_sim(W_c[a_id], W_c[f_id])
            skipped = [s for s in (row.get("skipped_ids") or [])
                       if isinstance(s, (int, np.integer)) and s < V]
            if skipped:
                sk_cos = [cosine_sim(W_c[s], W_c[f_id]) for s in skipped]
                mean_sk = float(np.mean(sk_cos))
            else:
                mean_sk = np.nan

            df.at[idx, "cos_a_f"] = cos_a_f
            df.at[idx, "mean_cos_skipped_f"] = mean_sk
            df.at[idx, "token_advantage"] = (cos_a_f - mean_sk
                                             if not np.isnan(mean_sk) else np.nan)

            # Test 3: distributional field mass
            try:
                bl = psyche.primary_process.logits(row["prompt"]).numpy()
                al = aligned.logits(row["prompt"]).numpy()
            except Exception:
                continue
            n = min(len(bl), len(al), V)
            base_p = softmax(bl[:n].astype(np.float64))
            al_p = softmax(al[:n].astype(np.float64))

            cos_all = W_n[:n] @ W_n[f_id]
            field = (cos_all > 0.3)
            field[f_id] = False

            if field.sum() == 0:
                continue

            flat_p = base_p.copy()
            flat_p[f_id] = 0
            flat_p /= flat_p.sum()

            df.at[idx, "aligned_field_mass"] = float(al_p[field].sum())
            df.at[idx, "flat_field_mass"] = float(flat_p[field].sum())
            df.at[idx, "field_advantage"] = float(al_p[field].sum() - flat_p[field].sum())
            df.at[idx, "n_field_tokens"] = int(field.sum())

        del W, W_c, W_n
        gc.collect()

    return df


# ── Summary ─────────────────────────────────────────────────────────

def print_summary(df, bl_df):
    from scipy.stats import mannwhitneyu, wilcoxon

    d = df[df["displaced"] == True].copy()
    trans = d[d["is_transgressive"] == True]
    neutral = d[d["category"] == "neutral"]

    print("\n" + "=" * 70)
    print("TEST 1: SKIP COUNT")
    print("Does alignment just pick the next-best base token?")
    print("=" * 70)

    for label, sub in [("transgressive", trans), ("neutral", neutral)]:
        if sub.empty:
            continue
        print(f"\n  {label} (n={len(sub)}):")
        print(f"    skip count — median={sub['skip_count'].median():.0f}  "
              f"mean={sub['skip_count'].mean():.1f}  "
              f"IQR=[{sub['skip_count'].quantile(0.25):.0f}, "
              f"{sub['skip_count'].quantile(0.75):.0f}]")
        print(f"    simple-mask match: {sub['simple_matches'].mean()*100:.1f}%")

    print(f"\n  baseline (mask argmax, renormalize): "
          f"median={bl_df['skip_count'].median():.0f}, "
          f"mean={bl_df['skip_count'].mean():.1f}")

    u, p = mannwhitneyu(trans["skip_count"], bl_df["skip_count"], alternative="greater")
    print(f"\n  Transgressive vs baseline: U={u:.0f}, p={p:.2e}")

    print("\n  Per-family (transgressive):")
    for fam in sorted(trans["family"].unique()):
        s = trans[trans["family"] == fam]
        print(f"    {fam:12s}  n={len(s):3d}  "
              f"skip_med={s['skip_count'].median():5.0f}  "
              f"simple={s['simple_matches'].mean()*100:4.0f}%")

    if "cos_a_f" in d.columns:
        has_token = d.dropna(subset=["token_advantage"])
        t_tok = has_token[has_token["is_transgressive"] == True]

        print("\n" + "=" * 70)
        print("TEST 2: TOKEN-LEVEL SEMANTIC RELATEDNESS")
        print("Is the aligned argmax closer to f than skipped tokens?")
        print("(mean-centered unembedding cosine)")
        print("=" * 70)

        for label, sub in [("transgressive", t_tok),
                           ("all", has_token)]:
            if sub.empty:
                continue
            adv = sub["token_advantage"]
            print(f"\n  {label} (n={len(sub)}):")
            print(f"    cos(a,f):    mean={sub['cos_a_f'].mean():.4f}")
            print(f"    mean skipped: mean={sub['mean_cos_skipped_f'].mean():.4f}")
            print(f"    advantage:   mean={adv.mean():+.4f}  "
                  f"median={adv.median():+.4f}  "
                  f"pct>0={100*(adv>0).mean():.0f}%")

        if len(t_tok) >= 10:
            stat, p = wilcoxon(t_tok["token_advantage"], alternative="greater")
            print(f"\n  Wilcoxon (trans advantage > 0): W={stat:.0f}, p={p:.4f}")

    if "field_advantage" in d.columns:
        has_field = d.dropna(subset=["field_advantage"])
        t_fld = has_field[has_field["is_transgressive"] == True]
        n_fld = has_field[has_field["category"] == "neutral"]

        print("\n" + "=" * 70)
        print("TEST 3: DISTRIBUTIONAL FIELD MASS")
        print("Does alignment shift mass toward or away from f's semantic field?")
        print("field = tokens with unembedding cos > 0.3 to f")
        print("=" * 70)

        for label, sub in [("transgressive", t_fld), ("neutral", n_fld)]:
            if sub.empty:
                continue
            fa = sub["field_advantage"]
            print(f"\n  {label} (n={len(sub)}):")
            print(f"    aligned field mass:  {sub['aligned_field_mass'].mean():.4f}")
            print(f"    flat-sup field mass: {sub['flat_field_mass'].mean():.4f}")
            print(f"    field advantage:     {fa.mean():+.4f}  "
                  f"pct>0={100*(fa>0).mean():.0f}%")

        if len(t_fld) >= 10:
            stat, p = wilcoxon(t_fld["field_advantage"], alternative="greater")
            print(f"\n  Wilcoxon (trans field_adv > 0): W={stat:.0f}, p={p:.4f}")
            stat2, p2 = wilcoxon(t_fld["field_advantage"], alternative="less")
            print(f"  Wilcoxon (trans field_adv < 0): W={stat2:.0f}, p={p2:.4f}")

        print("\n  Per-category:")
        for cat in sorted(has_field["category"].unique()):
            s = has_field[has_field["category"] == cat]
            fa = s["field_advantage"]
            print(f"    {cat:25s}  n={len(s):2d}  "
                  f"adv={fa.mean():+.5f}  pct>0={100*(fa>0).mean():.0f}%")

    # Examples
    print("\n" + "=" * 70)
    print("EXAMPLES (transgressive, highest skip counts)")
    print("=" * 70)

    ex = trans.sort_values("skip_count", ascending=False).head(15)
    for _, r in ex.iterrows():
        extras = []
        if pd.notna(r.get("token_advantage")):
            extras.append(f"tok_adv={r['token_advantage']:+.3f}")
        if pd.notna(r.get("field_advantage")):
            extras.append(f"fld_adv={r['field_advantage']:+.4f}")
        extra_str = "  " + "  ".join(extras) if extras else ""
        p = r['prompt'][:55] + "..." if len(r['prompt']) > 55 else r['prompt']
        print(f"\n  [{r['family']}] {r['prompt_key']}")
        print(f"    \"{p}\"")
        print(f"    {r['f_token']} → {r['a_token']}  "
              f"skip={r['skip_count']:.0f}  "
              f"simple={r['simple_argmax_token']}{extra_str}")


def main():
    parser = argparse.ArgumentParser(description="F01 euphemism vs proximity test")
    parser.add_argument("--families", nargs="+",
                        default=["olmo", "llama", "amber", "qwen", "tulu", "olmo-tiny"])
    parser.add_argument("--embed-families", nargs="+", default=None)
    parser.add_argument("--suppress-threshold", type=float, default=0.5)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("Test 1: skip count analysis")
    df, bl_df = run_skip_analysis(args.families, args.suppress_threshold)

    if not args.no_embeddings:
        embed_fams = args.embed_families or args.families
        print(f"\nTests 2-3: semantic relatedness ({', '.join(embed_fams)})")
        df = add_semantic_tests(df, embed_fams)

    if "skipped_ids" in df.columns:
        df = df.drop(columns=["skipped_ids"])

    print_summary(df, bl_df)

    if args.save:
        out = os.path.join(PATH_DATA, "euphemism_test.csv")
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows to {out}")
        bl = os.path.join(PATH_DATA, "euphemism_baseline.csv")
        bl_df.to_csv(bl, index=False)
        print(f"Saved {len(bl_df)} baseline rows to {bl}")


if __name__ == "__main__":
    main()
