"""F36 census: euphemism vs. proximity across all 44 cached families.

Pass 1 (no model loading): skip count + simple-mask match for all families.
Pass 2 (sequential model loading): mass-flow test with unembedding matrices.

Skips 32B/70B for unembedding (too large for CPU). Shares unembedding
matrices across families with the same base model.

Usage:
    uv run python scripts/euphemism_census.py --save
    uv run python scripts/euphemism_census.py --skip-only --save   # pass 1 only
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

TOO_LARGE = {"meta-llama/Llama-3.1-70B", "allenai/Olmo-3-1125-32B"}


def prompt_category(key):
    parts = key.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else key


def is_transgressive(key):
    return prompt_category(key) in TRANSGRESSIVE_CATS


# ── Pass 1: skip count (no models) ─────────────────────────────────

def run_skip_pass(families):
    from transformers import AutoTokenizer

    rows = []
    baseline_rows = []

    for fkey in families:
        fam = MODEL_FAMILIES[fkey]
        psyche = Psyche.from_family(fkey)
        aligned = psyche.superego or psyche.ego
        if aligned is None:
            continue

        try:
            tok = AutoTokenizer.from_pretrained(fam.base, trust_remote_code=True)
        except Exception as e:
            print(f"  {fkey:25s}  SKIP (tokenizer: {e})")
            continue
        n_prompts = 0

        for pkey, prompt in DEFAULT_PROMPTS.items():
            cat = prompt_category(pkey)
            try:
                bl = psyche.primary_process.logits(prompt).numpy()
                al = aligned.logits(prompt).numpy()
            except Exception:
                continue

            n = min(len(bl), len(al))
            base_p = softmax(bl[:n].astype(np.float64))
            al_p = softmax(al[:n].astype(np.float64))

            f_id = int(np.argmax(base_p))
            a_id = int(np.argmax(al_p))

            if f_id == a_id:
                rows.append(dict(
                    family=fkey, prompt_key=pkey, prompt=prompt,
                    category=cat, is_transgressive=is_transgressive(pkey),
                    displaced=False))
                continue

            ranking = np.argsort(-base_p)
            a_rank = int(np.where(ranking == a_id)[0][0])

            simple_p = base_p.copy()
            simple_p[f_id] = 0
            simple_p /= simple_p.sum()
            simple_id = int(np.argmax(simple_p))

            rows.append(dict(
                family=fkey, prompt_key=pkey, prompt=prompt,
                category=cat, is_transgressive=is_transgressive(pkey),
                displaced=True,
                f_id=f_id, f_token=tok.decode([f_id]).strip(),
                f_prob_base=float(base_p[f_id]),
                a_id=a_id, a_token=tok.decode([a_id]).strip(),
                a_prob_base=float(base_p[a_id]),
                a_base_rank=a_rank,
                skip_count=max(0, a_rank - 1),
                simple_matches=(simple_id == a_id),
                simple_argmax_token=tok.decode([simple_id]).strip(),
            ))

            # Baseline
            mp = base_p.copy(); mp[f_id] = 0; mp /= mp.sum()
            nid = int(np.argmax(mp))
            nrk = int(np.where(ranking == nid)[0][0])
            baseline_rows.append(dict(
                family=fkey, prompt_key=pkey, category=cat,
                is_transgressive=is_transgressive(pkey),
                skip_count=max(0, nrk - 1)))
            n_prompts += 1

        print(f"  {fkey:25s}  {n_prompts:3d} prompts")

    return pd.DataFrame(rows), pd.DataFrame(baseline_rows)


# ── Pass 2: mass-flow (needs unembedding) ───────────────────────────

def load_unembedding_normed(model_id):
    import torch
    from transformers import AutoModelForCausalLM
    print(f"    loading {model_id}...")
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True)
    if hasattr(m, 'lm_head'):
        W = m.lm_head.weight.detach().float().numpy()
    elif hasattr(m, 'embed_out'):
        W = m.embed_out.weight.detach().float().numpy()
    else:
        del m; gc.collect()
        raise AttributeError(f"No lm_head or embed_out on {type(m).__name__}")
    del m; gc.collect()
    W -= W.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    W /= norms
    return W


def mass_flow_advantage(base_p, al_p, W_n, n, threshold=1e-4):
    delta = al_p - base_p
    losers = delta < -threshold
    gainers = delta > threshold
    if losers.sum() == 0 or gainers.sum() == 0:
        return np.nan

    lw = -delta[losers]; lw /= lw.sum()
    centroid = W_n[:n][losers].T @ lw
    cn = np.linalg.norm(centroid)
    if cn < 1e-10:
        return np.nan
    centroid /= cn

    gw = delta[gainers]; gw /= gw.sum()
    g_cos = float(gw @ (W_n[:n][gainers] @ centroid))

    fp = base_p.copy(); fp[losers] = 0; fp /= fp.sum()
    fd = fp - base_p
    fg = fd > threshold
    if fg.sum() == 0:
        return np.nan
    fw = fd[fg]; fw /= fw.sum()
    f_cos = float(fw @ (W_n[:n][fg] @ centroid))

    return g_cos - f_cos


def run_mass_flow_pass(df, families):
    # Group families by base model to avoid reloading
    base_groups = {}
    for fkey in families:
        fam = MODEL_FAMILIES[fkey]
        if fam.base in TOO_LARGE:
            continue
        base_groups.setdefault(fam.base, []).append(fkey)

    print(f"  {len(base_groups)} unique base models to load")

    for base_id, fkeys in base_groups.items():
        try:
            W_n = load_unembedding_normed(base_id)
        except Exception as e:
            print(f"    SKIP {base_id}: {e}")
            continue
        V = W_n.shape[0]

        for fkey in fkeys:
            psyche = Psyche.from_family(fkey)
            aligned = psyche.superego or psyche.ego
            if aligned is None:
                continue

            mask = (df["family"] == fkey) & (df["displaced"] == True)
            n_sites = mask.sum()
            n_computed = 0

            for idx in df[mask].index:
                row = df.loc[idx]
                try:
                    bl = psyche.primary_process.logits(row["prompt"]).numpy()
                    al = aligned.logits(row["prompt"]).numpy()
                except Exception:
                    continue

                n = min(len(bl), len(al), V)
                base_p = softmax(bl[:n].astype(np.float64))
                al_p = softmax(al[:n].astype(np.float64))

                adv = mass_flow_advantage(base_p, al_p, W_n, n)
                df.at[idx, "mass_flow_advantage"] = adv
                n_computed += 1

            print(f"    {fkey:25s}  {n_computed}/{n_sites} sites")

        del W_n
        gc.collect()

    return df


# ── Summary ─────────────────────────────────────────────────────────

def print_summary(df, bl_df):
    from scipy.stats import mannwhitneyu, wilcoxon

    d = df[df["displaced"] == True].copy()
    trans = d[d["is_transgressive"] == True]

    print("\n" + "=" * 70)
    print(f"SKIP COUNT CENSUS ({len(d['family'].unique())} families, "
          f"{len(d)} displaced sites)")
    print("=" * 70)

    t_sk = trans["skip_count"]
    bl_sk = bl_df["skip_count"]
    print(f"\n  Transgressive (n={len(trans)}):")
    print(f"    skip: median={t_sk.median():.0f}  mean={t_sk.mean():.1f}  "
          f"IQR=[{t_sk.quantile(0.25):.0f}, {t_sk.quantile(0.75):.0f}]")
    print(f"    simple-mask match: {trans['simple_matches'].mean()*100:.1f}%")
    print(f"  Baseline: median={bl_sk.median():.0f}  mean={bl_sk.mean():.1f}")

    u, p = mannwhitneyu(t_sk, bl_sk, alternative="greater")
    print(f"  Trans vs baseline: U={u:.0f}, p={p:.2e}")

    print("\n  Per-family (transgressive, sorted by median skip):")
    fam_stats = []
    for fam in sorted(trans["family"].unique()):
        s = trans[trans["family"] == fam]
        fam_stats.append((fam, len(s), s["skip_count"].median(),
                          s["skip_count"].mean(),
                          s["simple_matches"].mean()))
    fam_stats.sort(key=lambda x: -x[2])
    for fam, n, med, mean, sm in fam_stats:
        print(f"    {fam:25s}  n={n:3d}  skip_med={med:6.0f}  "
              f"skip_mean={mean:7.1f}  simple={sm*100:4.0f}%")

    if "mass_flow_advantage" in d.columns:
        has_mf = d.dropna(subset=["mass_flow_advantage"])
        t_mf = has_mf[has_mf["is_transgressive"] == True]

        print("\n" + "=" * 70)
        print(f"MASS-FLOW CENSUS ({len(has_mf['family'].unique())} families, "
              f"{len(t_mf)} transgressive sites)")
        print("adv > 0 = euphemistic (mass toward losers' field)")
        print("adv < 0 = anti-euphemistic (mass flees the field)")
        print("=" * 70)

        mfa = t_mf["mass_flow_advantage"]
        print(f"\n  All transgressive: mean={mfa.mean():+.4f}  "
              f"median={mfa.median():+.4f}  pct>0={100*(mfa>0).mean():.0f}%")

        if len(mfa) >= 20:
            stat, p = wilcoxon(mfa, alternative="less")
            print(f"  Wilcoxon adv < 0: W={stat:.0f}, p={p:.6f}")

        print("\n  Per-family (sorted by mean advantage):")
        fam_mf = []
        for fam in sorted(t_mf["family"].unique()):
            s = t_mf[t_mf["family"] == fam]
            a = s["mass_flow_advantage"]
            fam_mf.append((fam, len(s), a.mean(), (a > 0).mean()))
        fam_mf.sort(key=lambda x: x[2])
        for fam, n, mean, pct in fam_mf:
            bar = "█" * int(abs(mean) * 200) if abs(mean) > 0.001 else ""
            sign = "+" if mean > 0 else "-" if mean < 0 else " "
            print(f"    {fam:25s}  n={n:3d}  adv={mean:+.4f}  "
                  f"pct>0={pct*100:4.0f}%  {sign}{bar}")

        print("\n  Per-category (transgressive only):")
        for cat in sorted(t_mf["category"].unique()):
            s = t_mf[t_mf["category"] == cat]
            a = s["mass_flow_advantage"]
            print(f"    {cat:25s}  n={len(s):4d}  adv={a.mean():+.5f}  "
                  f"pct>0={100*(a>0).mean():.0f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-only", action="store_true",
                        help="Run only pass 1 (skip counts, no model loading)")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    # Find all families with cached logits
    test_prompt = list(DEFAULT_PROMPTS.values())[0]
    families = []
    for fkey in sorted(MODEL_FAMILIES.keys()):
        try:
            p = Psyche.from_family(fkey)
            aligned = p.superego or p.ego
            if aligned is None:
                continue
            if (p.primary_process._cache and
                p.primary_process._cache.has_logits(
                    MODEL_FAMILIES[fkey].base, test_prompt) and
                aligned._cache and
                aligned._cache.has_logits(aligned.model_id, test_prompt)):
                families.append(fkey)
        except Exception:
            pass

    print(f"Pass 1: skip counts for {len(families)} families")
    df, bl_df = run_skip_pass(families)

    if not args.skip_only:
        print(f"\nPass 2: mass-flow for ≤10B families")
        df = run_mass_flow_pass(df, families)

    print_summary(df, bl_df)

    if args.save:
        out = os.path.join(PATH_DATA, "euphemism_census.csv")
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows to {out}")
        bl = os.path.join(PATH_DATA, "euphemism_census_baseline.csv")
        bl_df.to_csv(bl, index=False)
        print(f"Saved {len(bl_df)} rows to {bl}")


if __name__ == "__main__":
    main()
