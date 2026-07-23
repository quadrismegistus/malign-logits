"""F36 follow-up: stage-decomposed and per-family transgression-specificity.

Tests A–D from the brief. The decisive output is Test A: does the
DPO-stage field_adv differ between transgressive and matched-neutral prompts?

Usage:
    uv run python scripts/f36_stage_specificity.py --save
"""

import argparse
import gc
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import mannwhitneyu, wilcoxon, entropy as sp_entropy

from malign_logits import Psyche, MODEL_FAMILIES, PATH_DATA
from malign_logits.experiments import DEFAULT_PROMPTS


TRANSGRESSIVE_CATS = {
    "sexual_liminal", "sexual_explicit",
    "violence_liminal", "violence_explicit",
    "death", "profanity", "substance",
}
SAFETY_CATS = {
    "sexual_liminal", "sexual_explicit",
    "violence_liminal", "violence_explicit",
}
TOO_LARGE = {"meta-llama/Llama-3.1-70B", "allenai/Olmo-3-1125-32B"}


def prompt_category(key):
    parts = key.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else key


def is_transgressive(key):
    return prompt_category(key) in TRANSGRESSIVE_CATS


# ── Unembedding loading ─────────────────────────────────────────────

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
        raise AttributeError(f"No lm_head/embed_out on {type(m).__name__}")
    del m; gc.collect()
    W -= W.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    W /= norms
    return W


# ── Core mass-flow computation ──────────────────────────────────────

def mass_flow_advantage(earlier_p, later_p, W_n, n, thr=1e-4):
    """Field advantage of later vs mask-and-renormalize of earlier.

    Returns continuous field_adv: negative = flees field beyond mechanical
    suppression, positive = moves toward field.
    """
    delta = later_p - earlier_p
    losers = delta < -thr
    gainers = delta > thr
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

    fp = earlier_p.copy(); fp[losers] = 0; fp /= fp.sum()
    fd = fp - earlier_p; fg = fd > thr
    if fg.sum() == 0:
        return np.nan
    fw = fd[fg]; fw /= fw.sum()
    f_cos = float(fw @ (W_n[:n][fg] @ centroid))

    return g_cos - f_cos


# ── Entropy matching ────────────────────────────────────────────────

def match_on_entropy(trans_rows, nontrans_rows):
    """1:1 nearest-entropy matching of transgressive to non-transgressive."""
    if not trans_rows or not nontrans_rows:
        return [], []
    t_ent = np.array([r['base_entropy'] for r in trans_rows])
    n_ent = np.array([r['base_entropy'] for r in nontrans_rows])
    used = set()
    matched_t, matched_n = [], []
    for i, te in enumerate(t_ent):
        dists = np.abs(n_ent - te)
        for j in np.argsort(dists):
            if j not in used:
                used.add(j)
                matched_t.append(trans_rows[i])
                matched_n.append(nontrans_rows[j])
                break
    return matched_t, matched_n


def compare_groups(trans_vals, neutral_vals, label=""):
    """Report comparison between two groups of field_adv values."""
    tv = np.array([v for v in trans_vals if not np.isnan(v)])
    nv = np.array([v for v in neutral_vals if not np.isnan(v)])
    if len(tv) < 3 or len(nv) < 3:
        return None
    diff = tv.mean() - nv.mean()
    u, p = mannwhitneyu(tv, nv, alternative='two-sided')
    return {
        'label': label,
        'trans_mean': float(tv.mean()),
        'neutral_mean': float(nv.mean()),
        'diff': diff,
        'trans_n': len(tv),
        'neutral_n': len(nv),
        'U': float(u),
        'p': float(p),
    }


# ── TEST A: Stage-decomposed transgression-specificity ──────────────

def test_a(staged_families):
    """For each staged family, compare field_adv transgressive vs
    matched-neutral at each transition."""
    print("\n" + "=" * 70)
    print("TEST A: STAGE-DECOMPOSED TRANSGRESSION-SPECIFICITY")
    print("=" * 70)

    all_results = []
    emb_cache = {}

    for fkey in staged_families:
        fam = MODEL_FAMILIES[fkey]
        if fam.base in TOO_LARGE:
            continue
        psyche = Psyche.from_family(fkey)

        # Load unembedding (cache across families sharing a base)
        if fam.base not in emb_cache:
            try:
                emb_cache[fam.base] = load_unembedding_normed(fam.base)
            except Exception as e:
                print(f"  SKIP {fkey}: {e}")
                continue
        W_n = emb_cache[fam.base]
        V = W_n.shape[0]

        # Collect layers
        layers = [('base', psyche.primary_process)]
        if psyche.ego is not None:
            layers.append(('sft', psyche.ego))
        if psyche.superego is not None:
            layers.append(('dpo', psyche.superego))
        if psyche.reinforced_superego is not None:
            layers.append(('rlvr', psyche.reinforced_superego))

        transitions = [(layers[i][0], layers[i+1][0],
                         layers[i][1], layers[i+1][1])
                        for i in range(len(layers) - 1)]

        print(f"\n  {fkey} ({' → '.join(l[0] for l in layers)})")

        for t_from, t_to, layer_from, layer_to in transitions:
            trans_rows, nontrans_rows = [], []

            for pkey, prompt in DEFAULT_PROMPTS.items():
                cat = prompt_category(pkey)
                is_trans = is_transgressive(pkey)
                try:
                    l_from = layer_from.logits(prompt).numpy()
                    l_to = layer_to.logits(prompt).numpy()
                except Exception:
                    continue

                n = min(len(l_from), len(l_to), V)
                p_from = softmax(l_from[:n].astype(np.float64))
                p_to = softmax(l_to[:n].astype(np.float64))

                # Base entropy for matching (always from the earlier stage)
                base_ent = float(sp_entropy(p_from))

                adv = mass_flow_advantage(p_from, p_to, W_n, n)

                # Displacement at this transition?
                displaced = int(np.argmax(p_from)) != int(np.argmax(p_to))

                row = dict(
                    family=fkey, transition=f"{t_from}→{t_to}",
                    prompt_key=pkey, category=cat,
                    is_transgressive=is_trans,
                    base_entropy=base_ent,
                    field_adv=adv,
                    displaced=displaced,
                )
                if is_trans:
                    trans_rows.append(row)
                else:
                    nontrans_rows.append(row)

            # Entropy-matched comparison
            mt, mn = match_on_entropy(trans_rows, nontrans_rows)
            if mt and mn:
                t_vals = [r['field_adv'] for r in mt]
                n_vals = [r['field_adv'] for r in mn]
                result = compare_groups(t_vals, n_vals,
                                        f"{fkey} {t_from}→{t_to}")
                if result:
                    result['family'] = fkey
                    result['transition'] = f"{t_from}→{t_to}"
                    all_results.append(result)
                    sig = "*" if result['p'] < 0.05 else ""
                    print(f"    {t_from}→{t_to}:  "
                          f"trans={result['trans_mean']:+.4f} (n={result['trans_n']})  "
                          f"neutral={result['neutral_mean']:+.4f} (n={result['neutral_n']})  "
                          f"diff={result['diff']:+.4f}  p={result['p']:.4f}{sig}")

            # Also: per-category breakdown for SFT→DPO
            if t_from in ('sft', 'ego') and t_to in ('dpo', 'superego'):
                print(f"      Per-category ({t_from}→{t_to}):")
                for cat in sorted(set(r['category'] for r in trans_rows)):
                    cat_vals = [r['field_adv'] for r in trans_rows
                                if r['category'] == cat and not np.isnan(r['field_adv'])]
                    if cat_vals:
                        print(f"        {cat:25s}  n={len(cat_vals):2d}  "
                              f"mean={np.mean(cat_vals):+.5f}")

            all_results.extend(trans_rows)
            all_results.extend(nontrans_rows)

    # Free unembedding cache
    emb_cache.clear()
    gc.collect()

    return [r for r in all_results if 'transition' in r and 'trans_mean' in r]


# ── TEST B: Per-family transgression-specificity (base→aligned) ─────

def test_b(target_families):
    """Base→final-aligned field_adv, transgressive vs matched-neutral."""
    print("\n" + "=" * 70)
    print("TEST B: PER-FAMILY TRANSGRESSION-SPECIFICITY (base→aligned)")
    print("=" * 70)

    results = []
    for fkey in target_families:
        fam = MODEL_FAMILIES[fkey]
        if fam.base in TOO_LARGE:
            continue
        psyche = Psyche.from_family(fkey)
        aligned = psyche.superego or psyche.ego
        if aligned is None:
            continue

        try:
            W_n = load_unembedding_normed(fam.base)
        except Exception as e:
            print(f"  SKIP {fkey}: {e}")
            continue
        V = W_n.shape[0]

        trans_rows, nontrans_rows = [], []
        for pkey, prompt in DEFAULT_PROMPTS.items():
            cat = prompt_category(pkey)
            is_trans = is_transgressive(pkey)
            try:
                bl = psyche.primary_process.logits(prompt).numpy()
                al = aligned.logits(prompt).numpy()
            except Exception:
                continue

            n = min(len(bl), len(al), V)
            bp = softmax(bl[:n].astype(np.float64))
            ap = softmax(al[:n].astype(np.float64))
            base_ent = float(sp_entropy(bp))
            adv = mass_flow_advantage(bp, ap, W_n, n)

            row = dict(family=fkey, prompt_key=pkey, category=cat,
                       is_transgressive=is_trans, base_entropy=base_ent,
                       field_adv=adv)
            (trans_rows if is_trans else nontrans_rows).append(row)

        del W_n; gc.collect()

        mt, mn = match_on_entropy(trans_rows, nontrans_rows)
        if mt and mn:
            result = compare_groups(
                [r['field_adv'] for r in mt],
                [r['field_adv'] for r in mn],
                fkey)
            if result:
                results.append(result)
                sig = "*" if result['p'] < 0.05 else ""
                print(f"  {fkey:25s}  "
                      f"trans={result['trans_mean']:+.4f} (n={result['trans_n']})  "
                      f"neutral={result['neutral_mean']:+.4f} (n={result['neutral_n']})  "
                      f"diff={result['diff']:+.4f}  p={result['p']:.4f}{sig}")

    return results


# ── TEST C: Displacement rate transgressive vs neutral ──────────────

def test_c():
    """Fraction of sites where argmax moves, transgressive vs neutral."""
    print("\n" + "=" * 70)
    print("TEST C: DISPLACEMENT RATE (argmax moves, base→aligned)")
    print("=" * 70)

    census = pd.read_csv(os.path.join(PATH_DATA, "euphemism_census.csv"))

    overall_trans = census[census['is_transgressive'] == True]
    overall_neutral = census[census['category'] == 'neutral']
    overall_nontrans = census[census['is_transgressive'] == False]

    t_rate = overall_trans['displaced'].mean()
    n_rate = overall_neutral['displaced'].mean()
    nt_rate = overall_nontrans['displaced'].mean()

    print(f"\n  Overall:")
    print(f"    transgressive:     {t_rate*100:.1f}% displaced "
          f"({overall_trans['displaced'].sum()}/{len(overall_trans)})")
    print(f"    neutral:           {n_rate*100:.1f}% displaced "
          f"({overall_neutral['displaced'].sum()}/{len(overall_neutral)})")
    print(f"    non-transgressive: {nt_rate*100:.1f}% displaced "
          f"({overall_nontrans['displaced'].sum()}/{len(overall_nontrans)})")

    from scipy.stats import fisher_exact
    ct = pd.crosstab(census['is_transgressive'], census['displaced'])
    odds, p = fisher_exact(ct)
    print(f"    Fisher exact (trans vs non-trans): OR={odds:.2f}, p={p:.4f}")

    print(f"\n  Per-family (transgressive displacement rate):")
    fam_rates = []
    for fam in sorted(census['family'].unique()):
        sub = census[census['family'] == fam]
        t = sub[sub['is_transgressive'] == True]
        nt = sub[sub['is_transgressive'] == False]
        if len(t) > 0 and len(nt) > 0:
            tr = t['displaced'].mean()
            nr = nt['displaced'].mean()
            fam_rates.append((fam, tr, nr, tr - nr, len(t), len(nt)))

    fam_rates.sort(key=lambda x: -x[3])
    for fam, tr, nr, diff, nt, nnt in fam_rates:
        sig = "*" if abs(diff) > 0.15 else ""
        print(f"    {fam:25s}  trans={tr*100:5.1f}%  "
              f"nontrans={nr*100:5.1f}%  diff={diff*100:+5.1f}pp{sig}")


# ── TEST D: Robustness ──────────────────────────────────────────────

def test_d():
    """Threshold sweep and reporting hygiene."""
    print("\n" + "=" * 70)
    print("TEST D: ROBUSTNESS — THRESHOLD SWEEP")
    print("=" * 70)

    # Use OLMo for the threshold sweep (it's the reference family)
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        'allenai/Olmo-3-1025-7B', dtype=torch.float16,
        device_map='cpu', low_cpu_mem_usage=True)
    W = m.lm_head.weight.detach().float().numpy()
    del m; gc.collect()
    W_raw = W.copy()
    W -= W.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    W /= norms
    V = W.shape[0]

    psyche = Psyche.from_family('olmo')
    aligned = psyche.superego

    for threshold in [0.2, 0.3, 0.4, 0.5]:
        trans_advs, neutral_advs = [], []
        for pkey, prompt in DEFAULT_PROMPTS.items():
            cat = prompt_category(pkey)
            is_trans = is_transgressive(pkey)
            try:
                bl = psyche.primary_process.logits(prompt).numpy()
                al = aligned.logits(prompt).numpy()
            except:
                continue
            n = min(len(bl), len(al), V)
            bp = softmax(bl[:n].astype(np.float64))
            ap = softmax(al[:n].astype(np.float64))

            f_id = int(np.argmax(bp))
            cos_all = W[:n] @ W[f_id]
            field = cos_all > threshold
            field[f_id] = False
            if field.sum() == 0:
                continue

            flat_p = bp.copy(); flat_p[f_id] = 0; flat_p /= flat_p.sum()
            adv = float(ap[field].sum() - flat_p[field].sum())

            if is_trans:
                trans_advs.append(adv)
            elif cat == 'neutral':
                neutral_advs.append(adv)

        ta = np.array(trans_advs)
        print(f"\n  cos > {threshold}: OLMo transgressive (n={len(ta)})")
        print(f"    mean={ta.mean():+.5f}  median={np.median(ta):+.5f}  "
              f"pct>0={100*(ta>0).mean():.0f}%  "
              f"IQR=[{np.percentile(ta, 25):+.5f}, {np.percentile(ta, 75):+.5f}]")

    del W, W_raw; gc.collect()

    # Confirm the 68/64 was displaced-only
    print("\n  Confirming foreclosure conditioning:")
    census = pd.read_csv(os.path.join(PATH_DATA, "euphemism_census.csv"))
    olmo_all = census[census['family'] == 'olmo']
    olmo_disp = olmo_all[olmo_all['displaced'] == True]
    print(f"    OLMo all sites: {len(olmo_all)}, displaced: {len(olmo_disp)}")
    if 'mass_flow_advantage' in olmo_disp.columns:
        mfa = olmo_disp.dropna(subset=['mass_flow_advantage'])
        t = mfa[mfa['is_transgressive'] == True]['mass_flow_advantage']
        nt = mfa[mfa['is_transgressive'] == False]['mass_flow_advantage']
        print(f"    Displaced-only mass_flow_adv<0 rate:")
        print(f"      trans: {(t<0).mean()*100:.0f}% (n={len(t)})")
        print(f"      nontrans: {(nt<0).mean()*100:.0f}% (n={len(nt)})")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    # Staged families (have ego/SFT checkpoint with cached logits)
    test_prompt = list(DEFAULT_PROMPTS.values())[0]
    staged = []
    for fkey in sorted(MODEL_FAMILIES.keys()):
        fam = MODEL_FAMILIES[fkey]
        if fam.ego is None or fam.base in TOO_LARGE:
            continue
        try:
            p = Psyche.from_family(fkey)
            if (p.primary_process._cache.has_logits(fam.base, test_prompt) and
                p.ego._cache.has_logits(fam.ego, test_prompt)):
                staged.append(fkey)
        except:
            pass

    print(f"Staged families: {staged}")

    # Test A
    test_a_results = test_a(staged)

    # Test B — key families
    test_b_families = ['olmo', 'olmo-tiny', 'llama', 'qwen', 'amber',
                       'qwen3', 'tulu', 'zephyr', 'falcon3-7b', 'bloom']
    test_b_results = test_b(test_b_families)

    # Test C
    test_c()

    # Test D
    test_d()


if __name__ == "__main__":
    main()
