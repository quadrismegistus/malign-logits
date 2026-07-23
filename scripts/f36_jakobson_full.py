"""F36 Jakobson plane: full run with residualized axes on full-API Dolma.

Computes per-site (ParaAdv, SynAdv) → rotated (Axis1=relatedness, Axis2=discriminator)
for all usable (non-polysemous) displaced sites across families.

Then runs Test A: SFT→DPO transgressive vs matched-neutral on Axis1/Axis2
for staged families.

Usage:
    uv run python scripts/f36_jakobson_full.py --save
"""

import argparse
import gc
import math
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import entropy as sp_entropy, mannwhitneyu, pearsonr
from sklearn.linear_model import LinearRegression

from malign_logits import Psyche, MODEL_FAMILIES, PATH_DATA
from malign_logits.experiments import DEFAULT_PROMPTS
from malign_logits.cache import open_stash

# ── Config ──────────────────────────────────────────────────────────

INFIGRAM_URL = "https://api.infini-gram.io/"
INFIGRAM_INDEX = "v4_dolma-v1_7_llama"
N_TOKENS = 2.60e12
MCF = 500000
WINDOW = 100
K_GAINERS = 20
TOO_LARGE = {"meta-llama/Llama-3.1-70B", "allenai/Olmo-3-1125-32B"}

TRANSGRESSIVE_CATS = {
    "sexual_liminal", "sexual_explicit",
    "violence_liminal", "violence_explicit",
    "death", "profanity", "substance",
}

# Scene-mates for functional polysemy screen
SCENE_MATES = {
    "sexual": ["body", "bed", "skin", "fingers", "lips"],
    "violence": ["scream", "blood", "dead", "weapon", "fight"],
    "death": ["body", "dead", "grave", "cold", "life"],
    "profanity": ["angry", "yelled", "frustration", "hell", "damn"],
    "substance": ["drunk", "bottle", "smoke", "high", "glass"],
}

# Regression sample words (common English, stable frequencies)
REG_WORDS = [
    "the", "and", "was", "his", "her", "they", "said", "have", "been",
    "this", "were", "will", "about", "into", "just", "like", "some",
    "time", "when", "come", "make", "know", "take", "good", "give",
    "most", "only", "tell", "back", "work", "first", "want", "way",
    "look", "more", "day", "man", "find", "thing", "well", "long",
    "right", "think", "world", "life", "hand", "part", "eye", "house",
    "place", "water", "room", "name", "old", "new", "big", "small",
    "high", "help", "turn", "door", "head",
]


def prompt_category(key):
    parts = key.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else key


def is_transgressive(key):
    return prompt_category(key) in TRANSGRESSIVE_CATS


def is_whole_word(s):
    return bool(s) and s.isalpha() and len(s) > 1


# ── Infini-gram API with HashStash ──────────────────────────────────

_stash = None

def get_stash():
    global _stash
    if _stash is None:
        _stash = open_stash(os.path.join(PATH_DATA, "raw", "cache", "infigram_api"))
    return _stash


def api_count(query, max_diff=None):
    stash = get_stash()
    is_cnf = " AND " in query
    key = {"q": query, "md": max_diff}
    if is_cnf:
        key["mcf"] = MCF
    if key in stash:
        return stash[key]

    import requests
    payload = {"index": INFIGRAM_INDEX, "query_type": "count", "query": query}
    if is_cnf:
        payload["max_clause_freq"] = MCF
    if max_diff is not None:
        payload["max_diff_tokens"] = max_diff

    for attempt in range(5):
        try:
            r = requests.post(INFIGRAM_URL, json=payload, timeout=60)
            r.raise_for_status()
            c = r.json()["count"]
            stash[key] = c
            return c
        except Exception as e:
            if attempt == 4:
                return None
            time.sleep(min(2 ** attempt, 15))


def ppmi_val(w1, w2, c1=None, c2=None):
    if c1 is None: c1 = api_count(w1)
    if c2 is None: c2 = api_count(w2)
    c12 = api_count(f"{w1} AND {w2}", max_diff=WINDOW)
    if any(x is None for x in [c1, c2, c12]):
        return None
    if c12 == 0 or c1 == 0 or c2 == 0:
        return 0.0
    k = 2 * WINDOW
    return max(0.0, math.log2(c12 * N_TOKENS / (c1 * c2 * k)))


# ── Polysemy screen ────────────────────────────────────────────────

def is_usable_trigger(word, category):
    """Functional screen: does this trigger have PPMI > 0 for any scene-mate?"""
    if not is_whole_word(word):
        return False
    cat_broad = category.split("_")[0]
    mates = SCENE_MATES.get(cat_broad, [])
    if not mates:
        return False
    for m in mates:
        p = ppmi_val(word, m)
        if p is not None and p > 0:
            return True
    return False


# ── Unembedding ─────────────────────────────────────────────────────

def load_unembedding_normed(model_id):
    import torch
    from transformers import AutoModelForCausalLM
    print(f"    loading unembedding: {model_id}")
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True)
    if hasattr(m, "lm_head"):
        W = m.lm_head.weight.detach().float().numpy()
    elif hasattr(m, "embed_out"):
        W = m.embed_out.weight.detach().float().numpy()
    else:
        del m; gc.collect()
        raise AttributeError(f"No lm_head/embed_out")
    del m; gc.collect()
    W -= W.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    W /= norms
    return W


# ── Frequency regression (per forbidden token) ─────────────────────

_reg_freqs = None

def get_reg_freqs():
    global _reg_freqs
    if _reg_freqs is None:
        _reg_freqs = {}
        for w in REG_WORDS:
            c = api_count(w)
            if c and c > 0:
                _reg_freqs[w] = c
        print(f"  Regression sample: {len(_reg_freqs)} words")
    return _reg_freqs


def fit_regression(f_word, f_tid, W_n, tokenizer):
    """Fit cosine ~ log_freq and ppmi ~ log_freq for this forbidden token."""
    f_vec = W_n[f_tid]
    f_c = api_count(f_word)
    if f_c is None:
        return None, None

    reg_freqs = get_reg_freqs()

    def get_tid(word):
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        return ids[0] if ids else None

    lf, cos_arr, ppmi_arr = [], [], []
    for sw, sc in reg_freqs.items():
        if sw == f_word:
            continue
        stid = get_tid(sw)
        if stid is None or stid >= W_n.shape[0]:
            continue
        c = float(np.dot(W_n[stid], f_vec))
        p = ppmi_val(f_word, sw, c1=f_c, c2=sc)
        if p is None:
            continue
        lf.append(math.log(sc + 1))
        cos_arr.append(c)
        ppmi_arr.append(p)

    if len(lf) < 20:
        return None, None

    X = np.array(lf).reshape(-1, 1)
    reg_cos = LinearRegression().fit(X, cos_arr)
    reg_ppmi = LinearRegression().fit(X, ppmi_arr)
    return reg_cos, reg_ppmi


# ── Per-site computation ────────────────────────────────────────────

def compute_site(base_p, aligned_p, W_n, tokenizer, n,
                 f_word, f_tid, reg_cos, reg_ppmi, f_c):
    """Compute residualized ParaAdv, SynAdv, Axis1, Axis2 for one site."""
    delta = aligned_p - base_p
    f_vec = W_n[f_tid]

    def get_tid(word):
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        return ids[0] if ids else None

    def score_gainers(d, p_ref):
        """Score gaining tokens: raw + residualized cosine and PPMI."""
        gainer_ids = np.where(d > 1e-6)[0]
        if len(gainer_ids) == 0:
            return None

        total_mass = float(d[gainer_ids].sum())
        top_idx = np.argsort(-d[gainer_ids])[:K_GAINERS * 3]

        candidates = []
        word_mass = 0.0
        for i in top_idx:
            tid = int(gainer_ids[i])
            if tid >= W_n.shape[0]:
                continue
            word = tokenizer.decode([tid]).strip()
            if is_whole_word(word):
                candidates.append((tid, word, float(d[tid])))
                word_mass += float(d[tid])
            if len(candidates) >= K_GAINERS:
                break

        if not candidates:
            return None

        total_gain = sum(c[2] for c in candidates)
        if total_gain <= 0:
            return None
        weights = [(tid, word, mass / total_gain) for tid, word, mass in candidates]
        coverage = word_mass / total_mass if total_mass > 0 else 0

        para, syn, r_para, r_syn = 0.0, 0.0, 0.0, 0.0
        n_ppmi = 0
        for tid, word, w in weights:
            cos = float(np.dot(W_n[tid], f_vec))
            tc = api_count(word)
            if tc is None or tc == 0:
                continue
            p = ppmi_val(f_word, word, c1=f_c, c2=tc)
            if p is None:
                continue

            lf = math.log(tc + 1)
            rc = cos - reg_cos.predict([[lf]])[0]
            rp = p - reg_ppmi.predict([[lf]])[0]

            para += w * cos
            syn += w * p
            r_para += w * rc
            r_syn += w * rp
            n_ppmi += 1

        return {
            "para": para, "syn": syn,
            "r_para": r_para, "r_syn": r_syn,
            "n_ppmi": n_ppmi, "coverage": coverage,
            "top_words": " ".join(w for _, w, _ in weights[:5]),
        }

    # Aligned gaining tokens
    aligned_scores = score_gainers(delta, aligned_p)
    if aligned_scores is None:
        return None

    # Flat counterfactual
    flat_p = base_p.copy()
    flat_p[int(np.argmax(base_p))] = 0
    flat_p /= flat_p.sum()
    flat_delta = flat_p - base_p
    flat_scores = score_gainers(flat_delta, flat_p)

    if flat_scores is None:
        return None

    para_adv = aligned_scores["r_para"] - flat_scores["r_para"]
    syn_adv = aligned_scores["r_syn"] - flat_scores["r_syn"]

    return {
        "f_word": f_word,
        "a_word": tokenizer.decode([int(np.argmax(aligned_p))]).strip(),
        "para_adv": para_adv,
        "syn_adv": syn_adv,
        "axis1": para_adv + syn_adv,  # relatedness
        "axis2": syn_adv - para_adv,  # discriminator: + = displacement, - = condensation
        "para_raw": aligned_scores["para"],
        "syn_raw": aligned_scores["syn"],
        "n_ppmi": aligned_scores["n_ppmi"],
        "whole_word_coverage": aligned_scores["coverage"],
        "top_gainers": aligned_scores["top_words"],
    }


# ── Main run ────────────────────────────────────────────────────────

def run(families):
    from transformers import AutoTokenizer

    rows = []
    emb_cache = {}
    reg_cache = {}

    for fkey in families:
        fam = MODEL_FAMILIES[fkey]
        if fam.base in TOO_LARGE:
            continue
        psyche = Psyche.from_family(fkey)

        # Determine layers to test
        layers = [("base", psyche.primary_process)]
        aligned = psyche.superego or psyche.ego
        if aligned is None:
            continue
        # For staged families, also compute per-transition
        if psyche.ego is not None and psyche.superego is not None:
            transitions = [
                ("base→aligned", psyche.primary_process, aligned),
            ]
            transitions.append(("base→sft", psyche.primary_process, psyche.ego))
            transitions.append(("sft→dpo", psyche.ego, psyche.superego))
            if psyche.reinforced_superego is not None:
                transitions.append(("dpo→rlvr", psyche.superego, psyche.reinforced_superego))
        else:
            transitions = [("base→aligned", psyche.primary_process, aligned)]

        if fam.base not in emb_cache:
            try:
                emb_cache[fam.base] = load_unembedding_normed(fam.base)
            except Exception as e:
                print(f"  SKIP {fkey}: {e}")
                continue
        W_n = emb_cache[fam.base]
        V = W_n.shape[0]

        tok = AutoTokenizer.from_pretrained(fam.base, trust_remote_code=True)

        def get_tid(word):
            ids = tok.encode(" " + word, add_special_tokens=False)
            return ids[0] if ids else None

        print(f"\n  {fkey} ({', '.join(t[0] for t in transitions)}):")

        for t_name, layer_from, layer_to in transitions:
            n_done, n_skip_poly, n_skip_other = 0, 0, 0

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

                f_id = int(np.argmax(p_from))
                a_id = int(np.argmax(p_to))
                f_word = tok.decode([f_id]).strip()

                if not is_whole_word(f_word):
                    n_skip_other += 1
                    continue

                # Polysemy screen
                poly_key = (f_word, cat)
                if poly_key not in reg_cache:
                    usable = is_usable_trigger(f_word, cat)
                    if usable:
                        reg_c, reg_p = fit_regression(f_word, f_id, W_n, tok)
                        reg_cache[poly_key] = (reg_c, reg_p, api_count(f_word))
                    else:
                        reg_cache[poly_key] = None

                if reg_cache[poly_key] is None:
                    n_skip_poly += 1
                    continue

                reg_cos, reg_ppmi, f_c = reg_cache[poly_key]
                if reg_cos is None:
                    n_skip_other += 1
                    continue

                result = compute_site(p_from, p_to, W_n, tok, n,
                                      f_word, f_id, reg_cos, reg_ppmi, f_c)
                if result is None:
                    n_skip_other += 1
                    continue

                result["family"] = fkey
                result["transition"] = t_name
                result["prompt_key"] = pkey
                result["category"] = cat
                result["is_transgressive"] = is_trans
                result["base_entropy"] = float(sp_entropy(p_from))
                result["displaced"] = (f_id != a_id)
                rows.append(result)
                n_done += 1

            print(f"    {t_name}: {n_done} sites  "
                  f"(poly_skip={n_skip_poly}, other_skip={n_skip_other})")

    emb_cache.clear()
    gc.collect()
    return pd.DataFrame(rows)


# ── Entropy matching ────────────────────────────────────────────────

def match_on_entropy(trans_rows, nontrans_rows, key="base_entropy"):
    if trans_rows.empty or nontrans_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    t_ent = trans_rows[key].values
    n_ent = nontrans_rows[key].values
    used = set()
    t_idx, n_idx = [], []
    for i, te in enumerate(t_ent):
        dists = np.abs(n_ent - te)
        for j in np.argsort(dists):
            if j not in used:
                used.add(j)
                t_idx.append(trans_rows.index[i])
                n_idx.append(nontrans_rows.index[j])
                break
    return trans_rows.loc[t_idx], nontrans_rows.loc[n_idx]


# ── Summary ─────────────────────────────────────────────────────────

def print_summary(df):
    # Noise floor: Axis2 on known-unrelated pairs
    print("\n" + "=" * 70)
    print("NOISE FLOOR (Axis2 on unrelated pairs)")
    print("=" * 70)
    unrelated = [("kill", "banana"), ("kill", "umbrella"), ("kill", "bicycle"),
                 ("blood", "bicycle"), ("blood", "calendar"), ("scream", "banana")]
    nf_vals = []
    for w1, w2 in unrelated:
        p = ppmi_val(w1, w2)
        if p is not None:
            nf_vals.append(p)
    if nf_vals:
        print(f"  PPMI noise floor: mean={np.mean(nf_vals):.3f}, max={max(nf_vals):.3f}")

    ba = df[df["transition"] == "base→aligned"]
    trans = ba[ba["is_transgressive"] == True]

    print("\n" + "=" * 70)
    print("JAKOBSON PLANE (residualized, rotated)")
    print("Axis1 = resid_cos + resid_ppmi (relatedness, low = foreclosure)")
    print("Axis2 = resid_ppmi - resid_cos (+ = displacement, - = condensation)")
    print("=" * 70)

    for label, sub in [("transgressive", trans), ("all", ba)]:
        a1 = sub["axis1"].dropna()
        a2 = sub["axis2"].dropna()
        if a1.empty:
            continue
        r, p = pearsonr(a1, a2) if len(a1) > 5 else (0, 1)
        wc = sub["whole_word_coverage"].dropna()
        print(f"\n  {label} (n={len(a1)}):")
        print(f"    Axis1: mean={a1.mean():+.4f}  median={a1.median():+.4f}")
        print(f"    Axis2: mean={a2.mean():+.4f}  median={a2.median():+.4f}  pct>0={100*(a2>0).mean():.0f}%")
        print(f"    Axis1-Axis2 correlation: r={r:.3f} (p={p:.3f})")
        if not wc.empty:
            print(f"    whole-word coverage: mean={wc.mean():.2f}")

    # Per-family
    print("\n  Per-family (transgressive base→aligned, sorted by Axis2):")
    fam_stats = []
    for fam in sorted(trans["family"].unique()):
        s = trans[trans["family"] == fam]
        a2 = s["axis2"].dropna()
        a1 = s["axis1"].dropna()
        if a2.empty:
            continue
        fam_stats.append((fam, len(a2), a2.mean(), a1.mean(), (a2 > 0).mean()))
    fam_stats.sort(key=lambda x: x[2])
    for fam, n, a2m, a1m, a2pct in fam_stats:
        print(f"    {fam:20s}  n={n:3d}  Axis2={a2m:+.4f}  "
              f"Axis1={a1m:+.4f}  disp>0={a2pct*100:.0f}%")

    # Per-category
    print("\n  Per-category (transgressive):")
    for cat in sorted(trans["category"].unique()):
        s = trans[trans["category"] == cat]
        a2 = s["axis2"].dropna()
        a1 = s["axis1"].dropna()
        if a2.empty:
            continue
        print(f"    {cat:25s}  n={len(a2):3d}  Axis2={a2.mean():+.5f}  "
              f"Axis1={a1.mean():+.5f}")

    # 2D quadrant classification
    print("\n  Quadrant classification (transgressive):")
    a1 = trans["axis1"].dropna()
    a2 = trans["axis2"].dropna()
    both = trans.dropna(subset=["axis1", "axis2"])
    n = len(both)
    if n > 0:
        q_disp = ((both.axis2 > 0) & (both.axis1 > 0)).sum()
        q_cond = ((both.axis2 < 0) & (both.axis1 > 0)).sum()
        q_fore = (both.axis1 < 0).sum()
        print(f"    displacement (Ax1>0, Ax2>0): {q_disp:3d} ({100*q_disp/n:.0f}%)")
        print(f"    condensation (Ax1>0, Ax2<0): {q_cond:3d} ({100*q_cond/n:.0f}%)")
        print(f"    foreclosure  (Ax1<0):        {q_fore:3d} ({100*q_fore/n:.0f}%)")

    # ── Test A: stage-decomposed ────────────────────────────────────
    staged = df[df["transition"].isin(["base→sft", "sft→dpo", "dpo→rlvr"])]
    if staged.empty:
        return

    print("\n" + "=" * 70)
    print("TEST A: STAGE-DECOMPOSED (Axis2, transgressive vs matched-neutral)")
    print("=" * 70)

    for fam in sorted(staged["family"].unique()):
        print(f"\n  {fam}:")
        for trans_name in ["base→sft", "sft→dpo", "dpo→rlvr"]:
            sub = staged[(staged["family"] == fam) & (staged["transition"] == trans_name)]
            t = sub[sub["is_transgressive"] == True]
            nt = sub[sub["is_transgressive"] == False]
            if t.empty or nt.empty:
                continue

            mt, mn = match_on_entropy(t, nt)
            if mt.empty:
                continue

            t_vals = mt["axis2"].dropna()
            n_vals = mn["axis2"].dropna()
            if len(t_vals) < 3 or len(n_vals) < 3:
                continue

            diff = t_vals.mean() - n_vals.mean()
            u, p = mannwhitneyu(t_vals, n_vals, alternative="two-sided")
            sig = "*" if p < 0.05 else ""
            print(f"    {trans_name:12s}  trans={t_vals.mean():+.4f} (n={len(t_vals)})  "
                  f"neutral={n_vals.mean():+.4f} (n={len(n_vals)})  "
                  f"diff={diff:+.4f}  p={p:.4f}{sig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+",
                        default=["olmo", "llama", "amber", "qwen", "qwen3",
                                 "tulu", "olmo-tiny", "zephyr"])
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("Jakobson plane: full API, residualized, rotated")
    print(f"Families: {args.families}")

    df = run(args.families)
    print_summary(df)

    if args.save:
        out = os.path.join(PATH_DATA, "jakobson_plane_full.csv")
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
