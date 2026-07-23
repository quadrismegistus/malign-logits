"""F36 Jakobson plane: paradigmatic (cosine) × syntagmatic (PPMI) displacement.

Uses infini-gram API (Dolma v1.7, Llama-2 tokenizer) for PPMI co-occurrence
and the model's own unembedding matrix for cosine. Reuses F36 displaced sites.

Usage:
    uv run python scripts/f36_jakobson_plane.py --save
    uv run python scripts/f36_jakobson_plane.py --families olmo llama
    uv run python scripts/f36_jakobson_plane.py --calibrate   # sanity-check PPMI pairs
"""

import argparse
import gc
import json
import math
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import softmax

from malign_logits import Psyche, MODEL_FAMILIES, PATH_DATA
from malign_logits.experiments import DEFAULT_PROMPTS

# ── Config ──────────────────────────────────────────────────────────

INFIGRAM_URL = "https://api.infini-gram.io/"
INFIGRAM_INDEX = "v4_dolma-v1_7_llama"
N_TOKENS = 2.60e12     # Dolma v1.7 total tokens
MAX_CLAUSE_FREQ = 500000
DEFAULT_WINDOW = 100
K_GAINERS = 20          # top-K gaining tokens per site
TOO_LARGE = {"meta-llama/Llama-3.1-70B", "allenai/Olmo-3-1125-32B"}

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


def is_whole_word(s):
    return bool(s) and s.isalpha() and len(s) > 1


# ── Infini-gram backend (API or local engine) ──────────────────────

_infigram_stash = None
_infigram_backend = "api"  # "api" or "local"
_local_engine = None
_local_tokenizer = None
_local_N = None

LOCAL_INDEX_DIR = os.path.join(PATH_DATA, "raw", "infigram", "v4_dolmasample_olmo")
LOCAL_TOKENIZER = "allenai/gpt-neox-olmo-dolma-v1_5"


def _get_stash():
    global _infigram_stash
    if _infigram_stash is None:
        from malign_logits.cache import open_stash
        suffix = "infigram_local" if _infigram_backend == "local" else "infigram_api"
        _infigram_stash = open_stash(
            os.path.join(PATH_DATA, "raw", "cache", suffix))
    return _infigram_stash


def _init_local():
    global _local_engine, _local_tokenizer, _local_N
    if _local_engine is not None:
        return
    from infini_gram.engine import InfiniGramEngine
    from transformers import AutoTokenizer
    print(f"  Loading local infini-gram index from {LOCAL_INDEX_DIR}...")
    _local_tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_TOKENIZER, add_bos_token=False, add_eos_token=False,
        trust_remote_code=True)
    _local_engine = InfiniGramEngine(
        index_dir=LOCAL_INDEX_DIR,
        eos_token_id=_local_tokenizer.eos_token_id)
    _local_N = _local_engine.count(input_ids=[])["count"]
    print(f"  Local index: {_local_N:,} tokens")


def use_local_backend():
    global _infigram_backend, _infigram_stash
    _infigram_backend = "local"
    _infigram_stash = None  # reset stash to pick up new suffix


def infigram_count(query, max_diff=None, retries=5):
    """Query infini-gram with HashStash cache. Uses API or local engine."""
    stash = _get_stash()
    is_cnf = " AND " in query or " OR " in query
    key = {"query": query, "max_diff": max_diff}
    if is_cnf:
        key["mcf"] = MAX_CLAUSE_FREQ
    if key in stash:
        return stash[key]

    if _infigram_backend == "local":
        count = _local_count(query, max_diff, is_cnf)
    else:
        count = _api_count(query, max_diff, is_cnf, retries)

    if count is not None:
        stash[key] = count
    return count


def _local_count(query, max_diff, is_cnf):
    _init_local()
    if is_cnf:
        parts = [p.strip() for p in query.split(" AND ")]
        cnf = [[_local_tokenizer.encode(p)] for p in parts]
        kwargs = {"max_clause_freq": MAX_CLAUSE_FREQ}
        if max_diff is not None:
            kwargs["max_diff_tokens"] = max_diff
        return _local_engine.count_cnf(cnf=cnf, **kwargs)["count"]
    else:
        ids = _local_tokenizer.encode(query)
        return _local_engine.count(input_ids=ids)["count"]


def _api_count(query, max_diff, is_cnf, retries):
    import requests
    payload = {
        "index": INFIGRAM_INDEX,
        "query_type": "count",
        "query": query,
    }
    if is_cnf:
        payload["max_clause_freq"] = MAX_CLAUSE_FREQ
    if max_diff is not None:
        payload["max_diff_tokens"] = max_diff

    for attempt in range(retries):
        try:
            r = requests.post(INFIGRAM_URL, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["count"]
        except Exception as e:
            if attempt == retries - 1:
                return None
            wait = min(2 ** attempt, 30)
            time.sleep(wait)


def marginal_count(word):
    return infigram_count(word)


def cooccur_count(w1, w2, window=DEFAULT_WINDOW):
    return infigram_count(f"{w1} AND {w2}", max_diff=window)


def ppmi(w1, w2, c1=None, c2=None, window=DEFAULT_WINDOW):
    """PPMI between two words. Returns None on API failure."""
    if c1 is None:
        c1 = marginal_count(w1)
    if c2 is None:
        c2 = marginal_count(w2)
    c12 = cooccur_count(w1, w2, window)
    if c1 is None or c2 is None or c12 is None:
        return None
    if c12 == 0 or c1 == 0 or c2 == 0:
        return 0.0
    N = _local_N if (_infigram_backend == "local" and _local_N) else N_TOKENS
    k = 2 * window
    pmi = math.log2(c12 * N / (c1 * c2 * k))
    return max(0.0, pmi)


# ── Polysemy proxy ──────────────────────────────────────────────────

# Category anchor sets for polysemy estimation
CATEGORY_ANCHORS = {
    "sexual": ["sex", "orgasm", "erotic", "nude", "genital"],
    "violence": ["weapon", "attack", "wound", "bleed", "assault"],
}


def polysemy_ratio(word, category=None):
    """Estimate what fraction of a word's occurrences are in-category.

    Crude proxy: co-occurrence with category anchors / marginal count.
    High ratio = low polysemy risk; low ratio = diluted marginal.
    """
    c_word = marginal_count(word)
    if c_word is None or c_word == 0:
        return None

    if category and category in CATEGORY_ANCHORS:
        anchors = CATEGORY_ANCHORS[category]
    else:
        anchors = []
        for cat_anchors in CATEGORY_ANCHORS.values():
            anchors.extend(cat_anchors)

    anchor_cooccur = 0
    n_anchors = 0
    for a in anchors:
        c = cooccur_count(word, a)
        if c is not None:
            anchor_cooccur += c
            n_anchors += 1

    if n_anchors == 0:
        return None
    return anchor_cooccur / (c_word * n_anchors)


# ── Noise floor calibration ─────────────────────────────────────────

NOISE_FLOOR_PAIRS = [
    ("kill", "banana"), ("kill", "umbrella"), ("kill", "bicycle"),
    ("cock", "umbrella"), ("cock", "bicycle"), ("cock", "calendar"),
    ("blood", "bicycle"), ("blood", "calendar"), ("scream", "banana"),
    ("murder", "umbrella"), ("die", "calendar"), ("kiss", "bicycle"),
]


def calibrate_noise_floor():
    """Compute PPMI on known-unrelated pairs to estimate the noise floor."""
    vals = []
    for w1, w2 in NOISE_FLOOR_PAIRS:
        p = ppmi(w1, w2)
        if p is not None:
            vals.append(p)
    if vals:
        return float(np.mean(vals)), float(np.max(vals))
    return 0.0, 0.0


# ── Unembedding loading ─────────────────────────────────────────────

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
        raise AttributeError(f"No lm_head/embed_out on {type(m).__name__}")
    del m; gc.collect()
    W_raw = W.copy()
    W -= W.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    W /= norms
    return W, W_raw


# ── Per-site computation ────────────────────────────────────────────

def compute_site(base_p, aligned_p, W_n, tokenizer, n, category=None, top_k=K_GAINERS):
    """Compute ParaScore, SynScore, and their flat-suppression counterfactuals."""
    f_id = int(np.argmax(base_p))
    a_id = int(np.argmax(aligned_p))

    delta = aligned_p - base_p

    # Gaining tokens — compute whole-word coverage
    gainer_ids = np.where(delta > 1e-6)[0]
    if len(gainer_ids) == 0:
        return None

    total_gained_mass = float(delta[gainer_ids].sum())
    gainer_deltas = delta[gainer_ids]
    top_idx = np.argsort(-gainer_deltas)[:top_k * 3]
    candidates = []
    word_mass = 0.0
    for i in top_idx:
        tid = int(gainer_ids[i])
        if tid >= W_n.shape[0]:
            continue
        word = tokenizer.decode([tid]).strip()
        if is_whole_word(word):
            candidates.append((tid, word, float(delta[tid])))
            word_mass += float(delta[tid])
        if len(candidates) >= top_k:
            break

    if not candidates:
        return None

    whole_word_coverage = word_mass / total_gained_mass if total_gained_mass > 0 else 0.0

    # Forbidden token
    f_word = tokenizer.decode([f_id]).strip()
    if not is_whole_word(f_word):
        return None

    # Normalize gainer weights
    total_gain = sum(c[2] for c in candidates)
    if total_gain <= 0:
        return None
    weights = [(tid, word, d / total_gain) for tid, word, d in candidates]

    # Polysemy proxy for the forbidden token
    f_poly = polysemy_ratio(f_word, category=category)

    # ParaScore (cosine to forbidden token in unembedding space)
    f_vec = W_n[f_id]
    para_score = sum(w * float(np.dot(W_n[tid], f_vec)) for tid, _, w in weights)

    # SynScore (PPMI to forbidden token via infini-gram)
    f_marginal = marginal_count(f_word)
    if f_marginal is None:
        return None  # API down, skip site entirely
    syn_score = 0.0
    n_ppmi = 0
    for tid, word, w in weights:
        p = ppmi(f_word, word, c1=f_marginal)
        if p is not None:
            syn_score += w * p
            n_ppmi += 1

    # Flat-suppression counterfactual
    flat_p = base_p.copy()
    flat_p[f_id] = 0
    flat_p /= flat_p.sum()
    flat_delta = flat_p - base_p

    flat_candidates = []
    flat_gainer_ids = np.where(flat_delta > 1e-6)[0]
    if len(flat_gainer_ids) > 0:
        flat_top = np.argsort(-flat_delta[flat_gainer_ids])[:top_k * 3]
        for i in flat_top:
            tid = int(flat_gainer_ids[i])
            if tid >= W_n.shape[0]:
                continue
            word = tokenizer.decode([tid]).strip()
            if is_whole_word(word):
                flat_candidates.append((tid, word, float(flat_delta[tid])))
            if len(flat_candidates) >= top_k:
                break

    if flat_candidates:
        flat_total = sum(c[2] for c in flat_candidates)
        flat_weights = [(tid, word, d / flat_total) for tid, word, d in flat_candidates]
        para_flat = sum(w * float(np.dot(W_n[tid], f_vec)) for tid, _, w in flat_weights)
        syn_flat = 0.0
        for tid, word, w in flat_weights:
            p = ppmi(f_word, word, c1=f_marginal)
            if p is not None:
                syn_flat += w * p
    else:
        para_flat = np.nan
        syn_flat = np.nan

    return {
        "f_id": f_id,
        "f_word": f_word,
        "a_word": tokenizer.decode([a_id]).strip(),
        "para_score": para_score,
        "syn_score": syn_score,
        "para_flat": para_flat,
        "syn_flat": syn_flat,
        "para_adv": para_score - para_flat if not np.isnan(para_flat) else np.nan,
        "syn_adv": syn_score - syn_flat if not np.isnan(syn_flat) else np.nan,
        "n_gainers": len(weights),
        "n_ppmi": n_ppmi,
        "whole_word_coverage": whole_word_coverage,
        "f_polysemy_ratio": f_poly,
        "top_gainers": " ".join(w for _, w, _ in weights[:5]),
    }


# ── Calibration ─────────────────────────────────────────────────────

def calibrate():
    """Sanity-check PPMI on known pairs."""
    print("PPMI calibration (Dolma v1.7, window=100):\n")
    pairs = [
        ("kill", "scream", "violence scene"),
        ("kill", "blood", "violence scene"),
        ("kill", "die", "violence scene"),
        ("kill", "murder", "violence paradigmatic+syntagmatic"),
        ("cock", "sucked", "sexual scene"),
        ("cock", "mouth", "sexual scene"),
        ("cock", "hard", "sexual scene"),
        ("cock", "penis", "paradigmatic synonym"),
        ("kill", "pointed", "F36 displacement"),
        ("aimed", "pointed", "F36 register shift"),
        ("kiss", "massage", "F36 sexual displacement"),
        ("stared", "pressed", "F36 sexual_liminal_1"),
        ("cock", "big", "F36 category shift"),
        ("kill", "banana", "cross-scene control"),
        ("kill", "Options", "template control"),
    ]

    n_ok, n_fail = 0, 0
    marginals = {}
    for w1, w2, _ in pairs:
        for w in [w1, w2]:
            if w not in marginals:
                c = marginal_count(w)
                if c is None:
                    n_fail += 1
                    print(f"  API down for '{w}' — will retry others")
                else:
                    n_ok += 1
                marginals[w] = c

    for w1, w2, label in pairs:
        c1, c2 = marginals.get(w1), marginals.get(w2)
        if c1 is None or c2 is None:
            print(f"  PPMI({w1:10s}, {w2:10s}) = ----  [API down]")
            continue
        c12 = cooccur_count(w1, w2)
        p = ppmi(w1, w2, c1, c2)
        if p is None or c12 is None:
            print(f"  PPMI({w1:10s}, {w2:10s}) = ----  [API down]")
        else:
            print(f"  PPMI({w1:10s}, {w2:10s}) = {p:5.2f}  "
                  f"co={c12:>10,}  [{label}]")

    print(f"\n  Cached: {n_ok} queries OK, {n_fail} failed")


# ── Main driver ─────────────────────────────────────────────────────

def run_plane(families):
    """Compute ParaAdv × SynAdv for all sites across families."""
    from transformers import AutoTokenizer
    from scipy.stats import entropy as sp_entropy

    rows = []
    emb_cache = {}

    for fkey in families:
        fam = MODEL_FAMILIES[fkey]
        if fam.base in TOO_LARGE:
            continue
        psyche = Psyche.from_family(fkey)
        aligned = psyche.superego or psyche.ego
        if aligned is None:
            continue

        if fam.base not in emb_cache:
            try:
                emb_cache[fam.base] = load_unembedding_normed(fam.base)
            except Exception as e:
                print(f"  SKIP {fkey}: {e}")
                continue
        W_n, W_raw = emb_cache[fam.base]
        V = W_n.shape[0]

        tok = AutoTokenizer.from_pretrained(fam.base, trust_remote_code=True)

        print(f"\n  {fkey}:")
        n_done = 0
        for pkey, prompt in DEFAULT_PROMPTS.items():
            cat = prompt_category(pkey)
            is_trans = is_transgressive(pkey)

            try:
                bl = psyche.primary_process.logits(prompt).numpy()
                al = aligned.logits(prompt).numpy()
            except Exception:
                continue

            n = min(len(bl), len(al), V)
            base_p = softmax(bl[:n].astype(np.float64))
            al_p = softmax(al[:n].astype(np.float64))

            cat_broad = cat.split("_")[0] if "_" in cat else cat
            result = compute_site(base_p, al_p, W_n, tok, n, category=cat_broad)
            if result is None:
                continue

            result["family"] = fkey
            result["prompt_key"] = pkey
            result["category"] = cat
            result["is_transgressive"] = is_trans
            result["base_entropy"] = float(sp_entropy(base_p))
            rows.append(result)
            n_done += 1

        print(f"    {n_done} sites computed")

    # Free embeddings
    emb_cache.clear()
    gc.collect()

    return pd.DataFrame(rows)


def print_summary(df):
    from scipy.stats import mannwhitneyu, wilcoxon

    trans = df[df["is_transgressive"] == True]

    # Noise floor
    nf_mean, nf_max = calibrate_noise_floor()
    print(f"\n  PPMI noise floor (unrelated pairs): mean={nf_mean:.3f}, max={nf_max:.3f}")
    print(f"  SynAdv values below ~{nf_max:.2f} are indistinguishable from noise.")

    print("\n" + "=" * 70)
    print("JAKOBSON PLANE: ParaAdv × SynAdv")
    print("ParaAdv > 0 = condensation (synonym swap)")
    print("SynAdv > 0, ParaAdv ≤ 0 = displacement (scene-preserving)")
    print("SynAdv < 0 = foreclosure (scene abandoned)")
    print("=" * 70)

    for label, sub in [("transgressive", trans), ("all", df)]:
        pa = sub["para_adv"].dropna()
        sa = sub["syn_adv"].dropna()
        if pa.empty:
            continue
        print(f"\n  {label} (n={len(pa)}):")
        print(f"    ParaAdv: mean={pa.mean():+.4f}  median={pa.median():+.4f}  pct>0={100*(pa>0).mean():.0f}%")
        print(f"    SynAdv:  mean={sa.mean():+.4f}  median={sa.median():+.4f}  pct>0={100*(sa>0).mean():.0f}%")
        wc = sub["whole_word_coverage"].dropna()
        if not wc.empty:
            print(f"    whole-word coverage: mean={wc.mean():.2f}  median={wc.median():.2f}")

    print("\n  Per-family (transgressive, sorted by SynAdv):")
    fam_stats = []
    for fam in sorted(trans["family"].unique()):
        s = trans[trans["family"] == fam]
        pa = s["para_adv"].dropna()
        sa = s["syn_adv"].dropna()
        wc = s["whole_word_coverage"].dropna()
        if pa.empty:
            continue
        fam_stats.append((fam, len(sa), sa.mean(), pa.mean(),
                          (sa > 0).mean(), wc.mean() if not wc.empty else 0))
    fam_stats.sort(key=lambda x: x[2])
    for fam, n, sa_m, pa_m, sa_pct, wc_m in fam_stats:
        print(f"    {fam:25s}  n={n:3d}  SynAdv={sa_m:+.4f}  "
              f"ParaAdv={pa_m:+.4f}  syn>0={sa_pct*100:3.0f}%  "
              f"ww_cov={wc_m:.2f}")

    print("\n  Per-category (transgressive, polysemy-flagged):")
    for cat in sorted(trans["category"].unique()):
        s = trans[trans["category"] == cat]
        sa = s["syn_adv"].dropna()
        pa = s["para_adv"].dropna()
        poly = s["f_polysemy_ratio"].dropna()
        if sa.empty:
            continue
        poly_flag = ""
        if not poly.empty and poly.mean() < 0.005:
            poly_flag = "  [!polysemy: low in-category ratio]"
        print(f"    {cat:25s}  n={len(sa):3d}  SynAdv={sa.mean():+.5f}  "
              f"ParaAdv={pa.mean():+.5f}{poly_flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+",
                        default=["olmo", "llama", "amber", "qwen", "qwen3",
                                 "tulu", "olmo-tiny", "zephyr"])
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--local", action="store_true",
                        help="Use local infini-gram index (v4_dolmasample_olmo, 8B tokens)")
    parser.add_argument("--poll", type=int, default=0, metavar="SECS",
                        help="Re-run calibrate every N seconds until API responds")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.local:
        use_local_backend()
        print("Using LOCAL infini-gram (v4_dolmasample_olmo, ~8B tokens)")
    else:
        print("Using API infini-gram (v4_dolma-v1_7_llama, ~2.6T tokens)")

    if args.calibrate or args.poll:
        calibrate()
        while args.poll:
            stash = _get_stash()
            # Check if we got all 15 calibration pairs cached
            n_cached = sum(1 for _ in stash)
            print(f"\n  [{n_cached} queries cached. Waiting {args.poll}s...]\n")
            time.sleep(args.poll)
            calibrate()
        return

    df = run_plane(args.families)
    print_summary(df)

    if args.save:
        out = os.path.join(PATH_DATA, "jakobson_plane.csv")
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
