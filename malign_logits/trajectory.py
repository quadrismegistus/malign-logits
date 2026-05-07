#!/usr/bin/env python
"""Trajectory drift through hidden state space.

Part A: feed identical fixed text through base/SFT/DPO/RLVR, capture per-token
hidden states at a late layer, compare trajectory geometry across alignment stages.
Uses pre-generated passages from stash_gen_battery for error bars (n per prompt).

Part B (intervention): test whether alignment is a fold (reachable by linear push
in residual space) or a wall (structural restructuring). Three methods:
  v2   — single-prompt (DPO - base) direction
  v2.5 — averaged direction across prompts
  v2.6 — learned steering vector via gradient descent

Usage:
    malign trajectory                          # default family, full run
    malign trajectory --family olmo            # specific family
    malign trajectory --family olmo-tiny       # 1B, full run
    malign trajectory --skip-intervention      # geometry only
    malign trajectory --n-passages 20          # passages per prompt (default: all)
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from malign_logits.analysis import js_divergence
from malign_logits.experiments import TIER1_PROMPTS
from malign_logits.psyche import Psyche

SUBSET_KEYS = [
    "sexual_liminal_1", "sexual_explicit_2",
    "violence_liminal_3", "violence_explicit_3",
    "profanity_2", "substance_2",
    "neutral_1", "neutral_7",
]

MAX_NEW = 100
ALPHAS_COARSE = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
ALPHAS_FINE = [-0.3, -0.1, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


def load_stash_passages(family, labels=None, n_per_prompt=None):
    """Load pre-generated passages from stash_gen_battery.

    Returns dict: label -> list of passage strings (from base model).
    """
    from malign_logits.embedding import load_generations_from_stash
    gen_df = load_generations_from_stash()
    gen_df = gen_df[gen_df.family == family]
    if labels is not None:
        gen_df = gen_df[gen_df.label.isin(labels)]
    # Group by label and model, return base passages
    result = {}
    for label in gen_df.label.unique():
        base = gen_df[(gen_df.label == label) & (gen_df.model == 'base')]
        passages = [str(r["psg"]).rstrip() for _, r in base.iterrows()]
        if n_per_prompt is not None:
            passages = passages[:n_per_prompt]
        if passages:
            result[label] = passages
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trajectory_metrics(traj):
    cos = F.cosine_similarity(traj[1:], traj[:-1], dim=-1)
    local_drift = (1 - cos).mean().item()

    unit = F.normalize(traj, dim=-1)
    mean_dir = F.normalize(unit.mean(dim=0), dim=-1)
    gyration_cos = (1 - F.cosine_similarity(unit, mean_dir.unsqueeze(0), dim=-1)).mean().item()

    diffs = traj[1:] - traj[:-1]
    path_length = diffs.norm(dim=-1).sum().item()
    centroid = traj.mean(dim=0)
    gyration = (traj - centroid).norm(dim=-1).mean().item()
    mean_norm = traj.norm(dim=-1).mean().item()

    return {
        "local_drift": local_drift,
        "gyration_cos": gyration_cos,
        "path_length": path_length,
        "gyration_radius": gyration,
        "mean_norm": mean_norm,
    }


def _get_layers(model):
    """Get the transformer layers list, handling different architectures."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers  # Llama, OLMo, Qwen, Mistral
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers  # GPT-NeoX (Pythia)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h  # GPT-2
    raise AttributeError(f"Cannot find layers in {type(model).__name__}")


def get_trajectory(model, tokenizer, token_ids, layer_idx):
    ids = token_ids.unsqueeze(0).to(model.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[layer_idx][0].float().cpu()


def get_trajectories_batched(model, tokenizer, token_id_list, layer_idx,
                              batch_size=16):
    """Batched forward passes with padding. Returns list of trajectories,
    each trimmed to its original (unpadded) length."""
    results = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    for start in range(0, len(token_id_list), batch_size):
        batch_ids = token_id_list[start:start + batch_size]
        lengths = [len(ids) for ids in batch_ids]
        max_len = max(lengths)

        padded = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            padded[i, :len(ids)] = ids
            attn_mask[i, :len(ids)] = 1

        padded = padded.to(model.device)
        attn_mask = attn_mask.to(model.device)

        with torch.no_grad():
            out = model(padded, attention_mask=attn_mask,
                        output_hidden_states=True)

        hidden = out.hidden_states[layer_idx].float().cpu()
        for i, length in enumerate(lengths):
            results.append(hidden[i, :length, :])

    return results


def last_hidden(model, tokenizer, prompt, layer_idx):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[layer_idx][0, -1, :].float().cpu()


def last_logits(model, tokenizer, prompt):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(ids)
    return out.logits[0, -1, :].float().cpu()


def intervene_logits(model, tokenizer, prompt, layer_idx, direction, alpha):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    addend = (alpha * direction).to(model.device).to(next(model.parameters()).dtype)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            h[0, -1, :] = h[0, -1, :] + addend
            return (h,) + output[1:]
        else:
            h = output.clone()
            h[0, -1, :] = h[0, -1, :] + addend
            return h

    handle = _get_layers(model)[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(ids)
        return out.logits[0, -1, :].float().cpu()
    finally:
        handle.remove()


def train_steering_vector(base_model, dpo_targets, tokenizer, prompts, layer_idx,
                          init_direction=None, n_epochs=30, lr=0.05, log_every=10):
    hidden_dim = base_model.config.hidden_size
    device = base_model.device

    if init_direction is not None:
        d = nn.Parameter(init_direction.clone().to(device).float())
    else:
        d = nn.Parameter((torch.randn(hidden_dim, device=device) * 0.1).float())

    optimizer = torch.optim.Adam([d], lr=lr)
    losses = []

    def make_hook(d_param):
        def hook_fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h_prefix = h[:, :-1, :]
            h_last = h[:, -1:, :] + d_param.to(h.dtype).view(1, 1, -1)
            h_new = torch.cat([h_prefix, h_last], dim=1)
            return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
        return hook_fn

    hook_fn = make_hook(d)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0

        for prompt in prompts:
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            target_probs = torch.softmax(dpo_targets[prompt].to(device), dim=-1)

            handle = _get_layers(base_model)[layer_idx].register_forward_hook(hook_fn)
            try:
                out = base_model(ids)
                logits = out.logits[0, -1, :].float()
            finally:
                handle.remove()

            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -(target_probs * log_probs).sum()
            (loss / len(prompts)).backward()
            epoch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_([d], max_norm=20.0)
        optimizer.step()

        avg_loss = epoch_loss / len(prompts)
        losses.append(avg_loss)
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}: loss={avg_loss:.4f}  ||d||={d.norm().item():.2f}")

    return d.detach().cpu().float(), losses


def train_steering_vectors(base_model, dpo_targets, tokenizer, prompts, layer_idx,
                           n_vectors=1, n_epochs=30, lr=0.05, log_every=10):
    """Learn a rank-N steering subspace via gradient descent.

    Learns a single direction d = sum(D[i]) where D is (n_vectors, hidden_dim).
    No orthogonalization during training — the effective rank is determined by
    the optimization landscape. Post-hoc SVD reveals the intrinsic dimensionality.

    For N=1 this is identical to train_steering_vector (v2.6).
    For N>1 the extra parameters give the optimizer more room to find
    a direction that generalizes across prompts.

    Returns (d_combined, D_raw, losses) where d_combined is the sum vector
    (hidden_dim,), D_raw is the raw matrix (n_vectors, hidden_dim).
    """
    hidden_dim = base_model.config.hidden_size
    device = base_model.device

    D = nn.Parameter((torch.randn(n_vectors, hidden_dim, device=device) * 0.1).float())
    optimizer = torch.optim.Adam([D], lr=lr)
    losses = []

    def make_hook(D_param):
        def hook_fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h_prefix = h[:, :-1, :]
            addend = D_param.sum(dim=0).to(h.dtype).view(1, 1, -1)
            h_last = h[:, -1:, :] + addend
            h_new = torch.cat([h_prefix, h_last], dim=1)
            return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
        return hook_fn

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0

        hook_fn = make_hook(D)
        for prompt in prompts:
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            target_probs = torch.softmax(dpo_targets[prompt].to(device), dim=-1)

            handle = _get_layers(base_model)[layer_idx].register_forward_hook(hook_fn)
            try:
                out = base_model(ids)
                logits = out.logits[0, -1, :].float()
            finally:
                handle.remove()

            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -(target_probs * log_probs).sum()
            (loss / len(prompts)).backward()
            epoch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_([D], max_norm=20.0)
        optimizer.step()

        avg_loss = epoch_loss / len(prompts)
        losses.append(avg_loss)
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            d_combined = D.sum(dim=0)
            print(f"  epoch {epoch:3d}: loss={avg_loss:.4f}  "
                  f"||d||={d_combined.norm().item():.2f}", flush=True)

    d_combined = D.detach().sum(dim=0).cpu().float()
    return d_combined, D.detach().cpu().float(), losses


def intervene_logits_multi(model, tokenizer, prompt, layer_idx, d):
    """Apply a steering vector (combined direction)."""
    return intervene_logits(model, tokenizer, prompt, layer_idx, d, alpha=1.0)


def token_level_report(base_model, tokenizer, prompt, layer_idx, direction, alpha,
                       logits_base, logits_dpo):
    logits_int = intervene_logits(base_model, tokenizer, prompt, layer_idx, direction, alpha)
    js_bd = js_divergence(logits_base, logits_dpo)
    closure = (js_bd - js_divergence(logits_int, logits_dpo)) / js_bd * 100

    p_b = torch.softmax(logits_base, dim=-1)
    p_d = torch.softmax(logits_dpo, dim=-1)
    p_p = torch.softmax(logits_int, dim=-1)

    print(f'\n  Prompt: "{prompt}"')
    print(f"  Closure on this prompt: {closure:.1f}%\n")
    print(f'  {"token":15s}  {"base":>8s}  {"dpo":>8s}  {"pushed":>8s}   direction')
    for v, idx in zip(*p_b.topk(10)):
        word = tokenizer.decode([idx]).strip()
        toward = (p_p[idx] - p_b[idx]) * (p_d[idx] - p_b[idx]) > 0
        label = "-> toward DPO" if toward else "x  away from DPO"
        print(f"  {word:15s}  {v.item():8.4f}  {p_d[idx]:8.4f}  {p_p[idx]:8.4f}   {label}")

    K = 20
    top_idx = p_b.topk(K).indices
    correct = sum(1 for idx in top_idx
                  if (p_p[idx] - p_b[idx]).item() * (p_d[idx] - p_b[idx]).item() > 0)
    print(f"\n  Top-{K} base tokens nudged toward DPO: {correct}/{K} ({correct/K*100:.0f}%)")


# ---------------------------------------------------------------------------
# Part A: trajectory geometry
# ---------------------------------------------------------------------------

def run_trajectory_geometry(psyche, family, layer_idx, out_dir, n_passages=None,
                            prompts_set="tier1"):
    print(f"\n{'=' * 60}")
    print(f"  Part A: trajectory geometry (family={family}, layer={layer_idx})")
    print(f"{'=' * 60}")

    tokenizer = psyche.primary_process.tokenizer
    base_model = psyche.primary_process.model

    model_layers = [
        ("base", psyche.primary_process),
        ("sft", psyche.ego),
        ("dpo", psyche.superego),
        ("rlvr", psyche.reinforced_superego),
    ]
    model_layers = [(n, l) for n, l in model_layers if l is not None]

    from .experiments import DEFAULT_PROMPTS
    if prompts_set == "all":
        subset = dict(DEFAULT_PROMPTS)
    else:
        subset = {k: TIER1_PROMPTS[k] for k in SUBSET_KEYS}

    # Load pre-generated passages from stash
    stash_passages = load_stash_passages(family, labels=list(subset.keys()),
                                          n_per_prompt=n_passages)
    has_stash = bool(stash_passages)
    if has_stash:
        total = sum(len(v) for v in stash_passages.values())
        print(f"  Loaded {total} pre-generated passages from stash"
              f" ({len(stash_passages)} prompts)")
    else:
        print("  No stash passages found, falling back to n=1 generation")

    # Tokenize all passages up front
    passage_data = []  # list of (label, passage_idx, token_ids)
    for label, prompt in subset.items():
        if has_stash and label in stash_passages:
            passages = stash_passages[label]
        else:
            torch.manual_seed(0)
            ids = tokenizer.encode(prompt, return_tensors="pt").to(base_model.device)
            with torch.no_grad():
                out = base_model.generate(
                    ids, max_new_tokens=MAX_NEW, do_sample=True,
                    temperature=1.0, top_p=0.95
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            passages = [text[len(prompt):]]

        for pi, passage in enumerate(passages):
            full_text = prompt + passage
            fixed = tokenizer.encode(full_text, return_tensors="pt")[0]
            passage_data.append((label, pi, fixed))
        n_done = len(passages) if has_stash and label in stash_passages else 1
        print(f"  {label}: {n_done} passages tokenized")

    print(f"  Total: {len(passage_data)} passages to process")
    all_token_ids = [pd[2] for pd in passage_data]

    # Determine batch size: use batching on CUDA, sequential on MPS/CPU
    use_batching = base_model.device.type == "cuda"
    batch_size = 16 if use_batching else 1

    rows = []
    for name, layer in model_layers:
        t0 = time.time()
        if use_batching and len(all_token_ids) > 1:
            trajs = get_trajectories_batched(
                layer.model, tokenizer, all_token_ids, layer_idx,
                batch_size=batch_size)
        else:
            from tqdm import tqdm
            trajs = [get_trajectory(layer.model, tokenizer, ids, layer_idx)
                     for ids in tqdm(all_token_ids, desc=f"  {name}")]

        for (label, pi, fixed), traj in zip(passage_data, trajs):
            m = trajectory_metrics(traj)
            rows.append({
                "label": label, "model": name, "passage_idx": pi,
                "n_tokens": len(fixed), **m,
            })
        elapsed = time.time() - t0
        print(f"  {name}: {len(passage_data)} passages in {elapsed:.1f}s"
              f" ({elapsed/len(passage_data):.2f}s/psg)"
              f"{' [batched]' if use_batching else ''}")

    agg = pd.DataFrame(rows)
    model_order = [n for n, _ in model_layers]
    agg["model"] = pd.Categorical(agg["model"], categories=model_order, ordered=True)
    agg["category"] = agg["label"].str.replace(r"_\d+$", "", regex=True)
    agg["transgressive"] = ~agg["category"].str.startswith("neutral")

    metric_cols = ["local_drift", "gyration_cos", "path_length", "gyration_radius", "mean_norm"]

    print(f"\n--- Overall mean (n={len(agg) // len(model_layers)} passages) ---")
    print(agg.groupby("model", observed=True)[metric_cols].mean().round(4).to_string())

    # Mean ± std for key metrics
    print(f"\n--- Overall mean ± std ---")
    for metric in ["local_drift", "gyration_cos", "mean_norm"]:
        summary = agg.groupby("model", observed=True)[metric].agg(["mean", "std"])
        print(f"\n  {metric}:")
        for model_name, row in summary.iterrows():
            print(f"    {model_name:6s}: {row['mean']:.4f} ± {row['std']:.4f}")

    print("\n--- Transgressive only ---")
    print(agg[agg["transgressive"]].groupby("model", observed=True)[metric_cols].mean().round(4).to_string())
    print("\n--- Neutral only ---")
    print(agg[~agg["transgressive"]].groupby("model", observed=True)[metric_cols].mean().round(4).to_string())

    # Bootstrap CI on base→aligned deltas
    print("\n--- Base→Aligned Δ (bootstrap 95% CI) ---")
    aligned_name = model_order[-1] if len(model_order) <= 2 else "dpo"
    if aligned_name not in model_order:
        aligned_name = model_order[-1]
    for metric in ["local_drift", "gyration_cos", "mean_norm"]:
        base_vals = agg[agg.model == "base"][metric].values
        aligned_vals = agg[agg.model == aligned_name][metric].values
        if len(base_vals) > 0 and len(aligned_vals) > 0:
            rng = np.random.default_rng(42)
            deltas = []
            for _ in range(10000):
                b = rng.choice(base_vals, size=len(base_vals), replace=True).mean()
                a = rng.choice(aligned_vals, size=len(aligned_vals), replace=True).mean()
                deltas.append(a - b)
            deltas = np.array(deltas)
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            print(f"  {metric}: Δ={np.median(deltas):+.4f} [{lo:+.4f}, {hi:+.4f}]")

    csv_path = f"{out_dir}/trajectory_geometry_{family}.csv"
    agg.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")

    # Figure
    plot_metrics = ["local_drift", "gyration_cos", "gyration_radius", "mean_norm"]
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4.5 * len(plot_metrics), 4))
    for ax, metric in zip(axes, plot_metrics):
        # Per-prompt means (aggregate across passages)
        prompt_means = agg.groupby(["label", "model"], observed=True)[metric].mean().reset_index()
        for label in sorted(prompt_means.label.unique()):
            sub = prompt_means[prompt_means.label == label].sort_values("model")
            is_neutral = label.startswith("neutral")
            color = "#999999" if is_neutral else "#4e79a7"
            ls = ":" if is_neutral else "-"
            ax.plot(sub["model"].astype(str), sub[metric], ls, marker="o",
                    alpha=0.4, color=color, markersize=4)
        for is_t, color, lbl in [(True, "black", "transgressive mean"),
                                  (False, "#bb5544", "neutral mean")]:
            sub_mean = agg[agg["transgressive"] == is_t].groupby("model", observed=True)[metric]
            means = sub_mean.mean()
            stds = sub_mean.std()
            ax.plot(means.index.astype(str), means.values, "-D",
                    color=color, linewidth=2.5, markersize=8, label=lbl)
            ax.fill_between(means.index.astype(str),
                            means.values - stds.values,
                            means.values + stds.values,
                            alpha=0.15, color=color)
        ax.set_title(metric)
        ax.set_xlabel("layer")
    axes[0].legend(fontsize=6, loc="best", ncol=2)
    n_psg = len(agg) // len(model_layers)
    plt.suptitle(f"Trajectory geometry ({family}, layer {layer_idx}, n={n_psg})", y=1.02)
    plt.tight_layout()
    fig_path = f"figures/trajectory_geometry.{family}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")

    return agg


# ---------------------------------------------------------------------------
# Part B: intervention experiments
# ---------------------------------------------------------------------------

def run_intervention(psyche, family, intervention_layers, out_dir, n_epochs=30,
                     lr=0.05, prompts_set="tier1"):
    from .experiments import DEFAULT_PROMPTS
    print(f"\n{'=' * 60}")
    print(f"  Part B: fold-vs-wall intervention ({family})")
    print(f"  Intervention layers: {intervention_layers}")
    print(f"{'=' * 60}")

    base_model = psyche.primary_process.model
    dpo_model = psyche.superego.model
    tokenizer = psyche.primary_process.tokenizer

    if prompts_set == "all":
        subset = dict(DEFAULT_PROMPTS)
    else:
        subset = {k: TIER1_PROMPTS[k] for k in SUBSET_KEYS}
    target_prompt = subset.get("violence_liminal_3", list(subset.values())[0])

    # ------------------------------------------------------------------
    # v2: per-prompt self-direction (extract from P, test on P)
    # ------------------------------------------------------------------
    print(f"\n--- v2: per-prompt self-direction ({len(subset)} prompts) ---")

    v2_rows = []
    v2_best_per_prompt = []
    for label, prompt in subset.items():
        directions = {}
        for L in intervention_layers:
            base_h = last_hidden(base_model, tokenizer, prompt, L)
            dpo_h = last_hidden(dpo_model, tokenizer, prompt, L)
            directions[L] = dpo_h - base_h

        logits_b = last_logits(base_model, tokenizer, prompt)
        logits_d = last_logits(dpo_model, tokenizer, prompt)
        js_bd = js_divergence(logits_b, logits_d)

        best_closure = -999
        best_row = None
        for L in intervention_layers:
            for alpha in ALPHAS_COARSE:
                if alpha == 0:
                    logits_int = logits_b
                else:
                    logits_int = intervene_logits(
                        base_model, tokenizer, prompt, L, directions[L], alpha)
                js_to_b = js_divergence(logits_int, logits_b)
                js_to_d = js_divergence(logits_int, logits_d)
                closure = (js_bd - js_to_d) / js_bd
                v2_rows.append({
                    "label": label, "layer": L, "alpha": alpha,
                    "js_to_base": js_to_b, "js_to_dpo": js_to_d,
                    "baseline_js": js_bd, "closure": closure,
                })
                if alpha != 0 and closure > best_closure:
                    best_closure = closure
                    best_row = {"label": label, "layer": L, "alpha": alpha,
                                "baseline_js": js_bd, "closure": closure}
        v2_best_per_prompt.append(best_row)

    df_v2 = pd.DataFrame(v2_rows)
    df_v2_best = pd.DataFrame(v2_best_per_prompt)

    print(f"\n  Per-prompt self-closure (best layer × alpha for each):")
    print(f"  {'label':30s} {'baseline_js':>12s} {'best_L':>7s} {'α':>5s} {'closure':>10s}")
    print("  " + "-" * 70)
    for _, r in df_v2_best.sort_values("closure", ascending=False).iterrows():
        print(f"  {r.label:30s} {r.baseline_js:12.4f} {int(r.layer):7d} {r.alpha:5.1f} {r.closure*100:9.1f}%")
    print(f"\n  Mean self-closure: {df_v2_best.closure.mean()*100:.1f}%")
    print(f"  Median: {df_v2_best.closure.median()*100:.1f}%")
    print(f"  Range: {df_v2_best.closure.min()*100:.1f}% – {df_v2_best.closure.max()*100:.1f}%")

    # v2 figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mean_v2 = df_v2.groupby(["layer", "alpha"])[["js_to_base", "js_to_dpo"]].mean().reset_index()
    for L in intervention_layers:
        sub = mean_v2[mean_v2["layer"] == L].sort_values("alpha")
        axes[0].plot(sub["alpha"], sub["js_to_base"], "-o", label=f"L={L}")
        axes[1].plot(sub["alpha"], sub["js_to_dpo"], "-o", label=f"L={L}")
    mean_baseline = df_v2_best.baseline_js.mean()
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="reach DPO")
    axes[1].axhline(mean_baseline, color="gray", linestyle=":", linewidth=1, alpha=0.6,
                     label=f"baseline={mean_baseline:.3f}")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xlabel("alpha")
    axes[0].set_ylabel("JS(intervened, base)")
    axes[0].set_title("Distance from base (mean across prompts)")
    axes[1].set_ylabel("JS(intervened, DPO)")
    axes[1].set_title("Distance from DPO (mean across prompts)")
    plt.suptitle(f"v2: per-prompt self-direction ({family}, {len(subset)} prompts)", y=1.02)
    plt.tight_layout()
    fig.savefig(f"figures/intervention_v2.{family}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Token-level report for the prompt with highest closure
    best_overall = df_v2_best.sort_values("closure", ascending=False).iloc[0]
    best_prompt = best_overall.label
    best_L = int(best_overall.layer)
    best_alpha = best_overall.alpha
    base_h = last_hidden(base_model, tokenizer, best_prompt, best_L)
    dpo_h = last_hidden(dpo_model, tokenizer, best_prompt, best_L)
    best_dir = dpo_h - base_h
    token_level_report(base_model, tokenizer, best_prompt,
                       best_L, best_dir, best_alpha,
                       last_logits(base_model, tokenizer, best_prompt),
                       last_logits(dpo_model, tokenizer, best_prompt))

    # ------------------------------------------------------------------
    # v2.5: averaged direction across prompts
    # ------------------------------------------------------------------
    print(f"\n--- v2.5: averaged direction across {len(subset)} prompts ---")

    hidden_pairs = {L: {"base": [], "dpo": []} for L in intervention_layers}
    for label, prompt in subset.items():
        for L in intervention_layers:
            hidden_pairs[L]["base"].append(last_hidden(base_model, tokenizer, prompt, L))
            hidden_pairs[L]["dpo"].append(last_hidden(dpo_model, tokenizer, prompt, L))

    avg_directions = {}
    print(f"  {'layer':>5s}  {'||avg_dir||':>12s}  {'||per-prompt||':>14s}  {'consistency':>12s}")
    for L in intervention_layers:
        diffs = torch.stack([d - b for d, b in
                             zip(hidden_pairs[L]["dpo"], hidden_pairs[L]["base"])])
        avg = diffs.mean(dim=0)
        avg_directions[L] = avg
        cos_to_avg = F.cosine_similarity(diffs, avg.unsqueeze(0), dim=-1).mean().item()
        print(f"  {L:5d}  {avg.norm().item():12.2f}  {diffs.norm(dim=-1).mean().item():14.2f}  {cos_to_avg:12.3f}")

    v25_rows = []
    for label, prompt in subset.items():
        logits_b = last_logits(base_model, tokenizer, prompt)
        logits_d = last_logits(dpo_model, tokenizer, prompt)
        js_bd = js_divergence(logits_b, logits_d)
        for L in intervention_layers:
            for alpha in ALPHAS_FINE:
                if alpha == 0:
                    logits_int = logits_b
                else:
                    logits_int = intervene_logits(base_model, tokenizer, prompt, L,
                                                  avg_directions[L], alpha)
                v25_rows.append({
                    "label": label, "layer": L, "alpha": alpha,
                    "js_to_base": js_divergence(logits_int, logits_b),
                    "js_to_dpo": js_divergence(logits_int, logits_d),
                    "baseline_js": js_bd,
                })

    df_v25 = pd.DataFrame(v25_rows)
    df_v25["closure"] = (df_v25["baseline_js"] - df_v25["js_to_dpo"]) / df_v25["baseline_js"]

    closure_summary = (df_v25[df_v25["alpha"] != 0]
                       .groupby(["layer", "alpha"])["closure"]
                       .agg(["mean", "std"]).reset_index())
    closure_summary["mean_closure_pct"] = closure_summary["mean"] * 100
    best_v25 = closure_summary.sort_values("mean_closure_pct", ascending=False).iloc[0]
    print(f"\n  Best v2.5: L={int(best_v25['layer'])}, alpha={best_v25['alpha']:.2f},"
          f" mean closure={best_v25['mean_closure_pct']:.2f}%")

    # v2.5 figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mean_curves = df_v25.groupby(["layer", "alpha"])[["js_to_base", "js_to_dpo"]].mean().reset_index()
    for L in intervention_layers:
        sub = mean_curves[mean_curves["layer"] == L].sort_values("alpha")
        axes[0].plot(sub["alpha"], sub["js_to_base"], "-o", label=f"L={L}")
        axes[1].plot(sub["alpha"], sub["js_to_dpo"], "-o", label=f"L={L}")
    mean_baseline = df_v25.groupby("label")["baseline_js"].first().mean()
    axes[1].axhline(mean_baseline, color="gray", linestyle=":", alpha=0.7, label=f"baseline={mean_baseline:.3f}")
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.5, label="reach DPO")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xlabel("alpha")
    axes[0].set_ylabel("mean JS(intervened, base)")
    axes[0].set_title("Distance from base")
    axes[1].set_ylabel("mean JS(intervened, DPO)")
    axes[1].set_title("Distance from DPO")
    plt.suptitle(f"v2.5: averaged direction across {len(subset)} prompts ({family})", y=1.02)
    plt.tight_layout()
    fig.savefig(f"figures/intervention_v25.{family}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    token_level_report(base_model, tokenizer, target_prompt,
                       int(best_v25["layer"]), avg_directions[int(best_v25["layer"])],
                       best_v25["alpha"],
                       last_logits(base_model, tokenizer, target_prompt),
                       last_logits(dpo_model, tokenizer, target_prompt))

    # ------------------------------------------------------------------
    # v2.6: learned steering vector
    # ------------------------------------------------------------------
    print(f"\n--- v2.6: learned steering vector ---")

    # Split: first half for eval (v2 self-direction already tested these),
    # second half for training the learned vector
    all_keys = list(subset.keys())
    split = len(all_keys) // 2
    eval_keys = all_keys[:split]
    train_keys = all_keys[split:]
    train_prompts = [subset[k] for k in train_keys]
    print(f"  Train: {len(train_keys)} prompts, Eval: {len(eval_keys)} prompts (held out)")

    for p in base_model.parameters():
        p.requires_grad_(False)
    for p in dpo_model.parameters():
        p.requires_grad_(False)

    print("  Pre-computing DPO targets...")
    dpo_targets = {}
    for k in train_keys:
        dpo_targets[subset[k]] = last_logits(dpo_model, tokenizer, subset[k])

    learned_directions = {}
    training_losses = {}
    for L in intervention_layers:
        for init_name in ["avg_init", "rand_init"]:
            print(f"\n  === Layer {L}, {init_name} ===")
            init = avg_directions[L].clone() if init_name == "avg_init" else None
            t0 = time.time()
            d_learned, losses = train_steering_vector(
                base_model, dpo_targets, tokenizer, train_prompts,
                layer_idx=L, init_direction=init, n_epochs=n_epochs, lr=lr, log_every=10,
            )
            learned_directions[(L, init_name)] = d_learned
            training_losses[(L, init_name)] = losses
            print(f"  done in {time.time() - t0:.1f}s, ||d||={d_learned.norm().item():.2f}")

    v26_rows = []
    for L in intervention_layers:
        for init_name in ["avg_init", "rand_init"]:
            d_learned = learned_directions[(L, init_name)]
            for label, prompt in subset.items():
                logits_b = last_logits(base_model, tokenizer, prompt)
                logits_d = last_logits(dpo_model, tokenizer, prompt)
                js_bd = js_divergence(logits_b, logits_d)
                for alpha in ALPHAS_FINE:
                    if alpha == 0:
                        logits_int = logits_b
                    else:
                        logits_int = intervene_logits(base_model, tokenizer, prompt, L,
                                                      d_learned, alpha)
                    v26_rows.append({
                        "init": init_name, "label": label, "layer": L, "alpha": alpha,
                        "js_to_base": js_divergence(logits_int, logits_b),
                        "js_to_dpo": js_divergence(logits_int, logits_d),
                        "baseline_js": js_bd,
                    })

    df_v26 = pd.DataFrame(v26_rows)
    df_v26["closure"] = (df_v26["baseline_js"] - df_v26["js_to_dpo"]) / df_v26["baseline_js"]

    v26_summary = (df_v26[df_v26["alpha"] != 0]
                   .groupby(["init", "layer", "alpha"])["closure"].mean().reset_index())
    v26_summary["mean_closure_pct"] = v26_summary["closure"] * 100
    best_per = v26_summary.loc[
        v26_summary.groupby(["init", "layer"])["closure"].idxmax()
    ].reset_index(drop=True)

    print("\n  Best alpha per (init, layer) by held-out mean closure:")
    print(best_per[["init", "layer", "alpha", "mean_closure_pct"]].round(2).to_string(index=False))

    # v2.6 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for L in intervention_layers:
        axes[0, 0].plot(training_losses[(L, "avg_init")], label=f"L={L}", alpha=0.8)
        axes[0, 1].plot(training_losses[(L, "rand_init")], label=f"L={L}", alpha=0.8)
    axes[0, 0].set_title("Training loss (avg_init)")
    axes[0, 1].set_title("Training loss (rand_init)")
    for ax in axes[0]:
        ax.set_xlabel("epoch")
        ax.set_ylabel("CE to DPO")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    mean_v26 = df_v26.groupby(["init", "layer", "alpha"])["js_to_dpo"].mean().reset_index()
    mean_baseline = df_v26.groupby("label")["baseline_js"].first().mean()
    for ax_idx, init_name in enumerate(["avg_init", "rand_init"]):
        ax = axes[1, ax_idx]
        sub = mean_v26[mean_v26["init"] == init_name]
        for L in intervention_layers:
            sub_L = sub[sub["layer"] == L].sort_values("alpha")
            ax.plot(sub_L["alpha"], sub_L["js_to_dpo"], "-o", label=f"L={L}", alpha=0.8)
        ax.axhline(mean_baseline, color="gray", linestyle=":", alpha=0.7,
                    label=f"baseline={mean_baseline:.3f}")
        ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="reach DPO")
        ax.set_xlabel("alpha")
        ax.set_ylabel("mean JS(intervened, DPO)")
        ax.set_title(f"Held-out: {init_name}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(f"v2.6: learned steering vectors ({family})", y=1.00)
    plt.tight_layout()
    fig.savefig(f"figures/intervention_v26.{family}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # v2.7: fold-rank analysis — closure vs N orthogonal vectors
    # ------------------------------------------------------------------
    print(f"\n--- v2.7: fold-rank analysis (closure vs N vectors) ---")

    # Use best layer from v2.6
    best_v26 = best_per.sort_values("mean_closure_pct", ascending=False).iloc[0]
    best_L = int(best_v26["layer"])
    print(f"  Using best layer from v2.6: L={best_L}")

    # SVD of the (DPO - base) difference matrix as passive baseline
    print(f"  Computing SVD of alignment shift (passive dimensionality)...")
    diff_vecs = []
    for label, prompt in subset.items():
        base_h = last_hidden(base_model, tokenizer, prompt, best_L)
        dpo_h = last_hidden(dpo_model, tokenizer, prompt, best_L)
        diff_vecs.append((dpo_h - base_h).numpy())
    diff_matrix = np.stack(diff_vecs)  # (n_prompts, hidden_dim)
    U, S, Vt = np.linalg.svd(diff_matrix, full_matrices=False)
    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    print(f"  SVD singular values (top 10): {', '.join(f'{s:.2f}' for s in S[:10])}")
    print(f"  Cumulative variance: {', '.join(f'{v:.1%}' for v in cumvar[:10])}")

    # Active fold-rank: learn N vectors and measure held-out closure
    N_VALUES = [1, 2, 3, 5, 10, 20]
    eval_prompts = {k: subset[k] for k in eval_keys}

    # Pre-compute eval baselines
    eval_baselines = {}
    for label, prompt in eval_prompts.items():
        logits_b = last_logits(base_model, tokenizer, prompt)
        logits_d = last_logits(dpo_model, tokenizer, prompt)
        eval_baselines[label] = {
            "logits_base": logits_b,
            "logits_dpo": logits_d,
            "js_bd": js_divergence(logits_b, logits_d),
        }

    fold_rank_results = []
    for N in N_VALUES:
        print(f"\n  === N={N} vectors, L={best_L} ===")
        t0 = time.time()
        d_combined, D_raw, losses = train_steering_vectors(
            base_model, dpo_targets, tokenizer, train_prompts,
            layer_idx=best_L, n_vectors=N, n_epochs=n_epochs, lr=lr,
            log_every=10,
        )
        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.1f}s, ||d||={d_combined.norm().item():.2f}")

        # Evaluate on held-out prompts
        closures = []
        for label, prompt in eval_prompts.items():
            bl = eval_baselines[label]
            logits_int = intervene_logits_multi(
                base_model, tokenizer, prompt, best_L, d_combined)
            js_int_dpo = js_divergence(logits_int, bl["logits_dpo"])
            closure = (bl["js_bd"] - js_int_dpo) / bl["js_bd"]
            closures.append(closure)

        mean_closure = np.mean(closures) * 100
        print(f"  Held-out closure: {mean_closure:.1f}% "
              f"(range {min(closures)*100:.1f}%–{max(closures)*100:.1f}%)")

        fold_rank_results.append({
            "n_vectors": N, "layer": best_L,
            "mean_closure_pct": mean_closure,
            "min_closure_pct": min(closures) * 100,
            "max_closure_pct": max(closures) * 100,
            "train_loss_final": losses[-1],
        })

    df_fold = pd.DataFrame(fold_rank_results)

    # Estimate fold rank: smallest N where closure reaches 90% of max
    max_closure = df_fold.mean_closure_pct.max()
    threshold = max_closure * 0.9
    fold_rank_rows = df_fold[df_fold.mean_closure_pct >= threshold]
    fold_rank = int(fold_rank_rows.n_vectors.min()) if not fold_rank_rows.empty else N_VALUES[-1]

    print(f"\n  === Fold-rank summary ===")
    print(f"  {'N':>4s}  {'closure':>10s}  {'range':>20s}")
    print(f"  {'-'*38}")
    for _, r in df_fold.iterrows():
        print(f"  {int(r.n_vectors):4d}  {r.mean_closure_pct:9.1f}%  "
              f"[{r.min_closure_pct:.1f}%, {r.max_closure_pct:.1f}%]")
    print(f"\n  Max closure: {max_closure:.1f}% at N={int(df_fold.loc[df_fold.mean_closure_pct.idxmax(), 'n_vectors'])}")
    print(f"  Fold rank (90% of max): K={fold_rank}")
    print(f"  SVD 90% variance at: k={int(np.searchsorted(cumvar, 0.9)) + 1}")

    # Figure: closure vs N + SVD spectrum
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df_fold.n_vectors, df_fold.mean_closure_pct, '-o', color='#4e79a7',
                 linewidth=2, markersize=8)
    axes[0].fill_between(df_fold.n_vectors, df_fold.min_closure_pct,
                         df_fold.max_closure_pct, alpha=0.2, color='#4e79a7')
    axes[0].axhline(threshold, color='red', linestyle=':', alpha=0.5,
                     label=f'90% of max ({threshold:.1f}%)')
    axes[0].set_xlabel('N (number of steering vectors)')
    axes[0].set_ylabel('Held-out closure (%)')
    axes[0].set_title(f'Fold rank: closure vs N ({family})')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    n_sv = min(20, len(S))
    axes[1].bar(range(1, n_sv + 1), S[:n_sv] ** 2 / (S ** 2).sum() * 100,
                color='#e15759', alpha=0.7)
    ax2 = axes[1].twinx()
    ax2.plot(range(1, n_sv + 1), cumvar[:n_sv] * 100, '-o',
             color='#59a14f', markersize=4)
    ax2.set_ylabel('Cumulative variance (%)', color='#59a14f')
    axes[1].set_xlabel('Singular value index')
    axes[1].set_ylabel('Variance explained (%)')
    axes[1].set_title(f'SVD of alignment shift (L={best_L})')
    axes[1].grid(alpha=0.3)

    plt.suptitle(f'Fold-rank analysis ({family})', y=1.02)
    plt.tight_layout()
    fig.savefig(f"figures/fold_rank.{family}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figures/fold_rank.{family}.png")

    # Summary comparison
    v2_mean_pct = df_v2_best.closure.mean() * 100
    v25_best_pct = best_v25["mean_closure_pct"]
    v26_best_avg = best_per[best_per["init"] == "avg_init"]["mean_closure_pct"].max()
    v26_best_rand = best_per[best_per["init"] == "rand_init"]["mean_closure_pct"].max()
    print(f"\n  === Closure summary ===")
    print(f"    v2  (self-direction, mean): {v2_mean_pct:6.2f}%  (per-prompt, not generalizable)")
    print(f"    v2.5 (averaged, held-out):  {v25_best_pct:6.2f}%")
    print(f"    v2.6 (learned 1-vec):       {max(v26_best_avg, v26_best_rand):6.2f}%")
    print(f"    v2.7 fold rank K={fold_rank}:       {max_closure:6.2f}%  (N={int(df_fold.loc[df_fold.mean_closure_pct.idxmax(), 'n_vectors'])})")

    # Token-level for best v2.6
    d_best = learned_directions[(int(best_v26["layer"]), best_v26["init"])]
    print(f"\n  Best v2.6: init={best_v26['init']}, L={int(best_v26['layer'])},"
          f" alpha={best_v26['alpha']}, closure={best_v26['mean_closure_pct']:.1f}%")
    token_level_report(base_model, tokenizer, target_prompt,
                       int(best_v26["layer"]), d_best, best_v26["alpha"],
                       last_logits(base_model, tokenizer, target_prompt),
                       last_logits(dpo_model, tokenizer, target_prompt))

    # Save all intervention results
    all_int = pd.concat([
        df_v2.assign(version="v2", init="self"),
        df_v25.assign(version="v2.5", init="averaged"),
        df_v26.assign(version="v2.6"),
    ], ignore_index=True)

    # Save fold-rank results
    df_fold["family"] = family
    fold_csv = f"{out_dir}/fold_rank_{family}.csv"
    df_fold.to_csv(fold_csv, index=False)
    print(f"  Saved {fold_csv}")

    csv_path = f"{out_dir}/intervention_{family}.csv"
    all_int.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", default="olmo-tiny",
                        help="Model family key (default: olmo-tiny)")
    parser.add_argument("--skip-intervention", action="store_true",
                        help="Run Part A only (trajectory geometry)")
    parser.add_argument("--n-passages", type=int, default=None,
                        help="Max passages per prompt from stash (default: all)")
    parser.add_argument("--n-epochs", type=int, default=30,
                        help="Training epochs for v2.6 steering vector")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate for v2.6 steering vector")
    args = parser.parse_args()

    psyche = Psyche.from_family(args.family, load=True)
    print(f"Loaded family={args.family}, n_layers={psyche.n_layers}")

    n_hidden = psyche.primary_process.model.config.num_hidden_layers
    layer = round(n_hidden * 0.8125)
    intervention_layers = [round(n_hidden * f) for f in (0.25, 0.5, 0.75, 0.875)]
    print(f"N_LAYERS={n_hidden}  LAYER={layer}  INTERVENTION_LAYERS={intervention_layers}")

    run_trajectory_geometry(psyche, args.family, layer, out_dir="data",
                            n_passages=args.n_passages)

    if not args.skip_intervention:
        if psyche.superego is None:
            print("\nSkipping intervention: need at least base + superego (2 layers)")
        else:
            run_intervention(psyche, args.family, intervention_layers, out_dir="data",
                             n_epochs=args.n_epochs, lr=args.lr)

    print("\nDone.")


if __name__ == "__main__":
    main()
