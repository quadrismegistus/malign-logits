"""
F26 — Three distributions on the anger prompt.

Publication-quality figure: base, DPO-aligned, and R1-reasoning
probability distributions over top base-model tokens for
"She was so angry she wanted to".
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.special import softmax
from transformers import AutoTokenizer

from malign_logits.probe import Probe

# ── Load data ────────────────────────────────────────────────────────
base = Probe("allenai/Olmo-3-1025-7B")
dpo = Probe("allenai/Olmo-3-7B-Instruct-DPO")
r1 = Probe("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

base_logits = base.logits("anger", gen=0, pos=0)
dpo_logits = dpo.logits("anger", gen=0, pos=0)
r1_logits = r1.logits("anger", gen=0, pos=0)

base_probs = softmax(base_logits)
dpo_probs = softmax(dpo_logits)
r1_probs = softmax(r1_logits)

# Tokenizers
base_tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")
r1_tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

# ── Top 30 tokens from base model ───────────────────────────────────
N_TOKENS = 30
top_idx = np.argsort(base_probs)[::-1][:N_TOKENS]

tokens, base_p, dpo_p, r1_p = [], [], [], []

for idx in top_idx:
    token_text = base_tok.decode([idx])
    label = token_text.strip()
    if not label:
        label = repr(token_text)
    tokens.append(label)
    base_p.append(float(base_probs[idx]))
    dpo_p.append(float(dpo_probs[idx]))

    # R1 uses Qwen tokenizer — look up by decoded text
    r1_ids = r1_tok.encode(token_text, add_special_tokens=False)
    if len(r1_ids) == 1:
        r1_p.append(float(r1_probs[r1_ids[0]]))
    else:
        r1_p.append(1e-8)

base_p = np.array(base_p)
dpo_p = np.array(dpo_p)
r1_p = np.array(r1_p)

# ── Figure ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

fig, ax = plt.subplots(figsize=(12, 4.5))

x = np.arange(N_TOKENS)
w = 0.26

ax.bar(x - w, base_p, w, label="Base  (primary process)",
       color="#2166ac", alpha=0.9, edgecolor="none", zorder=3)
ax.bar(x,     dpo_p,  w, label="DPO  (aligned)",
       color="#d6604d", alpha=0.9, edgecolor="none", zorder=3)
ax.bar(x + w, r1_p,   w, label="R1-Distill  (reasoning)",
       color="#4daf4a", alpha=0.9, edgecolor="none", zorder=3)

ax.set_yscale("log")
ax.set_ylabel("Probability", fontsize=11, labelpad=8)
ax.set_xticks(x)
ax.set_xticklabels(tokens, rotation=50, ha="right", fontsize=8.5,
                   fontfamily="monospace")

# Clean axes
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.6)
ax.spines["bottom"].set_linewidth(0.6)
ax.grid(False)
ax.tick_params(axis="both", which="both", length=3, pad=3)
ax.tick_params(axis="x", which="minor", bottom=False)

# Y-axis limits and ticks
ax.set_ylim(1e-5, 1.0)
ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

# Legend
ax.legend(frameon=False, fontsize=9.5, loc="upper right",
          handlelength=1.2, handletextpad=0.5)

# Title — italic prompt text
ax.set_title('“She was so angry she wanted to”',
             fontsize=12, style="italic", pad=12, fontfamily="serif")

# Tight x-range
ax.set_xlim(-0.6, N_TOKENS - 0.4)

fig.tight_layout()

out_path = "figures/F26_three_distributions.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")

# Diagnostics
print(f"\nTokens shown: {N_TOKENS}")
print("Top-5 base:", [(tokens[i], f"{base_p[i]:.4f}")
                       for i in np.argsort(base_p)[::-1][:5]])
print("Top-5 DPO: ", [(tokens[i], f"{dpo_p[i]:.4f}")
                       for i in np.argsort(dpo_p)[::-1][:5]])
print("Top-5 R1:  ", [(tokens[i], f"{r1_p[i]:.4f}")
                       for i in np.argsort(r1_p)[::-1][:5]])

plt.close()
