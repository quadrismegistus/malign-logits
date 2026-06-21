"""
F25 reasoning phase boundary figure: think H vs response H.

Shows entropy in thinking vs response phases across R1-Llama, R1-Qwen,
and SmolLM3-Think, broken down by prompt category.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

FILES = {
    "R1-Llama": "data/mega_gen_r1_reasoning.csv",
    "R1-Qwen": "data/mega_gen_reasoning_r1_qwen.csv",
    "SmolLM3": "data/mega_gen_reasoning_smol3_think.csv",
}

PROMPTS = ["anger", "violence", "sexual", "worker", "love"]
PROMPT_COLORS = {
    "anger": "#e74c3c",
    "violence": "#2c3e50",
    "sexual": "#8e44ad",
    "worker": "#e67e22",
    "love": "#27ae60",
}

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True, sharex=True)

for idx, (model_name, fpath) in enumerate(FILES.items()):
    ax = axes[idx]
    df = pd.read_csv(fpath)

    for prompt in PROMPTS:
        sub = df[df["prompt_key"] == prompt]
        think_h = sub[sub["phase"] == "think"]["entropy"].mean()
        resp_h = sub[sub["phase"] == "response"]["entropy"].mean()

        ax.scatter(think_h, resp_h, color=PROMPT_COLORS[prompt],
                  s=120, zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(prompt, (think_h, resp_h), fontsize=8,
                   xytext=(5, 5), textcoords='offset points')

    # Diagonal line (think = response)
    lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1]) + 0.2]
    ax.plot(lim, lim, 'k--', alpha=0.3, linewidth=1)

    ax.set_title(model_name, fontsize=12, fontweight='bold')
    ax.set_xlabel("Thinking entropy (nats)", fontsize=10)
    if idx == 0:
        ax.set_ylabel("Response entropy (nats)", fontsize=10)

    # Add region labels
    ax.text(0.95, 0.05, "think > response\n(narrowing)",
           transform=ax.transAxes, fontsize=7, alpha=0.5,
           ha='right', va='bottom')
    ax.text(0.05, 0.95, "response > think\n(broadening)",
           transform=ax.transAxes, fontsize=7, alpha=0.5,
           ha='left', va='top')

fig.suptitle("Reasoning Phase Boundary: Thinking vs Response Entropy\n"
             "R1 narrows at </think>, SmolLM3 broadens",
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("figures/F25_reasoning_phase_boundary.png", dpi=200,
            bbox_inches='tight', facecolor='white')
print("Saved figures/F25_reasoning_phase_boundary.png")

# Summary stats
print("\nPhase entropy summary:")
print(f"{'Model':<12} {'Prompt':<10} {'Think H':>10} {'Resp H':>10} {'Delta':>10} {'Direction':>12}")
for model_name, fpath in FILES.items():
    df = pd.read_csv(fpath)
    for prompt in PROMPTS:
        sub = df[df["prompt_key"] == prompt]
        th = sub[sub["phase"] == "think"]["entropy"].mean()
        rh = sub[sub["phase"] == "response"]["entropy"].mean()
        delta = rh - th
        direction = "broadens" if delta > 0 else "narrows"
        print(f"{model_name:<12} {prompt:<10} {th:>10.3f} {rh:>10.3f} {delta:>+10.3f} {direction:>12}")
