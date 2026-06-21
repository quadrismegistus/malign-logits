"""
Chinese displacement figure: EN vs ZH argmax comparison for Qwen.

Shows how alignment operates differently across languages — Chinese
prompts bypass the exam-question genre collapse seen in English.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib
import matplotlib.font_manager as fm

cjk_font_path = "/System/Library/Fonts/PingFang.ttc"
if not __import__('os').path.exists(cjk_font_path):
    cjk_font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
fm.fontManager.addfont(cjk_font_path)
cjk_prop = fm.FontProperties(fname=cjk_font_path)

df = pd.read_csv("data/qwen_chinese_logits.csv")

PROMPTS = ["anger", "violence", "sexual", "love", "worker"]

# Separate base and aligned
base = df[df["model"] == "qwen_base"].copy()
aligned = df[df["model"] == "qwen_aligned"].copy()

fig, axes = plt.subplots(2, 5, figsize=(16, 7))

for j, prompt in enumerate(PROMPTS):
    for lang_idx, lang in enumerate(["en", "zh"]):
        ax = axes[lang_idx, j]
        pk = f"{prompt}_{lang}"

        b = base[base["prompt_key"] == pk]
        a = aligned[aligned["prompt_key"] == pk]

        if len(b) == 0 or len(a) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                   transform=ax.transAxes)
            continue

        b_words = str(b.iloc[0]["top5_words"]).split("|")
        b_probs = [float(x) for x in str(b.iloc[0]["top5_probs"]).split("|")]
        a_words = str(a.iloc[0]["top5_words"]).split("|")
        a_probs = [float(x) for x in str(a.iloc[0]["top5_probs"]).split("|")]

        # Plot base and aligned top-5 as horizontal bars
        y_pos = np.arange(5)

        bars_b = ax.barh(y_pos + 0.2, b_probs[:5], 0.35, label='Base',
                        color='#95a5a6', edgecolor='white')
        bars_a = ax.barh(y_pos - 0.2, a_probs[:5], 0.35, label='Aligned',
                        color='#2c3e50', edgecolor='white')

        # Label bars with token text (use CJK font for ZH)
        fp = cjk_prop if lang == "zh" else None
        for k, (word, prob) in enumerate(zip(b_words[:5], b_probs[:5])):
            ax.text(prob + 0.01, k + 0.2, word, va='center', fontsize=7,
                   color='#666', fontproperties=fp)
        for k, (word, prob) in enumerate(zip(a_words[:5], a_probs[:5])):
            ax.text(prob + 0.01, k - 0.2, word, va='center', fontsize=7,
                   fontweight='bold', color='#2c3e50', fontproperties=fp)

        ax.set_yticks([])
        ax.set_xlim(0, max(max(b_probs[:5]), max(a_probs[:5])) * 1.6)
        ax.invert_yaxis()

        if lang_idx == 0:
            ax.set_title(prompt.capitalize(), fontsize=12, fontweight='bold')
        if j == 0:
            lang_label = "English" if lang == "en" else "Chinese (中文)"
            ax.set_ylabel(lang_label, fontsize=11, fontweight='bold')
        if lang_idx == 1:
            ax.set_xlabel("Probability", fontsize=9)
        if j == 0 and lang_idx == 0:
            ax.legend(fontsize=7, loc='lower right')

fig.suptitle("Qwen 2.5: Cross-Lingual Displacement\n"
             "English triggers exam blanks, Chinese preserves semantic content",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("figures/F25_chinese_displacement.png", dpi=200,
            bbox_inches='tight', facecolor='white')
print("Saved figures/F25_chinese_displacement.png")

# Key comparison table
print("\nKey argmax shifts:")
print(f"{'Prompt':<10} {'Lang':<5} {'Base argmax':<15} {'Aligned argmax':<15} {'Entropy shift':>15}")
for prompt in PROMPTS:
    for lang in ["en", "zh"]:
        pk = f"{prompt}_{lang}"
        b = base[base["prompt_key"] == pk]
        a = aligned[aligned["prompt_key"] == pk]
        if len(b) > 0 and len(a) > 0:
            b_top = b.iloc[0]["top1"]
            a_top = a.iloc[0]["top1"]
            b_h = b.iloc[0]["entropy"]
            a_h = a.iloc[0]["entropy"]
            delta_h = a_h - b_h
            print(f"{prompt:<10} {lang:<5} {str(b_top):<15} {str(a_top):<15} {delta_h:>+.2f}")
