# %% [markdown]
# # Libidinal Toolkit — Quick Start
# 
# Psychoanalytic analysis of LLM probability distributions.
# Compares base (id), instruct (ego), and prohibition-prefixed (superego)
# models to map repression, displacement, and condensation.
#
# **Requirements**: Colab Pro with GPU runtime, HuggingFace account with
# Llama 3.1 access.

# %% Install dependencies
# !pip install transformers accelerate bitsandbytes huggingface_hub pandas tqdm -q

# %% Authenticate
# from huggingface_hub import login
# login()

# %% Load models
from malign_logits import load_models

base_model, instruct_model, tokenizer = load_models()

# %% [markdown]
# ## 1. Discover word-level distributions
#
# `discover_top_words` explores the model's top first-token paths and
# greedily completes each to a full word, filtering to alphabetic words.

# %%
from malign_logits import discover_top_words

prompt = "He lay naked in his bed and"

ego_words = discover_top_words(instruct_model, tokenizer, prompt)
print("Ego top 10:", list(ego_words.keys())[:10])

# %% [markdown]
# ## 2. Ego vs Superego comparison
#
# The superego is the same instruct model with a prohibitive prefix.
# The delta between them reveals repression and amplification.

# %%
from malign_logits import (
    make_superego_prompt,
    discover_top_words,
    compute_repression,
)

superego_prompt = make_superego_prompt(prompt)
superego_words = discover_top_words(instruct_model, tokenizer, superego_prompt)

df = compute_repression(ego_words, superego_words)

print("\n=== REPRESSED ===")
print(df[df["repressed"]][["word", "ego", "superego", "delta"]].head(10).to_string())

print("\n=== AMPLIFIED ===")
amplified = df[df["amplified"]].sort_values("delta")
print(amplified[["word", "ego", "superego", "delta"]].head(10).to_string())

# %% [markdown]
# ## 3. Full prompt battery
#
# Run ego/superego comparison across liminal, explicit, and neutral prompts.

# %%
from malign_logits import run_prompt_battery, summarize_battery, print_repression_report

# Two-layer analysis (ego vs superego only — faster)
results = run_prompt_battery(instruct_model, tokenizer)

# Or three-layer analysis including base model (id):
# results = run_prompt_battery(instruct_model, tokenizer, base_model=base_model)

# %%
summary = summarize_battery(results)
print(summary.to_string())

# %%
# Detailed report for a specific prompt
print_repression_report(results, "sexual_liminal_1")

# %% [markdown]
# ## 4. Three-layer id analysis
#
# The id is not any single model — it emerges from the relationship
# between all three layers. A word scores high when the ego wants it,
# the superego forbids it, AND the base model has raw drive energy behind it.

# %%
from malign_logits import compute_id

base_words = discover_top_words(base_model, tokenizer, prompt)
id_scores, id_analysis = compute_id(base_words, ego_words, superego_words)

print("Top id-content:")
for word, score in list(id_scores.items())[:10]:
    a = id_analysis[word]
    print(f"  {word:20s}  id={score:.4f}  base={a['base_drive']:.4f}  "
          f"ego={a['ego_desire']:.4f}  superego={a['superego_allows']:.4f}")

# %% [markdown]
# ## 5. Displacement analysis
#
# Redistribute repressed probability mass onto semantically adjacent
# permitted words, weighted by base-model drive energy.

# %%
from malign_logits import (
    compute_displacement,
    get_base_logits,
    get_embeddings,
)

base_logits = get_base_logits(base_model, tokenizer, prompt)
embeddings = get_embeddings(base_model)

displaced_dist, condensation_log = compute_displacement(
    base_words, ego_words, superego_words,
    base_logits, embeddings, tokenizer,
)

print("Neurotic distribution (top 15):")
for word, prob in list(displaced_dist.items())[:15]:
    sup = superego_words.get(word, 0)
    gained = prob - sup
    marker = " ← SYMPTOMATIC" if gained > 0.01 else ""
    print(f"  {word:20s}  neurotic: {prob:.4f}  superego: {sup:.4f}  "
          f"gained: {gained:+.4f}{marker}")

print("\nCondensation points (words absorbing mass from multiple sources):")
for word, sources in sorted(condensation_log.items(),
                             key=lambda x: len(x[1]), reverse=True)[:5]:
    source_words = [s["source"] for s in sources]
    total_mass = sum(s["mass"] for s in sources)
    print(f"  '{word}' ← {source_words} (total mass: {total_mass:.4f})")