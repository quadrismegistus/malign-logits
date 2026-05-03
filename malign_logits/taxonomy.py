"""Displacement taxonomy.

Classifies each (source, target) repression/sublimation pair from a
``Psyche.analyze(...).displacement_map()`` into one of four types using
contextual spaCy POS tagging and wordfreq corpus frequencies:

- **register_shift** — same POS, high similarity. Same referent, different
  social/lexical register (kill -> hurt).
- **category_shift** — different POS. Charge migrates across grammatical
  category (kill[V] -> harm[N]).
- **genre_change** — target is a meta/function token from ``GENRE_TOKENS``.
  The model abandons synonym substitution and breaks the syntagmatic chain
  (refusal, format change, multiple-choice template).
- **archaic** — target Zipf frequency below ``ARCHAIC_ZIPF`` (e.g. smite,
  hath). Modern vocabulary displaced onto rare/literary register.

Also computes ``syntagmatic_js`` per pair: the JS divergence between
``p(next | prompt + source)`` and ``p(next | prompt + target)`` under the
base model. High JS means the substitute jars the next-token chain even
when the substitute is paradigmatically close to the source — a continuous
measure of syntagmatic-axis disruption complementary to the categorical
``displacement_type`` and the existing ``similarity`` (paradigmatic) score.
"""
import gc

import pandas as pd
import torch

from . import MODEL_FAMILIES
from .analysis import js_divergence
from .experiments import DEFAULT_PROMPTS, TIER1_PROMPTS

# Tokens that cluster around literary archaism. Anything below this Zipf
# corpus frequency is treated as a low-frequency / archaic displacement.
# Reference: smite=2.93, hath=3.62, kill=5.09, gulped=2.7.
ARCHAIC_ZIPF = 3.0

# Function/meta tokens that signal genre change when displaced onto.
GENRE_TOKENS = {
    "what", "who", "where", "when", "why", "how", "which",
    "What", "Who", "Where", "When", "Why", "How", "Which",
    "WHAT", "WHO", "WHERE", "WHEN", "WHY", "HOW", "WHICH",
    "Options", "options", "Question", "question",
    "____", "___", "__", "...", "the", "a", "an",
    "is", "are", "was", "were", "it", "this", "that",
    "to", "of", "for", "in", "on", "at", "by", "with",
    "she", "he", "her", "his", "they", "them",
}


def get_pos_and_freq(words, prompt, nlp, zipf_frequency):
    """Contextual POS tag (spaCy on prompt+word) and Zipf corpus frequency."""
    result = {}
    for w in words:
        doc = nlp(prompt + " " + w)
        pos = doc[-1].pos_ if len(doc) > 0 else "X"
        zipf = zipf_frequency(w, "en")
        result[w] = (pos, round(zipf, 2))
    return result


def classify_pair(src, tgt, src_pos, tgt_pos, tgt_freq):
    """Classify a (source, target) displacement pair into a taxonomy type."""
    if tgt in GENRE_TOKENS:
        return "genre_change"
    if tgt_freq is not None and tgt_freq < ARCHAIC_ZIPF:
        return "archaic"
    if src_pos == tgt_pos:
        return "register_shift"
    return "category_shift"


def syntagmatic_js(model, tokenizer, prompt, src, tgt):
    """JS divergence between p(next | prompt + src) and p(next | prompt + tgt).

    Higher = the substitute jars the syntagmatic chain even if the substitute
    is paradigmatically close. Two forward passes per pair, no generation."""
    ids_src = tokenizer.encode(prompt + " " + src, return_tensors="pt").to(model.device)
    ids_tgt = tokenizer.encode(prompt + " " + tgt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits_src = model(ids_src).logits[0, -1, :].cpu().float()
        logits_tgt = model(ids_tgt).logits[0, -1, :].cpu().float()
    return float(js_divergence(logits_src, logits_tgt))


def run_taxonomy(family_key="olmo", all_prompts=False, output_path=None,
                 measure_syntagmatic=True, psyche=None):
    """Classify displacement pairs and write them to a CSV.

    Args:
        family_key: Model family registered in ``MODEL_FAMILIES``.
        all_prompts: Use all ``DEFAULT_PROMPTS`` (47) instead of
            ``TIER1_PROMPTS`` (18).
        output_path: CSV destination (default
            ``data/displacement_taxonomy.csv``).
        measure_syntagmatic: Compute ``syntagmatic_js`` for every paired
            row. Adds two forward passes per pair through the base model.
        psyche: Optional pre-loaded ``Psyche``. If ``None`` we load and
            tear down. If provided, the caller owns it.

    Returns:
        DataFrame with one row per displacement pair (and one row per
        orphan repressed source word, treated as ``genre_change``).
    """
    import spacy
    from wordfreq import zipf_frequency

    if family_key not in MODEL_FAMILIES:
        raise ValueError(f"Unknown family: {family_key}")
    prompts = DEFAULT_PROMPTS if all_prompts else TIER1_PROMPTS
    output_path = output_path or f"data/taxonomy_{family_key}.csv"

    owns_psyche = psyche is None
    if owns_psyche:
        from .psyche import Psyche
        print(f"Loading {family_key} models...")
        psyche = Psyche.from_family(family_key, load=True)

    base_model = psyche.primary_process.model
    tokenizer = psyche.primary_process.tokenizer

    print("Loading spaCy + wordfreq...")
    nlp = spacy.load("en_core_web_sm")

    all_rows = []
    for label, prompt in prompts.items():
        print(f"\n  {label}: {prompt[:60]}")
        analysis = psyche.analyze(prompt)
        try:
            dm = analysis.displacement_map()
        except Exception as e:
            print(f"    Skipping: {e}")
            continue

        for axis in ["repression", "sublimation"]:
            axis_data = dm.get(axis, {})
            pairs = axis_data.get("pairs", [])
            source_words = axis_data.get("source", [])
            paired_sources = set(src for src, _, _, _ in pairs)

            if pairs:
                words = set()
                for src, tgt, _, _ in pairs:
                    words.add(src)
                    words.add(tgt)
                word_info = get_pos_and_freq(list(words), prompt, nlp, zipf_frequency)

                for src, tgt, sim, layer in pairs:
                    src_pos, _ = word_info.get(src, ("X", None))
                    tgt_pos, tgt_freq = word_info.get(tgt, ("X", None))
                    dtype = classify_pair(src, tgt, src_pos, tgt_pos, tgt_freq)
                    row = {
                        "family": family_key,
                        "label": label,
                        "prompt": prompt[:60],
                        "axis": axis,
                        "source": src,
                        "target": tgt,
                        "similarity": sim,
                        "layer": layer,
                        "source_pos": src_pos,
                        "target_pos": tgt_pos,
                        "target_freq": tgt_freq,
                        "displacement_type": dtype,
                    }
                    if measure_syntagmatic:
                        try:
                            row["syntagmatic_js"] = round(
                                syntagmatic_js(base_model, tokenizer, prompt, src, tgt), 6
                            )
                        except Exception as e:
                            row["syntagmatic_js"] = None
                            print(f"    syntagmatic_js failed for ({src} -> {tgt}): {e}")
                    all_rows.append(row)

            # Orphan repressed sources (no semantic-pair target) are genre_change:
            # the model didn't substitute, it broke the chain.
            orphans = [w for w in source_words if w not in paired_sources]
            if orphans:
                orphan_info = get_pos_and_freq(orphans, prompt, nlp, zipf_frequency)
                for w in orphans:
                    w_pos, _ = orphan_info.get(w, ("X", None))
                    all_rows.append({
                        "family": family_key,
                        "label": label,
                        "prompt": prompt[:60],
                        "axis": axis,
                        "source": w,
                        "target": None,
                        "similarity": None,
                        "layer": None,
                        "source_pos": w_pos,
                        "target_pos": None,
                        "target_freq": None,
                        "displacement_type": "genre_change",
                        **({"syntagmatic_js": None} if measure_syntagmatic else {}),
                    })

        prompt_rows = [r for r in all_rows if r["label"] == label]
        if prompt_rows:
            types = {}
            for r in prompt_rows:
                types[r["displacement_type"]] = types.get(r["displacement_type"], 0) + 1
            print(f"    {len(prompt_rows)} pairs: {types}")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} pairs to {output_path}")

    if not df.empty:
        _print_summary(df)

    if owns_psyche:
        del psyche
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return df


def _print_summary(df):
    print(f"\n{'=' * 60}\nDISPLACEMENT TYPE SUMMARY\n{'=' * 60}")
    summary = df.groupby(["axis", "displacement_type"]).size().reset_index(name="count")
    summary["pct"] = (
        summary["count"] / summary.groupby("axis")["count"].transform("sum") * 100
    ).round(1)
    print(summary.to_string(index=False))

    df = df.copy()
    df["category"] = df["label"].str.replace(r"_\d+$", "", regex=True)
    cat_summary = df.groupby(["category", "displacement_type"]).size().unstack(fill_value=0)
    print(f"\n{'=' * 60}\nBY CONTENT CATEGORY\n{'=' * 60}")
    print(cat_summary.to_string())

    if "syntagmatic_js" in df.columns and df["syntagmatic_js"].notna().any():
        print(f"\n{'=' * 60}\nSYNTAGMATIC JS BY DISPLACEMENT TYPE\n{'=' * 60}")
        synt = (
            df.dropna(subset=["syntagmatic_js"])
              .groupby("displacement_type")["syntagmatic_js"]
              .agg(["mean", "std", "count"])
              .round(4)
        )
        print(synt.to_string())

        print(f"\n{'=' * 60}\nSYNTAGMATIC JS BY CONTENT CATEGORY\n{'=' * 60}")
        synt_cat = (
            df.dropna(subset=["syntagmatic_js"])
              .groupby("category")["syntagmatic_js"]
              .agg(["mean", "std", "count"])
              .round(4)
        )
        print(synt_cat.to_string())
