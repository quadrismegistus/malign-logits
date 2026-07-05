"""Displacement taxonomy and cross-family analysis.

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
    aligned_model = psyche.superego.model if psyche.superego else None
    aligned_tokenizer = psyche.superego.tokenizer if psyche.superego else None

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
                        "prompt_full": prompt,
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
                        if aligned_model is not None:
                            try:
                                row["syntagmatic_js_aligned"] = round(
                                    syntagmatic_js(aligned_model, aligned_tokenizer, prompt, src, tgt), 6
                                )
                            except Exception as e:
                                row["syntagmatic_js_aligned"] = None
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
                        **({"syntagmatic_js_aligned": None} if measure_syntagmatic and aligned_model is not None else {}),
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


def add_aligned_baseline(family_key="olmo", input_path=None, output_path=None,
                         psyche=None):
    """Add syntagmatic_js_aligned column to an existing taxonomy CSV.

    Reads pairs from the CSV, computes syntagmatic_js under the aligned
    (superego) model, and writes the augmented CSV. Skips pairs that
    already have a non-null syntagmatic_js_aligned value.
    """
    if family_key not in MODEL_FAMILIES:
        raise ValueError(f"Unknown family: {family_key}")
    input_path = input_path or f"data/taxonomy_{family_key}.csv"
    output_path = output_path or input_path

    df = pd.read_csv(input_path)
    if "syntagmatic_js_aligned" in df.columns:
        todo = df["syntagmatic_js_aligned"].isna() & df["target"].notna()
    else:
        df["syntagmatic_js_aligned"] = None
        todo = df["target"].notna()

    n_todo = todo.sum()
    if n_todo == 0:
        print("All pairs already have aligned baseline. Nothing to do.")
        return df

    owns_psyche = psyche is None
    if owns_psyche:
        from .psyche import Psyche
        print(f"Loading {family_key} models...")
        psyche = Psyche.from_family(family_key, load=True)

    aligned = psyche.superego
    if aligned is None or aligned.model is None:
        print("No superego model available. Cannot compute aligned baseline.")
        return df

    # Full prompt text: the CSV's 'prompt' column is truncated to 60 chars for
    # display, which silently changed the context for prompts longer than that.
    # Prefer prompt_full; fall back to resolving the label for older CSVs.
    def _full_prompt(row):
        pf = row.get("prompt_full")
        if isinstance(pf, str) and pf:
            return pf
        from .experiments import DEFAULT_PROMPTS, INSTITUTIONAL_PROMPTS
        label_map = {**DEFAULT_PROMPTS, **INSTITUTIONAL_PROMPTS}
        if row["label"] in label_map:
            return label_map[row["label"]]
        p = row["prompt"]
        if isinstance(p, str) and len(p) < 60:
            return p  # short enough that truncation never applied
        raise ValueError(f"cannot recover full prompt for label {row['label']!r} "
                         f"(truncated at 60 chars; re-run taxonomy to get prompt_full)")

    print(f"Computing aligned syntagmatic_js for {n_todo} pairs...")
    done = 0
    for idx in df.index[todo]:
        row = df.loc[idx]
        try:
            val = syntagmatic_js(
                aligned.model, aligned.tokenizer,
                _full_prompt(row), row["source"], row["target"],
            )
            df.at[idx, "syntagmatic_js_aligned"] = round(val, 6)
        except Exception:
            pass
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{n_todo}")

    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({n_todo} aligned values added)")

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

    if "syntagmatic_js_aligned" in df.columns and df["syntagmatic_js_aligned"].notna().any():
        print(f"\n{'=' * 60}\nBASELINE CHECK: BASE vs ALIGNED SYNTAGMATIC JS\n{'=' * 60}")
        both = df.dropna(subset=["syntagmatic_js", "syntagmatic_js_aligned"]).copy()
        both["category"] = both["label"].str.replace(r"_\d+$", "", regex=True)
        baseline = both.groupby("category").agg(
            base_synt_js=("syntagmatic_js", "mean"),
            aligned_synt_js=("syntagmatic_js_aligned", "mean"),
            count=("syntagmatic_js", "count"),
        ).round(4)
        baseline["delta"] = (baseline["aligned_synt_js"] - baseline["base_synt_js"]).round(4)
        print(baseline.to_string())
        print(f"\nInterpretation: negative delta = aligned model smooths its own substitutions")
        print(f"If neutral delta ≈ 0, the metric is noise. If neutral delta < 0,")
        print(f"alignment produces background syntagmatic smoothing even on safe content.")


def analyze_taxonomy(data_dir="data", output_path="data/taxonomy_summary.csv"):
    """Cross-family analysis of all taxonomy CSVs. No models needed.

    Reads every ``taxonomy_*.csv`` in *data_dir*, computes per-family:
    - Jakobsonian correlation (similarity vs syntagmatic_js)
    - Within-category and within-type correlations
    - Displacement type profile by category
    - Aligned baseline comparison (where available)

    Writes a summary CSV and prints a report.
    """
    import glob
    from scipy.stats import pearsonr, spearmanr

    csvs = sorted(glob.glob(f"{data_dir}/taxonomy_*.csv"))
    if not csvs:
        print(f"No taxonomy CSVs found in {data_dir}/")
        return None

    frames = []
    for path in csvs:
        df = pd.read_csv(path)
        if "family" not in df.columns:
            family = path.split("taxonomy_")[-1].replace(".csv", "")
            df["family"] = family
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["category"] = all_df["label"].str.replace(r"_\d+$", "", regex=True)

    families = sorted(all_df["family"].unique())
    print(f"Loaded {len(all_df)} pairs across {len(families)} families: {', '.join(families)}")

    # ── 1. Jakobsonian correlation per family ──
    print(f"\n{'=' * 70}")
    print("JAKOBSONIAN AXIS CORRELATION (similarity vs syntagmatic_js)")
    print(f"{'=' * 70}")

    corr_rows = []
    for fam in families:
        fdf = all_df[(all_df["family"] == fam)].dropna(subset=["similarity", "syntagmatic_js"])
        if len(fdf) < 10:
            continue
        r, p = pearsonr(fdf["similarity"], fdf["syntagmatic_js"])
        rho, sp = spearmanr(fdf["similarity"], fdf["syntagmatic_js"])
        corr_rows.append({
            "family": fam, "n_pairs": len(fdf),
            "pearson_r": round(r, 4), "pearson_p": f"{p:.2e}",
            "spearman_rho": round(rho, 4),
        })
        print(f"\n  {fam} (n={len(fdf)}): Pearson r={r:.3f}, Spearman ρ={rho:.3f}")

        # Within-category
        cats = sorted(fdf["category"].unique())
        for cat in cats:
            cdf = fdf[fdf["category"] == cat]
            if len(cdf) < 10:
                continue
            rc, _ = pearsonr(cdf["similarity"], cdf["syntagmatic_js"])
            corr_rows.append({
                "family": fam, "n_pairs": len(cdf),
                "pearson_r": round(rc, 4), "pearson_p": "",
                "spearman_rho": None, "category": cat,
            })
        cat_corrs = {cat: pearsonr(
            fdf[fdf["category"] == cat]["similarity"],
            fdf[fdf["category"] == cat]["syntagmatic_js"]
        )[0] for cat in cats if len(fdf[fdf["category"] == cat]) >= 10}
        if cat_corrs:
            rng = [f"{min(cat_corrs.values()):.2f}", f"{max(cat_corrs.values()):.2f}"]
            print(f"    Within-category r ∈ [{rng[0]}, {rng[1]}]")

    # ── 2. Category means per family ──
    print(f"\n{'=' * 70}")
    print("PARADIGMATIC vs SYNTAGMATIC BY CATEGORY (mean per family)")
    print(f"{'=' * 70}")

    summary_rows = []
    for fam in families:
        fdf = all_df[(all_df["family"] == fam)].dropna(subset=["similarity", "syntagmatic_js"])
        cat_means = fdf.groupby("category").agg(
            similarity=("similarity", "mean"),
            syntagmatic_js=("syntagmatic_js", "mean"),
            n=("similarity", "count"),
        ).round(4)
        for cat, row in cat_means.iterrows():
            summary_rows.append({
                "family": fam, "category": cat,
                "paradigmatic_sim": row["similarity"],
                "syntagmatic_js": row["syntagmatic_js"],
                "n_pairs": int(row["n"]),
            })

    summary_df = pd.DataFrame(summary_rows)
    pivot_sim = summary_df.pivot(index="category", columns="family", values="paradigmatic_sim")
    pivot_synt = summary_df.pivot(index="category", columns="family", values="syntagmatic_js")

    print("\nParadigmatic similarity:")
    print(pivot_sim.round(3).to_string())
    print("\nSyntagmatic JS:")
    print(pivot_synt.round(3).to_string())

    # ── 3. Displacement type profile per family ──
    print(f"\n{'=' * 70}")
    print("DISPLACEMENT TYPE PROFILE (% of pairs)")
    print(f"{'=' * 70}")

    type_rows = []
    for fam in families:
        fdf = all_df[all_df["family"] == fam]
        total = len(fdf)
        for dtype in ["register_shift", "category_shift", "genre_change", "archaic"]:
            n = (fdf["displacement_type"] == dtype).sum()
            type_rows.append({
                "family": fam, "displacement_type": dtype,
                "count": n, "pct": round(100 * n / total, 1) if total else 0,
            })

    type_df = pd.DataFrame(type_rows)
    type_pivot = type_df.pivot(index="family", columns="displacement_type", values="pct")
    col_order = ["register_shift", "category_shift", "genre_change", "archaic"]
    type_pivot = type_pivot[[c for c in col_order if c in type_pivot.columns]]
    print(type_pivot.round(1).to_string())

    # Per category × family
    print(f"\n{'=' * 70}")
    print("DISPLACEMENT TYPE BY CATEGORY × FAMILY")
    print(f"{'=' * 70}")

    for fam in families:
        fdf = all_df[all_df["family"] == fam]
        cat_type = fdf.groupby(["category", "displacement_type"]).size().unstack(fill_value=0)
        cat_pct = cat_type.div(cat_type.sum(axis=1), axis=0).mul(100).round(0).astype(int)
        cat_pct = cat_pct[[c for c in col_order if c in cat_pct.columns]]
        print(f"\n  {fam}:")
        print(f"  {cat_pct.to_string()}")

    # ── 4. Aligned baseline (where available) ──
    has_aligned = "syntagmatic_js_aligned" in all_df.columns and all_df["syntagmatic_js_aligned"].notna().any()
    if has_aligned:
        print(f"\n{'=' * 70}")
        print("BASELINE CHECK: BASE vs ALIGNED SYNTAGMATIC JS")
        print(f"{'=' * 70}")

        for fam in families:
            fdf = all_df[all_df["family"] == fam].dropna(
                subset=["syntagmatic_js", "syntagmatic_js_aligned"]
            )
            if fdf.empty:
                continue
            baseline = fdf.groupby("category").agg(
                base=("syntagmatic_js", "mean"),
                aligned=("syntagmatic_js_aligned", "mean"),
                n=("syntagmatic_js", "count"),
            ).round(4)
            baseline["delta"] = (baseline["aligned"] - baseline["base"]).round(4)
            print(f"\n  {fam}:")
            print(f"  {baseline.to_string()}")

    # ── Save summary ──
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved {output_path}")

    corr_df = pd.DataFrame(corr_rows)
    corr_path = output_path.replace(".csv", "_correlations.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"Saved {corr_path}")

    return summary_df
