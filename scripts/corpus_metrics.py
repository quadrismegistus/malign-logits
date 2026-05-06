"""Compute passage metrics across all corpora with length normalization.

Loads model generations, dream reports, waking narratives (Hippocorpus),
and C20 fiction. Truncates each passage to the minimum number of sentences
needed to exceed 100 words, so all corpora are compared at matched length.

Outputs a single CSV with corpus/family/model columns for unified analysis.

Usage:
    python scripts/corpus_metrics.py [--min-words 100] [--output data/corpus_metrics.csv]
"""
import argparse
import json

import nltk
import pandas as pd

from malign_logits.embedding import compute_passage_metrics, load_generations_from_stash


def truncate_to_min_sentences(text, min_words=100):
    """Return the fewest complete sentences that exceed min_words.

    Uses NLTK sentence tokenizer. Returns None if the text can't reach
    min_words even with all sentences included.
    """
    sents = nltk.sent_tokenize(str(text))
    words = 0
    kept = []
    for s in sents:
        kept.append(s)
        words += len(s.split())
        if words >= min_words:
            return " ".join(kept), words, len(kept)
    if words >= min_words:
        return " ".join(kept), words, len(kept)
    return None, words, len(kept)


def load_model_generations():
    """Load from generation stash, truncate each passage."""
    raw = load_generations_from_stash()
    if raw.empty:
        raw = pd.read_parquet("data/gen_battery_raw.parquet")
    rows = []
    for _, r in raw.iterrows():
        rows.append({
            "corpus": r["family"],
            "subcorpus": r["model"],
            "prompt": str(r.get("prompt", "")),
            "label": r.get("label", ""),
            "text": str(r["psg"]),
        })
    return pd.DataFrame(rows)


def load_dreams():
    """Load cleaned dream reports."""
    path = "data/dreams_sample_500_cleaned.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.read_csv("data/dreams_sample_500.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "corpus": "dreams",
            "subcorpus": "dream",
            "prompt": "",
            "label": "dream",
            "text": str(r["text"]),
        })
    return pd.DataFrame(rows)


def load_hippocorpus():
    """Load waking narrative sample."""
    df = pd.read_csv("data/hippocorpus_sample_500.csv")
    col = "story" if "story" in df.columns else "text"
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "corpus": "waking",
            "subcorpus": "recalled",
            "prompt": "",
            "label": "waking",
            "text": str(r[col]),
        })
    return pd.DataFrame(rows)


def load_fiction():
    """Load C20 fiction narration passages."""
    rows = []
    with open("data/markmark_c20_narration_500.jsonl") as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "corpus": "c20_fiction",
                "subcorpus": "narration",
                "prompt": "",
                "label": "fiction",
                "text": d["text"],
            })
    return pd.DataFrame(rows)


def load_abstracts():
    """Load arxiv abstracts."""
    df = pd.read_csv("data/arxiv_abstracts_500.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "corpus": "abstracts",
            "subcorpus": "arxiv",
            "prompt": "",
            "label": "abstract",
            "text": str(r["text"]),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-words", type=int, default=75,
                        help="Minimum words per truncated passage (default: 75)")
    parser.add_argument("--output", "-o", default="data/corpus_metrics.csv")
    parser.add_argument("--ref-model", default="gpt2",
                        help="Reference model for surprisal (default: gpt2)")
    parser.add_argument("--add-ref", default=None, action="append",
                        help="Additional reference model(s) for surprisal (repeatable)")
    parser.add_argument("--all-refs", action="store_true",
                        help="Run all standard reference models (GPT-2, Llama, Mistral)")
    parser.add_argument("--add-embedder", default=None, action="append",
                        help="Additional sentence embedder(s) for drift (repeatable)")
    parser.add_argument("--all-embedders", action="store_true",
                        help="Run all standard embedders (MiniLM, mpnet, bge-m3)")
    args = parser.parse_args()

    ALL_REFS = [
        "meta-llama/Llama-3.1-8B",
        "mistralai/Mistral-7B-v0.1",
    ]
    ALL_EMBEDDERS = [
        "paraphrase-multilingual-mpnet-base-v2",
        "BAAI/bge-m3",
    ]

    if args.all_refs:
        args.add_ref = (args.add_ref or []) + ALL_REFS
    if args.all_embedders:
        args.add_embedder = (args.add_embedder or []) + ALL_EMBEDDERS

    print("Loading corpora...")
    frames = []
    for name, loader in [
        ("model_generations", load_model_generations),
        ("dreams", load_dreams),
        ("hippocorpus", load_hippocorpus),
        ("c20_fiction", load_fiction),
        ("abstracts", load_abstracts),
    ]:
        try:
            df = loader()
            print(f"  {name}: {len(df)} passages")
            frames.append(df)
        except Exception as e:
            print(f"  {name}: SKIPPED ({e})")

    all_df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal before truncation: {len(all_df)}")

    # Truncate each passage
    print(f"Truncating to min {args.min_words} words at sentence boundary...")
    trunc_rows = []
    skipped = 0
    for _, r in all_df.iterrows():
        result, n_words, n_sents = truncate_to_min_sentences(r["text"], args.min_words)
        if result is None:
            skipped += 1
            continue
        trunc_rows.append({
            "corpus": r["corpus"],
            "subcorpus": r["subcorpus"],
            "prompt": r["prompt"],
            "label": r["label"],
            "text": result,
            "n_words_truncated": n_words,
            "n_sents_truncated": n_sents,
        })
    trunc_df = pd.DataFrame(trunc_rows)
    print(f"After truncation: {len(trunc_df)} passages ({skipped} skipped < {args.min_words} words)")

    # Word count stats by corpus
    print(f"\nWord counts after truncation:")
    for corpus in sorted(trunc_df["corpus"].unique()):
        sub = trunc_df[trunc_df["corpus"] == corpus]
        wc = sub["n_words_truncated"]
        print(f"  {corpus:15s}  n={len(sub):5d}  words: {wc.mean():.0f} ± {wc.std():.0f}  (min={wc.min()}, max={wc.max()})")

    # Build DataFrame for compute_passage_metrics
    psg_df = pd.DataFrame({
        "prompt": trunc_df["prompt"],
        "model": trunc_df["subcorpus"],
        "psg": trunc_df["text"],
        "family": trunc_df["corpus"],
        "label": trunc_df["label"],
    })

    # Compute metrics with primary reference model
    result = compute_passage_metrics(psg_df, min_sentences=3,
                                     ref_model_name=args.ref_model)
    print(f"\nComputed metrics for {len(result)} passages")

    # Additional reference models for surprisal
    if args.add_ref:
        from malign_logits.embedding import passage_surprisal, _load_surprisal_model, _get_gen_stash
        import numpy as np
        from tqdm import tqdm
        stash = _get_gen_stash()

        for ref in args.add_ref:
            ref_short = ref.split("/")[-1].replace("-", "_").replace(".", "_").lower()
            col_name = f"surprisal_{ref_short}"
            print(f"\nComputing surprisal with {ref}...")

            # Reset the global model so _load_surprisal_model loads the new one
            import malign_logits.embedding as _emb
            _emb._surprisal_model = None
            _emb._surprisal_tokenizer = None

            ref_model, ref_tok = _load_surprisal_model(ref)

            surp_vals = []
            cached = computed = 0
            for _, r in tqdm(result.iterrows(), total=len(result),
                             desc=f"{ref_short}"):
                text = str(r["psg"]).rstrip()
                prompt = str(r.get("prompt", "")).strip()
                ts_key = ("token_surprisals_v3", ref, prompt, text)
                if ts_key in stash:
                    tok_surps = stash[ts_key]
                    cached += 1
                else:
                    s = passage_surprisal(text, model=ref_model, tokenizer=ref_tok,
                                          prompt_prefix=prompt)
                    tok_surps = s["token_surprisals"]
                    stash[ts_key] = tok_surps
                    computed += 1
                if tok_surps:
                    surp_vals.append(round(float(np.mean([v for _, v in tok_surps])), 4))
                else:
                    surp_vals.append(None)
            result[col_name] = surp_vals
            print(f"  {cached} cached, {computed} computed")

            # Free memory
            del ref_model, ref_tok
            _emb._surprisal_model = None
            _emb._surprisal_tokenizer = None
            import gc; gc.collect()

    # Additional sentence embedders for drift
    if args.add_embedder:
        from malign_logits.embedding import (_get_embedder, _get_gen_stash,
                                             _split_sentences, drift_metrics_from_embeddings,
                                             DEFAULT_EMBEDDER)
        import numpy as np
        from tqdm import tqdm
        stash = _get_gen_stash()

        for emb_name in args.add_embedder:
            emb_short = emb_name.split("/")[-1].replace("-", "_").replace(".", "_").lower()
            print(f"\nComputing drift with {emb_name}...")

            # Reset global embedder
            import malign_logits.embedding as _emb
            _emb._embedder = None
            _emb._embedder_name = None

            embedder = _get_embedder(emb_name)

            drift_vals = []
            dir_vals = []
            cached = computed = 0
            for _, r in tqdm(result.iterrows(), total=len(result),
                             desc=f"{emb_short}"):
                text = str(r["psg"]).rstrip()
                prompt = str(r.get("prompt", "")).strip()
                se_key = ("sent_embeddings_v3", emb_name, prompt, text)
                if se_key in stash:
                    sent_vecs = stash[se_key]
                    cached += 1
                else:
                    sents = _split_sentences(text)
                    if len(sents) < 3:
                        drift_vals.append(None)
                        dir_vals.append(None)
                        continue
                    if prompt and sents:
                        sents[0] = prompt + " " + sents[0]
                    vecs = embedder.encode(sents, show_progress_bar=False)
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
                    sent_vecs = (vecs / norms).tolist()
                    stash[se_key] = sent_vecs
                    computed += 1

                if sent_vecs and len(sent_vecs) >= 3:
                    d = drift_metrics_from_embeddings(sent_vecs)
                    drift_vals.append(d.get("total_drift"))
                    dir_vals.append(d.get("directedness"))
                else:
                    drift_vals.append(None)
                    dir_vals.append(None)

            result[f"drift_{emb_short}"] = drift_vals
            result[f"directedness_{emb_short}"] = dir_vals
            print(f"  {cached} cached, {computed} computed")

            # Free memory
            _emb._embedder = None
            _emb._embedder_name = None

    # Save
    result.to_csv(args.output, index=False)
    print(f"Saved {args.output}")

    # Summary comparison
    print(f"\n{'=' * 90}")
    print("SUMMARY BY CORPUS")
    print(f"{'=' * 90}")
    print(f"{'corpus':15s} {'subcorpus':12s} {'drift':>8s} {'surprisal':>10s} {'directed':>10s} {'metonymy':>10s} {'n':>6s}")
    print("-" * 75)

    for corpus in sorted(result["family"].unique()):
        for sub in sorted(result[result["family"] == corpus]["model"].unique()):
            s = result[(result["family"] == corpus) & (result["model"] == sub)]
            print(f"{corpus:15s} {sub:12s} {s.total_drift.mean():8.3f} {s.mean_surprisal.mean():10.3f} {s.directedness.mean():10.3f} {s.metonymy_idx.mean():10.3f} {len(s):6d}")

    # Model-only z-scores
    models = result[~result["family"].isin(["dreams", "waking", "c20_fiction"])]
    if not models.empty:
        print(f"\n{'=' * 90}")
        print("Z-SCORES (relative to model generation distribution)")
        print(f"{'=' * 90}")
        for corpus in ["dreams", "waking", "c20_fiction"]:
            sub = result[result["family"] == corpus]
            if sub.empty:
                continue
            zs = []
            for col in ["total_drift", "mean_surprisal", "directedness", "metonymy_idx"]:
                m, s = models[col].mean(), models[col].std()
                z = (sub[col].mean() - m) / s if s > 0 else 0
                zs.append(f"{col.replace('total_drift','drift').replace('mean_surprisal','surp').replace('directedness','dir').replace('metonymy_idx','met')}={z:+.2f}σ")
            print(f"  {corpus:15s}  {'  '.join(zs)}")


def summary(csv_path="data/corpus_metrics.csv"):
    """Print markdown-ready summary tables from existing corpus_metrics.csv."""
    import re
    import numpy as np

    df = pd.read_csv(csv_path)

    HUMAN = {'dreams', 'waking', 'c20_fiction', 'abstracts'}
    label_map = {'base': 'BASE', 'ego': 'SFT', 'superego': 'DPO',
                 'instruct': 'RLVR', 'dream': 'dream', 'recalled': 'waking',
                 'narration': 'fiction', 'arxiv': 'abstract'}

    # Genre classifier
    def is_template(row):
        gt = row.get('genre_type', 'narrative')
        return gt != 'narrative' if pd.notna(gt) else False

    df['_is_template'] = df.apply(is_template, axis=1)
    df['_layer'] = df['model'].map(lambda m: label_map.get(m, m.upper()))
    df['_is_ai'] = ~df['family'].isin(HUMAN)
    df['_texttype'] = df['family'].apply(lambda f: 'AI' if f not in HUMAN else f)
    df['_category'] = df['label'].str.replace(r'_\d+$', '', regex=True)

    # Compute median z-scores
    surp_cols = [c for c in ['mean_surprisal', 'surprisal_llama', 'surprisal_mistral'] if c in df.columns]
    drift_cols = [c for c in ['total_drift', 'drift_mpnet', 'drift_bge_m3'] if c in df.columns]

    for col in surp_cols + drift_cols:
        vals = df[col].dropna()
        m, s = vals.mean(), vals.std()
        df[f'_{col}_z'] = (df[col] - m) / s

    if len(surp_cols) > 1:
        df['_surp_z'] = df[[f'_{c}_z' for c in surp_cols]].median(axis=1)
    else:
        df['_surp_z'] = df[f'_{surp_cols[0]}_z']

    if len(drift_cols) > 1:
        df['_drift_z'] = df[[f'_{c}_z' for c in drift_cols]].median(axis=1)
    else:
        df['_drift_z'] = df[f'_{drift_cols[0]}_z']

    # ── Table 1: By text type (sorted by surprisal desc) ──
    print("## Median z-scores by text type")
    print()
    print("| Text type | Surprisal (z) | Drift (z) | n |")
    print("|---|---|---|---|")
    tt_rows = []
    for tt in ['c20_fiction', 'abstracts', 'dreams', 'waking', 'AI']:
        sub = df[df['_is_ai']] if tt == 'AI' else df[df['family'] == tt]
        if sub.empty:
            continue
        name = {'c20_fiction': 'C20 fiction', 'abstracts': 'Arxiv abstracts',
                'dreams': 'Dream reports', 'waking': 'Waking narratives',
                'AI': '**AI generations**'}.get(tt, tt)
        tt_rows.append((sub._surp_z.mean(), name, sub._drift_z.mean(), len(sub)))
    for sz, name, dz, n in sorted(tt_rows, reverse=True):
        print(f"| {name} | {sz:+.2f} | {dz:+.2f} | {n} |")
    print()

    # ── Table 2: AI by family × layer (narrative only, sorted by surprisal desc) ──
    ai = df[df['_is_ai'] & ~df['_is_template']]

    print("## AI narrative-only: median z-scores by family × layer")
    print()
    print("| Family | Layer | Surprisal (z) | Drift (z) | n |")
    print("|---|---|---|---|---|")
    fl_rows = []
    for fam in sorted(ai['family'].unique()):
        for layer in ['BASE', 'SFT', 'DPO', 'RLVR']:
            sub = ai[(ai['family'] == fam) & (ai['_layer'] == layer)]
            if sub.empty:
                continue
            fl_rows.append((sub._surp_z.mean(), fam, layer, sub._drift_z.mean(), len(sub)))
    for sz, fam, layer, dz, n in sorted(fl_rows, reverse=True):
        print(f"| {fam} | {layer} | {sz:+.2f} | {dz:+.2f} | {n} |")
    print()

    # ── Table 3: DPO - BASE deltas by family (narrative only, sorted by surprisal delta desc) ──
    print("## AI narrative-only: DPO − BASE delta (median z)")
    print()
    print("| Family | Surprisal Δ | Drift Δ | n (base) | n (dpo) |")
    print("|---|---|---|---|---|")
    delta_rows = []
    for fam in sorted(ai['family'].unique()):
        b = ai[(ai['family'] == fam) & (ai['model'] == 'base')]
        a = ai[(ai['family'] == fam) & (ai['model'] == 'superego')]
        if b.empty or a.empty:
            continue
        ds = a['_surp_z'].mean() - b['_surp_z'].mean()
        dd = a['_drift_z'].mean() - b['_drift_z'].mean()
        delta_rows.append((ds, fam, dd, len(b), len(a)))
    for ds, fam, dd, nb, na in sorted(delta_rows, reverse=True):
        print(f"| {fam} | {ds:+.2f} | {dd:+.2f} | {nb} | {na} |")
    print()

    # ── Table 4: By content category (narrative only, sorted by surprisal delta desc) ──
    print("## AI narrative-only: DPO − BASE delta by content category")
    print()
    print("| Category | BASE surp (z) | DPO surp (z) | Δ surp | BASE drift (z) | DPO drift (z) | Δ drift |")
    print("|---|---|---|---|---|---|---|")
    cat_rows = []
    for cat in sorted(ai['_category'].unique()):
        b = ai[(ai['_category'] == cat) & (ai['model'] == 'base')]
        a = ai[(ai['_category'] == cat) & (ai['model'] == 'superego')]
        if b.empty or a.empty:
            continue
        ds = a['_surp_z'].mean() - b['_surp_z'].mean()
        cat_rows.append((ds, cat, b['_surp_z'].mean(), a['_surp_z'].mean(),
                         b['_drift_z'].mean(), a['_drift_z'].mean(),
                         a['_drift_z'].mean() - b['_drift_z'].mean()))
    for ds, cat, bs, as_, bd, ad, dd in sorted(cat_rows, reverse=True):
        print(f"| {cat} | {bs:+.2f} | {as_:+.2f} | {ds:+.2f} | {bd:+.2f} | {ad:+.2f} | {dd:+.2f} |")
    print()

    # ── Table 4b: By family × content category (narrative only) ──
    print("## AI narrative-only: DPO − BASE delta by family × content category")
    print()
    print("| Family | Category | Δ surp | Δ drift | n (base) | n (dpo) |")
    print("|---|---|---|---|---|---|")
    fc_rows = []
    for fam in sorted(ai['family'].unique()):
        for cat in sorted(ai['_category'].unique()):
            b = ai[(ai['family'] == fam) & (ai['_category'] == cat) & (ai['model'] == 'base')]
            a = ai[(ai['family'] == fam) & (ai['_category'] == cat) & (ai['model'] == 'superego')]
            if len(b) < 3 or len(a) < 3:
                continue
            ds = a['_surp_z'].mean() - b['_surp_z'].mean()
            dd = a['_drift_z'].mean() - b['_drift_z'].mean()
            fc_rows.append((ds, fam, cat, dd, len(b), len(a)))
    for ds, fam, cat, dd, nb, na in sorted(fc_rows, reverse=True):
        print(f"| {fam} | {cat} | {ds:+.2f} | {dd:+.2f} | {nb} | {na} |")
    print()

    # ── Table 5: Template prevalence (sorted by DPO % desc) ──
    print("## Template prevalence by family")
    print()
    print("| Family | BASE % template | DPO % template | n |")
    print("|---|---|---|---|")
    tp_rows = []
    for fam in sorted(df[df['_is_ai']]['family'].unique()):
        sub = df[df['family'] == fam]
        b = sub[sub['model'] == 'base']
        a = sub[sub['model'] == 'superego']
        if b.empty or a.empty:
            continue
        bp = b['_is_template'].mean() * 100
        ap = a['_is_template'].mean() * 100
        tp_rows.append((ap, fam, bp, len(sub)))
    for ap, fam, bp, n in sorted(tp_rows, reverse=True):
        print(f"| {fam} | {bp:.1f}% | {ap:.1f}% | {n} |")
    print()


if __name__ == "__main__":
    import sys
    if '--summary' in sys.argv:
        csv = 'data/corpus_metrics.csv'
        for i, a in enumerate(sys.argv):
            if a == '--output' and i + 1 < len(sys.argv):
                csv = sys.argv[i + 1]
        summary(csv)
    else:
        main()
