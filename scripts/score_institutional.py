"""Score institutional alignment generations using AlignmentAsymmetryTask.

Reads generations from lmdb cache, scores via LLM tagger (parallel, resumable).
Ctrl-C safe: re-run picks up where it left off via task stash.

Usage:
    python scripts/score_institutional.py                    # DeepSeek primary
    python scripts/score_institutional.py --model sonnet     # Sonnet validation
    python scripts/score_institutional.py --dry-run          # count without scoring
    python scripts/score_institutional.py --harvest          # save task stash -> gen_annotations
    python scripts/score_institutional.py --export           # export scored CSV
"""
import argparse
from malign_logits.cache import get_cache
from malign_logits.experiments import INSTITUTIONAL_PROMPTS
from malign_logits import MODEL_FAMILIES
from largeliterarymodels.tasks import AlignmentAsymmetryTask, prepare_text

FRONTIER_MODELS = [
    ("openai/gpt-4o-mini-raw", "gpt-4o-mini", "aligned"),
    ("anthropic/claude-haiku-4-5-raw", "claude-haiku", "aligned"),
    ("deepseek/deepseek-chat-raw", "deepseek-chat", "aligned"),
    ("anthropic/claude-sonnet-4-6-raw", "claude-sonnet", "aligned"),
    ("google/gemini-2.5-flash-raw", "gemini-flash", "aligned"),
]

MODEL_ALIASES = {
    "deepseek": "deepseek/deepseek-chat",
    "sonnet": "claude-sonnet-4-6",
    "gpt": "openai/gpt-4o-mini",
}


def collect_all(cache):
    """Collect all institutional generations from cache with metadata."""
    rows = []

    for fam_key, fam in MODEL_FAMILIES.items():
        layers = [(fam.base, "base")]
        if fam.ego:
            layers.append((fam.ego, "ego"))
        if fam.superego:
            layers.append((fam.superego, "super"))
        if hasattr(fam, "reinforced_superego") and fam.reinforced_superego:
            layers.append((fam.reinforced_superego, "rlvr"))

        for model_id, layer_name in layers:
            for prompt_key, prompt_text in INSTITUTIONAL_PROMPTS.items():
                n = cache.count_generations(model_id, prompt_text, temp=1.0)
                for idx in range(n):
                    gen = cache.get_generation(model_id, prompt_text,
                                               temp=1.0, idx=idx)
                    if gen and gen.strip():
                        rows.append({
                            "prompt_key": prompt_key,
                            "prompt_text": prompt_text,
                            "generation_text": gen,
                            "model_id": model_id,
                            "family": fam_key,
                            "layer_name": layer_name,
                            "is_frontier": False,
                            "idx": idx,
                        })

    for model_id, family_label, layer_name in FRONTIER_MODELS:
        for prompt_key, prompt_text in INSTITUTIONAL_PROMPTS.items():
            n = cache.count_generations(model_id, prompt_text, temp=1.0)
            for idx in range(n):
                gen = cache.get_generation(model_id, prompt_text,
                                           temp=1.0, idx=idx)
                if gen and gen.strip():
                    rows.append({
                        "prompt_key": prompt_key,
                        "prompt_text": prompt_text,
                        "generation_text": gen,
                        "model_id": model_id,
                        "family": family_label,
                        "layer_name": layer_name,
                        "is_frontier": True,
                        "idx": idx,
                    })

    return rows


def harvest(cache, tagger_id):
    """Harvest from task.df (with metadata) into gen_annotations cache."""
    task = AlignmentAsymmetryTask(model=tagger_id)
    df = task.df

    required = ['meta_family', 'meta_model_id', 'meta_prompt_key', 'meta_idx']
    if not all(c in df.columns for c in required):
        print("No metadata columns in task.df — run scoring with metadata first.")
        print(f"Columns: {list(df.columns)}")
        return

    stored = 0
    skipped = 0
    score_cols = [
        'apology_present', 'specific_rights_named', 'concrete_action_recommended',
        'homework_assigned', 'delay_advised', 'agency', 'institutional_deference',
        'assertiveness', 'power_acknowledgment', 'strategy_specificity',
        'emotional_tone', 'action_verbs', 'hedging_phrases',
    ]

    for _, row in df.iterrows():
        model_id = row['meta_model_id']
        prompt_key = row['meta_prompt_key']
        idx = int(row['meta_idx'])
        prompt_text = INSTITUTIONAL_PROMPTS.get(prompt_key, '')
        if not prompt_text:
            continue

        if cache.has_gen_annotation(tagger_id, model_id, prompt_text,
                                    temp=1.0, idx=idx):
            skipped += 1
            continue

        scores = {c: row[c] for c in score_cols if c in row}
        cache.set_gen_annotation(tagger_id, model_id, prompt_text,
                                 scores, temp=1.0, idx=idx)
        stored += 1

    print(f"Harvest: {stored} new, {skipped} already cached")


def export_csv(tagger_id, output_path):
    """Export scored annotations from task.df to CSV."""
    task = AlignmentAsymmetryTask(model=tagger_id)
    df = task.df

    required = ['meta_family', 'meta_model_id', 'meta_prompt_key', 'meta_idx']
    if not all(c in df.columns for c in required):
        print("No metadata columns in task.df — run scoring with metadata first.")
        return

    # Derive side from prompt_key
    def get_side(pk):
        for tag in ['_worker_', '_tenant_', '_patient_', '_citizen_']:
            if tag in str(pk):
                return 'individual'
        for tag in ['_mgmt_', '_landlord_', '_doctor_', '_officer_', '_agency_', '_party_']:
            if tag in str(pk):
                return 'institution'
        return 'unknown'

    # Derive domain from prompt_key
    def get_domain(pk):
        for d in ['labor', 'housing', 'medical', 'police', 'govt', 'political']:
            if f'_{d}_' in str(pk):
                return d
        return 'unknown'

    df['side'] = df['meta_prompt_key'].apply(get_side)
    df['domain'] = df['meta_prompt_key'].apply(get_domain)

    # Rename meta columns
    df = df.rename(columns={
        'meta_family': 'family', 'meta_layer_name': 'layer_name',
        'meta_model_id': 'model_id', 'meta_prompt_key': 'prompt_key',
        'meta_is_frontier': 'is_frontier', 'meta_idx': 'idx',
    })

    # Select and order columns
    cols = [
        'family', 'layer_name', 'model_id', 'prompt_key', 'side', 'domain',
        'is_frontier', 'idx',
        'apology_present', 'specific_rights_named',
        'concrete_action_recommended', 'homework_assigned', 'delay_advised',
        'agency', 'institutional_deference', 'assertiveness',
        'power_acknowledgment', 'strategy_specificity',
        'emotional_tone', 'action_verbs', 'hedging_phrases',
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # Convert lists to pipe-separated strings
    for col in ['action_verbs', 'hedging_phrases']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: '|'.join(x) if isinstance(x, list) else x)

    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} scored annotations to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek",
                        help="Tagger model: deepseek, sonnet, gpt (default: deepseek)")
    parser.add_argument("--n", type=int, default=None,
                        help="Score at most N generations (default: all)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers for API calls (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count generations without scoring")
    parser.add_argument("--harvest", action="store_true",
                        help="Harvest from task stash into gen_annotations")
    parser.add_argument("--export", action="store_true",
                        help="Export scored annotations to CSV")
    args = parser.parse_args()

    tagger_id = MODEL_ALIASES.get(args.model, args.model)
    cache = get_cache()
    rows = collect_all(cache)

    print(f"Total generations: {len(rows)}")
    print(f"  Local: {sum(1 for r in rows if not r['is_frontier'])}")
    print(f"  Frontier: {sum(1 for r in rows if r['is_frontier'])}")
    print(f"  Families: {sorted(set(r['family'] for r in rows))}")

    if args.export:
        export_csv(tagger_id, f"data/institutional_scored_{args.model}.csv")
        return

    if args.harvest:
        harvest(cache, tagger_id)
        return

    if args.dry_run:
        return

    # Prepare texts and score (parallel, resumable via task stash)
    import random
    random.shuffle(rows)
    if args.n:
        rows = rows[:args.n]
        print(f"  Limiting to {args.n} generations")

    prepared = [prepare_text(r["generation_text"], prompt_text=r["prompt_text"])
                for r in rows]
    metadata_list = [{
        "family": r["family"],
        "layer_name": r["layer_name"],
        "model_id": r["model_id"],
        "prompt_key": r["prompt_key"],
        "is_frontier": r["is_frontier"],
        "idx": r["idx"],
    } for r in rows]

    print(f"\nScoring with {tagger_id}... (ctrl-C safe, re-run to resume)")
    task = AlignmentAsymmetryTask(model=tagger_id)
    results = task.map(prepared, metadata_list=metadata_list,
                       num_workers=args.workers)

    # Write results to gen_annotations
    stored = 0
    for row, result in zip(rows, results):
        if result is not None:
            scores = result if isinstance(result, dict) else result.model_dump()
            cache.set_gen_annotation(
                tagger_id, row["model_id"], row["prompt_text"],
                scores, temp=1.0, idx=row["idx"],
            )
            stored += 1

    print(f"\nStored {stored} annotations in gen_annotations cache")
    print(f"Export with: python scripts/score_institutional.py --export")


if __name__ == "__main__":
    main()
