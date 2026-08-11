#!/usr/bin/env python
"""One-time class-mass table: resolved twp mass by pos_class for every
(checkpoint, prompt) cell on the battery, both ladders.

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_class_mass.py

The expensive join, done once: ~146k cells (250 checkpoints x 584 prompts)
read through the word_probs choke point and collapsed onto the frozen tag
table (data/m05_syntax_tags.parquet, pos_class). Every syntax-curve
variant -- any coder's licit sets, strict or permissive, the format band,
future coders -- is then a cheap reweighting of this table; no store reads
at curve time.

Writes data/m05_class_mass.parquet, one row per (ladder, checkpoint,
prompt, pos_class): mass (sum of p), plus per-cell resolved_mass, n_rows,
payload_empty carried on every row of the cell. Populations stay separate
via the `ladder` column; nothing here pools them.
"""
import json
import os
import sys

os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = "data/m05_class_mass.parquet"
ROLE_ORDER = {"base_step": 0, "base_endpoint": 1, "sft_step": 2,
              "sft_endpoint": 3, "dpo_endpoint": 4, "rlvr_step": 5}
STAGE_ORDER = {"stage1": 0, "stage2": 1, "stage3": 2}


def model_string(c):
    return (c["model_id"] if c["revision"] == "main"
            else f"{c['model_id']}@{c['revision']}")


def load_population(path):
    pop = json.load(open(path))["checkpoints"]
    pop = sorted(pop, key=lambda c: (ROLE_ORDER[c["role"]],
                                     STAGE_ORDER.get(c.get("stage"), 9),
                                     c.get("step", 0)))
    return pop


def battery_texts():
    b = json.load(open("data/m05_battery.json"))
    texts = []
    for blk in b["blocks"].values():
        for t in blk["texts"]:
            texts.append(t if isinstance(t, str) else
                         t.get("text", t.get("prompt")))
    return list(dict.fromkeys(texts))


def main():
    from collections import defaultdict

    import pandas as pd

    from malign_logits.movement import word_probs

    tags = pd.read_parquet("data/m05_syntax_tags.parquet")
    tagmap = {(r.prompt, r.word): r.pos_class for r in tags.itertuples()}
    texts = battery_texts()

    rows = []
    for ladder, path in [("olmo", "data/m05_checkpoint_population.json"),
                         ("pythia", "data/pythia_population.json")]:
        pop = load_population(path)
        print(f"{ladder}: {len(pop)} checkpoints x {len(texts)} prompts")
        gaps = 0
        for idx, c in enumerate(pop):
            m = model_string(c)
            for p in texts:
                wp = word_probs(m, p)
                if wp is None:
                    gaps += 1
                    continue
                masses = defaultdict(float)
                for w, prob in wp.probs.items():
                    masses[tagmap.get((p, w), "UNTAGGED")] += prob
                resolved = sum(wp.probs.values())
                base = dict(ladder=ladder, ckpt_idx=idx, model=m,
                            role=c["role"], stage=c.get("stage"),
                            step=c.get("step"), prompt=p,
                            resolved_mass=resolved, n_rows=wp.n_rows,
                            payload_empty=(wp.n_rows == 0))
                if not masses:
                    rows.append(dict(base, pos_class="NONE", mass=0.0))
                for cls, mass in masses.items():
                    rows.append(dict(base, pos_class=cls, mass=mass))
            if (idx + 1) % 25 == 0:
                print(f"  {idx + 1}/{len(pop)} checkpoints", flush=True)
        print(f"  {ladder} gaps (cell not in store): {gaps}")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(f"wrote {OUT}: {len(df)} rows, "
          f"{df.groupby('ladder').ckpt_idx.nunique().to_dict()} checkpoints")
    untag = df[df.pos_class == "UNTAGGED"].mass.sum() / max(df.mass.sum(),
                                                            1e-9)
    print(f"UNTAGGED mass share (words outside the tag table): {untag:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
