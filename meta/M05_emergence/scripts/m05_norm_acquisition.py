"""Norm-acquisition table: K-scale composition of the resolved distribution
per (ladder, checkpoint, prompt) cell, both ladders — plan_norm_acquisition.

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_norm_acquisition.py

Mirrors the sense/class mass pipeline: same populations, same word_probs
choke point, same cell discipline. Per cell, for each of the seven K
scales: the mass-weighted mean over K-RATED words (renormalised within
rated mass), with k_rated_mass_share as the cell's coverage figure —
unrated mass is censored, never zero. Raw rows first; curves are a
separate reader's job.

k_ riders travel (fields.py): one coder; register_level descriptor-only;
vulgarity sparse; ranks-not-levels; CHARGE IS NOT AROUSAL.

Writes data/m05_norm_mass.parquet, one row per (ladder, checkpoint role/
stage/step, prompt).
"""
import json
import os
import sys

os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = "data/m05_norm_mass.parquet"
ROLE_ORDER = {"base_step": 0, "base_endpoint": 1, "sft_step": 2,
              "sft_endpoint": 3, "dpo_endpoint": 4, "rlvr_step": 5}
STAGE_ORDER = {"stage1": 0, "stage2": 1, "stage3": 2}
K_SCALES = ("vulgarity", "register_level", "transgressiveness", "charge",
            "valence", "bodily_harm", "concreteness")


def battery_texts():
    b = json.load(open("data/m05_battery.json"))
    texts = []
    for blk in b["blocks"].values():
        for t in blk["texts"]:
            texts.append(t if isinstance(t, str) else
                         t.get("text", t.get("prompt")))
    return list(dict.fromkeys(texts))


def load_population(path):
    pop = json.load(open(path))["checkpoints"]
    pop = sorted(pop, key=lambda c: (ROLE_ORDER[c["role"]],
                                     STAGE_ORDER.get(c.get("stage"), 9),
                                     c.get("step", 0)))
    return pop


def main():
    import pandas as pd
    from malign_logits.fields import _k
    from malign_logits.movement import word_probs

    ratings, _meta = _k("en")  # word -> tuple in K_SCALES order
    print(f"k_ratings: {len(ratings)} en words", flush=True)

    rows = []
    for ladder, path in [("olmo", "data/m05_checkpoint_population.json"),
                         ("pythia", "data/pythia_population.json")]:
        pop = load_population(path)
        print(f"== {ladder}: {len(pop)} checkpoints", flush=True)
        prompts = battery_texts()
        for ci, c in enumerate(pop):
            m = c["model"]
            for p in prompts:
                try:
                    wp = word_probs(m, p)
                except Exception:
                    continue
                if not wp:
                    continue
                total = sum(wp.values())
                if total <= 0:
                    continue
                sums = [0.0] * len(K_SCALES)
                rated_mass = 0.0
                n_rated = 0
                for w, pr in wp.items():
                    r = ratings.get(w.lower())
                    if r is None:
                        continue
                    rated_mass += pr
                    n_rated += 1
                    for i, v in enumerate(r):
                        sums[i] += pr * v
                row = {"ladder": ladder, "model": m, "role": c["role"],
                       "stage": c.get("stage"), "step": c.get("step", 0),
                       "prompt": p, "total_mass": total,
                       "k_rated_mass_share": rated_mass / total,
                       "n_rated_words": n_rated}
                for i, sc in enumerate(K_SCALES):
                    row[f"dist_mean_k_{sc}"] = (sums[i] / rated_mass
                                                if rated_mass > 0 else None)
                rows.append(row)
            if ci % 10 == 0:
                print(f"   {ci+1}/{len(pop)} rungs, {len(rows)} rows",
                      flush=True)
                pd.DataFrame(rows).to_parquet(OUT)
    pd.DataFrame(rows).to_parquet(OUT)
    print(f"wrote {OUT}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
