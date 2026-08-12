#!/usr/bin/env python
"""One-time sense-mass table: resolved twp mass by SENSE BAND for every
(checkpoint, prompt) cell on the battery, both ladders.

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_sense_mass.py

Mirrors m05_class_mass.py exactly (same populations, same choke point,
same cell discipline); the collapse key is the tier-3 sense verdict
instead of the pos_class. Band of a (prompt, word):

    JUDGE bucket          -> its coder verdict (natural | odd |
                             ungrammatical | not_a_word), from
                             data/m05_sense_verdicts.parquet
                             (118,129 rows, canaries 10/10,
                             sha 9060957ed8050b42)
    ungrammatical_auto    -> ungrammatical  (both syntax coders illicit;
                             the paid instrument already ruled)
    format_auto           -> format         (PUNCT/X/SYM band)
    absent from census    -> unclassified   (below both census floors;
                             max p < 0.003 everywhere and < 0.002 in the
                             early window -- tail mass, never judged)

Writes data/m05_sense_mass.parquet, one row per (ladder, checkpoint,
prompt, band): mass, plus per-cell resolved_mass, n_rows, payload_empty.
Curve producers reweight this table; no store reads at curve time.
"""
import json
import os
import sys

os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = "data/m05_sense_mass.parquet"
ROLE_ORDER = {"base_step": 0, "base_endpoint": 1, "sft_step": 2,
              "sft_endpoint": 3, "dpo_endpoint": 4, "rlvr_step": 5}
STAGE_ORDER = {"stage1": 0, "stage2": 1, "stage3": 2}


def model_string(c):
    return (c["model_id"] if c["revision"] == "main"
            else f"{c['model_id']}@{c['revision']}")


def load_population(path):
    pop = json.load(open(path))["checkpoints"]
    return sorted(pop, key=lambda c: (ROLE_ORDER[c["role"]],
                                      STAGE_ORDER.get(c.get("stage"), 9),
                                      c.get("step", 0)))


def battery_texts():
    b = json.load(open("data/m05_battery.json"))
    texts = []
    for blk in b["blocks"].values():
        for t in blk["texts"]:
            texts.append(t if isinstance(t, str) else
                         t.get("text", t.get("prompt")))
    return list(dict.fromkeys(texts))


def band_map():
    import pandas as pd
    census = pd.read_parquet("data/m05_sense_census.parquet")
    verdicts = pd.read_parquet("data/m05_sense_verdicts.parquet")
    vmap = {(r.prompt, r.word): r.verdict for r in verdicts.itertuples()}
    bands = {}
    missing = 0
    for r in census.itertuples():
        k = (r.prompt, r.word)
        if r.bucket == "JUDGE":
            v = vmap.get(k)
            if v is None:
                missing += 1
                continue
            bands[k] = v
        elif r.bucket == "ungrammatical_auto":
            bands[k] = "ungrammatical"
        else:
            bands[k] = "format"
    if missing:
        print(f"WARNING: {missing} JUDGE pairs without a verdict")
    return bands


def main():
    from collections import defaultdict

    import pandas as pd

    from malign_logits.movement import word_probs

    bands = band_map()
    print(f"band map: {len(bands)} (prompt, word) pairs")
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
                    masses[bands.get((p, w), "unclassified")] += prob
                resolved = sum(wp.probs.values())
                base = dict(ladder=ladder, ckpt_idx=idx, model=m,
                            role=c["role"], stage=c.get("stage"),
                            step=c.get("step"), prompt=p,
                            resolved_mass=resolved, n_rows=wp.n_rows,
                            payload_empty=(wp.n_rows == 0))
                if not masses:
                    rows.append(dict(base, band="NONE", mass=0.0))
                for band, mass in masses.items():
                    rows.append(dict(base, band=band, mass=mass))
            if (idx + 1) % 25 == 0:
                print(f"  {idx + 1}/{len(pop)} checkpoints", flush=True)
        print(f"  {ladder} gaps (cell not in store): {gaps}")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(f"wrote {OUT}: {len(df)} rows, "
          f"{df.groupby('ladder').ckpt_idx.nunique().to_dict()} checkpoints")
    uncl = df[df.band == "unclassified"].mass.sum() / max(df.mass.sum(), 1e-9)
    print(f"unclassified mass share (below both census floors): {uncl:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
