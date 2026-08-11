#!/usr/bin/env python
"""The frozen licit-set run: all 584 battery prompts through
code_m05_licit_v1, witness-checked, stability-probed, written as the
secondary-5 artifact. Plan: meta/M05_emergence/plans/syntax_curve.md.

    uv run python meta/M05_emergence/scripts/m05_licit_run.py

Writes data/m05_licit_sets.json:
  _instrument: task name, instrument sha, model of record, spacy version,
               battery sha, witness agreement, probe results
  prompts: {prompt: {frame, licit: [{pos, example}], marginal: [...],
                     witness_disagreements: [...]}}
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = "data/m05_licit_sets.json"
PROBE_N = 30
PROBE_MODEL = "anthropic/claude-haiku-4-5"
EQUIV = [{"ADP", "PART"}, {"NUM", "NOUN"}, {"AUX", "VERB"}]


def eq(a, b):
    return a == b or any(a in g and b in g for g in EQUIV)


def battery_texts():
    b = json.load(open("data/m05_battery.json"))
    texts = []
    for blk in b["blocks"].values():
        for t in blk["texts"]:
            texts.append(t if isinstance(t, str) else
                         t.get("text", t.get("prompt")))
    return list(dict.fromkeys(texts)), b["sha256_16_over_texts"]


def main():
    import random

    import spacy

    from malign_logits.tasks.code_m05_licit_v1 import LicitSetTask
    sys.path.insert(0, HERE)
    from m05_syntax_tags import pos_class as pc

    texts, battery_sha = battery_texts()
    task = LicitSetTask()
    print(f"coding {len(texts)} prompts on {task.model} at temp "
          f"{task.temperature}")
    results = task.map([f"TEXT:\n{p}" for p in texts], num_workers=8,
                       verbose=True)

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    out, agree, tries, failed = {}, 0, 0, []
    for p, res in zip(texts, results):
        if res is None:
            failed.append(p)
            continue
        bad = []
        for w in list(res.licit) + list(res.marginal):
            doc = nlp(f"{p} {w.example}")
            start = len(p) + 1
            toks = [t for t in doc if t.idx >= start] or [doc[-1]]
            got = pc(toks[0].tag_, toks[0].pos_)
            tries += 1
            if eq(got, w.pos):
                agree += 1
            else:
                bad.append(f"{w.pos}:{w.example} tagged {got}")
        out[p] = dict(
            frame=res.frame,
            licit=[dict(pos=w.pos, example=w.example) for w in res.licit],
            marginal=[dict(pos=w.pos, example=w.example)
                      for w in res.marginal],
            witness_disagreements=bad)
    print(f"\ncoded {len(out)}/{len(texts)} (failed {len(failed)}); "
          f"witness/tagger agreement {agree}/{tries} "
          f"({agree / max(tries, 1):.1%})")
    sizes = [len(v["licit"]) for v in out.values()]
    import numpy as np
    print(f"licit-set sizes: min {min(sizes)}, median "
          f"{int(np.median(sizes))}, max {max(sizes)}; "
          f"share with <=3 classes {np.mean([s <= 3 for s in sizes]):.0%}")

    # ---- stability probe: second family on 30 seeded prompts -------------
    rng = random.Random(20260811)
    probe_prompts = rng.sample([p for p in texts if p in out], PROBE_N)
    probe_task = LicitSetTask()
    probe_task.model = PROBE_MODEL
    print(f"\nstability probe: {PROBE_N} prompts on {PROBE_MODEL}")
    probe_res = probe_task.map([f"TEXT:\n{p}" for p in probe_prompts],
                               num_workers=8, verbose=False)
    jaccards = []
    for p, r in zip(probe_prompts, probe_res):
        if r is None:
            continue
        a = {w.pos for w in r.licit}
        b = {w["pos"] for w in out[p]["licit"]}
        j = len(a & b) / len(a | b) if (a | b) else 1.0
        jaccards.append((p, j, sorted(a - b), sorted(b - a)))
    js = [j for _, j, _, _ in jaccards]
    print(f"strict-set Jaccard vs {task.model}: median "
          f"{np.median(js):.2f}, min {min(js):.2f}, "
          f"<0.5 on {sum(j < 0.5 for j in js)}/{len(js)} prompts")
    worst = sorted(jaccards, key=lambda x: x[1])[:5]
    for p, j, only_h, only_d in worst:
        print(f"  J={j:.2f} {p[:50]!r}  haiku-only {only_h} "
              f"deepseek-only {only_d}")

    task_sha = getattr(task, "instrument_sha256", None)
    payload = {
        "_instrument": dict(
            task="m05_licit_v1",
            instrument_sha256=(task_sha() if callable(task_sha)
                               else task_sha),
            model_of_record=task.model,
            spacy_model="en_core_web_sm", spacy_version=spacy.__version__,
            battery_sha=battery_sha,
            n_coded=len(out), n_failed=len(failed), failed=failed,
            witness_agreement=f"{agree}/{tries}",
            probe=dict(model=PROBE_MODEL, n=len(js),
                       jaccard_median=float(np.median(js)),
                       jaccard_min=float(min(js)),
                       below_half=int(sum(j < 0.5 for j in js)))),
        "prompts": out,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16]
    print(f"\nwrote {OUT} (sha256_16 {sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
