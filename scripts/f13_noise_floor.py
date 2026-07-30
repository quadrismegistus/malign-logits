"""How much of the flash-vs-pro disagreement is coder difference, and how much is
run-to-run noise at temperature 0?

    uv run .venv/bin/python scripts/f13_noise_floor.py [--n 50] [--passes 3]

WHY THIS IS NOW POSSIBLE. The HashStash behind Task is append_mode, so `force=True`
APPENDS a version rather than overwriting; every recode of a key stays on disk and
`Task.results_history` yields all retained versions. Before knowing that, a
same-model same-prompt rerun could only ever hit cache, so "temperature 0" was
untestable and the flash-vs-pro number had no baseline to be read against.

THE POINT, stated before the number exists. Flash and pro disagreed on 156
identical items at 76.9% (direction), 73.7% (relation), 69.2% (speech_act). Those
figures are only interpretable against WITHIN-MODEL agreement. If one model
re-coding the same item disagrees with itself at a similar rate, the cross-model
number is measuring noise and not coder difference, and the concurrence claim the
registration is waiting on cannot be built from either.

    IDENTICAL OUTPUT ACROSS TWO ORDINARY RUNS DEMONSTRATES CACHING.
    DETERMINISM IS ONLY TESTABLE UNDER force=True.

The instrument digest is recorded with the result so a later reader can prove the
same instrument was administered: `instrument_sha256()` moves on a
field-description-only change, verified upstream, so it is a claim about the
scheme's content and not only its field names.
"""
from __future__ import annotations
import argparse, itertools, os, sys, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from malign_logits.tasks.code_displacement_relation import (
    DisplacementRelationTask, prepare)

FIELDS = ("direction", "relation", "speech_act",
          "a_is_content_word", "b_is_content_word", "confidence")
SRC = "data/f13_amber_stage_codings.parquet"


def main(n=50, passes=3, model=None, workers=6):
    d = pd.read_parquet(SRC)
    task = DisplacementRelationTask()
    if model:
        task.model = model
    print(f"instrument sha256: {task.instrument_sha256()}")
    print(f"model: {task.model}   temperature: {task.temperature}   passes: {passes}")

    # Sample the SAME items the concurrence figure was computed on, so the noise
    # floor and the cross-model number share a population. Deterministic seed.
    s = d.sample(min(n, len(d)), random_state=20260730).reset_index(drop=True)
    items = [prepare(r.prompt, r.a, r.b) for r in s.itertuples()]
    uniq = list(dict.fromkeys(items))
    print(f"{len(s)} rows sampled -> {len(uniq)} distinct item strings "
          f"(map de-dupes; only distinct strings are recoded)\n")

    runs = []
    for p in range(passes):
        errs = {}
        anns = task.map(uniq, num_workers=workers, verbose=False,
                        force=True, errors=errs)
        runs.append([a.model_dump() if a is not None else None for a in anns])
        got = sum(a is not None for a in anns)
        print(f"  pass {p + 1}: {got}/{len(uniq)} coded, {len(errs)} failed")

    print("\n" + "=" * 74)
    print("WITHIN-MODEL AGREEMENT ACROSS FORCED PASSES, pairwise mean")
    print("=" * 74)
    pairs = list(itertools.combinations(range(passes), 2))
    print(f"{'field':<22}{'pairwise agree':>16}{'items unanimous':>18}")
    out = {}
    for f in FIELDS:
        agrees, unan = [], 0
        for i in range(len(uniq)):
            vals = [r[i][f] if r[i] else None for r in runs]
            if any(v is None for v in vals):
                continue
            agrees += [vals[a] == vals[b] for a, b in pairs]
            unan += len(set(vals)) == 1
        out[f] = sum(agrees) / len(agrees) if agrees else float("nan")
        print(f"  {f:<20}{out[f]:>15.1%}{unan / len(uniq):>17.1%}")

    print("\nTHE COMPARISON THE REGISTRATION NEEDS:")
    cross = {"direction": .769, "relation": .737, "speech_act": .692,
             "a_is_content_word": .821, "b_is_content_word": .891,
             "confidence": .628}
    print(f"{'field':<22}{'within-model':>14}{'flash vs pro':>14}{'headroom':>11}")
    for f in FIELDS:
        print(f"  {f:<20}{out[f]:>13.1%}{cross[f]:>14.1%}"
              f"{out[f] - cross[f]:>+11.1%}")
    print("\nHEADROOM is within-model minus cross-model. Near zero means the "
          "cross-model\nfigure is NOISE, not coder difference, and neither can "
          "support a concurrence\nclaim. Large and positive means the models "
          "genuinely differ and the gap is real.")

    # per-item instability, for the schema work
    print("\nMOST UNSTABLE ITEMS (disagree with themselves across passes):")
    rows = []
    for i, it in enumerate(uniq):
        if any(r[i] is None for r in runs):
            continue
        flips = sum(len({r[i][f] for r in runs}) > 1 for f in FIELDS)
        if flips:
            rows.append((flips, it.replace("\n", " | ")[:96],
                         {f: [r[i][f] for r in runs] for f in FIELDS
                          if len({r[i][f] for r in runs}) > 1}))
    for flips, it, det in sorted(rows, reverse=True)[:10]:
        print(f"  {flips} fields unstable  {it}")
        for f, vs in det.items():
            print(f"      {f}: {vs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    main(a.n, a.passes, a.model, a.workers)
