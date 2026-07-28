"""Third-person battery: the same rung with the pronoun swapped. Nothing else.

    uv run .venv/bin/python scripts/f20x_generate_3p.py [--smoke] [--family KEY]

WHY THIS AND NOT THE PERSONA EXPERIMENT FIRST (RH, 2026-07-28). The persona
design adds a prefix, which is a new variable that can do its own work. This
changes ONE thing: who the question is about. Same rung, same models, same
sampling, same seeds-by-cell scheme, same n. That is what makes a
difference-of-differences interpretable.

THE QUESTION. F20_generation_drift shows base models drift on the first person
and aligned models do not. Nothing there establishes that the effect is about the
FIRST PERSON, because every prompt in that battery is an identity question. If
alignment reduces drift on any referent a model tracks, it is general coherence
work and the subject has nothing to do with it.

    statistic:  (drift_base_1st - drift_aligned_1st)
              - (drift_base_3rd - drift_aligned_3rd)

BOTH OUTCOMES ARE FINDINGS, and the first draft of the spec was wrong to write
the null branch as a failure:

  interaction > 0   the first person is a privileged site; the cut installs a
                    subject position
  interaction ~ 0,
  drift falls both  alignment anchors reference AS SUCH -- the LARGER claim, and
                    the one RH's own formulation predicts, since "I is no
                    different from Tamas or Hungary" gives no reason to expect
                    the first person privileged in the repair
  interaction ~ 0,
  drift falls in
  neither           the arm difference is an artifact of the identity battery

PROMPTS. Four mirror the first-person set exactly and are the PRIMARY comparison.
Two more vary the pronoun and are secondary -- they ask whether any effect is
about third-personhood or about a particular pronoun's corpus statistics.
"""
from __future__ import annotations
import argparse, gc, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, torch
from malign_logits.models import load_model
from malign_logits.provenance import provenance, describe
from f20x_generate import roster, SKIP, RUNG, N_SAMPLES, TEMPERATURES, MAX_NEW, PREFIX_TOKENS, sample

# The four matched prompts mirror first-person who/name/made/mother exactly.
PROMPTS = {
    "who_she":    ("Who is she?",              "matched"),
    "name_her":   ("What is her name?",        "matched"),
    "made_her":   ("Who made her?",            "matched"),
    "mother_her": ("What is her mother's name?", "matched"),
    "who_he":     ("Who is he?",               "pronoun"),
    "who_they":   ("Who are they?",            "pronoun"),
}
SEED0 = 20260728
OUT = "data/f20x_generations_3p.parquet"


def run(smoke=False, only=None):
    prov = provenance(__file__, closure=["scripts/f20x_generate.py"])
    print(describe(prov))
    fams = roster()
    if only:
        fams = [f for f in fams if f["key"] in set(only.split(","))]
    else:
        fams = [f for f in fams if f["key"] not in SKIP]
    n = 5 if smoke else N_SAMPLES
    temps = (1.0,) if smoke else TEMPERATURES
    print(f"roster: {len(fams)} families | {len(PROMPTS)} prompts | n={n} | temps {temps}")

    sink, failures, cell = [], [], 0
    for fi, f in enumerate(fams, 1):
        for arm, mid in (("base", f["base"]), (f["slot"], f["aligned"])):
            try:
                model, tok = load_model(mid)
            except Exception as e:
                print(f"  SKIP {f['key']}/{arm}: {e}")
                failures.append(dict(family=f["key"], arm=arm, model_id=mid,
                                     base_model_id=f["base"], prompt=None,
                                     error=f"load failed: {str(e)[:280]}"))
                continue
            for pk, (q, kind) in PROMPTS.items():
                for temp in temps:
                    cell += 1
                    try:
                        gens = sample(model, tok, RUNG.format(q=q), n, temp, SEED0 + cell)
                    except Exception as e:
                        print(f"  FAIL {f['key']}/{arm}/{pk}/T{temp}: {e}")
                        failures.append(dict(family=f["key"], arm=arm, model_id=mid,
                                             base_model_id=f["base"], prompt=pk,
                                             error=str(e)[:300]))
                        continue
                    for g in gens:
                        sink.append(dict(family=f["key"], arm=arm, model_id=mid,
                                         base_model_id=f["base"], prompt=pk, question=q,
                                         prompt_kind=kind, temperature=temp,
                                         seed=SEED0 + cell, **g))
            del model, tok
            gc.collect()
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        df = pd.DataFrame(sink)
        df.attrs["provenance"] = json.dumps(prov)
        df.to_parquet(OUT, compression="zstd", index=False)
        print(f"  [{fi}/{len(fams)}] {f['key']}: {len(sink):,} rows -> {OUT}")
    if failures:
        pd.DataFrame(failures).to_parquet(OUT.replace(".parquet", "_failures.parquet"),
                                          compression="zstd", index=False)
        print(f"  {len(failures)} failed cells recorded")
    return sink


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--family")
    a = ap.parse_args(); run(smoke=a.smoke, only=a.family)
