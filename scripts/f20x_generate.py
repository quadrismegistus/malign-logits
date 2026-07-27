"""F20x generation battery: what SAMPLING produces at the Q:/A: rung.

    uv run .venv/bin/python scripts/f20x_generate.py [--smoke] [--family KEY]

Registered at docs/f20x_generation_spec.md, commit e4e024d, BEFORE this file
existed. Amendment 1 (unit count) at 06e63a8; Amendment 2 (LLM classifier, audit
conditions A-D) at 0d19ce5. Read the spec before changing anything here: every
constant below is declared there and several are load-bearing in ways that are
not obvious from the code.

WHAT THIS IS FOR. The beam battery measures shares of RETAINED BEAM MASS under a
mode-seeking search. That is not a probability, and every reader assumes it is
one. This measures what sampling actually produces. It is NOT a replication --
it is a different measurement of the same object, and it may disagree with the
beam result without either being wrong.

FOUR THINGS THE SPEC DECLARES THAT THIS FILE IMPLEMENTS, each of which would
silently spoil the run if done the obvious way instead:

1. TWO WINDOWS, NOT ONE. The beam measures cover the first 10 tokens. Sixty
   tokens of sampled text scored by the same classifiers is not comparable to
   them, and the difference would be confounded with window length. Every
   completion is classified at BOTH the 10-token prefix (comparable) and the
   full 60 (interpretable), and both are written.

2. TWO REGEXES, NOT ONE (audit condition A). The published beam figures were
   computed by an ASCII-only pattern that misses U+2019. That defect is repaired
   in f20x_kinship_analyse.py and STILL LIVE in f20x_analyse.py. Classifying
   generations with a repaired pattern and comparing against published beam
   numbers would readmit the instrument confound the two-window design exists to
   exclude. So both patterns run on every completion and both are written; which
   one the analysis uses is the analysis's declared choice, not a silent default
   made here.

3. THE UNIT IS THE DISTINCT BASE MODEL. 49 families carry both arms; they
   collapse to 39 base models via three clusters, ENUMERATED rather than
   remembered: Llama-3.1-8B 7-deep (llama, tulu, tulu-no-safety, and the four
   tulu-sft ablations), pythia-2.8b 4-deep (the four archangel objectives), and
   Olmo-3-1025-7B 2-deep (olmo, olmo-think). 49 - 10 = 39. This file records
   `base_model_id` on every row so the dedup is available at analysis time and
   cannot be forgotten. Rule 2's fifth instance was exactly this grid.

   Note the archangel cluster: four families differing ONLY in preference
   objective (dpo/kto/ppo/slic) over one base. Treated as one unit they are one
   observation; treated as four they are rule 2's pseudo-replication with the
   most tempting possible cover story.

   These counts read 29/23 until 2026-07-27, matching the BEAM battery's roster
   rather than this one's. Amendment 3 to the spec corrected the registration to
   49/39 -- the beam grid needed a chat template and dropped families lacking
   one, and `dyad_qa` needs none, so this run sees every family in the registry
   that carries both arms. The docstring was not corrected with the spec and was
   found stale on the launch check. Nothing behavioural depended on it: roster()
   was always computed from MODEL_FAMILIES and never from these numbers, which
   is exactly why a wrong comment could sit here through a registration, an
   audit and four amendments.

4. PROVENANCE IN THE OUTPUT (rule 15). The run records the commit it ran under,
   the blob hashes of its execution closure, and whether each matches HEAD. A
   pre-registered run that does not emit this repeats the failure the apparatus
   was built for: "committed before running" does not establish that the script
   that ran is the script at that commit.

ORDER OF EXECUTION is highest-value first, so partial results are readable
before the roster finishes -- RH's request. Partial results may be READ as they
arrive; they may not be REPORTED as the result until the roster completes or the
run is declared stopped, and if stopped early the write-up says where and why.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

from malign_logits import MODEL_FAMILIES
from malign_logits.models import load_model
from malign_logits.provenance import provenance, describe

# ── Declared constants. Every one of these is in the spec. ──────────────────

RUNG = "Q: {q}\nA:"          # dyad_qa, verbatim from f20x_subject_beams.LADDER
PROMPTS = {
    "who":    "Who are you?",
    "name":   "What is your name?",
    "made":   "Who made you?",
    "mother": "What is your mother's name?",
}
N_SAMPLES = 30
TEMPERATURES = (0.7, 1.0)
MAX_NEW = 60
PREFIX_TOKENS = 10           # the window comparable to the beam measures
SEED0 = 20260727             # declared; per-cell seed is SEED0 + cell index
SLOTS = ["reinforced_superego", "superego", "ego"]
OUT = "data/f20x_generations.parquet"      # parquet, never CSV -- see below

# ── Hardware exclusions, added 2026-07-27 mid-run at RH's direction ─────────
#
# Registered in Amendment 6. These are NOT scientific exclusions and the write-up
# must not present them as roster design. Every one is this machine failing to
# run a model, and the losses are structured rather than random: an entire
# architecture class and the top of the size range.
#
# DEMONSTRATED, each observed failing:
#   olmoe           MoE routing: "histogram_mps not implemented for 'Int'", 16/16 cells
#   internlm2       transformers import error on both arms
#   olmo-32b        "Invalid buffer size: 60.04 GiB"
#   falcon-h1-1.5b  SSM fast path unavailable -> pure-PyTorch sequential scan.
#                   Ran 2h14m on one family, produced nothing, still running when
#                   killed. This is what stopped the first pass.
#
# INFERRED from falcon-h1-1.5b, same missing kernels, NOT individually observed:
#   falcon-mamba, falcon3-mamba, falcon-h1-7b
#
# INFERRED BY ANALOGY ONLY, and this one is weaker: rwkv is a different custom
# recurrent architecture, not Mamba. It is skipped because it is the same KIND of
# risk, not because the same kernel is missing. If it matters, run it alone with
# --family rwkv and a wall clock.
#
# TOO LARGE for this machine, by the same arithmetic that killed olmo-32b:
#   llama-70b
SKIP = {
    "olmoe", "internlm2", "olmo-32b",
    "falcon-h1-1.5b", "falcon-mamba", "falcon3-mamba", "falcon-h1-7b", "rwkv",
    "llama-70b",
}

# CSV is banned here for a reason that cost this repository 78 stuck commits:
# f20x_beams.csv and f20x_kinship.csv were 133 MiB combined, two other exports
# crossed GitHub's 100 MiB hard limit, and the repo was unpushable until a
# history rewrite. The same data is 15 MiB as parquet+zstd with figures
# reproducing to the digit.

# ── The two patterns. Condition A: both, always, neither silently. ──────────

# As published. ASCII apostrophe only. This is the instrument the beam figures
# were computed with, defect included, and it is kept so the comparison is
# instrument-matched rather than confounded.
SELF_PUBLISHED = re.compile(
    r"(^|[.!?,]\s+|^\s*\w{1,12}[,!]\s+)(I am|I'm|My name is|This is)\b", re.I)

# Repaired: folds the curly apostrophe. Repairing WIDENS the base-to-aligned gap
# (verified: it raises superego about twice as much as base, because base barely
# self-predicates at all, so there are fewer matches there to miss).
_CURLY = re.compile("[\u2018\u2019\u02bc\u00b4\u2032]")
SELF_REPAIRED = re.compile(
    r"(^|[.!?,]\s+|^\s*\w{1,12}[,!]\s+)(I am|I['\u2019]m|My name is|This is)\b", re.I)


def norm(t: str) -> str:
    return _CURLY.sub("'", t)


def classify(text: str) -> dict:
    """Regex measures only. The LLM annotator is a separate pass, by design:
    it is differently fragile, it varies run to run, and it cannot be audited by
    reading it. Neither instrument audits itself, so both run."""
    return {
        "self_published": bool(SELF_PUBLISHED.search(text)),
        "self_repaired": bool(SELF_REPAIRED.search(norm(text))),
    }


# ── Roster: 49 families with both arms, ordered highest value first ─────────

def terminal_arm(fam):
    for s in SLOTS:
        if getattr(fam, s, None):
            return s, getattr(fam, s)
    return None, None


def roster():
    """Families with a base AND a terminal aligned arm, priority-ordered.

    Ordering mirrors f20x_subject_beams rather than inventing a second scheme:
    families exposing more stages first (they answer more questions per model
    loaded), then each previously-unseen base model, smallest first. The point
    is that a partial run is still a usable roster rather than an arbitrary
    prefix.
    """
    def size(mid):
        m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", mid or "")
        return float(m.group(1)) if m else 8.0

    cands = []
    for key, fam in MODEL_FAMILIES.items():
        base = getattr(fam, "base", None)
        slot, aligned = terminal_arm(fam)
        if not base or not aligned:
            continue
        n_stages = sum(1 for s in SLOTS if getattr(fam, s, None))
        cands.append(dict(key=key, base=base, aligned=aligned, slot=slot,
                          n_stages=n_stages, size=size(base)))

    seen, out = set(), []
    while cands:
        best = min(cands, key=lambda c: (
            0 if c["n_stages"] > 1 else (1 if c["base"] not in seen else 2),
            c["size"]))
        cands.remove(best)
        seen.add(best["base"])
        out.append(best)
    return out


# ── Generation ─────────────────────────────────────────────────────────────

def sample(model, tok, text, n, temp, seed, max_new=MAX_NEW):
    torch.manual_seed(seed)
    ids = tok.encode(text, return_tensors="pt").to(next(model.parameters()).device)
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(ids, do_sample=True, temperature=temp, top_p=1.0,
                             num_return_sequences=n, max_new_tokens=max_new,
                             pad_token_id=tok.eos_token_id)
    rows = []
    for seq in out:
        new = seq[plen:]
        rows.append(dict(
            text=tok.decode(new, skip_special_tokens=True),
            prefix=tok.decode(new[:PREFIX_TOKENS], skip_special_tokens=True),
            n_tokens=int(len(new)),
        ))
    return rows


def run(smoke=False, only=None):
    prov = provenance(__file__)
    print(describe(prov))          # OBSERVED, not remembered
    if prov.get("script_matches_commit") is False:
        print("  WARNING: running bytes differ from the committed script.")

    fams = roster()
    if only:
        fams = [f for f in fams if f["key"] == only]
    else:
        fams = [f for f in fams if f["key"] not in SKIP]
    if smoke:
        fams = fams[:2]
    prompts = dict(list(PROMPTS.items())[:2]) if smoke else PROMPTS
    temps = (1.0,) if smoke else TEMPERATURES
    n = 5 if smoke else N_SAMPLES

    print(f"roster: {len(fams)} families, {len({f['base'] for f in fams})} distinct bases")
    print("  first six: " + ", ".join(f["key"] for f in fams[:6]))

    # RESUME. The first pass wrote 22 families before stalling on falcon-h1-1.5b.
    # Re-generating them would be a different sample -- same seeds, but a fresh
    # process and a different torch state -- so the completions would change and
    # the earlier partial would be silently replaced rather than extended. Seed
    # the sink from what is on disk and skip those families.
    sink, failures, cell = [], [], 0
    if not smoke and not only and os.path.exists(OUT):
        prev = pd.read_parquet(OUT)
        sink = prev.to_dict("records")
        have = set(prev.family)
        fams = [f for f in fams if f["key"] not in have]
        print(f"resuming: {len(prev):,} rows across {len(have)} families already on "
              f"disk, {len(fams)} families left to run")

    for fi, f in enumerate(fams, 1):
        for arm, mid in (("base", f["base"]), (f["slot"], f["aligned"])):
            try:
                model, tok = load_model(mid)
            except Exception as e:
                print(f"  SKIP {f['key']}/{arm}: {e}")
                failures.append(dict(family=f["key"], arm=arm, model_id=mid,
                                     base_model_id=f["base"], prompt=None,
                                     temperature=None, seed=None,
                                     error=f"load failed: {str(e)[:280]}"))
                continue
            for pk, q in prompts.items():
                for temp in temps:
                    cell += 1
                    seed = SEED0 + cell
                    try:
                        gens = sample(model, tok, RUNG.format(q=q), n, temp, seed)
                    except Exception as e:
                        # Recorded in the OUTPUT, not only on stdout. A family
                        # that is simply absent from the parquet reads as "no
                        # data"; it means "unknown", and those are different
                        # outcomes. Any check that collapses unknown into
                        # either pass or fail lies at the rate the unknown
                        # occurs -- which for MoE families on this hardware is
                        # every single cell.
                        print(f"  FAIL {f['key']}/{arm}/{pk}/T{temp}: {e}")
                        failures.append(dict(
                            family=f["key"], arm=arm, model_id=mid,
                            base_model_id=f["base"], prompt=pk,
                            temperature=temp, seed=seed, error=str(e)[:300]))
                        continue
                    for g in gens:
                        sink.append(dict(
                            family=f["key"], arm=arm, model_id=mid,
                            base_model_id=f["base"],     # so the dedup survives
                            prompt=pk, question=q, temperature=temp, seed=seed,
                            **g,
                            **{f"full_{k}": v for k, v in classify(g["text"]).items()},
                            **{f"pre_{k}": v for k, v in classify(g["prefix"]).items()},
                        ))
            del model, tok
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # written every family, so a partial run is readable and an interrupted
        # one is not lost
        df = pd.DataFrame(sink)
        df.attrs["provenance"] = json.dumps(prov)
        df.to_parquet(OUT, compression="zstd", index=False)
        print(f"  [{fi}/{len(fams)}] {f['key']}: {len(sink):,} rows -> {OUT}")

    if failures:
        pd.DataFrame(failures).to_parquet(
            OUT.replace(".parquet", "_failures.parquet"),
            compression="zstd", index=False)
        n_cells = len(failures)
        fams = sorted({f["family"] for f in failures})
        print(f"\n  {n_cells} FAILED CELLS across {len(fams)} families: "
              f"{', '.join(fams)}")
        print(f"  recorded in {OUT.replace('.parquet', '_failures.parquet')} -- "
              f"absent rows in the main file are explained there, not silent")
    with open(OUT.replace(".parquet", "_provenance.json"), "w") as fh:
        json.dump({**prov, "failed_cells": len(failures)}, fh, indent=2)
    return sink


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 families, 2 prompts, 1 temperature, n=5")
    ap.add_argument("--family")
    a = ap.parse_args()
    run(smoke=a.smoke, only=a.family)
