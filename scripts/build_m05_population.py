#!/usr/bin/env python
"""Enumerate the M05 checkpoint population -> data/m05_checkpoint_population.json.

    cd ~/github/malign-logits && uv run python scripts/build_m05_population.py

The population is CHECKPOINTS (model_id, revision), enumerated as strings, per
the [5148] plan-documents standard. Two arms:

- SFT arm: every step branch of allenai/Olmo-3-7B-Think-SFT (43, contiguous at
  1000) plus the SFT endpoint `main`.
- BASE arm (added 2026-08-10 on RH's word, "include the olmo BASE checkpoints
  too"): a DECLARED log-spaced subsample of the allenai/Olmo-3-1025-7B ladder
  (1,486 step branches: stage1 1,421 / stage2 52 / stage3 13), plus `main`.
  The rule, stated so the subsample is a computation and not a curation:
    stage1  step0 (the init anchor) + geometric grid from step1000 at ratio
            sqrt(2) (half-octave; dense early, where F24's sequence lives),
            each target snapped to the nearest available step, deduplicated,
            + the final step  -> 22
    stage2  geometric grid at ratio 2 (full octave) + final step      ->  7
    stage3  ALL 13 (it is the anneal, it is short, and its tips are the
            join candidates for secondary 3)                          -> 13

Source of record: data/model_revisions.json (scripts/fetch_model_revisions.py,
2026-08-10). This script refuses to write if the SFT step set is not
contiguous at 1000-step spacing, or if any snapped base step is not an actual
branch -- a missing rung is a fact the plan must state, not a gap to discover
mid-run.
"""
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "model_revisions.json")
OUT = os.path.join(ROOT, "data", "m05_checkpoint_population.json")

SFT = "allenai/Olmo-3-7B-Think-SFT"
BASE = "allenai/Olmo-3-1025-7B"
RLVR = "allenai/Olmo-3-7B-Think"          # final model: RLVR on Think-DPO
DPO = "allenai/Olmo-3-7B-Think-DPO"       # main only, everywhere
#: v3 (RH's word 2026-08-11): "include a few from RLVR too -- these maybe
#: only need a few if there's very little difference in RLVR (which we've
#: found everywhere)". SPARSE BY DESIGN: geometric doubling from step_0025 +
#: the final step (7 rungs of the 55 released), plus Think-DPO@main so the
#: DPO jump is bracketed (SFT-final -> DPO endpoint -> RLVR ladder). The
#: RLVR/DPO repos are not in the registry, so their branches are fetched
#: LIVE at run time and every picked step is asserted to exist.
RLVR_GRID = [25, 50, 100, 200, 400, 800, 1375]


def stage_steps(branches, stage):
    pat = re.compile(re.escape(stage) + r"step(\d+)")
    return sorted(int(m.group(1)) for b in branches if (m := pat.fullmatch(b)))


def geo_snap(avail, ratio):
    """Geometric targets from 1000 up, snapped to nearest available step,
    deduplicated, final step always included."""
    picks, t = [], 1000.0
    while t <= avail[-1]:
        picks.append(min(avail, key=lambda a: abs(a - t)))
        t *= ratio
    picks.append(avail[-1])
    return sorted(set(picks))


def main():
    revs = json.load(open(SRC))["models"]

    sft_branches = revs[SFT]["branches"]
    sft_steps = sorted(int(m.group(1)) for b in sft_branches
                       if (m := re.fullmatch(r"step(\d+)", b)))
    assert sft_steps == list(range(1000, sft_steps[-1] + 1, 1000)), \
        f"SFT step set not contiguous: {sft_steps}"

    base_branches = set(revs[BASE]["branches"])
    s1 = stage_steps(base_branches, "stage1-")
    s2 = stage_steps(base_branches, "stage2-")
    s3 = stage_steps(base_branches, "stage3-")
    grid = ([("stage1", s) for s in [0] + geo_snap([x for x in s1 if x > 0],
                                                   2 ** 0.5)]
            + [("stage2", s) for s in geo_snap(s2, 2.0)]
            + [("stage3", s) for s in s3])
    base_revs = [f"{stage}-step{s}" for stage, s in grid]
    missing = [r for r in base_revs if r not in base_branches]
    assert not missing, f"snapped to non-existent branches: {missing}"

    from huggingface_hub import HfApi
    api = HfApi()
    rlvr_branches = {b.name for b in api.list_repo_refs(RLVR).branches}
    missing = [f"step_{n:04d}" for n in RLVR_GRID
               if f"step_{n:04d}" not in rlvr_branches]
    assert not missing, f"RLVR grid steps not on HF: {missing}"
    dpo_branches = {b.name for b in api.list_repo_refs(DPO).branches}
    assert "main" in dpo_branches

    ckpts = (
        [{"model_id": BASE, "revision": r, "role": "base_step",
          "stage": r.split("-")[0], "step": int(r.rsplit("step", 1)[1])}
         for r in base_revs]
        + [{"model_id": BASE, "revision": "main", "role": "base_endpoint"}]
        + [{"model_id": SFT, "revision": f"step{s}", "role": "sft_step",
            "step": s} for s in sft_steps]
        + [{"model_id": SFT, "revision": "main", "role": "sft_endpoint"}]
        + [{"model_id": DPO, "revision": "main", "role": "dpo_endpoint"}]
        + [{"model_id": RLVR, "revision": f"step_{n:04d}", "role": "rlvr_step",
            "step": n} for n in RLVR_GRID]
    )
    n_stage = {st: sum(1 for c in ckpts if c.get("stage") == st)
               for st in ("stage1", "stage2", "stage3")}
    out = {
        "_about": ("M05 population: the Think-SFT acquisition ladder plus a "
                   "declared log-spaced subsample of the Olmo-3-1025-7B base "
                   "ladder. Checkpoints enumerated as (model_id, revision) "
                   "strings; the unit of this experiment is the checkpoint, "
                   "not the base/aligned pair. Subsample rule in the producer "
                   "docstring -- a computation, not a curation."),
        "_producer": "scripts/build_m05_population.py",
        "_source_of_record": "data/model_revisions.json",
        "_generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_checkpoints": len(ckpts),
        "n_sft_steps": len(sft_steps),
        "n_rlvr_steps": len(RLVR_GRID),
        "rlvr_available": len({b for b in rlvr_branches
                               if b.startswith("step_")}),
        "n_base_steps": len(base_revs),
        "base_steps_by_stage": n_stage,
        "base_ladder_available": {"stage1": len(s1), "stage2": len(s2),
                                  "stage3": len(s3)},
        "checkpoints": ckpts,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}: {len(ckpts)} checkpoints "
          f"({len(base_revs)} base steps {n_stage} + {len(sft_steps)} SFT "
          f"steps + {len(RLVR_GRID)} RLVR steps + 3 endpoints "
          f"[base/sft/dpo mains])")


if __name__ == "__main__":
    main()
