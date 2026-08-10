#!/usr/bin/env python
"""Enumerate the M05 phase-1 checkpoint population -> data/m05_checkpoint_population.json.

    cd ~/github/malign-logits && uv run python scripts/build_m05_population.py

The population is CHECKPOINTS (model_id, revision), enumerated as strings, per
the [5148] plan-documents standard. Phase 1 is the Think-SFT acquisition ladder:
every step branch of allenai/Olmo-3-7B-Think-SFT plus its two anchors (the SFT
endpoint `main` and the base endpoint `allenai/Olmo-3-1025-7B` @ `main`).

Source of record: data/model_revisions.json (produced by
scripts/fetch_model_revisions.py, 2026-08-10). This script refuses to write if
the step set is not contiguous at 1000-step spacing -- a missing rung is a fact
the plan must state, not a gap to discover mid-run.
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


def main():
    revs = json.load(open(SRC))["models"]
    branches = revs[SFT]["branches"]
    steps = sorted(int(m.group(1)) for b in branches
                   if (m := re.fullmatch(r"step(\d+)", b)))
    expected = list(range(1000, steps[-1] + 1, 1000))
    assert steps == expected, f"step set not contiguous: {steps}"

    ckpts = ([{"model_id": BASE, "revision": "main", "role": "base_endpoint"}]
             + [{"model_id": SFT, "revision": f"step{s}", "role": "sft_step",
                 "step": s} for s in steps]
             + [{"model_id": SFT, "revision": "main", "role": "sft_endpoint"}])
    out = {
        "_about": ("M05 phase-1 population: the Think-SFT acquisition ladder, "
                   "checkpoints enumerated as (model_id, revision) strings. "
                   "The unit of this experiment is the checkpoint, not the "
                   "base/aligned pair."),
        "_producer": "scripts/build_m05_population.py",
        "_source_of_record": "data/model_revisions.json",
        "_generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_checkpoints": len(ckpts),
        "n_sft_steps": len(steps),
        "step_spacing": 1000,
        "checkpoints": ckpts,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}: {len(ckpts)} checkpoints "
          f"({len(steps)} SFT steps + 2 anchors)")


if __name__ == "__main__":
    main()
