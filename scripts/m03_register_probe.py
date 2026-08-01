"""REGISTER PROBE — is `should` vs `ought to` loaded differently by SPEAKER arm?

Commissioned at [1883].3, "@malign, BEFORE FREEZE OR IT IS WORTHLESS".

    .venv/bin/python scripts/m03_register_probe.py

WHAT IT ANSWERS AND WHY IT GATES A DECISION. M03 crosses FORM (marker-final /
medial / absent) and reads `ought to` as a second marker beside `should`. The
second marker is supposed to be a REFERENCE. [1883].1: **a reference with its
own arm-dependent register loading is not one** — `should` sits at 35 of 55
institutional prompts and 0 elsewhere, so an observed marker x SPEAKER
interaction could be predicted by register asymmetry alone, with nothing about
alignment in it.

    ratio SHIFTS by arm   asymmetry real and measured; [1883].2's one-sided
                          readout applies WITH FORCE
    ratio FLAT            the drafter heard something the model does not carry;
                          the interaction is clean and the rule costs nothing

**AND THE READOUT WAS DECLARED BEFORE THE RUN** ([1883].2), which is why this
file states it above the numbers rather than beside them: markers AGREE is
informative and markers DISAGREE is UNINTERPRETABLE, and that asymmetry was
ruled in advance and not fitted to whatever came out.

BLIND-SAFE. Base model only. No alignment arm, no outcome variable, nothing
about the hypothesis under test. It measures a property of ENGLISH as the base
model carries it.

METHOD. Each M03 `_absent` cell ends on the bare pronoun -- exactly the marker
position. For each stem we teacher-force both continuations and take the total
log-probability of the marker STRING, not of a first token:

    logP(" should")  vs  logP(" ought to")

**`ought to` IS TWO TOKENS AND `should` IS ONE.** Comparing first-token
probabilities would compare P(" ought") against P(" should") and silently omit
the ` to`, which is the same class of defect as a count without its unit. The
per-stem statistic is the LOG RATIO, so it is paired by construction and needs
no cross-stem normalisation.

POPULATION. All `_absent` cells in `pair_drafts/m03_scenarios_{A,B}.yaml`:
18 scenarios x 4 arm-person cells = 72 stems, 36 institutional and 36
individual. The scenarios are under redraft ([1906]) and this measures a
property of the MARKER, not of the scenario -- but the dependence is declared,
and `--kernel` re-runs it on lacan's kernel stems as a second population.
"""

import argparse
import os
import statistics
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "allenai/Olmo-3-1025-7B"
MARKERS = (" should", " ought to")


def stems(kernel=False):
    """(speaker, person, scenario_id, stem). The stem ends on the bare pronoun."""
    out = []
    if kernel:
        import json
        rows = json.load(open(os.path.join(
            ROOT, "meta/M03_proceduralization/m03_kernel.json")))
        for r in rows:
            for key, text in r["cells"].items():
                if key.endswith("_absent"):
                    sp, pn, _ = key.split("_")
                    out.append((sp, pn, r["scenario_id"], text))
        return out
    for name in ("m03_scenarios_A.yaml", "m03_scenarios_B.yaml"):
        for s in yaml.safe_load(open(os.path.join(ROOT, "pair_drafts", name))):
            for key, text in s["cells"].items():
                if key.endswith("_absent"):
                    sp, pn, _ = key.split("_")
                    out.append((sp, pn, s["scenario_id"], text))
    return out


def marker_logprob(model, tok, stem, marker, device):
    """Total log P of `marker` given `stem`, summed over the marker's tokens."""
    a = tok(stem, return_tensors="pt").input_ids
    b = tok(stem + marker, return_tensors="pt").input_ids
    assert b.shape[1] > a.shape[1], f"empty continuation for {marker!r}"
    ids = b.to(device)
    with torch.no_grad():
        logits = model(ids).logits.float()
    lp = torch.log_softmax(logits[0, :-1], dim=-1)
    tgt = ids[0, 1:]
    # only the positions the marker occupies
    start = a.shape[1] - 1
    return lp[start:, :].gather(1, tgt[start:].unsqueeze(1)).sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--kernel", action="store_true",
                    help="run on lacan's kernel stems instead of the 18 drafts")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = stems(args.kernel)[: args.limit]
    n_inst = sum(1 for r in rows if r[0] == "inst")
    print(f"REGISTER PROBE — {args.model}")
    print(f"{len(rows)} stems: {n_inst} institutional, {len(rows) - n_inst} "
          f"individual\n")
    print("DECLARED IN ADVANCE ([1883].2): markers AGREE is INFORMATIVE; "
          "markers\nDISAGREE is UNINTERPRETABLE. Stated here before the "
          "numbers, not after.\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).to(device).eval()

    by = {"inst": [], "indiv": []}
    per_row = []
    for sp, pn, sid, stem in rows:
        ls = marker_logprob(model, tok, stem, MARKERS[0], device)
        lo = marker_logprob(model, tok, stem, MARKERS[1], device)
        by[sp].append(ls - lo)
        per_row.append((sid, sp, pn, ls, lo, ls - lo))

    print(f"{'arm':<8}{'n':<5}{'median logP(should) - logP(ought to)':<40}"
          f"{'mean':<10}{'sd'}")
    for sp in ("indiv", "inst"):
        v = by[sp]
        print(f"{sp:<8}{len(v):<5}{statistics.median(v):<40.3f}"
              f"{statistics.mean(v):<10.3f}{statistics.stdev(v):.3f}")

    gap = statistics.median(by["inst"]) - statistics.median(by["indiv"])
    print(f"\nARM GAP (inst - indiv, median of per-stem log ratios): {gap:+.3f} nats")
    print(f"  ratio SHIFTS -> the register asymmetry is real and [1883].2 "
          f"applies with force")
    print(f"  ratio FLAT   -> the drafter heard something the model does not "
          f"carry\n")

    # Paired by scenario+person, which is the unit the design pairs on.
    pairs = {}
    for sid, sp, pn, _, _, d in per_row:
        pairs.setdefault((sid, pn), {})[sp] = d
    both = [(v["inst"] - v["indiv"]) for v in pairs.values() if len(v) == 2]
    pos = sum(1 for d in both if d > 0)
    print(f"PAIRED by scenario x person, n={len(both)}: "
          f"median {statistics.median(both):+.3f}, {pos} of {len(both)} positive")

    out = os.path.join(ROOT, "data",
                       f"m03_register_probe{'_kernel' if args.kernel else ''}.csv")
    with open(out, "w") as f:
        f.write("scenario_id,speaker,person,logp_should,logp_ought_to,log_ratio\n")
        for r in per_row:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nwrote {out} ({len(per_row)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
