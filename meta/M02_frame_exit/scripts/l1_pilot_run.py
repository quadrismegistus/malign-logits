"""Run the L1 coding pilot: 2 variants x 2 vendors, plus a batch-integrity arm.

    uv run python l1_pilot_run.py --dry-run     # show the calls, code nothing
    uv run python l1_pilot_run.py

## WHAT THE PILOT HAS TO ANSWER, AND IT IS THREE THINGS NOT ONE

**1. The resolved share**, per language, which is what the buy decision on 31,815
units reads. Reported over all mass AND over content mass, side by side, because
which denominator you use is a choice and it should be visible rather than baked
in ([5176].4).

**2. Whether the EXAMPLES are driving it.** Two prompt variants over the same
units: zero-shot with illustrations inside the class definitions, and the
balanced ten-example block. **Kappa cannot answer this** -- both vendors read the
same examples, anchor the same way, and agree, so an example prior raises
agreement while biasing the share. Reliability and validity come apart exactly
here, and only varying the prompt detects it.

**3. Whether BATCH 50 IS SAFE, which the stratified draw cannot test.** 150 units
over 21 groups is ~7 per group, and batches are per-pair, so the stratified arm
runs 7-long batches while the full pass would run 50. Misalignment is covered at
any size by the echo validator. ATTENTION DRIFT is not: a coder that classifies
item 3 carefully and item 44 lazily degrades silently. So two full-size batches
run from the largest groups and the first half is compared to the second.
Otherwise we price the design at 7 and buy it at 50.

## BLINDNESS

A unit is (surface, group, prompt_a, prompt_b). No model, no arm, no role, no
probability reaches the coder -- there is nothing in the record to be unblinded
by, which is structural rather than procedural.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

VENDORS = ["deepseek/deepseek-v4-flash", "openai/gpt-5.4-mini"]
BATCH = 50
INTEGRITY_GROUPS = 2          #: how many full-size batches for the drift check


def batches(units, size):
    """Group units by pair, then chunk. Batches never span pairs: the pair is
    the constant that makes one call cover many surfaces."""
    by = collections.defaultdict(list)
    for u in units:
        by[u["group"]].append(u)
    for g in sorted(by):
        us = by[g]
        for i in range(0, len(us), size):
            yield g, us[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from malign_logits.tasks import code_l1_surface_v1 as L

    S = json.load(open(os.path.join(ROOT, "data", "f11_l1_pilot_sample.json")))
    units = S["units"]
    print("pilot sample: %s" % {k: len(v) for k, v in units.items()})

    #: THE INTEGRITY ARM. Full-size batches from the largest groups, drawn from
    #: the FRAME rather than from the stratified sample -- the sample has ~7 per
    #: group by construction and cannot supply 50.
    F = json.load(open(os.path.join(ROOT, "data", "f11_k2_units.json")))
    surv = [r for r in F["units"] if r["survives"]]
    per_group = collections.Counter(r["group"] for r in surv)
    Q = {q["group"]: q for q in
         json.load(open(os.path.join(ROOT, "data", "f11_quintuplets.json")))["quintuplets"]}
    integ = []
    for g, _ in per_group.most_common():
        if len([x for x in integ if x[0] == g]) or len(integ) >= INTEGRITY_GROUPS:
            continue
        rows = [r for r in surv if r["group"] == g][:BATCH]
        if len(rows) == BATCH:
            integ.append((g, rows))
    print("integrity arm: %s" % [(g, len(r)) for g, r in integ])

    plan = []
    for variant, task_cls in (("zeroshot", L.L1SurfaceTask),
                              ("fewshot", L.L1SurfaceFewshotTask)):
        for vendor in VENDORS:
            for lang in ("en", "zh"):
                for g, chunk in batches(units[lang], BATCH):
                    plan.append(("stratified", variant, vendor, lang, g, chunk))
            for g, rows in integ:
                chunk = [{"group": g, "surface": r["surface"],
                          "pole_first": Q[g]["pole_a"], "pole_second": Q[g]["pole_b"]}
                         for r in rows]
                plan.append(("integrity", variant, vendor, Q[g]["language"], g, chunk))

    n_units = sum(len(c[-1]) for c in plan)
    print("\nPLAN: %d calls, %d unit-codings" % (len(plan), n_units))
    for kind in ("stratified", "integrity"):
        sub = [p for p in plan if p[0] == kind]
        print("   %-11s %4d calls  %6d unit-codings  batch sizes %s"
              % (kind, len(sub), sum(len(p[-1]) for p in sub),
                 sorted({len(p[-1]) for p in sub})))
    print("   variants %s   vendors %s"
          % (sorted({p[1] for p in plan}), [v.split("/")[-1] for v in VENDORS]))

    if a.dry_run:
        g, chunk = plan[0][4], plan[0][5]
        print("\nFIRST CALL, verbatim as the coder sees it:\n")
        print(L.prepare_batch(chunk[0]["pole_first"], chunk[0]["pole_second"],
                              [u["surface"] for u in chunk])[:900])
        return 0

    out, errors = [], []
    for kind, variant, vendor, lang, g, chunk in plan:
        surfaces = [u["surface"] for u in chunk]
        task = (L.L1SurfaceFewshotTask if variant == "fewshot" else L.L1SurfaceTask)()
        prompt = L.prepare_batch(chunk[0]["pole_first"], chunk[0]["pole_second"], surfaces)
        try:
            res = task.run(prompt, model=vendor)
        except Exception as e:
            errors.append((kind, variant, vendor, g, "%s: %s" % (type(e).__name__, str(e)[:90])))
            continue
        recs = [r.model_dump() if hasattr(r, "model_dump") else dict(r)
                for r in (res.records if hasattr(res, "records") else res["records"])]
        ok, why = L.validate_batch(surfaces, recs)
        if not ok:
            #: REFUSED, NOT REPAIRED. A partially-misaligned batch cannot be
            #: salvaged into a trustworthy one.
            errors.append((kind, variant, vendor, g, "REFUSED: %s" % why))
            continue
        for i, r in enumerate(recs):
            out.append({"kind": kind, "variant": variant, "vendor": vendor,
                        "lang": lang, "group": g, "pos": i + 1, "of": len(recs),
                        **{k: r[k] for k in ("n", "s", "cls", "content", "why")}})
        print("  %-11s %-9s %-22s %-20s %3d ok" % (kind, variant, vendor.split("/")[-1], g, len(recs)))

    p = os.path.join(CAMP, "results", "l1_pilot_coded.jsonl")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\ncoded %d unit-codings, %d batches refused or errored" % (len(out), len(errors)))
    for e in errors[:10]:
        print("   %s" % (e,))
    print("wrote %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
