"""What does alignment do GENERALLY, and what does it do to the arms?

    uv run python e_general_vs_institutional.py --run
    uv run python e_general_vs_institutional.py --report

RH: compare M03's field results against registrar's general M05 field flow, to
separate "what alignment does generally" from what is specific to the
institutional arm. This is plan C's question -- the one plan C could not answer
because its baseline was narrative continuations and its target was advice
prompts, so the vocabulary comparison compared prompt FORM.

## HOW THAT CONFOUND IS AVOIDED HERE

By running **registrar's instrument unchanged** on my corpus and comparing
MOVEMENTS rather than levels.

`m05_field_flow_fine.py` measures, per (checkpoint, prompt), the twp
probability mass falling in each of 287 fine fields -- USAS at 145 categories
decoded to labels, RID's 27, WordNet's 14 supersenses, and the trichotomised
Warriner/Brysbaert norms -- then takes the median over prompts and reports
`med[RLVR] - med[base_endpoint]`, with a floor of 0.003 on base-endpoint mass
so a field must be non-trivially present to qualify. It is REFERENCE-FREE: each
checkpoint on its own distribution, no anchor.

**A level differs between corpora because the prompts differ. A MOVEMENT is a
within-corpus change, so the form is held constant inside each one, and the
comparison of movements is a difference-in-differences.** That is what makes
this legitimate where plan C's was not.

Three corpora, one instrument, one ladder, one lineage:

    GENERAL         the 105 transgressive/neutral pairs (registrar's, reused
                    from data/m05_field_flow_fine.parquet -- not recomputed,
                    because recomputing it would be a second implementation
                    of a committed instrument)
    INSTITUTIONAL   the 30 institutional-arm prompts (M03's 18 + F21's 12)
    INDIVIDUAL      the 30 individual-arm prompts

## WHAT IT CAN AND CANNOT SAY

CAN: whether a field that moves in the institutional arm also moves generally.
If it does, it is alignment's ordinary vocabulary and the arm merely gets more
of it; if it moves in the arm and not generally, it is arm-specific.

CANNOT: attribute the difference to institutionality as opposed to any other
property that separates these 60 prompts from those 210. One lineage, no null,
and the corpora differ in more than their topic.

The dominance result this is testing came from the arm CONTRAST
(`d_ladder_fields.py`): norms:dominance/dominant +0.1220, 25 of 29 scenarios,
p=0.0001, replicated separately in F21 (10/11) and M03 (15/18) and matching
plan B's 45-of-46-lineage cross-section. The question here is whether
alignment raises dominance everywhere or only on the institutional side.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "meta", "M05_emergence", "scripts"))

OUT = os.path.join(CAMP, "results", "e_field_flow_arms.parquet")
GENERAL = os.path.join(ROOT, "data", "m05_field_flow_fine.parquet")
FLOOR = 0.003          #: registrar's presence floor, inherited not chosen


def run():
    import pandas as pd
    from d_ladder import labels
    #: REGISTRAR'S HELPERS, IMPORTED. The field mapping and the population
    #: order are a committed instrument; a copy here would be a second
    #: implementation of it, and the campaign books that as a defect class.
    import m05_field_flow_fine as FF
    from malign_logits.movement import word_probs

    L = labels()
    pop = FF.population()
    print("arms %d prompts | checkpoints %d" % (len(L), len(pop)))

    cache = {}
    #: registrar's per-word field set, built with ITS `usas_label` decoder so
    #: the 145 USAS codes resolve to the same strings its parquet holds --
    #: otherwise the two tables would not join on `field` and the comparison
    #: would silently be over an intersection of nothing.
    from malign_logits import fields

    def fine_fields(w):
        k = w.strip()
        if k in cache:
            return cache[k]
        fs = set()
        try:
            for c in fields.count(k, source="usas", all_tags=True,
                                  content_only=True)["counts"]:
                fs.add("USAS: " + FF.usas_label(c))
            for c in fields.count(k, source="rid", all_tags=True,
                                  content_only=True)["counts"]:
                fs.add("RID: " + c.rstrip(":"))
            for c in fields.count(k, source="wordnet", all_tags=True,
                                  content_only=True)["counts"]:
                fs.add("WN: " + c)
            for dim, r in fields.norms(k).items():
                for b in r["counts"]:
                    fs.add("NORM: %s=%s" % (dim, b))
        except Exception:
            pass
        cache[k] = fs
        return fs

    rows = []
    for idx, mid, role in pop:
        for t, (arm, scen, stratum) in L.items():
            wp = word_probs(mid, t)
            if wp is None or not wp.probs:
                continue
            fmass = collections.defaultdict(float)
            for w, p in wp.probs.items():
                for f in fine_fields(w):
                    fmass[f] += p
            for f, m in fmass.items():
                rows.append(dict(ckpt_idx=idx, role=role, arm=arm,
                                 stratum=stratum, scenario=scen,
                                 field=f, mass=m))
        print("  [%2d] %-44s %s" % (idx, mid.split("/")[-1][:42], role), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT)
    print("wrote %s: %d rows, %d fields" % (os.path.relpath(OUT, ROOT),
                                            len(df), df.field.nunique()))


def movement(df, group=None):
    """med[RLVR] - med[base_endpoint] per field, registrar's definition."""
    import pandas as pd
    if group is not None:
        df = df[df.arm == group]
    med = (df.groupby(["field", "ckpt_idx", "role"]).mass.median().reset_index())
    be = med[med.role == "base_endpoint"].set_index("field").mass
    rl = med[med.role == "rlvr_step"]
    rl = rl[rl.ckpt_idx == rl.ckpt_idx.max()].set_index("field").mass
    j = pd.DataFrame({"base_end": be, "rlvr": rl}).dropna()
    j["move"] = j.rlvr - j.base_end
    return j[j.base_end >= FLOOR]


def report(top=16):
    import pandas as pd
    G = pd.read_parquet(GENERAL)
    A = pd.read_parquet(OUT)
    gen = movement(G)
    inst = movement(A, "inst")
    indiv = movement(A, "indiv")
    j = pd.DataFrame({"general": gen.move, "inst": inst.move,
                      "indiv": indiv.move}).dropna()
    j["arm_gap"] = j["inst"] - j["indiv"]
    print("=" * 82)
    print("ALIGNMENT MOVEMENT (base endpoint -> RLVR end), ONE INSTRUMENT, THREE CORPORA")
    print("=" * 82)
    print("  general = registrar's 105 transgressive/neutral pairs")
    print("  inst / indiv = the 30 institutional / 30 individual arm prompts")
    print("  fields present at >= %.3f in all three: %d" % (FLOOR, len(j)))
    print("\n  A LEVEL would differ by prompt form; a MOVEMENT is within-corpus,")
    print("  so form is constant inside each column and the comparison is a")
    print("  difference-in-differences.")

    print("\n  --- fields where the INSTITUTIONAL arm moves and the GENERAL corpus does not ---")
    sel = j[(j["inst"].abs() > 2 * j["general"].abs())]
    for f, r in sel.reindex(sel["inst"].abs().sort_values(ascending=False).index).head(top).iterrows():
        print("    %-46s general %+8.4f  inst %+8.4f  indiv %+8.4f"
              % (f[:46], r.general, r["inst"], r["indiv"]))

    print("\n  --- the DOMINANCE fields, which the arm contrast flagged ---")
    for f, r in j.iterrows():
        if "dominance" in f.lower():
            print("    %-46s general %+8.4f  inst %+8.4f  indiv %+8.4f  gap %+8.4f"
                  % (f[:46], r.general, r["inst"], r["indiv"], r.arm_gap))

    print("\n  --- the largest GENERAL movers, and what the arms do with them ---")
    for f, r in j.reindex(j.general.abs().sort_values(ascending=False).index).head(top).iterrows():
        same = "same" if r.general * r["inst"] > 0 else "OPPOSITE"
        print("    %-46s general %+8.4f  inst %+8.4f  %s"
              % (f[:46], r.general, r["inst"], same))

    import numpy as np
    print("\n  correlation of movement across corpora, over %d shared fields:" % len(j))
    print("    general vs inst   Spearman %.3f" % j.general.corr(j["inst"], method="spearman"))
    print("    general vs indiv  Spearman %.3f" % j.general.corr(j["indiv"], method="spearman"))
    print("    inst vs indiv     Spearman %.3f" % j["inst"].corr(j["indiv"], method="spearman"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    if a.report:
        report()
    if not (a.run or a.report):
        ap.print_help()


if __name__ == "__main__":
    main()
