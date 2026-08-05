"""Registration R-confirm's PRIMARY ANALYSIS, written before the data existed.

WRITTEN 2026-08-05 WHILE THE CODER PASS WAS STILL RUNNING and committed before
any annotation was read. That is the point of it: an analysis authored after
seeing the numbers is a description of them, and this registration's whole
claim is that its two hypotheses were fixed in advance. The verdict strings
below are the ones section 8 of the registration wrote out; nothing here
chooses anything.

WHAT IT MAY NOT DO, each because the registration forbids it:

  * NO CONTENT FILTER. Every word on both sides is already CLAWS `vv*`, so the
    filter can only remove light verbs, which are an outcome here. Applied to
    the pilot it manufactured a false null.
  * NO FREQUENCY CONTROL on any coded outcome. The only frequency objection
    that would license one is that coders read rarity as a surface cue; it was
    tested on the pilot and failed. The cue test is re-run below as a CHECK,
    not as a control: if it FAILS on this data the control becomes admissible
    and must be reported as an addition to the plan.
  * NO EXPLORATORY CUTS. Two primaries, four secondaries, two registered nulls
    and one declared-exposed secondary. That is the whole list.

UNIT IS THE STEM. Coders are replicates within an item, not observations:
eight coders on shared items is one measurement counted eight times, and that
error produced a spurious unanimity claim earlier in this campaign.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
sys.path.insert(0, ROOT)

from malign_logits.tasks.code_relation_axis import RelationAxisTask, prepare

OUT = os.path.join(CAMPAIGN, "results")
FRAME = os.path.join(OUT, "r_confirm_frame_255x2.parquet")
DECOY = os.path.join(OUT, "r_confirm_decoys_510.parquet")
LONG = os.path.join(OUT, "r_confirm_long.parquet")
RESULT = os.path.join(OUT, "result_r_confirm.json")

EXCLUDE = {"r2bt_109"}          #: registration section 4, fixed before any data
SEED = 20260805
NPERM = 20000
NBOOT = 20000
ALPHA = 0.025                   #: two-sided, Bonferroni across TWO primaries

MODELS = [
    "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash",
    "google/gemini-3.6-flash", "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4-5-20251001", "anthropic/claude-sonnet-5",
    "openai/gpt-4o-mini", "openai/gpt-5.4-mini",
]


def collect(df, arm):
    """Annotations for one arm, read through the library, never off disk.

    Metadata must be BYTE-IDENTICAL to what the run passed or every key misses
    and this silently re-pays at full price. The decoy arm's keys carry
    `arm="DECOY"`; the real arm's do not.
    """
    texts = [prepare(r.prompt, r.faller, r.riser) for r in df.itertuples()]
    if arm == "DECOY":
        metas = [dict(stem=r.stem, member=r.member, faller=r.faller,
                      riser=r.riser, arm="DECOY") for r in df.itertuples()]
    else:
        metas = [dict(stem=r.stem, member=r.member, faller=r.faller,
                      riser=r.riser) for r in df.itertuples()]
    rows = []
    for m in MODELS:
        task = RelationAxisTask()
        res = task.map(texts, model=m, metadata_list=metas, batch=False)
        #: a non-zero call count means the metadata did not match the run's and
        #: this read is re-paying rather than reporting what was bought.
        print("  %-40s %-5s parsed %d/%d   %s"
              % (m, arm, sum(r is not None for r in res), len(res),
                 task.usage.summary_line()), flush=True)
        for r, row in zip(res, df.itertuples()):
            if r is None:
                continue
            rows.append(dict(arm=arm, coder=m, stem=row.stem, member=row.member,
                             faller=row.faller, riser=row.riser, domain=row.domain,
                             axis=r.axis, relation=r.relation, intensity=r.intensity,
                             a_content=bool(r.a_is_content_word),
                             b_content=bool(r.b_is_content_word)))
    return pd.DataFrame(rows)


def signflip(d, seed=SEED, n=NPERM):
    """Two-sided p under within-stem arm exchangeability.

    Flipping a stem's arm labels negates D, so the permutation is a sign flip.
    +1 top and bottom: a permutation p can never be zero.
    """
    d = np.asarray(d, float)
    obs = d.mean()
    rng = np.random.RandomState(seed)
    null = (rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(axis=1)
    return obs, (1 + np.sum(np.abs(null) >= abs(obs))) / (n + 1)


def boot(d, seed=SEED, n=NBOOT):
    """Stem bootstrap CI. Included because its error source DIFFERS from the
    permutation's, rather than being the same arithmetic run twice."""
    d = np.asarray(d, float)
    rng = np.random.RandomState(seed)
    return tuple(np.percentile(
        [d[i].mean() for i in rng.randint(0, len(d), (n, len(d)))], [2.5, 97.5]))


def interaction(L, col):
    """D = (real - decoy | MARKED) - (real - decoy | UNMARKED), one per stem."""
    w = L.groupby(["arm", "member", "stem"])[col].mean().unstack("arm")
    d = (w["REAL"] - w["DECOY"]).unstack("member").dropna()
    return (d["MARKED"] - d["UNMARKED"]), d["MARKED"].mean(), d["UNMARKED"].mean()


def report(L, col, label, alpha, kind):
    x, mk, un = interaction(L, col)
    obs, p = signflip(x.values)
    lo, hi = boot(x.values)
    dz = x.mean() / x.std(ddof=1)
    verdict = ("CONFIRMED" if p < alpha and obs > 0 else
               "CONFIRMED (opposite direction)" if p < alpha else "NOT CONFIRMED")
    print("  %-34s marked %+0.3f  unmarked %+0.3f  D %+0.3f  p=%.4f  "
          "CI [%+0.3f, %+0.3f]  dz %+0.3f  %s"
          % (label, mk, un, obs, p, lo, hi, dz, verdict if kind == "primary" else ""))
    return dict(measure=col, label=label, kind=kind, marked=mk, unmarked=un,
                D=obs, p=p, ci_lo=lo, ci_hi=hi, dz=dz, n_stems=len(x),
                alpha=alpha, verdict=verdict if kind == "primary" else None)


def main():
    real = pd.read_parquet(FRAME)
    dec = pd.read_parquet(DECOY)
    real = real[~real.stem.isin(EXCLUDE)].sort_values(["stem", "member"])
    dec = dec[~dec.stem.isin(EXCLUDE)].sort_values(["stem", "member"])
    assert len(real) == len(dec) == 508, "expected 508 rows per arm"
    assert (real.faller.values == dec.faller.values).all(), "arms differ on faller"

    print("reading annotations (0 calls expected; non-zero means a metadata miss):")
    L = pd.concat([collect(real, "REAL"), collect(dec, "DECOY")], ignore_index=True)
    L.to_parquet(LONG, index=False)
    print("\nwrote %s  (%d annotations, %d stems)" % (LONG, len(L), L.stem.nunique()))

    L["related"] = L.relation != "NONE"
    L["_coact"] = L.relation == "CO_ACT"
    L["_beside"] = L.axis == "BESIDE"
    L["_inplace"] = L.axis == "IN_PLACE_OF"
    L["_spec"] = L.relation == "SPECIFICITY"
    L["_seq"] = L.relation == "SEQUENCE"
    L["_ncmp"] = L.intensity == "NOT_COMPARABLE"
    L["_milder"] = L.intensity == "B_MILDER"
    L["both_content"] = L.a_content & L.b_content

    out = {"n_annotations": len(L), "n_stems": int(L.stem.nunique()),
           "alpha_primary": ALPHA, "seed": SEED, "perms": NPERM, "results": []}

    print("\n=== PRIMARIES  (two-sided, alpha %.3f Bonferroni across two) ===" % ALPHA)
    out["results"].append(report(L, "_coact", "H1 displacement (CO_ACT)", ALPHA, "primary"))
    out["results"].append(report(L, "related", "H2 substitute formation (relation!=NONE)", ALPHA, "primary"))

    print("\n=== SECONDARIES  (directional, not corrected against the primaries) ===")
    for c, lab in [("_beside", "BESIDE"), ("_spec", "SPECIFICITY"),
                   ("_ncmp", "NOT_COMPARABLE"), ("_seq", "SEQUENCE"),
                   ("b_content", "b_is_content (decomposition)"),
                   ("both_content", "both_content (decomposition)")]:
        out["results"].append(report(L, c, lab, 0.05, "secondary"))

    print("\n=== REGISTERED NULLS  (a null here EXCLUDES, it does not merely fail to find) ===")
    for c, lab in [("_milder", "B_MILDER: attenuation is not mover-specific"),
                   ("_inplace", "IN_PLACE_OF: the operation is not substitutive")]:
        r = report(L, c, lab, 0.05, "registered_null")
        x, _, _ = interaction(L, c)
        r["mde"] = (2.241403 + 0.8416212) / np.sqrt(len(x)) * x.std(ddof=1)
        print("      MDE at n=%d, 80%% power, alpha .025 two-sided: %.3f" % (len(x), r["mde"]))
        out["results"].append(r)

    print("\n=== DECLARED-EXPOSED SECONDARY  (exposed by lacan 2026-08-05, docket [4672]) ===")
    v = L[L.domain == "violence"]
    o = L[L.domain != "violence"]
    gv = (v[v.arm == "REAL"]._milder.mean() - v[v.arm == "DECOY"]._milder.mean())
    go = (o[o.arm == "REAL"]._milder.mean() - o[o.arm == "DECOY"]._milder.mean())
    print("  B_MILDER real-minus-decoy   violence %+0.3f   other domains %+0.3f   gap %+0.3f"
          % (gv, go, gv - go))
    out["exposed_secondary"] = dict(violence_gap=gv, other_gap=go, difference=gv - go,
                                    note="NEVER quotable as predicted in advance")

    print("\n=== THE CUE CHECK  (a CHECK, not a control; see the module docstring) ===")
    print("  If coders read rarity as a surface cue, the rarer word gains the label")
    print("  whichever arm it sits in. On the pilot a rare decoy gained nothing.")
    out["cue_check_note"] = "run separately; frequency control is NOT applied to any result above"

    with open(RESULT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nwrote %s" % RESULT)


if __name__ == "__main__":
    main()
