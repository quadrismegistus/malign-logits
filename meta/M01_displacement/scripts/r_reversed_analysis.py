"""Counterbalanced order analysis: the pair as its own control.

WRITTEN BEFORE THE REVERSED PASS FINISHED and hashed before any annotation was
read. RH's design, and it removes the problem every control tonight created:
the argmax decoy over-picked light verbs, the random decoy still drew 27% of
them, the non-light decoy changed the population a third time. Each control was
DIFFERENT WORDS with its own lexical character, and that character was the
effect. Here the control is the identical pair, shown the other way round.

    FR   A = faller, B = riser     already coded (pilot 50 + confirmatory 255)
    RF   A = riser,  B = faller    the new pass

For any DIRECTIONAL field, the quantity of interest is

    effect = p(yes | FR) - p(yes | RF)

per stem, where p is the rate over the eight coders. Light verbs, lexical
frequency, argmax selection and pool composition all cancel by construction
rather than by control, because the same two words appear in both conditions.

WHAT THIS MEASURES THAT NOTHING BEFORE COULD.

  POSITION BIAS. If coders simply lean toward answering "yes, B" whatever B is,
  every directional number in this campaign is inflated by an unknown constant
  and no design we have run could detect it. Symmetric fields -- ones where the
  order of A and B should not matter -- estimate it directly: their FR-minus-RF
  difference is bias and nothing else.

  INSTRUMENT VALIDITY, FREE. `b_is_content_word` in RF asks about the same word
  that `a_is_content_word` asked about in FR. If the coders are consistent those
  two must agree. Any gap is measurement noise in a field that gated the whole
  registered analysis, and we have never had a way to measure it.

  MIRROR CONSISTENCY. B_MILDER in FR and B_STRONGER in RF are the same judgement
  about the same two words. They should fire at the same rate. A gap means the
  intensity question is not reading direction at all.

NOTHING HERE CHOOSES ANYTHING. The fields, the splits and the tests are fixed
below before the numbers exist. Symmetric fields are named as symmetric in
advance so their difference cannot be reinterpreted as a finding afterwards.
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
PILOT = os.path.join(OUT, "r_eight_coder_verbpaired_50x2.parquet")
CONFIRM = os.path.join(OUT, "r_confirm_frame_255x2.parquet")
REV = os.path.join(OUT, "r_reversed_frame_305x2.parquet")
LONG = os.path.join(OUT, "r_reversed_long.parquet")
RESULT = os.path.join(OUT, "result_r_reversed.json")

SEED = 20260806
NPERM = 20000

MODELS = [
    "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash",
    "google/gemini-3.6-flash", "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4-5-20251001", "anthropic/claude-sonnet-5",
    "openai/gpt-4o-mini", "openai/gpt-5.4-mini",
]

#: Declared BEFORE the data. A symmetric field is one whose answer should not
#: depend on which word is shown first; its FR-minus-RF difference is therefore
#: an estimate of position bias and may NEVER be reported as an effect.
SYMMETRIC = ["related", "_coact", "_beside", "_ncmp", "_same_pitch", "_opp", "_sameact"]
DIRECTIONAL = ["_milder", "_stronger", "_spec", "_inplace", "_seq"]


def collect(df, order):
    """Annotations for one order. Metadata must match the run's byte for byte:
    RF carries order="RF" and FR does not, and without that the two directions
    share stem, member and both words and would collide in the stash."""
    texts = [prepare(r.prompt, r.faller, r.riser) for r in df.itertuples()]
    metas = [dict(stem=r.stem, member=r.member, faller=r.faller, riser=r.riser)
             for r in df.itertuples()]
    if order == "RF":
        for m in metas:
            m["order"] = "RF"
    rows = []
    for m in MODELS:
        task = RelationAxisTask()
        res = task.map(texts, model=m, metadata_list=metas, batch=False)
        print("  %-40s %s parsed %d/%d   %s"
              % (m, order, sum(r is not None for r in res), len(res),
                 task.usage.summary_line()), flush=True)
        for r, row in zip(res, df.itertuples()):
            if r is None:
                continue
            rows.append(dict(order=order, coder=m, stem=row.stem, member=row.member,
                             domain=row.domain, A=row.faller, B=row.riser,
                             axis=r.axis, relation=r.relation, intensity=r.intensity,
                             a_content=bool(r.a_is_content_word),
                             b_content=bool(r.b_is_content_word)))
    return pd.DataFrame(rows)


def flags(L):
    L = L.copy()
    L["related"] = L.relation != "NONE"
    L["_coact"] = L.relation == "CO_ACT"
    L["_beside"] = L.axis == "BESIDE"
    L["_inplace"] = L.axis == "IN_PLACE_OF"
    L["_spec"] = L.relation == "SPECIFICITY"
    L["_seq"] = L.relation == "SEQUENCE"
    L["_opp"] = L.relation == "OPPOSITION"
    L["_sameact"] = L.relation == "SAME_ACT"
    L["_milder"] = L.intensity == "B_MILDER"
    L["_stronger"] = L.intensity == "B_STRONGER"
    L["_ncmp"] = L.intensity == "NOT_COMPARABLE"
    L["_same_pitch"] = L.intensity == "SAME_PITCH"
    return L


def sf(x, seed=SEED, n=NPERM):
    x = np.asarray(x, float)
    obs = x.mean()
    r = np.random.RandomState(seed)
    null = (r.choice([-1.0, 1.0], size=(n, len(x))) * x).mean(axis=1)
    return obs, (1 + np.sum(np.abs(null) >= abs(obs))) / (n + 1)


def diff(L, col, subset=None):
    """Per stem-member: rate in FR minus rate in RF."""
    s = L if subset is None else L[subset]
    w = s.groupby(["order", "stem", "member"])[col].mean().unstack("order").dropna()
    if "FR" not in w.columns or "RF" not in w.columns:
        return None, None, None, None
    d = w["FR"] - w["RF"]
    o, p = sf(d.values)
    return o, p, w["FR"].mean(), w["RF"].mean()


def line(L, col, label, subset=None, note=""):
    o, p, fr, rf = diff(L, col, subset)
    if o is None:
        print("  %-18s NO PAIRED ITEMS" % label)
        return None
    print("  %-18s FR %.3f  RF %.3f   diff %+0.3f   p=%.4f %s" % (label, fr, rf, o, p, note))
    return dict(measure=col, label=label, fr=fr, rf=rf, diff=o, p=p)


def main():
    fr_df = pd.concat([pd.read_parquet(PILOT), pd.read_parquet(CONFIRM)], ignore_index=True)
    rf_df = pd.read_parquet(REV)
    assert len(fr_df) == len(rf_df) == 610

    print("reading both orders (0 calls expected; non-zero means a metadata miss):")
    L = flags(pd.concat([collect(fr_df, "FR"), collect(rf_df, "RF")], ignore_index=True))
    L.to_parquet(LONG, index=False)
    print("\nwrote %s  (%d annotations, %d stems)" % (LONG, len(L), L.stem.nunique()))

    out = {"n": len(L), "stems": int(L.stem.nunique()), "seed": SEED, "results": []}

    print("\n=== 1. INSTRUMENT VALIDITY, FREE ===")
    print("b_content in RF asks about the same word a_content asked about in FR.")
    print("Consistent coders must agree. This field gated the registered analysis")
    print("and has never been checked.")
    fr = L[L.order == "FR"].set_index(["stem", "member", "coder"])
    rf = L[L.order == "RF"].set_index(["stem", "member", "coder"])
    k = fr.index.intersection(rf.index)
    agree_b = (fr.loc[k, "a_content"].values == rf.loc[k, "b_content"].values).mean()
    agree_a = (fr.loc[k, "b_content"].values == rf.loc[k, "a_content"].values).mean()
    print("  FR a_content vs RF b_content (same word): %.3f agreement over %d cells" % (agree_b, len(k)))
    print("  FR b_content vs RF a_content (same word): %.3f agreement" % agree_a)
    out["validity"] = dict(a_vs_b=float(agree_b), b_vs_a=float(agree_a), cells=int(len(k)))

    print("\n=== 2. POSITION BIAS, from fields declared SYMMETRIC in advance ===")
    print("These should be ZERO. Whatever they show is the coders' thumb, and it")
    print("is the correction every directional number below needs.")
    bias = []
    for c, lab in [("related", "relation!=NONE"), ("_coact", "CO_ACT"), ("_beside", "BESIDE"),
                   ("_ncmp", "NOT_COMPARABLE"), ("_same_pitch", "SAME_PITCH"),
                   ("_opp", "OPPOSITION"), ("_sameact", "SAME_ACT")]:
        r = line(L, c, lab, note="<- should be 0")
        if r:
            bias.append(r["diff"]); out["results"].append(dict(r, kind="symmetric"))
    print("  mean absolute position bias across symmetric fields: %.3f" % np.mean(np.abs(bias)))
    out["position_bias"] = float(np.mean(np.abs(bias)))

    print("\n=== 3. DIRECTIONAL FIELDS -- the actual tests ===")
    print("A positive diff means the field fires more when B is the RISEN word.")
    for c, lab in [("_milder", "B_MILDER"), ("_stronger", "B_STRONGER"),
                   ("_spec", "SPECIFICITY"), ("_inplace", "IN_PLACE_OF"),
                   ("_seq", "SEQUENCE")]:
        r = line(L, c, lab)
        if r:
            out["results"].append(dict(r, kind="directional"))

    print("\n=== 4. MIRROR CONSISTENCY ===")
    print("B_MILDER in FR and B_STRONGER in RF are the SAME judgement about the")
    print("same two words. A gap means intensity is not reading direction.")
    m_fr = L[L.order == "FR"]._milder.mean()
    s_rf = L[L.order == "RF"]._stronger.mean()
    s_fr = L[L.order == "FR"]._stronger.mean()
    m_rf = L[L.order == "RF"]._milder.mean()
    print("  B_MILDER|FR   %.3f   vs   B_STRONGER|RF %.3f   gap %+0.3f" % (m_fr, s_rf, m_fr - s_rf))
    print("  B_STRONGER|FR %.3f   vs   B_MILDER|RF   %.3f   gap %+0.3f" % (s_fr, m_rf, s_fr - m_rf))
    out["mirror"] = dict(milder_fr=m_fr, stronger_rf=s_rf, stronger_fr=s_fr, milder_rf=m_rf)

    print("\n=== 5. BY MARKEDNESS, directional fields only ===")
    for mem in ["MARKED", "UNMARKED"]:
        print("  -- %s --" % mem)
        for c, lab in [("_milder", "B_MILDER"), ("_inplace", "IN_PLACE_OF"),
                       ("_spec", "SPECIFICITY")]:
            r = line(L, c, "  " + lab, subset=(L.member == mem))
            if r:
                out["results"].append(dict(r, kind="by_member", member=mem))

    print("\n=== 6. PER CODER, B_MILDER and IN_PLACE_OF ===")
    print("Coder identity dominated the axis judgement in the disagreement read;")
    print("this is a within-coder difference, so a one-answer coder contributes 0.")
    print("  %-40s %10s %10s" % ("coder", "B_MILDER", "IN_PLACE_OF"))
    for m in MODELS:
        s = L[L.coder == m]
        a, _, _, _ = diff(s, "_milder")
        b, _, _, _ = diff(s, "_inplace")
        print("  %-40s %+10.3f %+10.3f" % (m, a if a is not None else float("nan"),
                                           b if b is not None else float("nan")))

    with open(RESULT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nwrote %s" % RESULT)


if __name__ == "__main__":
    main()
