"""The boundary test: does drift fall ACROSS self-written turn boundaries, or WITHIN
turns? Docket [246] through [256], [333]; run on RH's instruction.

    uv run .venv/bin/python scripts/f20x_boundary_test.py

WHAT IT DISCRIMINATES. It is the only instrument on the table that separates the two
live accounts of the drift finding:

    standing dispositional-answering   fixity is scoped to the RESPONSE. A bounded
    (RH)                               reply owes the previous one nothing, so drift
                                       should concentrate ACROSS boundaries.
    general fixing                     fixity is on the signifier's mobility. Turn
    (lacan)                            structure is irrelevant, so rates should be
                                       COMPARABLE across and within.

WITHIN-ARM by construction: aligned passages only. The arms differ enormously in
boundary rate (0.673 base against 0.459 aligned), and that asymmetry -- which has
sunk three conditioning analyses this campaign -- cannot reach a comparison made
inside one arm.

FOUR ARTEFACTS, THREE CONTROLLED AND ONE REGISTERED.

  (a) COUNTS. Across-turn sentence pairs outnumber within-turn pairs 2.56:1 ([256]),
      so a raw comparison finds across-concentration on an empty corpus. Handled by
      making each passage its OWN null: the expected across-share of a conflict is
      that passage's across-share of PAIRS. The bias becomes the null hypothesis
      instead of a confound.
  (b) TOPIC CHANGE. A self-written `Q: Where does 'The Starry Night' come from?` has
      changed the subject, and a difference across it is topic drift, not referent
      drift. Handled by the coder's `referent_status` field ([247] fix b) -- never a
      regex over coder prose, which measured the wrong thing at [216].
  (c) DISTANCE. Across-pairs are separated by 2.3x the tokens of within-pairs
      ([256]), and incompatibility accumulates with distance. Handled by caliper
      matching on token separation; 96% of within-pairs find a partner.
  (d) LEGIBILITY. Two accounts separated by a turn boundary are more legible AS two
      accounts than two adjacent sentences. No blinding removes this -- the boundary
      is in the text the coder reads. REGISTERED as a property of the object
      ([215]-class), not controlled.

lacan's concession sentence, frozen before the number existed: across-concentration
surviving (a), (b) and (c) concedes the fork, at any effect size significant on the
per-pair test at 29 base models, WITHOUT requiring (d) to be excluded first.
"""
import argparse
import json
import os
import re
import sys
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402
from malign_logits.embedding import _split_sentences  # noqa: E402
from malign_logits.tasks.code_conflict_locate import (  # noqa: E402
    ConflictLocationTask, prepare)

SRC = os.path.join(PATH_DATA, "f20x_codings.parquet")
OUT = os.path.join(PATH_DATA, "f20x_boundary_test.parquet")
ALIGNED = ("ego", "superego", "reinforced_superego")
BOUND = re.compile(r"\n\s*(?:Q|Question)\s*:", re.I)
DRIFT = {"quiet_drift", "bothness", "marked_contradiction", "dissolution",
         "name_arbitrary", "number_shift", "origin_displaced", "split_trace"}
CALIPER = 3   # tokens; matched-separation restriction


def has_drift(c):
    """`codes` is a JSON STRING on disk. Parsed, never membership-tested ([204])."""
    if not isinstance(c, str) or not c.strip():
        return False
    try:
        p = json.loads(c)
    except json.JSONDecodeError:
        return False
    if isinstance(p, str):
        p = [p]
    return bool(DRIFT & {str(x).strip().lower() for x in p})


def segments(text):
    """Sentences with their turn-segment index and token offset."""
    out, seg, tok = [], 0, 0
    parts = BOUND.split(text)
    for si, part in enumerate(parts):
        for s in _split_sentences(part):
            out.append({"sent": s, "seg": si, "tok": tok})
            tok += len(s.split())
    return out


def pair_stats(sents):
    """Across-share of PAIRS -- this passage's own null (artefact a)."""
    n = len(sents)
    if n < 2:
        return None
    within = across = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sents[i]["seg"] == sents[j]["seg"]:
                within += 1
            else:
                across += 1
    tot = within + across
    return {"n_sent": n, "within_pairs": within, "across_pairs": across,
            "across_share_pairs": across / tot if tot else np.nan}


def locate(sents, quote):
    """Which sentence contains this quote? Verbatim first, then normalised."""
    q = (quote or "").strip()
    if len(q) < 8:
        return None
    for norm in (lambda s: s, lambda s: " ".join(s.split()).lower()):
        nq = norm(q)
        for i, s in enumerate(sents):
            if nq and nq in norm(s["sent"]):
                return i
    return None


def build():
    d = pd.read_parquet(SRC)
    d = d[d.arm.isin(ALIGNED)].copy()
    d["text"] = d.text.fillna("")
    d = d[d.text.map(lambda t: bool(BOUND.search(t)))]
    d = d[d.codes.map(has_drift)]
    print(f"{len(d)} aligned, boundary-carrying, drift-coded passages "
          f"| {d.base_model_id.nunique()} base models")

    task = ConflictLocationTask()
    res = task.map([prepare(q, t) for q, t in zip(d.question, d.text)],
                   num_proc=8, desc="locate conflicts")
    d = d.reset_index(drop=True)
    for f in ("found", "quote_1", "quote_2", "attribute", "referent_status", "marked"):
        d[f] = [getattr(r, f) if r else None for r in res]
    d.to_parquet(OUT, compression="zstd", index=False)
    print(f"wrote {OUT}")
    return d


def analyse(d):
    rows = []
    for r in d.itertuples():
        if not r.found:
            continue
        sents = segments(r.text)
        ps = pair_stats(sents)
        if not ps:
            continue
        i, j = locate(sents, r.quote_1), locate(sents, r.quote_2)
        if i is None or j is None or i == j:
            continue
        a, b = sents[i], sents[j]
        rows.append({
            "base_model_id": r.base_model_id, "family": r.family,
            "referent_status": r.referent_status, "marked": r.marked,
            "across": a["seg"] != b["seg"],
            "sep": abs(a["tok"] - b["tok"]),
            **ps})
    e = pd.DataFrame(rows)
    print(f"\nlocated {len(e)} conflicts of {int(d.found.sum())} found "
          f"({len(d)} coded); unlocatable quotes are dropped, not guessed")
    return e


def report(e, label, min_models=8):
    if len(e) < 10:
        print(f"\n[{label}] n={len(e)} -- too few to test")
        return
    obs = e.across.mean()
    null = e.across_share_pairs.mean()
    print(f"\n[{label}] n={len(e)} conflicts, {e.base_model_id.nunique()} base models")
    print(f"   observed across-share   {obs:.3f}")
    print(f"   expected from pairs     {null:.3f}   <- each passage its own null")
    print(f"   difference              {obs - null:+.3f}")
    # Per-model, the unit that Rule 2 makes binding.
    g = e.groupby("base_model_id").apply(
        lambda x: x.across.mean() - x.across_share_pairs.mean(), include_groups=False)
    g = g[e.groupby("base_model_id").size() >= 2].dropna()
    if len(g) >= min_models:
        pos = int((g > 0).sum())
        p = stats.binomtest(pos, len(g), 0.5, alternative="greater").pvalue
        print(f"   per-model excess >0     {pos}/{len(g)}   sign p={p:.4f} (one-sided)")
        print(f"   per-model mean excess   {g.mean():+.3f}   median {g.median():+.3f}")
    else:
        print(f"   per-model: only {len(g)} models with >=2 conflicts -- not tested")


def main():
    d = pd.read_parquet(OUT) if os.path.exists(OUT) else build()
    if os.path.exists(OUT):
        print(f"{len(d)} coded rows from {OUT}")
    e = analyse(d)
    if not len(e):
        print("no locatable conflicts")
        return

    print("\n" + "=" * 72)
    print("BOUNDARY TEST -- does drift concentrate ACROSS turn boundaries?")
    print("=" * 72)
    report(e, "ALL located conflicts (artefact a controlled only)")

    keep = e[e.referent_status.isin(["all_keep_referent", "no_further_questions"])]
    report(keep, "+ (b) SAME-REFERENT boundaries only")

    # (c) caliper matching on token separation between the two accounts.
    w = keep[~keep.across]
    a = keep[keep.across]
    if len(w) and len(a):
        matched = [r for r in a.itertuples()
                   if ((w.sep - r.sep).abs() <= CALIPER).any()]
        mi = pd.DataFrame(matched)
        if len(mi):
            report(pd.concat([w, mi.drop(columns=["Index"], errors="ignore")]),
                   f"+ (c) DISTANCE-MATCHED (caliper +/-{CALIPER} tok)")
            print(f"   separation: within median {w.sep.median():.0f} tok, "
                  f"across median {mi.sep.median():.0f} tok (matched)")

    print(f"\n  unmarked only: {int((~keep.marked).sum())} of {len(keep)} "
          f"(the construct is QUIET drift)")
    print("\n  (d) legibility is REGISTERED, not controlled: a boundary makes two")
    print("      accounts more legible AS two, and no blinding removes it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if a.rebuild and os.path.exists(OUT):
        os.remove(OUT)
    main()
