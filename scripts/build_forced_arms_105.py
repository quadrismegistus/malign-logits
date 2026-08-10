#!/usr/bin/env python
"""The forced-word table for a passage run on the 105 minimal pairs.

WHAT THIS IS FOR. M04's Finding A found that forcing a demoted word makes the
aligned model's following region less probable, and left one confound open in
its own spec (SS8), written before the run: the faller is by construction
low-probability UNDER ALIGNED, so conditioning on it puts the aligned model in
a state it already assigns low probability to, and the next token inherits that
mechanically. "Separating them requires a word matched on improbability-under-
aligned but NOT demoted by alignment. No collected corpus has one."

This builds that word, per (pair, prompt), so a generation run can force it.

FOUR ARMS PER CELL:

    UNDISTURBED      no forcing. Where every effect measured this week lives:
                     entropy, KL, predictability, the span chain.
    FALLER           the vv* faller with the most negative delta. A's manipulation.
    FALLER-MATCHED   the unmoved vv* word closest to the faller in ALIGNED
                     probability. A's missing control: probability held,
                     movement varied.
    RISER            the vv* riser with the greatest EXCESS. A's original
                     contrast partner, and a median 12.9x MORE PROBABLE than the
                     faller -- so this arm carries A's confound, by design,
                     for continuity.
    RISER-MATCHED    the vv* riser closest to the faller in aligned probability.
                     Completes the three-way at fixed Q.

MATCHED ON THE POST ARM. The confound is what the ALIGNED model finds
improbable, so the control must be a word IT finds equally improbable and did
not demote. Matching on the base arm controls for what the base expected, which
is a different question. On Y, post-matching also yields more at every practical
tolerance (33 vs 27 of 167 cells at tau 0.005, tol 1.0).

NO TOLERANCE GATE AT GENERATION TIME. The closest available non-mover is taken
in every cell and `log2_ratio` is carried as a COLUMN. On the campaign's exhibit
cell -- Llama, "She was so angry she wanted to" -- there is NO match at tau
0.005 / tol 1.0: `kill` lands at 0.048 and the most probable unmoved word is
`take` at 0.017, a factor of 2.8. A fixed threshold would have dropped that cell
before any data existed. Match quality is something the analysis conditions on,
under a rule declared then; it is not something the corpus should lose now.
This is [5233]/[5236]'s gates-are-columns, applied one level earlier.

WORDS ARE DERIVED PER PAIR, NOT TAKEN FROM THE CSV. `beam_sample_105.csv`
declares one `faller` and one `riser` per PROMPT, across all pairs -- a design
prior, true on average and not necessarily on any given edge. Movement is a
property of the edge. A's contrast is within-pair, so per-pair derivation is
both correct and what A itself did. The CSV's columns are carried alongside as
`csv_faller` / `csv_riser` so the prior can be checked against the measurement.

    build_forced_arms_105.py --out data/forced_arms_105.json
"""
import argparse
import csv
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")

BATTERY = os.path.join(ROOT, "data", "beam_sample_105.csv")
TAU = 0.005          #: |Q-P| below which a word counts as unmoved
MIN_MASS = 0.001     #: a word must reach this in one arm to be a candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default=BATTERY,
                    help="prompt CSV. Defaults to the pinned 210-row "
                         "beam_sample_105.csv; pass another to extend it.")
    ap.add_argument("--out", default="data/forced_arms_105.json")
    ap.add_argument("--pairs", default=None, help="file with one A>B per line")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    import numpy as np
    from malign_logits import fields as F
    from malign_logits.step import Step
    from malign_logits.movement import CANONICAL

    #: LEXICAL VERBS ONLY -- CLAWS vv*, THE BATTERY'S OWN ELIGIBILITY TEST.
    #: A's spec says "top OPEN-CLASS faller" and the first build of this table
    #: dropped the filter entirely: `top_faller()` ranks by raw delta, so it
    #: returned discourse particles and the risers came back `the`, `she`, `they`.
    #:
    #: The second build used `fields.is_content_word`, which is still wrong
    #: here: it counts UNKNOWN forms as content (so `I` passes, byu=None) and
    #: admits adverbs (`then` is `rr`). `build_beam_sample.py:61` defines the
    #: battery's eligible frame as CLAWS `vv*` on BOTH sides for BOTH members,
    #: and these prompts all end mid-clause expecting a verb. Using any other
    #: definition means the forced word is drawn from a different population
    #: than the one the prompts were selected for.
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location(
        "_bbs", os.path.join(ROOT, "scripts", "build_beam_sample.py"))
    _bbs = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_bbs)
    IS_VV = _bbs.claws_vv()

    def open_class(m, keys):
        return [k for k in keys if IS_VV(k)]

    rows = list(csv.DictReader(open(a.battery)))
    print("battery: %d prompts, %d stems, %d domains"
          % (len(rows), len({r["stem"] for r in rows}),
             len({r["domain"] for r in rows})))

    if a.pairs:
        pairs = [l.strip() for l in open(a.pairs) if ">" in l]
    else:
        p = os.path.join(ROOT, "data", "base_aligned_pairs.json")
        raw = json.load(open(p))
        pairs = ["%s>%s" % (x["base"], x["aligned"]) if isinstance(x, dict) else x
                 for x in (raw if isinstance(raw, list) else raw.get("pairs", []))]
    if a.limit:
        pairs = pairs[:a.limit]
    print("pairs: %d\n" % len(pairs))

    out, skipped = [], {}
    for pr in pairs:
        b, al = pr.split(">")
        try:
            st = Step(b, al)
        except Exception as exc:
            skipped[pr] = "step: %s" % str(exc)[:50]
            continue
        n_ok = 0
        for r in rows:
            try:
                c = st.cell(r["prompt"])
                if not c.is_present:
                    continue
                m = c.movement(CANONICAL)
            except Exception:
                continue
            if m is None:
                continue
            fal_oc = open_class(m, m.fallers)
            ris_oc = open_class(m, m.risers)
            f = min(fal_oc, key=lambda w: m.delta.get(w, 0.0)) if fal_oc else None
            key = m.excess if m.rule.null_test else m.delta
            ri = max(ris_oc, key=lambda w: key.get(w, 0.0)) if ris_oc else None
            if f is None:
                continue
            #: closest available, no tolerance gate -- see the module docstring
            #: the control must be open-class too, or it is not comparable
            cands = [w for w in m.nonmovers(tau=a.tau, min_mass=MIN_MASS)
                     if w != f and IS_VV(w) and m.post.get(w, 0.0) > 0]
            qf0 = m.post.get(f, 0.0)
            nm = (min(cands, key=lambda w: abs(math.log2(m.post[w] / qf0)))
                  if cands and qf0 > 0 else None)
            #: FOURTH ARM: a riser matched to the faller on aligned probability.
            #: `ri` above is the max-EXCESS riser, which runs a median 12.9x more
            #: probable than the faller -- so faller-vs-riser reproduces A's
            #: original contrast WITH A's original confound. This one holds Q
            #: fixed. Where the cell has few qualifying risers the two collapse
            #: onto the same word, so that is flagged rather than hidden.
            ris_pos = [w for w in ris_oc if m.post.get(w, 0.0) > 0]
            rm = (min(ris_pos, key=lambda w: abs(math.log2(m.post[w] / qf0)))
                  if ris_pos and qf0 > 0 else None)
            qf = m.post.get(f, 0.0)
            rec = dict(pair=pr, stem=r["stem"], member=r["member"],
                       domain=r["domain"], stratum=r["stratum"],
                       prompt=r["prompt"],
                       csv_faller=r["faller"], csv_riser=r["riser"],
                       faller=f, faller_p=m.pre.get(f, 0.0), faller_q=qf,
                       faller_delta=m.delta.get(f),
                       riser=ri,
                       riser_q=(m.post.get(ri, 0.0) if ri else None),
                       riser_delta=(m.delta.get(ri) if ri else None),
                       riser_matched=rm,
                       riser_matched_q=(m.post.get(rm, 0.0) if rm else None),
                       riser_matched_delta=(m.delta.get(rm) if rm else None),
                       riser_matched_log2=(math.log2(m.post[rm] / qf)
                                           if rm and qf > 0 and m.post.get(rm, 0) > 0
                                           else None),
                       riser_arms_collapse=(rm is not None and rm == ri),
                       matched=nm,
                       matched_q=(m.post.get(nm, 0.0) if nm else None),
                       matched_delta=(m.delta.get(nm) if nm else None),
                       log2_ratio=(math.log2(m.post[nm] / qf)
                                   if nm and qf > 0 and m.post.get(nm, 0) > 0
                                   else None),
                       n_nonmovers=len(m.nonmovers(tau=a.tau, min_mass=MIN_MASS)),
                       n_scored=len(m.post))
            out.append(rec)
            n_ok += 1
        print("  %-56s %3d/%d cells" % (pr.split(">")[0][:56], n_ok, len(rows)))

    print("\ncells with a faller: %s" % format(len(out), ","))
    got = [r for r in out if r["matched"]]
    print("of which a matched non-mover exists at all: %s (%.0f%%)"
          % (format(len(got), ","), 100 * len(got) / max(len(out), 1)))
    if got:
        lr = np.abs([r["log2_ratio"] for r in got if r["log2_ratio"] is not None])
        print("\nMATCH QUALITY, |log2(Q_matched / Q_faller)| -- the column, not a gate")
        for t in (0.5, 1.0, 1.5, 2.0, 3.0):
            print("    within %.1f  (factor %.1f)   %5d cells  %4.0f%%"
                  % (t, 2 ** t, int((lr <= t).sum()), 100 * (lr <= t).mean()))
        print("    median |log2 ratio| %.2f" % np.median(lr))
    rs = [r for r in out if r["riser"]]
    print("\ncells with a riser too: %s" % format(len(rs), ","))
    both = [r for r in out if r["riser"] and r["matched"]]
    print("cells with ALL THREE arms: %s" % format(len(both), ","))
    if skipped:
        print("\nskipped pairs: %d" % len(skipped))
        for k, v in list(skipped.items())[:6]:
            print("    %-52s %s" % (k.split(">")[0][:52], v))

    p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    json.dump(dict(tau=a.tau, min_mass=MIN_MASS, rule="CANONICAL",
                   basis="post", n_cells=len(out), cells=out), open(p, "w"))
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
