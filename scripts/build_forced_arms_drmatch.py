#!/usr/bin/env python
"""LACAN VARIANT of build_forced_arms_105.py -- ADD-BESIDE, not an edit of
malign's builder. Identical in every respect except the `matched` selection.

WHY. The faller arm is a speech verb 15.5% of the time against the matched
arm's 7.6% (v4's own flags), and the measured dialogue rate splits the same
way in the TAIL: 12.1% of fallers sit above a 20% dialogue rate against 6.7%
of matched. The medians are identical (5.3% vs 5.2%), so this is not a centre
shift -- it is an upper-tail imbalance, and the faller-vs-matched contrast
therefore varies speech-inducing-ness alongside movement history.

EXCLUDING speech verbs costs 34% of cells and cannot work anyway: `nodded`
(34.3%), `sighed`, `smiled`, `shook`, `laughed` induce dialogue harder than
`answered` and are not speech verbs by any tag (`M04_syntagmatic/scripts/
dialogue_rate.py`). A class filter misses exactly the words that matter.

MATCHING costs no cells. The non-mover pool has a median of 104 candidates and
every cell has >= 20, so the selection has room to satisfy two criteria.

THE RULE, declared here before any output exists:

    candidates      unchanged -- non-movers, IS_VV, != faller, post > 0
    within TOL      |log2(post[w] / post[faller])| <= DR_TOL
    choose          the candidate minimising |rate(w) - rate(faller)|
    fallback        if NO candidate is within TOL, the original rule
                    (closest on probability) and `dr_matched` is False

TOL IS 0.15 IN LOG2, A FACTOR OF 1.11, AND THE RULE IS @registrar's [5515].2:
the probability match IS the construct and the dialogue gap is a confound, so a
residual gap is carried by a column while a degraded construct match is carried
by nothing. Spend tolerance on the confound only while the construct is
untouched.

TWO EARLIER TOLERANCES WERE PROPOSED AND BOTH ARE SUPERSEDED. I guessed 0.5
(rejected by my own sweep), then argued 0.25 from a rule of my own -- "median
must stay 3x inside the spec's 1.4x bar" -- which was arbitrary where
@registrar's is principled. Recorded because the tolerance is the only free
parameter here and an undeclared or post-hoc one is the defect class this
campaign spent two days on.

THE CORPUS-TRANSFER QUESTION IS ANSWERED, NOT ASSUMED. The rates are measured on
f11_l2 and applied to a different battery; the matcher uses ORDERING only.
Recomputed on `y`, an unrelated domain: **Spearman rho +0.686 over 4,858 shared
words**, levels differing as expected (median 5.5% against 8.8%). The ordering
transfers. `dialogue_rate.py --xcheck` reproduces it.

Rates from `meta/M04_syntagmatic/results/dialogue_rate.json`, measured on
f11_l2. A word's dialogue rate is a property of word AND genre: the ORDERING
should transfer to this battery, the LEVELS will not. Words absent from the
table are treated as unknown and take the corpus median rather than zero --
zero would make every unmeasured word a perfect match for a low-rate faller.
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

#: ── LEXICAL CLASS, CARRIED AS COLUMNS AND NEVER AS A FILTER ──────────────
#:
#: **THE ARMS ARE MATCHED ON PROBABILITY AND NOT ON WHAT KIND OF WORD THEY ARE**,
#: and RH asked the question that surfaced it: what if the risers are words like
#: `have`? Measured on the 8,169-cell table, the faller arm is a SPEECH verb
#: 16.4% of the time against the matched arm's 9.6%, and **21.5% of matched pairs
#: disagree on speech-verb status inside the same cell** (lacan, [5465]). `said`
#: alone is 616 fallers.
#:
#: That matters for this instrument specifically: A forces a word and measures
#: the surprisal of what FOLLOWS, and **a speech verb induces quoted dialogue.**
#: Forcing `said` and forcing `found` differ in more than movement history, and
#: a probability match cannot close a lexical-class gap because they are
#: different axes.
#:
#: **COLUMNS, NOT GATES** ([5233]/[5236]): a fixed class filter would drop cells
#: before any data exists — the same argument that keeps `log2_ratio` a column,
#: one level up. With these the analysis reports the primary on all cells and on
#: `class_match` cells, and the difference between the two is itself a result.
#: Which direction it moves is not predictable from here: M03 saw an effect GROW
#: from +0.0807 to +0.1303 when light verbs were removed, so removal can reveal
#: a dilution as easily as it can remove a confound.
SPEECH_VERBS = frozenset("""
say says said saying tell tells told telling ask asks asked asking
answer answers answered answering reply replies replied replying
whisper whispers whispered whispering shout shouts shouted shouting
mutter mutters muttered muttering murmur murmurs murmured murmuring
cry cries cried crying scream screams screamed screaming
call calls called calling speak speaks spoke speaking talk talks talked talking
add adds added adding repeat repeats repeated repeating
explain explains explained explaining admit admits admitted admitting
""".split())

#: "light" in the delexical sense -- high-frequency verbs carrying little
#: semantic content of their own. M03's dominance repair turned on exactly this
#: class: "dominance falls" was `have` falling.
LIGHT_VERBS = frozenset("""
be is are was were been being am
have has had having do does did doing done
get gets got getting gotten make makes made making
go goes went going gone come comes came coming
take takes took taking taken put puts putting
give gives gave giving given keep keeps kept keeping
let lets letting seem seems seemed seeming
""".split())


def _sha16(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def _input_digests(a):
    """sha256_16 of every file this build READ, so the run is reproducible."""
    out = {}
    for label, path in (("battery", a.battery),
                        ("pairs", a.pairs or os.path.join(
                            ROOT, "data", "base_aligned_pairs.json"))):
        try:
            out[label] = {"path": (os.path.relpath(path, ROOT)
                                   if path.startswith(ROOT) else path),
                          "sha256_16": _sha16(path)}
        except OSError as e:            #: named, never silently omitted
            out[label] = {"path": str(path), "sha256_16": None,
                          "error": type(e).__name__}
    return out


DR_PATH = os.path.join(ROOT, "meta", "M04_syntagmatic", "results",
                       "dialogue_rate.json")
DR_TOL = 0.15         #: log2; a factor of 1.11. SET BY @registrar's RULE at
                      #: [5515].2, which supersedes mine and is better:
                      #: THE ASYMMETRY OF REPAIRS. The probability match IS the
                      #: construct ("matched on improbability-under-aligned");
                      #: dialogue rate is a CONFOUND. A residual confound gap is
                      #: repairable downstream -- it is a measured COLUMN, so an
                      #: analysis reports the primary on all cells and again on
                      #: low-gap cells. A DEGRADED CONSTRUCT MATCH IS REPAIRABLE
                      #: BY NOTHING. So spend the tolerance on the confound only
                      #: as far as the construct is untouched: 0.15 holds the
                      #: median at 1.07x and takes the gap 2.19 -> 1.25pp; 0.25
                      #: would buy 0.30pp more for 1.07x -> 1.12x on the
                      #: construct's own axis, which a column can carry instead.
                      #: Frontier: `meta/M04_syntagmatic/results/drmatch_tolerance.json`.


def _dialogue_rates():
    """word -> P(quote within 6 tokens). Unknown words take the MEDIAN.

    Zero would be wrong and in a specific direction: it would make every
    unmeasured word a perfect match for a low-rate faller, so the matcher would
    systematically prefer words it knows nothing about.
    """
    import statistics
    d = json.load(open(DR_PATH))["rates"]
    r = {w: v["rate"] for w, v in d.items()}
    return r, statistics.median(r.values())


DR, DR_MED = _dialogue_rates()
dr_of = lambda w: DR.get(w.lower(), DR_MED)          # noqa: E731


def lex_flags(w):
    """(is_speech, is_light) for a surface form. Closed lists, declared above."""
    if not w:
        return None, None
    k = w.strip().lower()
    return (k in SPEECH_VERBS), (k in LIGHT_VERBS)


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
            #: TWO CRITERIA. Probability is a HARD gate at DR_TOL; dialogue
            #: rate is the objective inside it. Ordering matters: making
            #: dialogue rate the gate and probability the objective would let
            #: the probability match drift without bound.
            near = ([w for w in cands
                     if abs(math.log2(m.post[w] / qf0)) <= DR_TOL]
                    if cands and qf0 > 0 else [])
            if near:
                nm = min(near, key=lambda w: abs(dr_of(w) - dr_of(f)))
                dr_matched = True
            else:
                nm = (min(cands, key=lambda w: abs(math.log2(m.post[w] / qf0)))
                      if cands and qf0 > 0 else None)
                dr_matched = False
            #: FOURTH ARM: a riser matched to the faller on aligned probability.
            #: `ri` above is the max-EXCESS riser, which runs a median 12.9x more
            #: probable than the faller -- so faller-vs-riser reproduces A's
            #: original contrast WITH A's original confound. This one holds Q
            #: fixed.
            #:
            #: **`w != ri` ADDED 2026-08-11: THE TWO RISER ARMS MUST BE DISTINCT
            #: WORDS.** Without it a thin riser pool collapsed them onto the same
            #: word on 443 of 8,169 cells (5.4%) and the collision was merely
            #: FLAGGED (`riser_arms_collapse`). A flag is the right response to a
            #: fact about the data; this was a fact about the SELECTION -- the
            #: matched-non-mover arm has always excluded the faller by exactly
            #: this test (`w != f` above) and the riser arm simply never did.
            #:
            #: A cell that cannot supply a SECOND distinct riser now has NO fifth
            #: arm (rm is None) rather than a duplicate of the fourth. That is the
            #: honest failure: a missing arm is visible to any consumer, a
            #: duplicate one silently halves the contrast it was bought for.
            #: Registrar's condition on RH's five-arm ruling ([5464].1): fixed in
            #: the BUILDER before generation, not carried as a caveat column.
            ris_pos = [w for w in ris_oc if w != ri and m.post.get(w, 0.0) > 0]
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
                       dr_matched=dr_matched,
                       dr_faller=dr_of(f),
                       dr_matched_val=(dr_of(nm) if nm else None),
                       n_within_tol=len(near),
                       matched_q=(m.post.get(nm, 0.0) if nm else None),
                       matched_delta=(m.delta.get(nm) if nm else None),
                       log2_ratio=(math.log2(m.post[nm] / qf)
                                   if nm and qf > 0 and m.post.get(nm, 0) > 0
                                   else None),
                       n_nonmovers=len(m.nonmovers(tau=a.tau, min_mass=MIN_MASS)),
                       n_scored=len(m.post))
            #: lexical class per arm, and the WITHIN-CELL agreement flag, which
            #: is the one the contrast actually needs -- the comparison is
            #: faller-against-matched inside one cell, so a fleet-level balance
            #: would not license it.
            for _arm, _w in (("faller", f), ("matched", nm),
                             ("riser", ri), ("riser_matched", rm)):
                _sp, _li = lex_flags(_w)
                rec["%s_is_speech" % _arm] = _sp
                rec["%s_is_light" % _arm] = _li
            _fs, _fl = lex_flags(f)
            _ms, _ml = lex_flags(nm)
            rec["class_match"] = (None if nm is None
                                  else (_fs == _ms and _fl == _ml))
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
    #: **RECORD THE INVOCATION, NOT THE SCRIPT NAME.** The previous table said
    #: only that "--out variants produced the 105/v2/v3 series", so when its
    #: riser arms needed rebuilding on 2026-08-11 nobody could reproduce it:
    #: `--battery` and `--pairs` were both unrecoverable, and the default
    #: battery is NOT the one the table was built from. **A producer field that
    #: names a script is a citation; one that names the invocation is a
    #: reproduction.**
    json.dump(dict(tau=a.tau, min_mass=MIN_MASS, rule="CANONICAL",
                   basis="post", n_cells=len(out), cells=out,
                   _producer="scripts/build_forced_arms_105.py",
                   _invocation=" ".join([os.path.basename(sys.argv[0])]
                                        + sys.argv[1:]),
                   #: **AN INVOCATION IS REPRODUCIBLE ONLY IF ITS INPUTS ARE
                   #: CONTENT-ADDRESSED.** `--battery data/beam_sample_105_plus_anger.csv`
                   #: names a PATH. That file gained a row under its own name
                   #: between 10 and 11 Aug (the `He was so angry he wanted to`
                   #: twin), so the identical command line produced 9,180 cells
                   #: one day and 9,227 the next -- and every internal check
                   #: passed on both. lacan, [5468].
                   #:
                   #: Third instance of one failure in two days: `step` is not a
                   #: key because stages restart numbering; `prefer=fleet` and
                   #: `prefer=wider` give different bytes for one (model, prompt);
                   #: and a battery path is not its contents. **In each case the
                   #: identifier was stable and the thing it identified was not.**
                   #: With these a rebuild either reproduces or fails loudly.
                   _inputs=_input_digests(a),
                   _battery=os.path.relpath(a.battery, ROOT)
                            if not os.path.isabs(a.battery) or ROOT in a.battery
                            else a.battery,
                   _pairs_source=(a.pairs or "data/base_aligned_pairs.json"),
                   _arms=["undisturbed", "faller", "faller-matched",
                          "riser", "riser-matched"],
                   _columns={
                       "log2_ratio": "match quality, faller vs matched. A COLUMN, "
                                     "never a gate.",
                       "riser_arms_collapse": "retained for continuity; must now "
                                              "be False everywhere -- the two "
                                              "riser arms are required distinct.",
                       "<arm>_is_speech / _is_light":
                           "lexical class per arm from the closed lists at the "
                           "head of this file.",
                       "class_match": "faller and matched agree on BOTH classes, "
                                      "within the cell. The contrast is "
                                      "within-cell, so fleet-level balance would "
                                      "not license it."}),
              open(p, "w"))
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
