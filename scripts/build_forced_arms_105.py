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

#: DIALOGUE-RATE MATCHING, lifted from @lacan's `build_forced_arms_drmatch.py`
#: so there is ONE implementation of the selection rule and not two ([5527].1).
#: Rates are the QUOTATION-ADJACENCY rate of `dialogue_rate.py` -- P(quote mark
#: within 6 tokens after the word) -- NOT a dialogue rate: the measure catches
#: metalinguistic quotation too (`phrase` 49.5%), and @registrar fixed the name
#: of record at [5519]. Matching on it stays coherent because quote-adjacency is
#: what shifts the generative regime of the continuation.
#:
#: The ordering transfers to the corpus of use: rho +0.686 f11_l2->y (@lacan)
#: and +0.653 f11_l2->battery (malign, `dialogue_rate_transfer.py`), measured
#: independently and neither seat seeing the other's first. The matcher consumes
#: ORDERING ONLY, so the level difference between corpora is harmless.
DR_PATH = os.path.join(ROOT, "meta", "M04_syntagmatic", "results",
                       "dialogue_rate.json")
DR_TOL = 0.15        #: log2; a factor of 1.11. @registrar's rule at [5515].2 --
                     #: THE ASYMMETRY OF REPAIRS. The probability match IS the
                     #: construct; quote-adjacency is a CONFOUND. A residual
                     #: confound gap is repairable downstream because it is a
                     #: measured COLUMN; a degraded construct match is repairable
                     #: by nothing. So spend tolerance on the confound only as
                     #: far as the construct is untouched: 0.15 holds the median
                     #: probability distance at 1.07x and takes the gap
                     #: 2.19 -> 1.25pp. 0.25 buys 0.30pp more at 1.12x, which a
                     #: column can carry instead.


def _dialogue_rates():
    """word -> P(quote within 6 tokens). Unknown words take the MEDIAN.

    Zero would be wrong in a specific direction: it would make every unmeasured
    word a perfect match for a low-rate faller, so the matcher would
    systematically prefer words it knows nothing about.
    """
    import statistics
    d = json.load(open(DR_PATH))["rates"]
    r = {w: v["rate"] for w, v in d.items()}
    return r, statistics.median(r.values())


DR, DR_MED = _dialogue_rates()
dr_of = lambda w: DR.get(w.lower(), DR_MED)          # noqa: E731

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
            #: TWO CRITERIA. Probability is a HARD gate at DR_TOL; quote-
            #: adjacency is the objective inside it. ORDERING MATTERS: gating on
            #: quote-adjacency and optimising probability would let the
            #: probability match -- the construct -- drift without bound.
            #:
            #: The single-criterion rule this replaces kept finding exact
            #: probability twins doing a different job in the sentence, because
            #: speech verbs cluster at particular probabilities in this slot:
            #: `replied` as the control for `walked` at |log2| 0.000, quote-
            #: adjacency 51.4% against 3.7%. It changed the control on 2,666 of
            #: 7,309 cells (36.5%), and nothing in the aggregate showed it --
            #: the match-quality column read 0.000, which is a perfect score.
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
                   dr_tol=DR_TOL, dr_rates=os.path.relpath(DR_PATH, ROOT),
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
