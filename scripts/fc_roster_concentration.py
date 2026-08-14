#!/usr/bin/env python
"""fc_roster_concentration.py — THE PRODUCER FOR THE LOW-CONCENTRATION QUERY.

    scripts/fc_roster_concentration.py

WHY IT IS BEING WRITTEN AFTER THE FACT, WHICH IS THE WRONG ORDER. The query it
runs was executed on 6 Aug 2026 as a shell heredoc and reported at docket
[4816]. Its conclusion — **THE LOW-CONCENTRATION REGIME IS UNRESOLVABLE WITH
ALIGNED-MODEL PAIRS** — retired a standing caveat and went into the register.
On 7 Aug a single number from that post (phi-4's entropy drop, +0.0812) failed
to reconcile against anything reconstructible, and the reason no one could
check it is that **no script on this machine computed it.** That is the same
position as the freed-mass figure this campaign withdrew: a load-bearing
quantity with no producer.

The difference, and it is the whole difference: the INPUTS still exist
(`true_word_probs` + the model registry), so this is rebuildable rather than
unrecoverable. Those are different states and must not be blurred.

WHAT THE QUERY IS. A deflationary competitor says the resist asymmetry is
really concentration — aligned models put more mass on fewer words, and that
alone makes the base find their continuations strange. Testing it needs pairs
that concentrate LITTLE but align STRONGLY. The declared criterion, fixed at
[4814] BEFORE looking and not touched since:

    concentrates little  =  entropy drop < 0.10 nats
    aligns strongly      =  fallers/site >= the roster median

If that cell is empty, the competitor cannot be separated from the effect using
published checkpoints — a statement about the world, not about n.

TWO QUANTITIES, TWO POPULATIONS, AND THE POPULATION IS PART OF THE NUMBER.
Entropy drop can be computed over every prompt in `true_word_probs` (~2,583) or
over the 210 beam prompts a pair contributes to the forced-continuation run.
For phi-4 those give +0.0448 and +0.1391 — **3.1x apart, and one crosses the
0.10 criterion while the other does not.** The heredoc did not say which it
used and neither did [4816]. Both are printed here, always, and the criterion
is evaluated on the FULL population because the query is about the roster
rather than about the beam sample.
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meta", "M01_displacement", "scripts"))

#: **THE CRITERION IS "BOTTOM TERCILE OF CONCENTRATION". 0.10 WAS ITS VALUE ON
#: THE POPULATION THEN IN USE, NOT THE CRITERION ITSELF.** [4814] declared it as
#: "the boundary of the low tercile in the 32-pair population, a value that
#: already exists rather than one chosen for this question". A rule of that form
#: says RECOMPUTE the boundary when the population changes; it does not say
#: carry the number across. Ruled by lacan at [4885].1 -- who declared no stake
#: first, the substitution reading resting on the damage null rather than on the
#: resist asymmetry.
#:
#: --prompts beam recomputes it on the 210-prompt beam sample, which is the
#: population the committed fit's drops live on. The interpolation method is a
#: declared choice, so it is named rather than inherited from a library default.
TERCILE = 1.0 / 3.0
DROP_MAX_LEGACY = 0.10        #: the superseded full-population instantiation
TWP = dict(dict_sha="b16011275c42955c", mode="raw", rule_version=3, theta=0.001)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", choices=("full", "beam"), default="full",
                    help="'beam' restricts to the 210-prompt sample the "
                         "committed fit's drops are computed on")
    ap.add_argument("--interp", choices=("linear", "nearest"), default="linear",
                    help="percentile interpolation — a declared choice, not a "
                         "library default")
    a = ap.parse_args()
    import fc_committed_entropy_test as C
    from malign_logits.cache import get_cache
    from malign_logits.movement import movement, CANONICAL, RESIDUAL_KEY
    from m05_sites import prepare

    cm = get_cache()
    st = cm._stash("true_word_probs")

    #: --- every (model, prompt) present, and the base->aligned map ----------
    have = {}
    for k in st.keys():
        if not isinstance(k, dict):
            continue
        if all(k.get(f) == v for f, v in TWP.items()):
            have.setdefault(k["model"], set()).add(k["prompt"])
    #: **THE 210 BEAM PROMPTS ARE AVAILABLE FOR EVERY PAIR, beams or not.** Most
    #: of the 65 roster pairs have no forced-continuation data, so a drop
    #: computed FROM beams does not exist for them -- but the drop over the beam
    #: PROMPT SET does, from twp, and that is what makes the 65 commensurate
    #: with the fit's 32. Restricting the prompt set is the whole fix.
    if a.prompts == "beam":
        import csv as _csv
        beam = {r["prompt"] for r in _csv.DictReader(
            open(os.path.join(ROOT, "data", "beam_sample_105.csv")))}
        have = {m: (ps & beam) for m, ps in have.items()}
        print("  RESTRICTED to the %d-prompt beam sample" % len(beam))
    #: **PAIRING IS `base_of`, THE REGISTRY'S OWN RELATION** -- not a lineage
    #: grouping, which pools across SIZE (Falcon3-1B/3B/7B/10B, Qwen2.5-0.5B
    #: with 7B) and would form cross-size pairs. Same rule m05_sites pins as
    #: `base := model_to_base[aligned]`, reached through the current API.
    from malign_logits.registry import Registry
    reg = Registry()
    pairs = []
    for mid in sorted(have):
        try:
            b = reg.base_of(mid)
        except Exception:
            b = None
        if b and b in have and b != mid:
            try:
                stage = reg.stage_of(mid)
            except Exception:
                stage = None
            pairs.append((b, mid, stage))
    if not pairs:
        sys.exit("registry returned no base_of pairs among scoreable models -- "
                 "this query is registry-defined and must not fall back to a "
                 "hand-list")
    print("ROSTER CONCENTRATION QUERY — the producer for [4816]")
    print("  candidate base>aligned pairs scoreable from true_word_probs: %d" % len(pairs))
    print()

    rows = []
    for base, aligned, stage in pairs:
        common = sorted(have[base] & have[aligned])
        if len(common) < 5:
            continue
        drops = []
        fall, nsite = 0, 0
        for p in common:
            eb = C.entropy(st, prepare, base, p)
            ea = C.entropy(st, prepare, aligned, p)
            if eb is None or ea is None:
                continue
            drops.append(eb - ea)
            try:
                kb = dict(TWP); kb["model"] = base; kb["prompt"] = p
                ka = dict(TWP); ka["model"] = aligned; ka["prompt"] = p
                ob, pb = prepare(st[kb]["rows"])
                oa, pa = prepare(st[ka]["rows"])
                mv = movement(pb, pa, CANONICAL)
                fall += len([w for w in mv.fallers if w != RESIDUAL_KEY])
                nsite += 1
            except Exception:
                pass
        if not drops or not nsite:
            continue
        rows.append(dict(base=base, aligned=aligned, n=len(drops), stage=stage,
                         drop=statistics.mean(drops),
                         fps=fall / nsite))

    #: **THE POPULATION IS THE WHOLE QUESTION AND [4816] NAMED ONE.** That post
    #: said "44 candidate base>SUPEREGO pairs". `base_of` returns EVERY child,
    #: including SFT rungs, which is a larger and different roster -- and the
    #: only pair that meets both criteria on the wider one is an SFT rung, so
    #: the two populations give OPPOSITE answers. Reported separately, never
    #: pooled, with the declared population first.
    import math
    _all = sorted(r["drop"] for r in rows)
    _h = TERCILE * (len(_all) - 1)
    _lo = int(math.floor(_h))
    if a.interp == "linear" and _lo + 1 < len(_all):
        DROP_MAX = _all[_lo] + (_h - _lo) * (_all[_lo + 1] - _all[_lo])
    else:
        DROP_MAX = _all[max(0, math.ceil(TERCILE * len(_all)) - 1)]
    print("  bottom-tercile boundary on THIS population (%s interp): %+.5f"
          % (a.interp, DROP_MAX))
    print("  (the superseded full-population instantiation was %.2f)" % DROP_MAX_LEGACY)
    print()
    SUPEREGO = ("dpo", "kto", "ppo", "slic", "rlvr", "reasoning")
    for label, sel in (("DECLARED at [4816]: base>SUPEREGO only",
                        [r for r in rows if r["stage"] in SUPEREGO]),
                       ("WIDER: every base_of child, SFT rungs included", rows)):
        if not sel:
            continue
        m_ = statistics.median([r["fps"] for r in sel])
        hit = [r for r in sel if r["drop"] < DROP_MAX and r["fps"] >= m_]
        print("  %s" % label)
        print("     %d pairs | median fallers/site %.1f | MEETING BOTH: %d"
              % (len(sel), m_, len(hit)))
        for r in hit:
            print("        %-34s stage=%-4s drop %+.4f  fallers/site %.1f"
                  % ((r["base"].split("/")[-1][:16] + ">" +
                      r["aligned"].split("/")[-1][:16]), r["stage"],
                     r["drop"], r["fps"]))
        print()

    med_f = statistics.median([r["fps"] for r in rows])
    print("  pairs with both quantities: %d | median fallers/site %.1f" % (len(rows), med_f))
    print("  CRITERION (declared [4814], before looking): drop < %.2f AND fallers/site >= %.1f"
          % (DROP_MAX, med_f))
    print()
    hits = [r for r in rows if r["drop"] < DROP_MAX and r["fps"] >= med_f]
    print("  *** PAIRS MEETING BOTH: %d ***" % len(hits))
    for r in hits:
        print("      %-30s drop %+.4f  fallers/site %.1f"
              % (r["base"].split("/")[-1][:30], r["drop"], r["fps"]))
    print()
    #: **EVERY pair under the threshold, not a top-N.** A top-8 is an arbitrary
    #: window that can hide a member of the very set the criterion selects --
    #: it did, for phi-4, while I was reading the list to check the criterion.
    #: The criterion defines the set; print the set.
    low = sorted((r for r in rows if r["drop"] < DROP_MAX), key=lambda r: r["drop"])
    print("  ALL %d PAIRS UNDER drop < %.2f — the set the criterion selects:" % (len(low), DROP_MAX))
    print("    %-34s %9s %6s %9s %7s" % ("pair", "drop", "stage", "fallers/site", "n"))
    for r in low:
        print("    %-34s %+9.4f %6s %9.1f %7d   %s"
              % ((r["base"].split("/")[-1][:16] + ">" + r["aligned"].split("/")[-1][:16]),
                 r["drop"], r["stage"] or "?", r["fps"], r["n"],
                 "at/above median" if r["fps"] >= med_f else "below median"))
    print()
    #: **THE CONCLUSION IS CONDITIONAL ON THE COUNT AND MUST BE PRINTED THAT
    #: WAY.** This block previously emitted the UNRESOLVABLE sentence in full on
    #: every run, prefixed "If the count above is 0" -- honest phrasing, and it
    #: still printed the conclusion verbatim on the beam-population run where
    #: the count was TWO. A reader skimming the output sees the sentence, not
    #: the conditional. Booked by lacan at [4909].4 as the worst form of the
    #: assert-while-computing defect: an assertion that ships INSIDE the
    #: producer is re-emitted every run by an artifact that looks authoritative.
    hits = [r for r in rows if r["drop"] < DROP_MAX and r["fps"] >= med_f]
    if not hits:
        print("  NO PAIR MEETS BOTH: every pair that concentrates little also")
        print("  displaces little, so the cell cannot be populated and the")
        print("  low-concentration regime is UNRESOLVABLE with aligned-model pairs.")
        print("  That is a fact about the published roster, not about our n.")
    else:
        print("  %d PAIR(S) MEET BOTH, so the cell IS populated on this population"
              % len(hits))
        print("  and the UNRESOLVABLE conclusion does NOT hold here. Which")
        print("  population and which threshold instantiation produced this")
        print("  must travel with the statement either way.")


if __name__ == "__main__":
    main()
