"""q_h6_denominator_pass.py — REGISTRATION Q's H6 DENOMINATOR. COUNTS ONLY.

**WHY IT EXISTS.** RH's word admits two arms so Q delivers the full 3x2:

    H5  `departed`      marked vs unmarked, PAIR unit  -> magnitude at the twin
    H6  `A_|valence|`   marked vs unmarked, PAIR unit  -> norms at the twin

**H5 NEEDS NO NEW COUNT.** §Q1.2 clause 7 puts `departed` on the ANALYSED
denominator — §Q3's own H4 note says `departed` is zero BY CONSTRUCTION in a
zero-faller cell — which is exactly `tail_excess`'s. So H5 inherits H1's
measured 24,606 both-sides keys over k = 684.

**H6's DENOMINATOR DOES NOT EXIST ANYWHERE.** `A` runs on A-CELLS only
(§Q1.2 clause 6: >= 3 qualifying fallers AND >= 3 qualifying risers).
§Q1.3 publishes `pair_marked` 18,241 and `pair_unmarked` 17,782, but those
are PARTITION totals, not BOTH-SIDES keys, and `p_yield_pass.json`'s
`role_counts` is an unkeyed LIST per partition. **A pair contrast needs the
key, so the number has to be measured.**

EGRESS CONTRACT — the property to audit:

  1 **NO `A` VALUE IS EVER FORMED.** `A` is `wmean(fallers) − wmean(risers)`
    weighted by |delta| over z-scores. This file computes NO weighted mean,
    reads NO z-value, and calls `N.lookup` for PRESENCE ONLY — the returned
    value is tested against None on the line it is fetched and is never
    bound to a name that outlives the expression. Identical to
    `p_yield_pass.py`'s treatment, which cleared at [4277].
  2 Every egress is a COUNT or a ratio of counts. No per-cell, per-stem or
    per-edge value leaves; no key leaves.
  3 **NO MARKED-vs-UNMARKED COMPARISON OF ANY QUANTITY.** Eligibility is a
    per-cell BOOLEAN and the only thing done with the two members' booleans
    is AND. A contrast needs two values; this pass never holds one.
  4 No intermediate file, no sampling, no seed.

**THE STATED LIMIT.** This bounds H6's n and nothing else. It cannot say
whether H6 will detect anything — that needs `sd(d_i)`, which is the paired
difference whose MEAN is H6's test statistic, and which therefore requires a
construction-blind pass at a third seat ([4322]'s form).
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

PAIRS = os.path.join(CAMPAIGN, "results", "population_d_684.json")
OUT = os.path.join(ROOT, "data", "q_h6_denominator.json")

SENTINEL = re.compile(r"^<<<.*>>>$")
CJK = re.compile(r"[一-鿿]")

#: public known answers, all on the docket ([4319]/[4321]/§Q1.3)
EXPECT_STEMS = 684
EXPECT_PAIR_TEXTS = 1368
EXPECT_BOTHSIDES_ANALYSED = 24606          # H1's, which H5 inherits
EXPECT_A_MARKED = 18241                    # §Q1.3 partition totals
EXPECT_A_UNMARKED = 17782


def refuse(msg):
    raise SystemExit("REFUSING: %s" % msg)


def main():
    from malign_logits.movement import movement, word_probs, CANONICAL, RESIDUAL_KEY
    import m01_concentration as CC
    import m01_norms as N
    import m01_registration_b as B

    from malign_logits.prompts import Prompts
    byid = {str(p.id): p for p in Prompts().all()}
    stems = json.load(open(PAIRS))["ids"]
    if len(stems) != EXPECT_STEMS:
        refuse("population carries %d stems, not %d" % (len(stems), EXPECT_STEMS))

    #: stem -> (marked text, unmarked text). §Q1.1's precedence, verbatim.
    pair = {}
    for s in stems:
        m_, u_ = byid.get(s + "_M"), byid.get(s + "_U")
        if m_ is None or u_ is None:
            continue
        t_m, t_u = m_.text, u_.text
        if SENTINEL.match(t_m) or CJK.search(t_m):
            continue
        if SENTINEL.match(t_u) or CJK.search(t_u):
            continue
        pair[s] = (t_m, t_u)
    if 2 * len(pair) != EXPECT_PAIR_TEXTS:
        refuse("pair map covers %d texts, not %d" % (2 * len(pair), EXPECT_PAIR_TEXTS))
    print("pairs  %d stems, %d texts" % (len(pair), 2 * len(pair)), flush=True)

    _p, mods, _h, _d = CC.frozen_population()
    edges, _drop = CC.operation_edges(mods)

    def mid(o):
        return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)

    edges = [(fam, mid(s.pre), mid(s.post)) for fam, _pos, s in edges]
    #: §Q3's declared key: an edge is (base, aligned); FAMILY LABELS COLLAPSE.
    #: `tulu` and `tulu-no-safety` share both checkpoints and are ONE edge.
    edges = sorted({(b, p) for _f, b, p in edges})
    print("edges  %d distinct (base, aligned) transitions" % len(edges), flush=True)

    norms, _f, _r = N.load_norms(verify=True)
    tabs = {d: norms[("en", d, "primary")] for d in ("arousal", "valence")}

    def eligible(text, pre, post):
        """-> (analysed?, A-cell?). BOOLEANS. No A value is formed here."""
        A_, B_ = word_probs(pre, text), word_probs(post, text)
        if A_ is None or B_ is None:
            return False, False
        m = movement({**A_.probs, RESIDUAL_KEY: A_.residual},
                     {**B_.probs, RESIDUAL_KEY: B_.residual}, CANONICAL)
        if not m.fallers:
            return False, False
        kf = kr = 0
        for w in m.fallers:
            k = N.norm_key(w, "en", fold=False)
            if N.is_function_word(k, "en"):
                continue
            #: PRESENCE ONLY — the value is tested and discarded on this line.
            if all(N.lookup(tabs[d], k.casefold(), "en")[0] is not None for d in tabs):
                kf += 1
        for w in m.risers:
            k = N.norm_key(w, "en", fold=False)
            if N.is_function_word(k, "en"):
                continue
            if all(N.lookup(tabs[d], k.casefold(), "en")[0] is not None for d in tabs):
                kr += 1
        return True, (kf >= B.QUALIFYING_MIN and kr >= B.QUALIFYING_MIN)

    C = collections.Counter()
    stems_with_a = set()
    stems_with_analysed = set()
    for ei, (pre, post) in enumerate(edges, 1):
        for s, (t_m, t_u) in pair.items():
            an_m, a_m = eligible(t_m, pre, post)
            an_u, a_u = eligible(t_u, pre, post)
            C["keys_seen"] += 1
            if an_m and an_u:
                C["bothsides_analysed"] += 1
                stems_with_analysed.add(s)
            elif an_m or an_u:
                C["onesided_analysed"] += 1
            C["A_marked"] += a_m
            C["A_unmarked"] += a_u
            #: the ONLY operation on the two members' eligibility is AND.
            if a_m and a_u:
                C["bothsides_A"] += 1
                stems_with_a.add(s)
            elif a_m or a_u:
                C["onesided_A"] += 1
        print("  [%2d/%d] %-52s keys %6d" % (ei, len(edges), "%s -> %s" % (pre[:24], post[:24]),
                                             C["keys_seen"]), flush=True)

    print("\n=== H6's DENOMINATOR (counts only; no A computed)", flush=True)
    print("  (stem, edge) keys enumerated        %7d" % C["keys_seen"])
    print("  BOTH sides analysed                 %7d   (H1/H5's denominator)"
          % C["bothsides_analysed"])
    print("  one side only, analysed             %7d" % C["onesided_analysed"])
    print("  stems surviving, analysed           %7d" % len(stems_with_analysed))
    print()
    print("  A-cells, marked side                %7d" % C["A_marked"])
    print("  A-cells, unmarked side              %7d" % C["A_unmarked"])
    print("  **BOTH sides A-cells  -> H6's k**   %7d" % C["bothsides_A"])
    print("  one side only, A                    %7d" % C["onesided_A"])
    print("  **STEMS SURVIVING for H6**          %7d   of %d"
          % (len(stems_with_a), len(pair)))
    print()
    print("  H6 retains %.1f%% of the both-sides analysed keys that H1/H5 use."
          % (100 * C["bothsides_A"] / C["bothsides_analysed"]
             if C["bothsides_analysed"] else 0))
    print("  **THIS BOUNDS H6's n AND NOTHING ELSE.** Whether H6 DETECTS")
    print("  anything needs sd(d_i), whose mean is H6's test statistic — a")
    print("  construction-blind pass at a third seat, never this one.")

    json.dump({"_what": "Registration Q H6 denominator — counts only.",
               "_egress": "counts and ratios of counts; no A value formed",
               "keys_enumerated": C["keys_seen"],
               "bothsides_analysed": C["bothsides_analysed"],
               "onesided_analysed": C["onesided_analysed"],
               "stems_analysed": len(stems_with_analysed),
               "A_marked": C["A_marked"], "A_unmarked": C["A_unmarked"],
               "bothsides_A": C["bothsides_A"], "onesided_A": C["onesided_A"],
               "stems_A": len(stems_with_a), "stems_total": len(pair)},
              open(OUT, "w"), indent=1, sort_keys=True)
    print("\nwrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
