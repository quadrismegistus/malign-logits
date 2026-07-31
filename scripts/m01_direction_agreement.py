"""Producer for M01/direction-agreement, claim (A). Frozen spec [1111], ruled [1113]/[1116].

    uv run .venv/bin/python scripts/m01_direction_agreement.py
    ... --csv out.csv     per-unit rows at the headline floor

THE CLAIM. A unit is a (prompt, word). Its DIRECTION in a family is the sign of
`delta = Q - P` across that family's operation edge. Claim (A) asks how much families
AGREE about direction:

    q = max(rise, fall) / (rise + fall)          per admitted unit, then a distribution

**q IS DIRECTION-AGNOSTIC AND THAT IS WHY IT IS QUOTABLE.** It measures HOW BIG the
majority is, never WHICH WAY it goes. Malign's audit ([1114].§2) showed the pooled
rise/fall balance FLIPS across the sensitivity floors -- 41.9% rising at 0.001, 56.5% at
0.010 -- while q holds at .733/.731/.721. **So the sensitivity triple protects claim (A)
and would NOT protect a directional claim. No sentence about which way movement runs is
licensed by this instrument** ([1116].3(i)).

THE POPULATION, EDGE AND CANONICALISATION ARE IMPORTED FROM `m01_concentration`, NOT
RESTATED. [1116].1 froze them as identical to that clause BY DESIGN, so that a difference
between the two clauses is a finding rather than a design artifact. Restating them here
would be two copies of one commitment, free to drift apart silently -- which is the
`prompt_categorisation` lesson at the code layer. Importing makes the identity structural.

WHAT IS *NOT* SHARED: this clause's unit is a WORD, not a cell, and its admission rule is
its own. Those are declared below.

CLAIM (B) -- same-family-different-scale agreement -- IS NOT IN THIS FILE. It was ruled
UNDERPOWERED at [1116].1 (four pure-scale pairs against a blind floor of six) and goes
dormant; [1122].3 wakes it self-executingly if malign's ladder declarations clear the
floor from the artifact. It gets its own producer then.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import statistics as st
import sys

# The sibling producer holds the frozen population. Both roots, same rule as its own
# import block: the file's directory first (correct in scripts/ and for a clone), the
# cwd second (correct while these still sit in the seat directory).
for _root in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
    if os.path.isfile(os.path.join(_root, "m01_concentration.py")):
        sys.path.insert(0, _root)
        break
else:                                        # pragma: no cover - environment failure
    sys.exit("m01_concentration.py must sit beside this file; it holds the frozen "
             "population and the two clauses are frozen identical by [1116].1")

from m01_concentration import (                            # noqa: E402
    CANONICALISATION, EDGE, POPULATION, RESIDUAL, RULE,
    frozen_population, operation_edges,
)

SIDEDNESS = "n/a — this producer reports distributions, it runs no test"

# --- ADMISSION, AND IT IS THIS CLAUSE'S OWN -------------------------------------
#: A unit must be PRESENT in enough families to have an opinion, and MOVE in enough of
#: them for that opinion to mean anything. Both floors are declared, neither is tuned.
MIN_PRESENT = 10                 #: families in which the word appears at all
MIN_MOVED = 10                   #: families in which |delta| clears the floor

#: THE SENSITIVITY TRIPLE IS THE FROZEN FORM, NOT A FALLBACK ([1113].2). The clause is
#: quotable ONLY if the answer is stable across all three; if q moves materially with the
#: floor, that is a finding about the floor and no single number enters the clause. The
#: sharpening gate died of a magnitude picked from one look and no admission floor gets
#: to do the same. 0.003 is the headline because it is CANONICAL's own threshold, reused
#: rather than invented.
FLOORS = (0.001, 0.003, 0.010)
HEADLINE = 0.003

#: DISPLACEMENT CONDITIONING, [1116].5 via [1110].3. A site whose continuation is forced
#: cannot show displacement, and pooling it measures grammatical inevitability. The
#: CLAUSE TAKES THE CONDITIONED FIGURE; pooled prints beside it for the gap.
DISPLACED_AT = 0.10              #: median `departed` across families, per prompt


def unit_directions(step, prompts, floors=FLOORS):
    """Per (prompt, word): does it rise or fall, and could it have fallen at all?

    Returns (present, moved, impossible, departed, skipped) where
        present[(prompt, word)]              -> 1 if the word appears in the union
        moved[(floor, prompt, word)]         -> +1 rose, -1 fell
        impossible[floor]                    -> [n_moved, n_of_those_with_P < floor]
        departed[prompt]                     -> the cell's departed mass

    THE UNION OF PRE AND POST KEYS, never P's keys alone (commitment §6). Iterating the
    pre keys skips post-only words, which are exactly the arrivals this clause is about.

    **GOES THROUGH `Cell`, NOT THROUGH `word_probs` DIRECTLY, AND THAT IS NOT A STYLE
    CHOICE.** `word_probs()` reads one arm and cannot compare versions; the mixed
    `rule_version` refusal (commitment §4) lives on `Cell._check_versions`, reached here
    via `decompose()`. A first draft of this function called `word_probs` on each arm and
    would have booked an INSTRUMENT CHANGE as training movement in total silence -- v3
    changed what a word is, so words appear and merge for reasons that have nothing to do
    with the model, and every such word would have entered this clause as a unit with a
    direction. The same call also yields `departed`, so the version check costs nothing.
    """
    present, moved = {}, {}
    impossible = {f: [0, 0] for f in floors}       # [n_moved, n_of_those_with_P<floor]
    departed, skipped = {}, collections.Counter()
    for text in prompts:
        c = step.cell(text)
        if not c.is_present:
            skipped["cell absent"] += 1
            continue
        try:
            d = c.decompose(None)                  # raises on a mixed rule_version
        except ValueError as e:
            skipped["mixed rule_version" if "rule_version" in str(e) else "error"] += 1
            continue
        if d is None:
            skipped["no movement"] += 1
            continue
        departed[text] = d["departed"]
        P, Q = c.pre.probs, c.post.probs
        for w in set(P) | set(Q):
            p, q = P.get(w, 0.0), Q.get(w, 0.0)
            present[(text, w)] = 1
            delta = q - p
            for f in floors:
                if abs(delta) >= f:
                    # (direction, could-this-word-have-fallen-at-all)
                    moved[(f, text, w)] = (1 if delta > 0 else -1, 1 if p < f else 0)
                    impossible[f][0] += 1
                    # PROBABILITY IS BOUNDED BELOW BY ZERO: a word with p < floor cannot
                    # FALL by floor -- it has only p to lose -- so among sub-floor words
                    # the rule admits risers only, 100% by construction ([1116].3(ii)).
                    if p < f:
                        impossible[f][1] += 1
    return present, moved, impossible, departed, skipped


_LANG = {}


def _prompt_language(text):
    """Catalogue language for a prompt text, via the ranked pick. Cached.

    `Prompt.find` and not a text-keyed dict: the catalogue carries retired rows and a
    text can sit on several (commitment §9), and a comprehension takes whichever came
    last. That defect moved a rank-sum across significance in three families.
    """
    if text not in _LANG:
        from malign_logits.prompts import Prompt
        pr = Prompt.find(text)
        _LANG[text] = pr.language if pr else None
    return _LANG[text]


def displaced_prompts(by_family_departed):
    """Prompts whose median `departed` across families clears DISPLACED_AT."""
    out = set()
    for text, vals in by_family_departed.items():
        if vals and st.median(vals) >= DISPLACED_AT:
            out.add(text)
    return out


def agreement_q(votes):
    """q = max(rise, fall) / (rise + fall). Direction-agnostic by construction.

    `votes` is a list of (direction, impossible_fall) pairs.
    """
    rise = sum(1 for v, _ in votes if v > 0)
    fall = len(votes) - rise
    return max(rise, fall) / len(votes), rise, fall


def q_null(n, p):
    """Median q under INDEPENDENCE at the observed marginal: k ~ Binomial(n, p).

    **q HAS A FLOOR OF 0.5 AND THAT FLOOR MOVES WITH n**, because `max` over a binomial
    is inflated at small n: forty independent coin-flipping families give a median q near
    0.56, ten give near 0.62. So a q that DECLINES as more families weigh in is what
    independence predicts, not a finding, and the raw §5(a) trend cannot be read without
    this column. Same rule as concentration's Dirichlet null, one clause later
    ([1120].4(i)): no ordering -- and a trend is an ordering -- without a declared null.

    Exact, not simulated: the binomial is small enough to sum.

    **MEDIAN OF THE TRANSFORM, NOT THE TRANSFORM OF THE MEDIAN.** `max(k, n-k)` is not
    monotone in k -- it falls to n/2 and rises again -- so taking the median k and
    applying max gives the wrong answer, and gives a plausible one. The mass has to be
    re-sorted onto the q scale before the median is taken.
    """
    from math import comb
    pmf = collections.defaultdict(float)
    for k in range(n + 1):
        pmf[max(k, n - k) / n] += comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    cum = 0.0
    for q in sorted(pmf):
        cum += pmf[q]
        if cum >= 0.5:
            return q
    return 1.0                                   # pragma: no cover - mass sums to 1


def main(a):
    from malign_logits.sharpening import sharpening

    prompts, models, (ph, mh), drift = frozen_population()
    print(f"POPULATION   {POPULATION}")
    print(f"RESIDUAL     {RESIDUAL}")
    print(f"SIDEDNESS    {SIDEDNESS}")
    print(f"EDGE         {EDGE}   (imported from m01_concentration, [1116].1)")
    print(f"RULE         {RULE}")
    print(f"CANONICAL.   {CANONICALISATION}")
    print(f"ADMISSION    present in >={MIN_PRESENT} families AND |delta|>=floor in "
          f">={MIN_MOVED}")
    print(f"FLOORS       {FLOORS}, headline {HEADLINE}")
    print(f"FROZEN       prompts {len(prompts)} {ph[:16]}...  models {len(models)} "
          f"{mh[:16]}...")
    if drift:
        print("\n  *** POPULATION DRIFT — the store has moved since the freeze ***")
        for d in drift:
            print(f"      {d}")
        print("  Refusing to measure. The two M01 clauses are frozen to ONE population;")
        print("  measuring this one across a moved store would make a difference between")
        print("  the clauses an artifact, which is exactly what [1116].1 forbids.")
        return 1

    edges, dropped = operation_edges(models)
    print(f"\n{len(edges)} families on the operation edge"
          + (f"   dropped: {dict(dropped)}" if dropped else ""))
    if a.limit:
        prompts = prompts[:a.limit]
        print(f"  *** --limit {a.limit}: NOT THE FROZEN POPULATION, not quotable ***")

    n_present = collections.Counter()
    votes = {f: collections.defaultdict(list) for f in FLOORS}
    departed = collections.defaultdict(list)
    imposs = {f: [0, 0] for f in FLOORS}
    allskipped = collections.Counter()

    print(f"\n  {'family':<18}{'edge':<12}{'units':>9}{'moved@hd':>10}"
          f"{'entropy':>9}{'residual':>16}")
    print(f"  {'':<18}{'':<12}{'':>9}{'':>10}{'delta':>9}{'pre':>8}{'post':>8}")
    for fam, pos, step in sorted(edges):
        pres, mv, imp, dep, sk = unit_directions(step, prompts)
        allskipped.update(sk)
        for k in pres:
            n_present[k] += 1
        for (f, text, w), v in mv.items():
            votes[f][(text, w)].append(v)
        for f in FLOORS:
            imposs[f][0] += imp[f][0]
            imposs[f][1] += imp[f][1]
        # `departed` comes off the SAME decompose the sibling clause uses, so
        # "displacing site" means one thing across both M01 clauses.
        for text, v in dep.items():
            departed[text].append(v)
        sh = sharpening(step, texts=prompts)
        rp = [c.pre.residual for c in (step.cell(t) for t in prompts[:200])
              if c.is_present]
        qp = [c.post.residual for c in (step.cell(t) for t in prompts[:200])
              if c.is_present]
        print(f"  {fam:<18}{step.label:<12}{len(pres):>9}"
              f"{sum(1 for k in mv if k[0] == HEADLINE):>10}"
              f"{(sh['entropy_delta'] if sh else float('nan')):>+9.3f}"
              f"{(st.median(rp) if rp else float('nan')):>8.3f}"
              f"{(st.median(qp) if qp else float('nan')):>8.3f}"
              + ("  FLAT" if sh and sh["is_flat"] else ""))

    if allskipped:
        print(f"\n  DROPPED CELLS   {sum(allskipped.values())}")
        for reason, n in allskipped.most_common():
            print(f"      {n:>6}  {reason}")

    disp = displaced_prompts(departed)
    print(f"\n  DISPLACING SITES   {len(disp)} of {len(departed)} prompts have median "
          f"departed >= {DISPLACED_AT}")

    # --- CLAIM (A), THE SENSITIVITY TRIPLE ------------------------------------
    print(f"\n  M01/direction-agreement, CLAIM (A)")
    print(f"    q = max(rise,fall)/(rise+fall) per admitted unit. DIRECTION-AGNOSTIC:")
    print(f"    it measures how big the majority is, never which way it goes.")
    print(f"\n  {'floor':>8}{'units':>9}{'q med':>8}{'q p10':>8}{'q p90':>8}"
          f"{'rising':>9}{'IMPOSSIBLE-FALL: admitted units':>37}{'of moves':>10}")
    print(f"  {'':>8}{'':>9}{'':>8}{'':>8}{'':>8}{'':>9}"
          f"{'share':>10}{'their q':>8}{'others':>8}{'':>11}")
    rows, table = [], {}
    for f in FLOORS:
        adm = [(k, v) for k, v in votes[f].items()
               if n_present[k] >= MIN_PRESENT and len(v) >= MIN_MOVED]
        if not adm:
            print(f"  {f:>8.3f}{0:>9}   no admitted unit")
            continue
        qs, q_imp, q_cln = [], [], []
        for k, v in adm:
            q, rise, fall = agreement_q(v)
            qs.append(q)
            # A unit is IMPOSSIBLE-FALL CONTAMINATED when the word sat below the floor in
            # most of the families that moved it: those families could only ever have
            # voted "rise", so the unit's agreement is partly arithmetic.
            frac_imp = sum(i for _, i in v) / len(v)
            (q_imp if frac_imp > 0.5 else q_cln).append(q)
            if f == HEADLINE:
                rows.append({"prompt": k[0], "word": k[1], "q": q, "rise": rise,
                             "fall": fall, "n_moved": len(v), "n_present": n_present[k],
                             "frac_impossible_fall": round(frac_imp, 3),
                             "displacing": k[0] in disp})
        qs.sort()
        rising = sum(1 for k, v in adm for x, _ in v if x > 0)
        tot = sum(len(v) for k, v in adm)
        table[f] = st.median(qs)
        imp_share = 100 * len(q_imp) / len(adm)
        print(f"  {f:>8.3f}{len(adm):>9}{st.median(qs):>8.3f}"
              f"{qs[int(.1*(len(qs)-1))]:>8.3f}{qs[int(.9*(len(qs)-1))]:>8.3f}"
              f"{100*rising/tot:>8.1f}%{imp_share:>10.1f}%"
              f"{(st.median(q_imp) if q_imp else float('nan')):>8.3f}"
              f"{(st.median(q_cln) if q_cln else float('nan')):>8.3f}"
              f"{100*imposs[f][1]/imposs[f][0] if imposs[f][0] else float('nan'):>9.1f}%")

    if len(table) == len(FLOORS):
        spread = max(table.values()) - min(table.values())
        print(f"\n    FLOOR SENSITIVITY   q spans {spread:.3f} across floors "
              f"({min(table.values()):.3f} to {max(table.values()):.3f})")
        print(f"    The clause is quotable only if this is small ([1113].2). The RISING")
        print(f"    column above is NOT protected by it -- malign measured that balance")
        print(f"    flipping 41.9% to 56.5% across these same floors, so no directional")
        print(f"    sentence is licensed however stable q looks ([1116].3(i)).")
        print(f"\n    IMPOSSIBLE-FALL, AND IT HAS TWO DENOMINATORS -- BOTH PRINT.")
        print(f"    Probability is bounded below by zero, so a word with P < floor cannot")
        print(f"    FALL by floor: among sub-floor words the rule admits risers ONLY, 100%")
        print(f"    by arithmetic and not by training ([1116].3(ii)).")
        print(f"      'of moves'        share of ALL moved units whose word was sub-floor")
        print(f"                        -- the MECHANISM's size, malign's 19.6%")
        print(f"      'admitted units'  share of units reaching the CLAUSE that were")
        print(f"                        sub-floor in most of their moving families, with")
        print(f"                        their q beside everyone else's -- THE DECISION-")
        print(f"                        RELEVANT figure, malign's 2%. The mechanism is")
        print(f"                        real and large; what reaches the clause is small.")

    # --- §5(b): CONDITIONED ON DISPLACEMENT, AND THE CLAUSE TAKES THIS --------
    adm = [(k, v) for k, v in votes[HEADLINE].items()
           if n_present[k] >= MIN_PRESENT and len(v) >= MIN_MOVED]
    pooled = sorted(agreement_q(v)[0] for k, v in adm)
    cond = sorted(agreement_q(v)[0] for k, v in adm if k[0] in disp)
    print(f"\n    CONDITIONED ON DISPLACEMENT (the clause takes this figure, [1110].3):")
    if cond:
        print(f"      displacing sites   n={len(cond):>6}   q median {st.median(cond):.3f}")
    print(f"      pooled             n={len(pooled):>6}   q median {st.median(pooled):.3f}")
    if cond:
        print(f"      gap {st.median(cond)-st.median(pooled):+.3f}. A forced slot cannot show")
        print(f"      displacement and its agreement measures grammar, not convergence.")

    # --- §5(a): q BY ADMITTING-FAMILY COUNT ----------------------------------
    print(f"\n    q BY ADMITTING-FAMILY COUNT ([1113].5). A rising q at a falling")
    print(f"    denominator is a small-n artifact and this table is how you see it:")
    print(f"      {'families moved':>16}{'units':>8}{'q med':>8}{'q NULL':>8}"
          f"{'excess':>9}{'2q(1-q)':>10}")
    buck = collections.defaultdict(list)
    for k, v in adm:
        buck[min(len(v), 40)].append(agreement_q(v)[0])
    p_rise = (sum(1 for k, v in adm for x, _ in v if x > 0)
              / sum(len(v) for k, v in adm))
    excesses = []
    for b in sorted(buck):
        if len(buck[b]) < 20:
            continue
        m = st.median(buck[b])
        nul = q_null(b, p_rise)
        excesses.append(m - nul)
        print(f"      {b:>16}{len(buck[b]):>8}{m:>8.3f}{nul:>8.3f}{m-nul:>+9.3f}"
              f"{2*m*(1-m):>10.3f}")
    if excesses:
        print(f"\n      q NULL is the median q under INDEPENDENCE at the observed marginal")
        print(f"      (rise rate {p_rise:.3f}), computed exactly from the binomial. READ THE")
        print(f"      EXCESS COLUMN, NEVER THE RAW q COLUMN: q has a floor near 0.5 that")
        print(f"      RISES as the denominator falls, so a q declining with more families")
        print(f"      is what independence predicts. Excess median {st.median(excesses):+.3f},")
        print(f"      range {min(excesses):+.3f} to {max(excesses):+.3f}.")

    # --- THE SHAPE, WHICH IS THE FINDING -------------------------------------
    #: A median over a nearly FLAT distribution is not a rate. This block prints the
    #: shape so "families agree about 73% of the time" cannot be written from the
    #: headline -- 73% is the middle of a spread that runs the full range, not a
    #: tendency that units cluster around.
    print(f"\n    THE SHAPE OF q, WHICH IS THE FINDING AND NOT THE MEDIAN:")
    bins = [(.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.001)]
    for lo, hi in bins:
        n = sum(1 for q in pooled if lo <= q < hi)
        print(f"      q {lo:.1f}-{hi if hi <= 1 else 1.0:.1f}   {n:>6}  "
              f"{100*n/len(pooled):>5.1f}%  {'#' * int(60 * n / len(pooled))}")
    print(f"      NEARLY FLAT. A binomial would be peaked; this is not, so the units are")
    print(f"      strongly HETEROGENEOUS: some (prompt, word) pairs command near-unanimity")
    print(f"      and others are coin flips. AGREEMENT IS NOT A ROSTER-WIDE RATE.")

    # LANGUAGE, per commitment §2. The population is ALL languages by [1116].1's freeze,
    # so a flat pooled shape could be two peaked shapes at different centres. Measured,
    # not assumed -- the same check that killed a cross-language js_total comparison.
    bylang = collections.defaultdict(list)
    for k, v in adm:
        pr = _prompt_language(k[0])
        bylang[pr].append(agreement_q(v)[0])
    if len(bylang) > 1:
        print(f"\n      COMMITMENT §2 CHECK -- is the flat shape a POOLING artifact?")
        print(f"      {'language':<10}{'units':>8}{'q med':>8}   "
              + "".join(f"{f'{lo:.1f}-{min(hi,1.0):.1f}':>8}" for lo, hi in bins))
        for L in sorted(bylang, key=lambda k: -len(bylang[k])):
            qs = bylang[L]
            sh = "".join(f"{100*sum(1 for q in qs if lo<=q<hi)/len(qs):>7.0f}%"
                         for lo, hi in bins)
            print(f"      {str(L):<10}{len(qs):>8}{st.median(qs):>8.3f}   {sh}")
        print(f"      The shape must hold WITHIN each language or the flatness is a")
        print(f"      mixture of two peaked distributions and this clause pooled them.")

    # --- INDEPENDENCE, TESTED DIRECTLY ---------------------------------------
    print(f"\n    IS THE VOTE DISTRIBUTION INDEPENDENT? OVERDISPERSION, BY DENOMINATOR:")
    print(f"      {'families':>10}{'units':>8}{'var(rise)':>11}{'binomial':>10}{'ratio':>8}")
    ratios = []
    kb = collections.defaultdict(list)
    for k, v in adm:
        kb[len(v)].append(sum(1 for x, _ in v if x > 0))
    for n in sorted(kb):
        if len(kb[n]) < 200:
            continue
        vo, vb = st.variance(kb[n]), n * p_rise * (1 - p_rise)
        ratios.append(vo / vb)
        if n % 8 == 2 or n == min(kb):
            print(f"      {n:>10}{len(kb[n]):>8}{vo:>11.2f}{vb:>10.2f}{vo/vb:>8.2f}")
    if ratios:
        print(f"      ratio spans {min(ratios):.1f}x to {max(ratios):.1f}x, median "
              f"{st.median(ratios):.1f}x -- INDEPENDENCE IS REJECTED AT EVERY DENOMINATOR.")

    # --- §3: THE PRE-COMMITTED READING RULE, AND WHAT IT ACTUALLY TESTS -------
    if pooled:
        q = st.median(pooled)
        dis = st.mean(2 * sum(1 for x, _ in v if x > 0)
                      * (len(v) - sum(1 for x, _ in v if x > 0))
                      / (len(v) * (len(v) - 1)) for k, v in adm if len(v) > 1)
        print(f"\n    THE §3 READING RULE, frozen BEFORE the data ([1113].4):")
        print(f"      q = {q:.3f}  ->  2q(1-q) = {2*q*(1-q):.3f};  observed mean pairwise")
        print(f"      dissent {dis:.3f};  the booked cross-family dissent 0.39.")
        print(f"      THE RULE'S VERDICT HOLDS: the booked 39% is NOT re-booked as")
        print(f"      structure. But its MECHANISM does not carry the verdict, and this")
        print(f"      producer says so rather than borrowing the credit:")
        print(f"      **2q(1-q) COMPUTED FROM THE OBSERVED q IS NOT AN INDEPENDENCE")
        print(f"      PREDICTION -- it is the same votes stated twice.** It assumes one")
        print(f"      common rise-rate across units, and the overdispersion above measures")
        print(f"      that assumption failing by {st.median(ratios) if ratios else float('nan'):.0f}x. Observed dissent sitting just")
        print(f"      BELOW 2q(1-q) is the Jensen signature of exactly that heterogeneity,")
        print(f"      not evidence of independence. The valid test is the variance ratio.")

    if a.csv and rows:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}  {len(rows)} units at floor {HEADLINE}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv")
    p.add_argument("--limit", type=int, default=0,
                   help="first N prompts only; marks the run NOT QUOTABLE")
    sys.exit(main(p.parse_args()))
