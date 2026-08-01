"""THE MEI TABLE — what S=18 can detect, against the four candidate targets.

Commission [1990](a), from [1988].3.

    .venv/bin/python scripts/m03_mei_table.py

WHY A TABLE AND NOT A NUMBER. RH is naming the MEI. **[1988].3's order of
operations is the whole point: THE MEI COMES FROM THE HYPOTHESIS, THEN THE
DESIGN IS CHECKED AGAINST IT. If S=18 cannot reach it, the answer is to change
the design or not run — NEVER to lower the MEI to what the design happens to
reach.** Two earlier attempts to source it from data produced artefacts.

So this file computes what the design reaches and what each candidate would
cost. **It does not recommend a target**, and the seat running it has no
standing to: a recommended MEI from the seat that computed the MDE is the
lowered-to-fit move wearing a table's name.

THE TWO ANCHORS, sourced rather than assumed ([1779], verified at two seats):

    0.0112   booked C1, rank-biserial r=0.283 -> PS 0.641 -> d 0.513 -> JS
             THE CEILING. C1 is institutional-vs-NEUTRAL BETWEEN strata; M03's
             contrast is individual-vs-institution WITHIN scenario, and a
             within-content contrast has every reason to be smaller than a
             between-content one.
    0.0050   entropy-trimmed C1, r=0.129 -> PS 0.565 -> d 0.230 -> JS
             BETTER-CONTROLLED, because C1's raw comparator confounded domain
             with continuation freedom (movement tracks entropy at r=+0.672).

CANDIDATE TARGETS are a THIRD and a HALF of each — lacan's weak prior at
[1988].3 is a third, ON THE RECORD AS A PRIOR AND NOT A VALUE.

RHO IS UNKNOWN AND BOTH COLUMNS TRAVEL. `rho_pair` is -0.054 with 95% CI
[-0.609, +0.536] on n=12 pairs; break-even for pairing to repay its halved unit
count is 0.500, **which is inside the interval, so the measurement cannot
decide it** ([1767]/[1773]). Reporting one column would pick a side of an
interval that contains the decision boundary.

THE sqrt(6) COLUMN IS OPTIMISM AND IS LABELLED AS SUCH. It assumes the six
within-scenario realisations per side have independent residuals — **and the
stem ICC is +0.356 within-family, so they demonstrably do not.** It is the most
favourable arithmetic available to the design and it is printed so that a
target unreachable even there is unreachable, full stop.
"""

import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGMA = 0.0219          #: SD of the family-averaged prompt value, m03_design_parameters
ALPHA, POWER = 0.05, 0.80
S_DESIGN = 18
ANCHORS = (("better-controlled (entropy-trimmed C1)", 0.0050),
           ("ceiling (booked C1)", 0.0112))
FRACTIONS = (("a third", 1 / 3), ("a half", 1 / 2))


def mde(S, sig=SIGMA, rho=0.0, sqrt6=False):
    t = stats.t.ppf(1 - ALPHA / 2, S - 1) + stats.t.ppf(POWER, S - 1)
    m = t * sig * np.sqrt(2 * (1 - rho)) / np.sqrt(S)
    return m / np.sqrt(6) if sqrt6 else m


def required_S(target, rho, sqrt6, cap=100000):
    """Smallest S whose MDE reaches `target`. Returns None past the cap."""
    for S in range(3, cap):
        if mde(S, rho=rho, sqrt6=sqrt6) <= target:
            return S
    return None


def main():
    print("THE MEI TABLE — commission [1990](a)\n")
    print(f"  sigma {SIGMA}  (SD of the family-averaged prompt value)")
    print(f"  alpha {ALPHA} two-sided, power {POWER}, n = S scenarios\n")

    print("WHAT S=18 REACHES")
    print(f"  {'':<26}{'rho=0':>10}{'rho=0.5':>10}")
    for tag, s6 in (("MDE", False), ("MDE / sqrt(6)  [optimistic]", True)):
        print(f"  {tag:<26}{mde(S_DESIGN, rho=0.0, sqrt6=s6):>10.4f}"
              f"{mde(S_DESIGN, rho=0.5, sqrt6=s6):>10.4f}")

    print("\nTHE FOUR CANDIDATE TARGETS, against S=18")
    print(f"  {'target':<40}{'value':>8}{'rho=0':>10}{'rho=0.5':>10}"
          f"{'rho=0 /s6':>11}{'rho=.5 /s6':>12}")
    rows = []
    for aname, a in ANCHORS:
        for fname, f in FRACTIONS:
            t = a * f
            cells = [mde(S_DESIGN, rho=r, sqrt6=s6) <= t
                     for r, s6 in ((0.0, False), (0.5, False),
                                   (0.0, True), (0.5, True))]
            print(f"  {fname + ' of ' + aname:<40}{t:>8.4f}" +
                  "".join(f"{'REACH' if c else 'no':>10}" if i < 2
                          else f"{'REACH' if c else 'no':>11}"
                          for i, c in enumerate(cells)))
            rows.append((aname, fname, t, cells))

    print("\nWHAT EACH TARGET WOULD COST — smallest S that reaches it")
    print(f"  {'target':<40}{'value':>8}{'rho=0':>10}{'rho=0.5':>10}"
          f"{'rho=0 /s6':>11}{'rho=.5 /s6':>12}")
    for aname, fname, t, _ in rows:
        ss = [required_S(t, r, s6) for r, s6 in
              ((0.0, False), (0.5, False), (0.0, True), (0.5, True))]
        print(f"  {fname + ' of ' + aname:<40}{t:>8.4f}" +
              "".join(f"{(str(s) if s else '>1e5'):>10}" if i < 2
                      else f"{(str(s) if s else '>1e5'):>11}"
                      for i, s in enumerate(ss)))

    print("\n  Prompts = S x 14 cells. S=18 is 252; the S column above is "
          "scenarios,\n  and each is fourteen authored-then-generated strings.")
    print("\n  NO TARGET IS RECOMMENDED HERE. [1988].3: the MEI comes from the "
          "hypothesis,\n  then the design is checked against it — never the "
          "other way round.")

    out = os.path.join(ROOT, "data", "m03_mei_table.csv")
    with open(out, "w") as f:
        f.write("anchor,fraction,target,mde_rho0,mde_rho50,mde_rho0_sqrt6,"
                "mde_rho50_sqrt6,S_rho0,S_rho50,S_rho0_sqrt6,S_rho50_sqrt6\n")
        for aname, fname, t, _ in rows:
            ms = [mde(S_DESIGN, rho=r, sqrt6=s6) for r, s6 in
                  ((0.0, False), (0.5, False), (0.0, True), (0.5, True))]
            ss = [required_S(t, r, s6) for r, s6 in
                  ((0.0, False), (0.5, False), (0.0, True), (0.5, True))]
            f.write(f"\"{aname}\",\"{fname}\",{t:.6f}," +
                    ",".join(f"{m:.6f}" for m in ms) + "," +
                    ",".join(str(s) if s else "" for s in ss) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
