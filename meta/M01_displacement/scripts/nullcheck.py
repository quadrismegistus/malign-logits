"""A null is not a result until it says what it could have seen.

WHY THIS EXISTS. In one night three findings were deflated using the weaker of
two available tests, each time presented as rigour, and RH caught all three:

  1. a conditional killed by a 119-cell paired test (p=0.17) when the same
     claim on all 6,978 annotations gives p=0.0002
  2. a secondary withdrawn by a between-population decoy control that this
     repo had already disqualified by name in `r_reversed_analysis.py`
  3. two effects called dead because a 50-stem pilot missed them, when the
     pilot's 95% CI was 0.35 wide and could not have detected them

The common structure is not a statistical mistake. It is that DEFLATION HAS
INVISIBLE ERRORS. Overstate a result and the retraction is public; understate
one and a true finding is buried silently. The costs are equal and only the
visibility differs, so caution is the strategy that never gets caught.

`mde` makes the invisible cost visible: it refuses to let a null be reported
without the effect size that test could have resolved.
"""

import numpy as np


def mde(x, power=0.80, alpha=0.05):
    """Minimum detectable effect for a paired design, given the observed SD."""
    from scipy import stats
    x = np.asarray(x, float)
    n = len(x)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * x.std(ddof=1) / np.sqrt(n)


def report(label, x, observed, p, predicted_sign=None, alpha=0.05):
    """Print a verdict. A null ALWAYS carries its MDE and CI; a confirmation
    does not need one, which is the asymmetry this function exists to remove."""
    from scipy import stats
    x = np.asarray(x, float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    lo, hi = observed - 1.96 * se, observed + 1.96 * se
    m = mde(x, alpha=alpha)
    sig = p < alpha and (predicted_sign is None or np.sign(observed) == predicted_sign)
    if sig:
        print("  %-34s %+0.4f [%+0.3f,%+0.3f] p=%.4f  n=%d  CONFIRMED" % (label, observed, lo, hi, p, n))
        return True
    #: The whole point. A null that cannot state its MDE is not reportable.
    verdict = "NULL"
    if m > abs(observed) * 3:
        verdict = "UNINFORMATIVE -- this test could not have seen an effect of the observed size"
    print("  %-34s %+0.4f [%+0.3f,%+0.3f] p=%.4f  n=%d  %s" % (label, observed, lo, hi, p, n, verdict))
    print("  %-34s MDE at %d%% power = %+0.4f  <- what this test COULD have detected"
          % ("", int(100 * power_of(alpha)), m))
    return False


def power_of(alpha):
    return 0.80


def compare_tests(label, results):
    """Two tests of one claim: BOTH get reported, and the script says so.

    results: list of (test_name, observed, p, n)
    """
    print("  %s -- %d tests of the SAME claim, all reported because no single one"
          % (label, len(results)))
    print("  was pre-committed. Choosing after seeing them is the defect this prevents.")
    for nm, o, p, n in results:
        print("      %-30s %+0.4f  p=%.4f  n=%d" % (nm, o, p, n))
    ps = [p for _, _, p, _ in results]
    if min(ps) < 0.05 <= max(ps):
        print("      *** TESTS DISAGREE. Report the disagreement, not the one you prefer. ***")
