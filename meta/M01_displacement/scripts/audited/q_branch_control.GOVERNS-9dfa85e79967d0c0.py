"""q_branch_control.py — CHECKLIST 9.3. EXERCISE THE BRANCH MAPPING AND READ
THE SENTENCE.

**WHY IT EXISTS.** Everything else built for Q points UPSTREAM: the known
answers guard the inputs, the structural counts guard the join, [4497]'s
eight mutants guard the gates. **The branch logic sits downstream of all of
it, one step from prose, and nothing read what the producer SAYS.**

The defect that produced this file: one hardcoded direction shared by two
measures that predict opposite signs, so a confirmation printed as a
**REVERSAL**. No number moved. Every gate passed. **A gate over inputs
cannot see a miswired output**, and it fails silently in both directions —
a confirmation shown as a reversal, or a reversal shown as a confirmation,
and both look like a clean run.

**IT IMPORTS THE PRODUCER AND CALLS THE PRODUCER'S OWN FUNCTION.** A test
against a re-implementation of the branch rule would be testing a sibling,
which is not the source. Importing `q_primary` executes module level only;
`main()` is never called and nothing is computed or written.

WHAT IT ASSERTS, per arm, on constructed observations of each sign:

  * the SIGN maps to the right one of THREE outcomes (5.1: matches /
    reversal / null -- never two);
  * **an arm whose direction is a CITED SIBLING PRIOR may not say "AS
    PREDICTED"** ([4501]). `tail_excess` has Q's own registered expectation
    at SQ3 L314; `departed` has none in H4's or H5's entries, and its sign
    comes from L845 where Q cites what F and G found. Both are frozen;
    they are not equally strong, and the sentence must not claim the
    weaker one is the stronger.
  * every arm's sentence names the FROZEN LINE its direction comes from
    (9.1), so the direction is falsifiable by citation rather than
    defensible by argument.

**AND THE FIRST POSITIVE CONTROL AIMED AT THIS FILE WAS MISCONCEIVED, WHICH
IS WORTH MORE THAN THE FILE.** To prove the control bites, `DIRECTION["H5"]`
was flipped +1 -> -1, reproducing the original defect exactly. **The control
PASSED.** It derives its own expectation from `DIRECTION[arm]` -- the same
constant the producer reads -- so flipping it moves the producer AND the
checker together and they cannot disagree.

    **A POSITIVE CONTROL MUST BREAK THE THING UNDER TEST, NEVER ONE OF ITS
    INPUTS.** An input read by both sides moves both sides.

That is `check-axes` (an audit taking its criterion from the artifact cannot
see a wrong criterion) and it is the mirror of [4497]'s inert blob mutant: a
mutant that moves NOTHING the gate observes proves nothing, and a mutant that
moves the observation AND the expectation proves nothing either. **The three
controls that do bite mutate the MAPPING** -- forcing `registered` True so a
cited-prior arm claims "AS PREDICTED" (4 failures), inverting the match test
(6), and dropping the citation (2).

**WHAT IT CANNOT DO.** It checks that the mapping from sign to sentence is
right. **It cannot check that the registered direction is right** -- that is
a claim about the frozen document, and the only defence there is 9.1's
citation discipline.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCER = os.path.join(HERE, "q_primary.py")


def load_producer():
    """Import the PRODUCER, not a copy of its rule. Module level only."""
    spec = importlib.util.spec_from_file_location("q_primary_under_test", PRODUCER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    qp = load_producer()
    fails = []

    print("=== 9.3 BRANCH CONTROL — the sentence, not the p")
    print("    producer: %s" % PRODUCER)
    for arm in ("H1", "H2", "H5", "H4"):
        want, cite, kind = qp.DIRECTION[arm]
        registered = "registered expectation" in kind
        #: **[4572]: AN ARM MAY DECLARE NO DIRECTION**, and two do. For
        #: those the assertions invert: the sentence must NOT claim a
        #: direction, must NOT say "wrong direction", and must say so.
        if want == 0:
            print("\n  %s   NO REGISTERED DIRECTION   %s" % (arm, kind))
            for obs, sig, label in ((-0.01, True, "negative, significant"),
                                    (+0.01, True, "positive, significant"),
                                    (+0.0001, False, "null")):
                sent, matches, reg, c, k = qp.branch_sentence(arm, obs, sig)
                print("    %-22s %s" % (label, sent[:104]))
                if matches is not None:
                    fails.append("%s: reports a match flag with no direction" % arm)
                if sig:
                    if "WRONG DIRECTION" in sent.upper() or "REVERSAL" in sent.upper():
                        if "RELATIONAL" not in sent.upper():
                            fails.append("%s %s: **claims a reversal with no "
                                         "registered direction**" % (arm, label))
                    if "AS PREDICTED" in sent:
                        fails.append("%s %s: **claims Q PREDICTED a direction "
                                     "it does not register**" % (arm, label))
                    if qp.NO_DIRECTION not in sent:
                        fails.append("%s %s: does not disclose that no "
                                     "direction is registered" % (arm, label))
                elif "BOUND" not in sent:
                    fails.append("%s null: not quoted as a bound" % arm)
            continue
        print("\n  %s   direction %+d   [%s]   %s" % (arm, want, cite, kind))
        for obs, sig, label in ((-0.01, True, "negative, significant"),
                                (+0.01, True, "positive, significant"),
                                (+0.0001, False, "null")):
            sent, matches, reg, c, k = qp.branch_sentence(arm, obs, sig)
            print("    %-22s %s" % (label, sent[:104]))

            expect_match = (obs < 0) if want < 0 else (obs > 0)
            if matches != expect_match:
                fails.append("%s %s: match flag wrong" % (arm, label))
            if not sig:
                if "BOUND" not in sent or "absence" not in sent:
                    fails.append("%s null: not quoted as a bound" % arm)
                continue
            if expect_match:
                if "REVERSAL" in sent:
                    fails.append("%s %s: a MATCH printed as a REVERSAL" % (arm, label))
                #: the load-bearing assertion, and the defect that produced
                #: this file in its other half.
                if registered and "AS PREDICTED" not in sent:
                    fails.append("%s: Q's own expectation not claimed" % arm)
                if not registered and "AS PREDICTED" in sent:
                    fails.append("%s: **claims Q PREDICTED a direction Q does "
                                 "not register**" % arm)
                if not registered and "REGISTERS NO DIRECTION" not in sent:
                    fails.append("%s: cited prior not disclosed as such" % arm)
            else:
                if "REVERSAL" not in sent:
                    fails.append("%s %s: a REVERSAL not printed as one" % (arm, label))
            if c not in sent:
                fails.append("%s %s: sentence does not cite its frozen line" % (arm, label))

    print("\n" + "=" * 62)
    if fails:
        print("**%d FAILURE(S)**" % len(fails))
        for f in fails:
            print("   - %s" % f)
        return 1
    print("ALL BRANCH SENTENCES CORRECT for every arm and every sign.")
    print("**This checks the mapping from sign to sentence. It cannot check")
    print("that the registered direction is right — that is 9.1's citation")
    print("discipline, and it is falsifiable only against the frozen text.**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
