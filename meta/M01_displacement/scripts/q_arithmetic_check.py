"""q_arithmetic_check.py — RECOMPUTE REGISTRATION Q's PUBLISHED CONSTANTS.

**WHY IT EXISTS.** [4461]'s sweep looked for the changed constant and found
every one of its eleven occurrences. [4462] then found six live stale values
it had missed, and the reason is general enough to be worth an instrument:

    **A SWEEP FOR THE CHANGED CONSTANT DOES NOT FIND THE VALUES DERIVED
    FROM IT.** `3.2356` and `0.00183` share no characters. Grep is an
    identity operation over strings; staleness is a relation over
    arithmetic. **The way to enumerate what changed is to RECOMPUTE it.**

Two further defects surfaced that even [4462]'s recompute-and-search missed,
and both are in the same family one level along:

  * a **share-of-scale** derived from a corrected MDE, in the same sentence
    as the MDE (`7.0%` of `0.0789`, stale because its numerator moved);
  * a **DECOMPOSITION** of the correct multiplier into the ingredients of
    the superseded one (`3.3393 = 2.3940 + 0.8416`, which sums to 3.2356).
    A sweep for `3.3393` lands on that line and reads past it, because the
    string it is looking for **is present and is right.**

So this file checks IDENTITIES, not occurrences. Every published constant is
recomputed from the inputs the registration itself states, and any line whose
stated inputs do not produce its stated output is reported.

**WHAT IT CANNOT DO.** It checks arithmetic, never design. A registration can
be internally consistent to the last digit and still test the wrong quantity
on the wrong unit; three of Q's defects this week were of that kind and none
of them would appear here. **A clean run of this file means the numbers agree
with each other, and nothing more.**
"""
import io
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
DOC = os.path.join(CAMPAIGN, "registrations", "registration_q_bridge.md")

Z_POWER80 = 0.8416                      # z(0.80), as the registration states it
TOL_5DP = 0.6e-5                        # half a unit in the last published place


def zcrit(alpha_two_sided):
    """Two-sided critical z. erfinv-free; scipy is not a dependency here."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alpha_two_sided / 2.0)


def check(label, stated, recomputed, tol=TOL_5DP):
    ok = abs(stated - recomputed) <= tol
    print("  %-42s stated %-10.5f recomputed %-10.5f  %s"
          % (label, stated, recomputed, "ok" if ok else "**MISMATCH**"))
    return ok


def main():
    text = io.open(DOC, encoding="utf-8").read()
    fails = []

    print("=== MULTIPLIERS: each split's z_crit + z(0.80)")
    for arms in (3, 4, 5):
        m = zcrit(0.05 / arms) + Z_POWER80
        print("  %d-way  alpha %.6f   multiplier %.4f" % (arms, 0.05 / arms, m))
    M4 = zcrit(0.0125) + Z_POWER80

    print("\n=== EXPLICIT FORMULA LINES: 'A x B / sqrt(C) = D', executed")
    #: matches e.g. **MDE(H5) = 3.3393 x 0.022282 / sqrt(684) = 0.00284**
    pat = re.compile(r"MDE\((H\d)\)\s*=\s*([\d.]+)\s*x\s*([\d.]+)\s*/\s*sqrt\((\d+)\)\s*=\s*([\d.]+)")
    seen = 0
    for m in pat.finditer(text):
        arm, mult, sd, k, stated = m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4)), float(m.group(5))
        seen += 1
        if not check("%s = %s x %s / sqrt(%d)" % (arm, m.group(2), m.group(3), k),
                     stated, mult * sd / math.sqrt(k)):
            fails.append("%s formula line" % arm)
    if not seen:
        fails.append("NO formula line matched — the pattern has drifted from the document")
        print("  **NO MATCHES — pattern drift, which is itself a failure**")

    print("\n=== POWER-TABLE ROWS: SE x multiplier, at each row's own alpha")
    #: matches e.g.  H2  tail_excess, k=33   0.001305   **0.00436**   0.0125
    row = re.compile(r"\**(H\d)\s+[`\w|_]+,\s*k=(\d+)\s+([\d.]+)\s+\**([\d.]+)\**\s+([\d.]+)")
    seen = 0
    for m in row.finditer(text):
        arm, k, se, stated, alpha = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
        seen += 1
        mult = zcrit(alpha) + Z_POWER80
        #: the SE column is itself published to 6 dp, so a row is allowed to
        #: disagree in the last place when the ROUNDED SE straddles a boundary.
        lo, hi = (se - 0.5e-6) * mult, (se + 0.5e-6) * mult
        ok = (round(lo, 5) <= stated <= round(hi, 5)) or abs(stated - se * mult) <= TOL_5DP
        print("  %-4s k=%-4d SE %.6f a=%.4f  stated %.5f  from SE %.5f  [%.7f, %.7f]  %s"
              % (arm, k, se, alpha, stated, se * mult, lo, hi, "ok" if ok else "**MISMATCH**"))
        if not ok:
            fails.append("%s table row" % arm)
    if seen < 6:
        print("  **only %d rows matched; the table has 6**" % seen)
        fails.append("power table: %d of 6 rows matched" % seen)

    print("\n=== SHARES OF SCALE: every '% of' that descends from an MDE")
    #: these are the derived-from-derived values; [4462] found one, and one
    #: more (H4's 7.0%) survived that round in the SAME SENTENCE as its MDE.
    SHARES = [("H1", 0.014765 / math.sqrt(684), 0.0738, "2.55"),
              ("H2", 0.001305, 0.0738, "5.90"),
              ("H4", 0.001711, 0.0789, "7.2"),
              ("H5", 0.022282 / math.sqrt(684), 0.0789, None)]   # NOT published
    for arm, se, scale, published in SHARES:
        share = 100 * M4 * se / scale
        if published is None:
            #: computed for the record, NOT asserted. H5's row is the only
            #: tested arm whose MDE the document never puts on a scale, and
            #: adding one here would be this file inventing a claim.
            print("  %-4s MDE/%.4f = %6.3f%%  -- INFORMATIONAL, document publishes no share for H5"
                  % (arm, scale, share))
            continue
        want = round(share, len(published.split(".")[1]))
        ok = abs(float(published) - want) < 1e-9
        print("  %-4s MDE/%.4f = %6.3f%%  -> %s%%  document says %s%%  %s"
              % (arm, scale, share, ("%%.%df" % len(published.split(".")[1])) % want,
                 published, "ok" if ok else "**MISMATCH**"))
        if not ok:
            fails.append("%s share-of-scale" % arm)
        if published not in text:
            print("       **%s%% NOT FOUND IN DOCUMENT** — the share moved or the sentence did" % published)

    print("\n=== DECOMPOSITIONS: 'X = a + b' where X is a multiplier")
    dec = re.compile(r"\**([\d.]{6})\**\s*=\s*([\d.]+)\s*\+\s*([\d.]+)")
    for m in dec.finditer(text):
        tot, a, b = float(m.group(1)), float(m.group(2)), float(m.group(3))
        ok = abs(tot - (a + b)) <= 1.5e-4
        print("  %s = %s + %s  -> %.4f  %s"
              % (m.group(1), m.group(2), m.group(3), a + b, "ok" if ok else "**MISMATCH**"))
        if not ok:
            fails.append("decomposition %s = %s + %s" % (m.group(1), m.group(2), m.group(3)))

    print("\n=== SUPERSEDED VALUES: must appear ONLY inside a fenced historical block")
    #: the block at 'THE EARLIER ROUND, KEPT AS RECORD' is allowed to carry them.
    STALE = ["0.00183", "0.00554", "0.00422", "0.00361", "5.72%", "0.00285", "7.0% of that SD"]
    lines = text.split("\n")
    fence = [i for i, l in enumerate(lines) if "THE EARLIER ROUND, KEPT AS RECORD" in l]
    fstart = fence[0] if fence else -1
    fend = fstart + 14 if fstart >= 0 else -1

    def historical(i, line, v):
        """A superseded value is ALLOWED in two self-labelling contexts."""
        if fstart <= i <= fend:                     # the fenced 'EARLIER ROUND' block
            return True
        #: `OLD -> **NEW**` states its own supersession on the line itself.
        return bool(re.search(re.escape(v) + r"\s*->\s*\**[\d.]", line))

    for v in STALE:
        hits = [i + 1 for i, l in enumerate(lines) if v in l and not historical(i, l, v)]
        print("  %-16s live occurrences outside the historical fence: %s"
              % (v, hits if hits else "none"))
        if hits:
            fails.append("%s live at %s" % (v, hits))

    print("\n=== ADJECTIVAL CLAIMS: ENUMERATED, NOT VERIFIED")
    #: [4465]'s fourth form. A qualitative claim DERIVED from a quantity
    #: contains no number, so recompute-and-search cannot reach it and
    #: execute-every-equation cannot either. Q's live instance rode its
    #: ratio from a claimed 3% to 5.90% while the adjective never moved.
    #:
    #: **THIS SECTION CANNOT PASS OR FAIL.** Whether "trivial" is true at
    #: 3.2% is a judgement, and a checker that scored it would be inventing
    #: the threshold it claims to be auditing. It ENUMERATES, and a human
    #: confirms each site carries a threshold or is scoped to a comparison
    #: the document can actually make.
    ADJ = ["wide margin", "trivial", "comfortably", "negligible", "well below",
           "far below", "best-powered", "underpowered", "powered by", "invisible at",
           "detectable at", "amply", "plenty", "easily"]
    hits = []
    for i, line in enumerate(lines, 1):
        low = line.lower()
        for a in ADJ:
            j = low.find(a)
            #: word-boundary on the left kills 'sAMPLE'-class false positives,
            #: which were 4 of 14 raw hits on the first sweep of this document.
            if j >= 0 and (j == 0 or not low[j - 1].isalpha()):
                hits.append((i, a, line.strip()[:96]))
                break
    for i, a, frag in hits:
        print("  L%-4d [%s]  %s" % (i, a, frag))
    print("  -- %d site(s). EACH MUST CARRY A THRESHOLD OR A SCOPED COMPARISON." % len(hits))
    print("  -- this section is a CHECKLIST. It has no verdict and never fails the run.")

    print("\n" + ("=" * 62))
    if fails:
        print("**%d PROBLEM(S)**" % len(fails))
        for f in fails:
            print("   - %s" % f)
        return 1
    print("ALL RECOMPUTED CONSTANTS AGREE WITH THE DOCUMENT.")
    print("**This says the numbers agree with each other. It says nothing")
    print("about whether they are the right numbers to have computed.**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
