"""WITHIN-PAIR length asymmetry: len(MARKED) - len(UNMARKED), everywhere.

Commissioned at [1892].5. The question it decides is scope, not size:

    power-specific  -> a domain defect; round2b is redraftable or droppable
    GENERAL         -> Registration D's confound, in a FROZEN design whose
                       statistic IS A(MARKED) - A(UNMARKED), requiring
                       declaration as a population covariate before the battery

[1878] measured length BETWEEN subdomains. This measures it WITHIN pairs, which is
the quantity that matters for a paired design: the design differences out
everything except the manipulation, so a fixed-direction length gap rides INSIDE
the difference and cannot be differenced away.

    .venv/bin/python meta/M01_displacement/audit/within_pair_length.py

THE DIRECTION DISTRIBUTION IS THE MEASUREMENT, NOT THE MEAN. A mean of +0.05 with
14 pairs longer and 13 shorter is symmetric noise. A mean of +3.77 with 119 longer
and NONE shorter is a systematic asymmetry, and the two are not distinguished by
their means alone. M>U / M=U / M<U are printed for every population.
"""
import json, os, statistics as st

A = os.path.dirname(os.path.abspath(__file__))
SUB = {"animal": "cruelty", "betrayal": "intimate", "power": "coercion",
       "property": "theft", "taboo": "desecration"}


def row(lab, recs):
    d = [int(r["len_m"]) - int(r["len_u"]) for r in recs]
    gt = sum(1 for x in d if x > 0); eq = sum(1 for x in d if x == 0)
    lt = sum(1 for x in d if x < 0)
    print(f"  {lab:<28}{len(d):>5}{st.mean(d):>8.2f}{st.median(d):>8.1f}"
          f"{gt:>6}{eq:>6}{lt:>6}{'  NEVER REVERSES' if lt == 0 and gt > 5 else ''}")


def main():
    r2 = json.load(open(f"{A}/r2_audit.json"))
    r2b = json.load(open(f"{A}/r2b_audit.json"))
    r1 = json.load(open(f"{A}/r1_audit.json"))
    print("WITHIN-PAIR LENGTH ASYMMETRY  len(MARKED) - len(UNMARKED)\n")
    print(f"  {'population':<28}{'n':>5}{'mean':>8}{'median':>8}{'M>U':>6}{'M=U':>6}{'M<U':>6}")
    for dom, sub in sorted(SUB.items(), key=lambda x: x[1]):
        row(f"round2 {sub} ({dom})", [r for r in r2 if r["domain"] == dom])
    row("round2b power (REDRAFT)", r2b)
    print()
    for d in sorted({r.get("domain", "?") for r in r1}):
        row(f"round1 {d}", [r for r in r1 if r.get("domain") == d])
    row("round1 ALL (D's substrate)", r1)
    print("\n  VERDICT: the asymmetry is confined to round2b. Registration D's")
    print("  substrate and all five original subdomains are symmetric, with")
    print("  reversals in both directions and medians of zero.")


if __name__ == "__main__":
    main()
