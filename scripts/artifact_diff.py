#!/usr/bin/env python3
"""Per-field, MAGNITUDE-SIZED diff between two run artifacts. Ruled [3838].

A HASH SAYS THE FILE CHANGED. IT DOES NOT SAY WHETHER THE QUANTITY DID.

On L's re-run, six fields moved across ~3,000 rows each. Four were summation-order
drift at 1e-15 -- the faller fix changed set membership, so accumulations land on
different last bits, same quantity, different bytes. Two were real: `arrived` at
3.43e5 relative, `concentration` moving None -> 1.0, a change in DEFINEDNESS.

**6,000 rows of 1e-15 drift looks exactly like 6,000 rows of real movement until
someone measures the magnitude**, and a "fields changed" list without magnitudes
would have reported them as one thing. That is the whole reason this file exists.

CLASSES REPORTED
    IDENTICAL   field absent from the diff entirely
    NOISE       max relative delta <= NOISE_REL (default 1e-9)
    REAL        anything larger, INCLUDING every change of definedness
                (None <-> value), which has no magnitude and is never noise

The definedness rule is not a convenience. `concentration = top/arrived` is None
when nothing arrived; a cell moving None -> 1.0 has changed what question it
answers, and a threshold on magnitude cannot see that.

USAGE
    python scripts/artifact_diff.py OLD.json NEW.json [--rows-key rows]
    python scripts/artifact_diff.py OLD.json NEW.json --noise-rel 1e-12
"""
import argparse
import json
import sys
from collections import Counter

NOISE_REL = 1e-9


def _rows(doc, key):
    if isinstance(doc, dict) and key in doc:
        return doc[key]
    if isinstance(doc, list):
        return doc
    raise SystemExit("no %r in this artifact; pass --rows-key" % key)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def diff(old, new, noise_rel=NOISE_REL):
    """-> (per-field report, row-count of any change). Pure; no I/O."""
    if len(old) != len(new):
        raise SystemExit("ROW COUNT DIFFERS: %d -> %d. Not a re-run of the same "
                         "population; the diff below would be meaningless." %
                         (len(old), len(new)))
    fields, rows_changed = {}, 0
    for a, b in zip(old, new):
        touched = False
        for k in set(a) | set(b):
            x, y = a.get(k), b.get(k)
            if x == y:
                continue
            touched = True
            f = fields.setdefault(k, {"n": 0, "max_abs": 0.0, "max_rel": 0.0,
                                      "definedness": 0, "example": None})
            f["n"] += 1
            #: DEFINEDNESS FIRST -- None <-> value has no magnitude and must never
            #: be averaged into one. A field with even ONE such change is REAL.
            if (x is None) != (y is None):
                f["definedness"] += 1
            elif _num(x) and _num(y):
                d = abs(x - y)
                f["max_abs"] = max(f["max_abs"], d)
                if x:
                    f["max_rel"] = max(f["max_rel"], d / abs(x))
            else:
                f["definedness"] += 1          # non-numeric change: never noise
            if f["example"] is None:
                f["example"] = (x, y)
        rows_changed += touched
    for k, f in fields.items():
        f["klass"] = ("REAL" if f["definedness"] or f["max_rel"] > noise_rel
                      else "NOISE")
    return fields, rows_changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--rows-key", default="rows")
    ap.add_argument("--noise-rel", type=float, default=NOISE_REL)
    a = ap.parse_args()

    o = _rows(json.load(open(a.old)), a.rows_key)
    n = _rows(json.load(open(a.new)), a.rows_key)
    fields, rows_changed = diff(o, n, a.noise_rel)

    print("rows %d, %d with any change (%.2f%%)   noise threshold rel <= %g"
          % (len(o), rows_changed, 100.0 * rows_changed / len(o), a.noise_rel))
    if not fields:
        print("\n**BYTE-EQUIVALENT ON EVERY FIELD.** Nothing moved.")
        return 0
    print("\n%-24s %7s %13s %13s %6s  %s"
          % ("field", "rows", "max |delta|", "max relative", "class", "example"))
    for k, f in sorted(fields.items(), key=lambda kv: (kv[1]["klass"] != "REAL",
                                                       -kv[1]["n"])):
        ex = "%r -> %r" % f["example"]
        print("%-24s %7d %13.3e %13.3e %6s  %s"
              % (k, f["n"], f["max_abs"], f["max_rel"], f["klass"], ex[:44]))
        if f["definedness"]:
            print("%-24s %7s   %d change DEFINEDNESS or type -- never noise"
                  % ("", "", f["definedness"]))
    real = [k for k, f in fields.items() if f["klass"] == "REAL"]
    print("\nREAL: %s" % (", ".join(sorted(real)) if real else "none"))
    print("NOISE (summation order, same quantity): %s"
          % ", ".join(sorted(k for k, f in fields.items() if f["klass"] == "NOISE")))
    #: The tool classifies; it does not adjudicate. A REAL field may be one this
    #: campaign already expects to move (`arrived`, `concentration`), and a NOISE
    #: field is not thereby cleared for a claim that depends on its last bits.
    print("\nCLASSIFICATION IS NOT A VERDICT. Every REAL field needs a named "
          "reason it moved, and a re-run's verdict is the finding's, not the file's.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
