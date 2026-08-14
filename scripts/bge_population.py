#!/usr/bin/env python3
"""Measure the bge population from the corpus and write it as a declared artifact.

    python scripts/bge_population.py            # writes data/bge_population.json

WHY THIS EXISTS. `bge_fleet_sweep.py` needs a denominator: how many passages a
run under `--mixed-policy refuse` will ever embed. That number was a constant in
the sweep with a comment explaining where it came from -- which is exactly the
class lacan names at [5978]: **a comment making a claim about another artifact
is an untested assertion.** The corpus is a different file; nothing in the sweep
could ever notice if the two disagreed, and the sweep would misreport progress
for a whole run while looking healthy.

So the number is measured here, stamped with the corpus it was measured on, and
the sweep reads it instead of asserting it. The claim moves from a comment
nobody can test into an artifact with a producer.

THE DENOMINATOR IS POLICY-DEPENDENT, which is the reason it cannot be a single
constant at all. Under `refuse` the mixed stratum is excluded; under `dominant`,
`zh` or `en` it is included. All four are written, so a sweep run against a
different policy reads the right one rather than silently using the wrong one.
"""
import gzip, hashlib, json, os, sys, collections

SRC = "data/raw/blt_passages.jsonl.gz"
OUT = "data/bge_population.json"


def main():
    if not os.path.exists(SRC):
        print("  missing %s" % SRC)
        return 1
    n = 0
    by = collections.Counter()
    h = hashlib.sha256()
    with gzip.open(SRC, "rt") as fh:
        for line in fh:
            n += 1
            by[json.loads(line).get("script")] += 1
    #: Hash the FILE, not the counts -- the counts are what we are declaring, so
    #: hashing them would make the receipt agree with itself by construction.
    with open(SRC, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)

    en, zh, mixed = by.get("en", 0), by.get("zh", 0), by.get("mixed", 0)
    assert en + zh + mixed == n, (
        "script values beyond en/zh/mixed present: %s -- the policy table below "
        "does not cover them and every denominator would be wrong" % dict(by))

    doc = {
        "_about": {
            "what": "Denominator for bge_fleet_sweep: passages a run will embed, "
                    "by --mixed-policy. NOT the BLT total, which is all passages.",
            "why": "This was a constant in the sweep justified by a comment. A "
                   "comment making a claim about another artifact is untested "
                   "(lacan [5978]); nothing could notice if corpus and constant "
                   "disagreed, and the sweep would misreport all run.",
            "producer": "scripts/bge_population.py",
            "source": SRC,
            "source_sha256": h.hexdigest(),
            "source_passages": n,
        },
        "by_script": {"en": en, "zh": zh, "mixed": mixed},
        #: One entry per policy, because the denominator is policy-dependent and
        #: a single number would be right for exactly one of the four.
        "total_by_policy": {
            "refuse": en + zh,
            "dominant": n,
            "zh": n,
            "en": n,
        },
    }
    json.dump(doc, open(OUT, "w"), indent=1)
    print("  %s passages: en %s | zh %s | mixed %s" % (f"{n:,}", f"{en:,}", f"{zh:,}", f"{mixed:,}"))
    print("  refuse -> %s   other policies -> %s" % (f"{en + zh:,}", f"{n:,}"))
    print("  wrote %s (source sha256 %s...)" % (OUT, h.hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
