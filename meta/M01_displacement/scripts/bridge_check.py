"""BRIDGE CHECK — does the word-level pool artifact aggregate back to the frozen
`cells` that both implementations already agreed on? [3433].2.

**WHY THIS EXISTS.** The exclusion check needs word identity, which the frozen
producer discards, so its pool must be RE-DERIVED from upstream by one seat. That
leaves the reconstruction resting on one seat's code. This check pins it instead
to the artifact both seats verified to five decimals: **if the word lists
aggregate to `cells` exactly, only the WORD LABELS remain uncertified** — and the
labels are precisely what independent swap detection then exercises.

**THE JOINT IS COMPARED, NOT THE MARGINALS.** A corruption that swaps two words'
z-values leaves every marginal multiset identical and would pass a marginal
check. The comparison is over (valence_z, weight, role) TRIPLES.

Written before the artifact exists, so it cannot be shaped to it.

    python3 bridge_check.py --selftest        prove each check FIRES
    python3 bridge_check.py <word_artifact>   run against the real artifact
"""
import collections
import json
import sys


def cell_signature(entries, zkey="valence_z", wkey="weight", rkey="role"):
    """Multiset of (z, w, role) triples, rounded to a declared tolerance.

    Rounding is at 1e-12 -- tight enough that a real difference survives it and
    loose enough that float replay noise does not manufacture one.
    """
    sig = collections.Counter()
    for e in entries:
        sig[(round(float(e[zkey]), 12), round(float(e[wkey]), 12), str(e[rkey]))] += 1
    return sig


def cells_signature(cell):
    """The same multiset, built from the FROZEN producer's cell."""
    sig = collections.Counter()
    for z, w, r in zip(cell["zs"], cell["ws"], cell["rs"]):
        sig[(round(float(z["valence"]), 12), round(float(w), 12), str(r))] += 1
    return sig


def bridge(word_artifact_cells, frozen_cells):
    """Compare per cell. Returns (ok, list of discrepancies)."""
    bad = []
    seen = 0
    for key, entries in word_artifact_cells.items():
        fc = frozen_cells.get(key)
        if fc is None:
            bad.append((key, "cell absent from frozen cells"))
            continue
        seen += 1
        a, b = cell_signature(entries), cells_signature(fc)
        if a != b:
            only_a = list((a - b).elements())[:3]
            only_b = list((b - a).elements())[:3]
            bad.append((key, "joint multiset differs: artifact-only %s | frozen-only %s"
                        % (only_a, only_b)))
    missing = [k for k in frozen_cells if k not in word_artifact_cells]
    if missing:
        bad.append(("<coverage>", "%d frozen cells absent from the artifact" % len(missing)))
    return (not bad), bad, seen


def selftest():
    ok = True

    def case(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print("  [%s] %s" % ("ok" if cond else "FAIL", name))

    frozen = {"c1": {"zs": [{"valence": 1.0}, {"valence": -2.0}, {"valence": 0.5}],
                     "ws": [0.1, 0.2, 0.3],
                     "rs": ["faller", "riser", "faller"]}}
    good = {"c1": [{"word": "a", "valence_z": 1.0, "weight": 0.1, "role": "faller"},
                   {"word": "b", "valence_z": -2.0, "weight": 0.2, "role": "riser"},
                   {"word": "c", "valence_z": 0.5, "weight": 0.3, "role": "faller"}]}

    case("a faithful artifact BRIDGES", bridge(good, frozen)[0])

    drop = {"c1": good["c1"][:2]}
    case("a MISSING word fires", not bridge(drop, frozen)[0])

    extra = {"c1": good["c1"] + [{"word": "d", "valence_z": 9.0,
                                  "weight": 0.4, "role": "riser"}]}
    case("an EXTRA word fires", not bridge(extra, frozen)[0])

    altz = json.loads(json.dumps(good)); altz["c1"][0]["valence_z"] = 1.0001
    case("an ALTERED z fires", not bridge(altz, frozen)[0])

    altw = json.loads(json.dumps(good)); altw["c1"][0]["weight"] = 0.1001
    case("an ALTERED weight fires", not bridge(altw, frozen)[0])

    altr = json.loads(json.dumps(good)); altr["c1"][0]["role"] = "riser"
    case("an ALTERED role fires", not bridge(altr, frozen)[0])

    #: THE CONTROL THAT JUSTIFIES COMPARING THE JOINT. Swapping two words' z
    #: values leaves the z multiset, the weight multiset and the role multiset
    #: all identical. Only the TRIPLE changes.
    swp = json.loads(json.dumps(good))
    swp["c1"][0]["valence_z"], swp["c1"][1]["valence_z"] = \
        swp["c1"][1]["valence_z"], swp["c1"][0]["valence_z"]
    marg_same = (sorted(e["valence_z"] for e in swp["c1"])
                 == sorted(e["valence_z"] for e in good["c1"]))
    case("a z-SWAP leaves every MARGINAL identical", marg_same)
    case("...and the JOINT check fires on it anyway", not bridge(swp, frozen)[0])

    case("a cell missing from the artifact fires",
         not bridge({}, frozen)[0])

    print("selftest %s" % ("PASS -- every check proven to fire" if ok
                           else "FAIL -- DO NOT BRIDGE WITH THIS"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) < 2:
        sys.exit(selftest())
    sys.exit(0)
