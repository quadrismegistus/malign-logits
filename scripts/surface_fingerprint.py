"""Per-(model, store) surface conformance for `twp_words`.

    uv run python scripts/surface_fingerprint.py
    -> data/surface_fingerprint.json   (small, committable)

WHY THIS EXISTS. On 2026-08-15 a countersigned query returned n=59 where every
reference pair returned n~2,198. The cause was that ONE model's stored cells
carry the SentencePiece boundary marker in the `word` field -- `▁and` against
`and` -- so every cross-model word join against it silently collapsed. It was
read as a finding about vocabulary overlap by three seats before @malign found
it ([6295], [6301]).

**NO EXISTING ARTIFACT COULD HAVE CAUGHT IT.**

    rule_version / dict_sha    identical across the base and the arm.
                               They describe the rule DECLARED, never the
                               normalisation APPLIED.
    tokenizer_properties.json  the two tokenizers differ by ONE token in
                               32,000 (vocab_len 32000 vs 32001).
    edge_token_overlap.json    n_shared ~32000, cover ~1.0, bos_matches true.
                               Every field would have read COMPARABLE.

And the vocab gap is not the cause: OpenHermes, openchat and Nous-Hermes all
carry a LARGER gap (vocab_len 32002) against the same parent with 0.0% marked
surfaces over ~290k cells each. Token-id comparability and stored-surface
conformance are INDEPENDENT axes, and only the first was measured anywhere.

**THE GRAIN IS (MODEL, STORE), WHICH IS NEITHER OF THE AXES THE PLAN HAS.**
Viability, scheduled and has-data are per-checkpoint; comparability is per-edge
([6305]). This is a fact about how a given model's cells were WRITTEN into a
given table -- the same checkpoint re-expanded would fingerprint differently
with nothing about the checkpoint having changed. The edge inherits the defect
from one endpoint's storage rather than possessing it.

**TWO MEASURES, BOTH ONE QUERY.**

    u2581_rate      fraction of `word` values beginning U+2581 (SentencePiece
                    word-boundary marker). Raw tokenizer surfaces, not words.
    bytefall_rate   fraction matching `<0xHH>` exactly -- raw byte-fallback
                    tokens sitting in a field that should hold words. This is
                    the SECOND defect on the same model and it is invisible to
                    the first: SentencePiece does not mark word-initial CJK, so
                    the Chinese cells carry no marker AND are separately broken.

**REPORTED BESIDE A ROSTER MEDIAN ON PURPOSE.** A rate alone does not say
whether it is anomalous; 0.4% byte-fallback is unremarkable and 3.9% is not, and
neither is legible without the other 150 models. See the campaign's standing
rule that a metric wants a null.

This does NOT rule. It emits the numbers and flags outliers against the roster
so a reader can decide; the declaration of what counts as non-conformant is not
this producer's to make.
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "surface_fingerprint.json")
STORE = "malign_logits.twp_words"
#: U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's word-boundary marker.
U2581_HEX = "E29681"
BYTEFALL_RE = r"^<0x[0-9A-Fa-f]{2}>$"
#: LITERAL class. The escape form [\\x{4e00}-\\x{9fff}] does NOT work in
#: ClickHouse and matches everything, including "hello world".
CJK_RE = "[\u4e00-\u9fff]"
#: Flag at 10x the roster median, floored so a roster of near-zeros does not
#: make every trace value an outlier. Both are stated, neither is a ruling.
OUTLIER_FACTOR = 10.0
OUTLIER_FLOOR = 0.01


def main():
    from malign_logits import ch

    #: PARTITIONED BY SCRIPT, AND THE PARTITION IS THE POINT. Pooled rates hid
    #: the second defect from my own first run: dolphin's byte-fallback is
    #: 0.75% over all its cells and 3.9% within its CJK ones, because
    #: SentencePiece does not mark word-initial CJK and the Chinese cells are
    #: broken a DIFFERENT way. A pooled rate under a threshold does not clear
    #: a model -- which is the same population error three seats made on this
    #: model tonight (99% true of English, 82% true of the pool, neither
    #: stated). Reported both ways so neither can hide the other.
    q = """select model,
                  count() AS n,
                  countIf(startsWith(word, unhex('%s'))) AS marked,
                  countIf(match(word, '%s')) AS bytefall,
                  countIf(match(prompt, '%s')) AS n_cjk,
                  countIf(match(prompt, '%s') AND match(word, '%s')) AS bytefall_cjk
           from %s group by model order by model""" % (
        U2581_HEX, BYTEFALL_RE, CJK_RE, CJK_RE, BYTEFALL_RE, STORE)
    rows = []
    for r in ch.query(q):
        n, ncjk = r["n"], r["n_cjk"]
        rows.append({"model": r["model"], "store": STORE, "n_cells": n,
                     "u2581_rate": r["marked"] / n if n else 0.0,
                     "bytefall_rate": r["bytefall"] / n if n else 0.0,
                     "n_cjk_cells": ncjk,
                     "bytefall_rate_cjk": (r["bytefall_cjk"] / ncjk
                                           if ncjk else None)})
    if not rows:
        raise SystemExit("no rows from %s -- store unreachable or empty" % STORE)

    for k in ("u2581_rate", "bytefall_rate", "bytefall_rate_cjk"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            continue
        med = st.median(vals)
        bar = max(OUTLIER_FACTOR * med, OUTLIER_FLOOR)
        for r in rows:
            r.setdefault("flags", [])
            if r.get(k) is not None and r[k] > bar:
                r["flags"].append(k)

    #: THE KNOWN CASE IS ASSERTED. If a re-expansion lands and this producer
    #: still reports the marker, the re-expansion did not take; if the model
    #: leaves the store, this fires and says so rather than passing silently.
    #: The check exists because a fingerprint that cannot fail is a fingerprint
    #: nobody has watched work.
    known = "cognitivecomputations/dolphin-2.6-mistral-7b-dpo"
    hit = [r for r in rows if r["model"] == known]
    if hit:
        assert hit[0]["u2581_rate"] > 0.5, (
            "%s fingerprints clean at u2581_rate %.4f. Either the cells were "
            "re-expanded -- in which case update this assert and say so -- or "
            "this producer is not measuring what it claims."
            % (known, hit[0]["u2581_rate"]))
    else:
        print("  NOTE: %s absent from %s; the known-case assert did not run."
              % (known, STORE))

    flagged = [r for r in rows if r["flags"]]
    out = {"_about":
           "Per-(model, store) surface conformance for %s. u2581_rate is the "
           "fraction of `word` values carrying the SentencePiece boundary "
           "marker; bytefall_rate the fraction that are raw <0xHH> tokens. "
           "Neither is derivable from rule_version, dict_sha, "
           "tokenizer_properties.json or edge_token_overlap.json -- see the "
           "module docstring. Flags are 10x the roster median (floor 0.01) "
           "and are descriptive, not a ruling." % STORE,
           "_producer": "scripts/surface_fingerprint.py",
           "store": STORE, "n_models": len(rows),
           "median_u2581_rate": st.median([r["u2581_rate"] for r in rows]),
           "median_bytefall_rate": st.median([r["bytefall_rate"] for r in rows]),
           "n_flagged": len(flagged),
           "flagged": [r["model"] for r in flagged],
           "models": rows}
    json.dump(out, open(OUT, "w"), indent=1)

    print("%s: %d models, %s cells"
          % (STORE, len(rows), format(sum(r["n_cells"] for r in rows), ",")))
    print("  roster median  u2581 %.5f | bytefall %.5f"
          % (out["median_u2581_rate"], out["median_bytefall_rate"]))
    print("  flagged: %d" % len(flagged))
    for r in sorted(flagged, key=lambda r: -r["u2581_rate"]):
        print("    %-52s u2581 %6.2f%%  bytefall %5.2f%%  n=%s  %s"
              % (r["model"][:52], 100 * r["u2581_rate"],
                 100 * r["bytefall_rate"], format(r["n_cells"], ","),
                 ",".join(r["flags"])))
    print("\n-> %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
