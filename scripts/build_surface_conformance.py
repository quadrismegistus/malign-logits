#!/usr/bin/env python3
"""Surface conformance per (MODEL, STORE): does a model's stored word field
actually conform to the rule its cells CLAIM?

    scripts/build_surface_conformance.py            report
    scripts/build_surface_conformance.py --write    emit data/surface_conformance.json

WHY THIS EXISTS. On 2026-08-15 the dolphin discriminator returned a verdict on
n=59 of a declared 2,200-prompt population. The cause was in the stored cells:

    base  rule_version=3  dict_sha=b16011275c42955c   U+2581   0% of its rows
    arm   rule_version=3  dict_sha=b16011275c42955c   U+2581  99% of its rows

**IDENTICAL STAMPS, INCOMPATIBLE NORMALISATIONS.** `rule_version` and `dict_sha`
describe the rule that was DECLARED, not the normalisation that was APPLIED
(@lacan, docket [6306]). A version asserted rather than produced diagnoses
nothing, and no consumer could have caught this from metadata -- none did. It
surfaced only because a pre-registration put a reference `n` in the same table
as the result.

AND THE EXISTING ARTIFACTS COULD NOT HAVE CAUGHT IT EITHER. `edge_token_overlap`
answers *can A embed B's ids*; for this pair it would have read `n_shared`
~32,000, `cover` ~1.0, `bos_matches` true -- **COMPARABLE on every field** --
while the word-level join lost 83% of prompts. Token-id overlap and stored
surface normalisation are INDEPENDENT properties and only the first was measured.

SO THIS IS NOT A PROPERTY OF A CHECKPOINT, AND NOT OF AN EDGE. It is a property
of (model x store): the same checkpoint re-expanded writes clean surfaces, and
an edge inherits the defect from one endpoint's storage rather than possessing
it. That is why it needs its own record.

WHAT IT MEASURES, both cheap and both would have fired:

    u2581_rate    fraction of word fields carrying the SentencePiece boundary
                  marker. Rule 3 strips it; a rule-3 cell must read ~0.
    byte_rate     fraction matching <0xNN> -- raw byte-fallback tokens, which
                  are not words at all. Found in dolphin's CJK cells, where the
                  marker is ABSENT because SentencePiece does not mark
                  word-initial CJK -- so the language whose specialist would
                  have been asked to confirm the model is the one where it looks
                  clean.

**PROBE BOTH SCRIPTS OR THE ANSWER IS A POPULATION, NOT A FACT.** Measured on
dolphin: 98.8% marked on English rows, 0.4% on CJK, 82% pooled. Two seats
reported 99% and 82% as a disagreement for an hour; neither was wrong and
neither named its population.
"""
import argparse, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BYTE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
CJK = re.compile(r"[一-鿿]")
#: A rule-3 cell must carry no boundary markers and no byte fallbacks. Set
#: generously: the observed defect is 98.8%, the observed clean state is 0%.
#: A threshold at 1% cannot be tripped by a stray surface and cannot miss this.
U2581_MAX = 0.01
BYTE_MAX = 0.01


def probe_prompts(n_each=6):
    """English AND CJK, because the defect hides in one of them."""
    P = json.load(open(os.path.join(ROOT, "data/prompt_categorisation.json")))["prompts"]
    en = [p["prompt"] for p in P if not CJK.search(p["prompt"])][:n_each]
    zh = [p["prompt"] for p in P if CJK.search(p["prompt"])][:n_each]
    return en, zh


def measure(cm, twp, mid, prompts):
    rows = marked = byte = 0
    seen = 0
    for q in prompts:
        c = cm.get_true_word_probs(mid, q, theta=twp.THETA)
        if not c or not c.get("rows"):
            continue
        seen += 1
        for r in c["rows"]:
            w = r["word"]
            rows += 1
            marked += w.startswith("▁")
            byte += bool(BYTE.match(w))
    if not rows:
        return None
    return {"prompts": seen, "rows": rows,
            "u2581_rate": round(marked / rows, 4),
            "byte_rate": round(byte / rows, 4)}


CH_STORE = "malign_logits.twp_words"
#: The same CJK class as CJK above. NOTE the escape form [\x{4e00}-\x{9fff}]
#: does NOT work in ClickHouse -- it matches everything, "hello world" included.
CH_CJK = "[一-鿿]"
CH_BYTE = r"^<0x[0-9A-Fa-f]{2}>$"
#: U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's word-boundary marker.
U2581_HEX = "E29681"


def census_rows():
    """Every cell in the ClickHouse store, not a probe sample.

    THE SECOND STORE IS THE POINT, not extra coverage (@malign, [6316]).
    `true_word_probs` (the CacheManager stash) and `malign_logits.twp_words`
    are two stores holding one fact, reconciled to synonymy on 10 Aug at RH's
    instruction -- 400,644 cells, zero difference either way, from a starting
    gap of ~127,000. **That was five days and several fleets ago.**
    Conformance established on one store is not a property of the other,
    however tightly they were once reconciled -- the argument @malign used
    against `landed_v3` this afternoon, turned on this producer.

    Both rows are emitted per model and never collapsed. If they never
    disagree the second row costs four lines and proves it; if they do, we
    learn on the day it starts rather than when a registration returns n=59.

    Returns (rows, None) or ([], reason) -- NEVER a silent empty. This half
    cannot run in every venv (`clickhouse_connect` is absent from some), and a
    producer that quietly emits one store having promised two is the defect
    this file exists to catch.
    """
    try:
        from malign_logits import ch
    except Exception as e:
        return [], "import failed: %s" % str(e)[:80]
    q = """select model,
                  countIf(NOT match(prompt, '%s')) AS en_rows,
                  countIf(NOT match(prompt, '%s') AND startsWith(word, unhex('%s'))) AS en_mark,
                  countIf(NOT match(prompt, '%s') AND match(word, '%s')) AS en_byte,
                  countIf(match(prompt, '%s')) AS zh_rows,
                  countIf(match(prompt, '%s') AND startsWith(word, unhex('%s'))) AS zh_mark,
                  countIf(match(prompt, '%s') AND match(word, '%s')) AS zh_byte
           from %s group by model order by model""" % (
        CH_CJK, CH_CJK, U2581_HEX, CH_CJK, CH_BYTE,
        CH_CJK, CH_CJK, U2581_HEX, CH_CJK, CH_BYTE, CH_STORE)
    try:
        res = list(ch.query(q))
    except Exception as e:
        return [], "query failed: %s" % str(e)[:80]
    rows = []
    for r in res:
        def part(pre):
            n = r["%s_rows" % pre]
            if not n:
                return None
            return {"prompts": None, "rows": n,
                    "u2581_rate": round(r["%s_mark" % pre] / n, 4),
                    "byte_rate": round(r["%s_byte" % pre] / n, 4)}
        e, z = part("en"), part("zh")
        rates = [x for x in (e, z) if x]
        if not rates:
            continue
        wu = max(x["u2581_rate"] for x in rates)
        wb = max(x["byte_rate"] for x in rates)
        rows.append({"model": r["model"], "store": CH_STORE, "en": e, "zh": z,
                     "worst_u2581": wu, "worst_byte": wb,
                     "conforms": wu <= U2581_MAX and wb <= BYTE_MAX})
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--n", type=int, default=6, help="probe prompts per script")
    ap.add_argument("--no-census", action="store_true",
                    help="stash probe only; the ClickHouse half needs "
                         "clickhouse_connect, absent from some venvs")
    a = ap.parse_args()

    from malign_logits import twp
    from malign_logits.cache import get_cache
    cm = get_cache()
    en, zh = probe_prompts(a.n)
    models = [m["model_id"] for m in json.load(
        open(os.path.join(ROOT, "data/model_registry.json")))["models"]]

    out, bad, nodata = [], [], 0
    for mid in models:
        e = measure(cm, twp, mid, en)
        z = measure(cm, twp, mid, zh)
        if e is None and z is None:
            nodata += 1
            continue
        #: THE VERDICT IS THE WORST SCRIPT, NOT THE POOL. A pooled rate on a
        #: model with many CJK cells dilutes an English defect below any
        #: threshold -- which is exactly how 82% looked milder than 99%.
        rates = [x for x in (e, z) if x]
        worst_u = max(x["u2581_rate"] for x in rates)
        worst_b = max(x["byte_rate"] for x in rates)
        ok = worst_u <= U2581_MAX and worst_b <= BYTE_MAX
        row = {"model": mid, "store": "true_word_probs", "en": e, "zh": z,
               "worst_u2581": worst_u, "worst_byte": worst_b, "conforms": ok}
        out.append(row)
        if not ok:
            bad.append(row)

    census, census_skip = ([], "--no-census") if a.no_census else census_rows()

    #: CROSS-STORE AGREEMENT IS THE DRIFT SIGNAL, and it is why both rows are
    #: kept. Synonymy was verified 10 Aug; a verdict that differs between the
    #: stash and ClickHouse means one of them has moved since.
    by_store = {}
    for r in out + census:
        by_store.setdefault(r["model"], {})[r["store"]] = r
    both = {m: v for m, v in by_store.items() if len(v) == 2}
    disagree = [m for m, v in both.items()
                if len({x["conforms"] for x in v.values()}) > 1]

    print("  surface conformance, store=true_word_probs, %d probes per script"
          % a.n)
    print("  %d models measured, %d with no cells on the probes"
          % (len(out), nodata))
    if census_skip:
        print("  CENSUS HALF DID NOT RUN (%s) -- this artifact covers ONE store"
              % census_skip)
    else:
        print("  census, store=%s: %d models, %s cells"
              % (CH_STORE, len(census),
                 format(sum((x["en"] or {}).get("rows", 0)
                            + (x["zh"] or {}).get("rows", 0)
                            for x in census), ",")))
        print("  models in BOTH stores: %d | verdicts DISAGREEING: %d %s"
              % (len(both), len(disagree),
                 disagree[:3] if disagree else ""))
    print()
    bad = [r for r in out + census if not r["conforms"]]
    if bad:
        print("  NON-CONFORMING:")
        for r in bad:
            print("     %-52s [%s] u2581 %.3f  byte %.3f" %
                  (r["model"][:52], r["store"], r["worst_u2581"],
                   r["worst_byte"]))
            for k in ("en", "zh"):
                if r[k]:
                    print("        %s  u2581 %.3f  byte %.3f  (%d rows)"
                          % (k, r[k]["u2581_rate"], r[k]["byte_rate"], r[k]["rows"]))
    else:
        print("  all conform")

    if a.write:
        p = os.path.join(ROOT, "data/surface_conformance.json")
        json.dump({
            "_about": "SURFACE CONFORMANCE per (model, store): does a model's "
                      "stored `word` field conform to the rule its cells claim? "
                      "rule_version and dict_sha describe the rule DECLARED, not "
                      "the normalisation APPLIED -- on 2026-08-15 a base and an "
                      "arm carried identical stamps with 0% and 99% U+2581. "
                      "Not a property of a checkpoint and not of an edge: the "
                      "same checkpoint re-expanded writes clean surfaces.",
            "_producer": "scripts/build_surface_conformance.py --write",
            "_thresholds": {"u2581_max": U2581_MAX, "byte_max": BYTE_MAX},
            "_verdict_rule": "the WORST script, never the pool -- a pooled rate "
                             "on a CJK-heavy model dilutes an English defect "
                             "below any threshold",
            "_n_probes_per_script": a.n,
            "_stores": ["true_word_probs"] if census_skip
                       else ["true_word_probs", CH_STORE],
            "_census_skipped": census_skip,
            "_cross_store": {"in_both": len(both),
                             "verdicts_disagree": len(disagree),
                             "disagreeing": sorted(disagree)},
            "models": out + census}, open(p, "w"), indent=1)
        print("\n  wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
