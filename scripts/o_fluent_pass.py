#!/usr/bin/env python3
"""REGISTRATION O's ONE PASS — three columns over (valid pairs x fluent edges).

Ordered [4058].2. **A MEASUREMENT, NOT AN INSTRUMENT.** It computes nothing
Registration O tests; it supplies the three numbers O's freeze waits on:

    A-YIELD, per arm      does a cell yield an A? (>= QUALIFYING_MIN words in
                          EACH role after the norm join) -- decides whether
                          H2/H3 stand at full strength or carry a stated MDE
    ZERO-FALLER, per arm  O2: N excludes these at §6.5 and O's §O3 is silent.
                          The campaign's figures are not this population's.
    COMPETENCE SHARES     per model: retained mass on CJK-only word forms, and
                          the residual share. `cjk_tier` is CAPACITY -- what the
                          tokenizer can represent. This is COMPETENCE -- what the
                          model actually does. Necessary, not sufficient.

**THE POPULATION IS READ FROM A DEPOSITED ENUMERATION, NEVER RE-DERIVED HERE.**
[4053].2 fixes it as the valid pairs whose both sides are declared stimuli;
[4058].1 deposits them. A pass that re-derives its own population answers a
question nobody pinned -- and §O1's 373 was irreproducible precisely because no
enumeration existed.

**THE SHARES POST BLIND-RANKED.** [4058].3: the qualifying threshold is ruled on
the distribution's gap BEFORE anyone sees which model falls which side. This
script prints ranks and withholds names unless `--names` is passed, so the
ordering of those two acts is enforced by the tool rather than by memory.

    python scripts/o_fluent_pass.py --pairs <enumeration.json> [--names]
"""

import argparse
import collections
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "meta", "M01_displacement", "scripts"))

CJK = re.compile(r"[一-鿿]")
#: A word form is CJK-ONLY if every character is CJK. A mixed form is a
#: tokenizer artefact of exactly the kind the competence share is measuring,
#: so it counts toward neither numerator nor the clean denominator.
CJK_ONLY = re.compile(r"^[一-鿿]+$")
OUT = os.path.join(ROOT, "data", "o_fluent_pass.json")


def fluent_edges():
    """[4053].1 / [4057]: edges FLUENT on BOTH sides. Measured, not listed.

    Not one of the 44 edges is mixed-tier -- the tier is a property of the
    LINEAGE, not of the alignment -- so this is a clean cut. Hard-coding the
    ten names would make the cut a list; deriving it keeps it a rule.
    """
    import m01_concentration as CC
    reg = json.load(open(os.path.join(ROOT, "data", "model_registry.json")))
    models = reg["models"] if isinstance(reg, dict) and "models" in reg else reg
    rows = models.values() if isinstance(models, dict) else models
    tier = {}
    for m in rows:
        if isinstance(m, dict):
            mid = m.get("id") or m.get("model_id") or m.get("hf_id")
            if mid:
                tier[mid] = m.get("cjk_tier") or ""
    _p, mods, _h, _d = CC.frozen_population()
    edges, _dropped = CC.operation_edges(mods)

    def mid(o):
        return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)

    out, skipped = [], collections.Counter()
    for fam, _pos, step in edges:
        a, b = mid(step.pre), mid(step.post)
        ta, tb = tier.get(a, "?"), tier.get(b, "?")
        if ta == "FLUENT" and tb == "FLUENT":
            out.append((fam, a, b))
        else:
            skipped[(ta, tb)] += 1
    return out, skipped


def competence(model, prompts, cm):
    """Retained mass on CJK-ONLY word forms, and the residual share.

    `cjk_tier` says the tokenizer CAN represent Chinese. This says what the
    model DOES with it: if a FLUENT-tier model puts its mass on Latin forms or
    into the unresolved tail on Chinese prompts, the tier is satisfied and the
    competence is not.
    """
    from malign_logits.movement import word_probs
    cjk_mass, tot_mass, resid, n = 0.0, 0.0, 0.0, 0
    for pr in prompts:
        wp = word_probs(model, pr)
        if wp is None:
            continue
        n += 1
        for w, p in wp.probs.items():
            tot_mass += p
            if CJK_ONLY.match(w):
                cjk_mass += p
        resid += wp.residual
    if not n:
        return None
    return {"cells": n,
            "cjk_only_share": cjk_mass / n,
            "retained_share": tot_mass / n,
            "residual_share": resid / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="the DEPOSITED enumeration ([4058].1). Never re-derived here.")
    ap.add_argument("--exclude", default="",
                    help="comma-separated model ids failing the COMPETENCE "
                         "threshold ([4061]: >= 0.30 CJK-only retained mass). "
                         "An edge is dropped if EITHER side is listed -- the "
                         "unit is the EDGE, and an edge with a competent base "
                         "and a hollowed child measures the collapse, not the "
                         "mechanism.")
    ap.add_argument("--names", action="store_true",
                    help="reveal model names beside the shares. WITHHELD by "
                         "default so the threshold is ruled on the gap first ([4058].3)")
    a = ap.parse_args()

    from malign_logits.cache import get_cache
    from malign_logits.movement import movement, word_probs, RESIDUAL_KEY, CANONICAL
    import m01_norms as N
    import m01_registration_b as B

    #: THE PIN, [4035]/[4040]: record which constants this pass ran under.
    #: An extension can move while a rule's text stands still -- pin what
    #: SELECTS, not what is written.
    assert (CANONICAL.min_prob, CANONICAL.fall_ratio) == (0.003, 0.5), \
        "CANONICAL moved; this pass's zero-faller counts are not comparable"

    blob = open(a.pairs, "rb").read()
    import hashlib
    pairs_sha = hashlib.sha256(blob).hexdigest()[:16]
    doc = json.loads(blob)
    pairs = doc["pairs"] if isinstance(doc, dict) and "pairs" in doc else doc
    print("population  %s @ %s   %d pairs" % (
        os.path.basename(a.pairs), pairs_sha, len(pairs)), flush=True)

    edges, skipped = fluent_edges()
    drop = {x.strip() for x in a.exclude.split(",") if x.strip()}
    if drop:
        kept = [(f, b, p) for f, b, p in edges if b not in drop and p not in drop]
        for f, b, p in edges:
            if b in drop or p in drop:
                print("  EXCLUDED EDGE  %-20s %s -> %s  (competence < 0.30, [4061])"
                      % (f, b, p), flush=True)
        edges = kept
    print("qualifying edges %d   clusters %d   (skipped tiers: %s)" % (
        len(edges), len({b for _, b, _ in edges}), dict(skipped)), flush=True)

    norms, _f, _r = N.load_norms(verify=True)
    tabs_en = {d: norms[("en", d, "primary")] for d in ("arousal", "valence")}
    tabs_zh = {d: norms[("zh", d, "primary")] for d in ("arousal", "valence")}

    cm = get_cache()
    res = {"en": collections.Counter(), "zh": collections.Counter()}
    per_model = {}
    for fam, pre, post in edges:
        for arm, key, tabs in (("en", "english", tabs_en), ("zh", "chinese", tabs_zh)):
            texts = [p[key] for p in pairs if p.get(key)]
            for t in texts:
                A, Bp = word_probs(pre, t), word_probs(post, t)
                if A is None or Bp is None:
                    res[arm]["absent"] += 1
                    continue
                res[arm]["cells"] += 1
                m = movement({**A.probs, RESIDUAL_KEY: A.residual},
                             {**Bp.probs, RESIDUAL_KEY: Bp.residual}, CANONICAL)
                if not m.fallers:
                    res[arm]["zero_faller"] += 1
                    continue
                keep_f = keep_r = 0
                for w in m.fallers:
                    k = N.norm_key(w, arm, fold=False)
                    if N.is_function_word(k, arm):
                        continue
                    if all(N.lookup(tabs[d], k.casefold(), arm)[0] is not None
                           for d in tabs):
                        keep_f += 1
                for w in m.risers:
                    k = N.norm_key(w, arm, fold=False)
                    if N.is_function_word(k, arm):
                        continue
                    if all(N.lookup(tabs[d], k.casefold(), arm)[0] is not None
                           for d in tabs):
                        keep_r += 1
                if keep_f >= B.QUALIFYING_MIN and keep_r >= B.QUALIFYING_MIN:
                    res[arm]["yields_A"] += 1
        zh_texts = [p["chinese"] for p in pairs if p.get("chinese")][:120]
        for mdl in (pre, post):
            if mdl not in per_model:
                c = competence(mdl, zh_texts, cm)
                if c:
                    per_model[mdl] = c
        print("  %-22s en cells %5d  zh cells %5d" % (
            fam, res["en"]["cells"], res["zh"]["cells"]), flush=True)

    print("\n=== THE THREE COLUMNS")
    for arm in ("en", "zh"):
        c = res[arm]
        analysed = c["cells"] - c["zero_faller"]
        print("  %s   cells %6d   zero-faller %5d (%.2f%%)   A-YIELD %5d "
              "(%.1f%% of analysed)" % (
                  arm, c["cells"], c["zero_faller"],
                  100.0 * c["zero_faller"] / c["cells"] if c["cells"] else 0,
                  c["yields_A"],
                  100.0 * c["yields_A"] / analysed if analysed else 0), flush=True)

    print("\n=== COMPETENCE SHARES, BLIND-RANKED ([4058].3)")
    ranked = sorted(per_model.items(), key=lambda kv: -kv[1]["cjk_only_share"])
    print("  rank  cjk_only_share  residual_share  retained" +
          ("  model" if a.names else "   (names WITHHELD)"))
    for i, (mdl, c) in enumerate(ranked, 1):
        print("  %4d  %14.4f  %14.4f  %8.4f%s" % (
            i, c["cjk_only_share"], c["residual_share"], c["retained_share"],
            ("  " + mdl) if a.names else ""), flush=True)
    gaps = [(ranked[i][1]["cjk_only_share"] - ranked[i + 1][1]["cjk_only_share"], i + 1)
            for i in range(len(ranked) - 1)]
    if gaps:
        g, at = max(gaps)
        print("\n  largest gap %.4f, between rank %d and %d" % (g, at, at + 1))
        print("  **THE THRESHOLD IS RULED ON THIS GAP BEFORE NAMES ARE REVEALED.**")

    json.dump({"_what": "Registration O's freeze-precondition pass. [4058].2",
               "_population": {"file": os.path.basename(a.pairs),
                               "sha16": pairs_sha, "pairs": len(pairs)},
               "_constants": {"min_prob": CANONICAL.min_prob,
                              "fall_ratio": CANONICAL.fall_ratio},
               "fluent_edges": [{"family": f, "base": b, "aligned": p}
                                for f, b, p in edges],
               "arms": {k: dict(v) for k, v in res.items()},
               "competence": per_model}, open(OUT, "w"), indent=1, sort_keys=True)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
