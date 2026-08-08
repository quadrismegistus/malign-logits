#!/usr/bin/env python
"""f11_ambient_manifest.py — the ambient null for the M02 redo.

    scripts/f11_ambient_manifest.py --show     print it, write nothing
    scripts/f11_ambient_manifest.py --write    emit data/f11_ambient_manifest.json

WHAT THE AMBIENT IS FOR. E-ASSIST-AMBIENT ([5052]/[5054]/[5057]) established
that aligned checkpoints emit assistant control tokens into raw continuation
unbidden -- 17/18 movers, p 7.2e-05, magnitude carried by Falcon3. That rate is
the FLOOR an exit-at-contradiction claim must beat. **A regex ambient nulls a
regex claim and a coded ambient nulls a coded one** ([5059].1), so the same
cells are read by both instruments and only a subset is coded.

**THE POOL IS THE F01 PROMPTS, NOT THE F01 RATES.** The rates in
`findings/M02_eassist_ambient.md` are measured on a different corpus under a
different producer and are INADMISSIBLE as this study's null ([5059].4). The
prompts, regenerated here under this producer with this decoder, are the null
(registrar [5101], amending the clause to name the distinction).

THREE PROPERTIES, and the third was found at the draw rather than declared at
the spec:

    CONTENT-EXCLUSION   not a contradiction cell and not one of its poles. An
                        ambient estimated on poles already contains whatever
                        the poles do, and the poles are inside the design.
    LENGTH              8-11 words, matching the BOTH cells.
    GENRE               third-person narrative opening. **DECLARED LATE.** I
                        named the first two at [5058] and discovered the third
                        only when the draw came up four short: `neutral_2` is
                        "To make a good risotto you should first" -- 8 words,
                        out of design, and a cooking instruction. A checkpoint's
                        leakage rate on an instruction is not its rate on a
                        narrative, and leakage-into-assistant-frame is exactly
                        what this null measures. **The pool should have been
                        checked when the properties were declared.**

THE RULED POOL (registrar [5101], lacan holding construct veto): all 28
qualifying F01 prompts -- 24 transgressive narrative openings STRATIFIED BY
CATEGORY, plus the 4 neutrals as a named floor stratum. The transgressive
ambient is the CONSERVATIVE null: an exit rate exceeding a floor measured on
explicit narrative openings is a stronger claim than one exceeding a recipe.
And the stratification is a measurement rather than a nuisance -- E-ASSIST is
already known to be content-sensitive, so "the ambient rate" was always several
rates wearing one name.
"""
import argparse
import collections
import json
import os
import math
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "data", "f11_ambient_manifest.json")
SEED = 20260808
N_GEN = 20          #: generated per prompt per checkpoint
N_CODED = 100       #: coded passages per checkpoint, drawn from the generated
WMIN, WMAX = 8, 11
NON_NARRATIVE = ("institutional",)   #: excluded on GENRE, named not silent


def pool():
    """(entries, excluded) — the qualifying set, computed not listed."""
    from malign_logits.experiments import DEFAULT_PROMPTS
    from f11_canonical_texts import load
    kept, _ = load(("ACTIVE",))
    in_design = {t for v in kept.values() for t in v.values()}

    entries, excluded = [], collections.Counter()
    for k, p in sorted(DEFAULT_PROMPTS.items()):
        cat = k.rsplit("_", 1)[0]
        if p in in_design:
            excluded["in_design"] += 1
            continue
        if not (WMIN <= len(p.split()) <= WMAX):
            excluded["length"] += 1
            continue
        if cat.startswith(NON_NARRATIVE):
            excluded["genre_non_narrative"] += 1
            continue
        entries.append({"prompt_id": k, "prompt": p, "category": cat,
                        "words": len(p.split()),
                        "stratum": "floor" if cat == "neutral" else "transgressive"})
    return entries, excluded


def coded_draw(entries, checkpoints, n_coded=N_CODED, seed=SEED):
    """Which (prompt, idx) go to the coder, per checkpoint. DETERMINISTIC.

    **NAMED IN THE MANIFEST, NOT CHOSEN AT CODING TIME.** A subsample picked
    when the coder runs is a subsample picked after seeing something; the same
    discipline as registrar's pilot draw on text_sha16. Stratified so the floor
    stratum cannot vanish from the coded slice by chance.
    """
    rnd = random.Random(seed)
    strata = collections.defaultdict(list)
    for e in entries:
        strata[e["stratum"]].append(e)
    draw = {}
    for ck in checkpoints:
        picks = []
        #: proportional to stratum size, floor guaranteed at least a quarter
        n_floor = max(n_coded // 4, 1)
        n_trans = n_coded - n_floor
        for stratum, n in (("transgressive", n_trans), ("floor", n_floor)):
            es = strata.get(stratum, [])
            if not es:
                continue
            for i in range(n):
                e = es[i % len(es)]
                picks.append({"prompt_id": e["prompt_id"],
                              "idx": rnd.randrange(N_GEN),
                              "stratum": stratum})
        draw[ck] = picks
    return draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    entries, excluded = pool()
    by = collections.Counter(e["category"] for e in entries)
    strat = collections.Counter(e["stratum"] for e in entries)

    print("AMBIENT POOL — computed from the three properties, not listed")
    print("  excluded: %s" % dict(excluded))
    print("  qualifying: %d   (%s)" % (len(entries), dict(strat)))
    for c, n in by.most_common():
        print("     %-22s %d" % (c, n))
    if strat.get("floor", 0) < 4:
        sys.exit("REFUSING: floor stratum has %d prompts; the ruled pool needs 4"
                 % strat.get("floor", 0))

    from malign_logits.registry import Registry
    pairs = Registry().base_aligned_pairs()
    #: L2 runs the 7-edge frame, not the full roster ([5051].3)
    ck = sorted({m for p in pairs for m in (p["base"], p["aligned"])})[:13]

    n_gen = len(entries) * N_GEN * 13
    print("\nVOLUMES")
    print("  generated  %d prompts x %d samples x 13 ckpt = %d passages"
          % (len(entries), N_GEN, n_gen))
    print("  coded      %d per checkpoint x 13            = %d"
          % (N_CODED, N_CODED * 13))
    print("  (the coded slice is drawn FROM the generated, not generated separately)")
    se_f = math.sqrt(0.10*0.90/325)*100
    se_b = math.sqrt(0.10*0.90/3900)*100
    se_d = math.sqrt(se_f**2 + se_b**2)
    print("\nPOWER OF THE SUPPRESSION READING (the floor stratum carries it)")
    print("  floor coded 325  SE +/-%.2fpp | BOTH coded 3900 SE +/-%.2fpp"
          % (se_f, se_b))
    print("  CONTRAST SE +/-%.2fpp -> a 4pp inversion is %.2f SE" % (se_d, 4/se_d))
    print("  detectable barely, POOLED ACROSS CHECKPOINTS ONLY.")

    if a.write:
        draw = coded_draw(entries, ck)
        json.dump({
            "_about": "ambient null for the M02 redo. The F01 PROMPTS "
                      "regenerated under this producer; the F01 RATES are "
                      "inadmissible (docket [5059].4/[5101]).",
            "_producer": "scripts/f11_ambient_manifest.py",
            "_properties": {
                "content_exclusion": "not a contradiction cell or one of its poles",
                "length": "%d-%d words, matching the BOTH cells" % (WMIN, WMAX),
                "genre": "third-person narrative opening; institutional and "
                         "instructional prompts excluded. DECLARED LATE -- "
                         "found at the draw, not at the spec.",
            },
            "_ruling": "registrar [5101]: all qualifying prompts, transgressive "
                       "stratified by category + neutral as a named floor "
                       "stratum. The transgressive ambient is the CONSERVATIVE "
                       "null.",
            "_power": {
                "note": "THE SUPPRESSION READING RESTS ON THE FLOOR STRATUM. "
                        "Stated here so it sits beside every use of the rate, "
                        "not once in a caveat (lacan [5102].2/[5104].2).",
                "floor_coded_n": 325, "floor_se_pp": 1.66,
                "both_coded_n": 3900, "both_se_pp": 0.48,
                "contrast_se_pp": 1.73,
                "inversion_pp": 4.0, "inversion_in_se": 2.31,
                "correction": "2.31, NOT 2.40. The suppression reading is a "
                              "CONTRAST between floor and BOTH, so the noise "
                              "floor is the difference SE sqrt(1.66^2+0.48^2), "
                              "not the floor SE alone. lacan [5104].1.",
                "reading": "detectable barely, POOLED ACROSS CHECKPOINTS ONLY. "
                           "Never per-checkpoint, never per-category within "
                           "the floor.",
            },
            "seed": SEED, "n_generated_per_cell": N_GEN,
            "n_coded_per_checkpoint": N_CODED,
            "prompts": entries, "coded_draw": draw,
        }, open(OUT, "w"), ensure_ascii=False, indent=1)
        print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    else:
        print("\n--show: nothing written")


if __name__ == "__main__":
    main()
