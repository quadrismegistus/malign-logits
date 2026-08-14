"""Canonical sites for the six lineages measured after the grid export.

    ./canonical_sites_newlineages.py                counts
    ./canonical_sites_newlineages.py --emit PATH    also write the triples
    ./canonical_sites_newlineages.py --known-answer reproduce a booked pair

WHICH RULE, AND WHY IT IS THIS ONE. A site is the LARGEST OPEN-CLASS FALLER and
the LARGEST OPEN-CLASS RISER at a prompt, under `Cell.movement(CANONICAL)` --
the same objects and rule as `pair_census_canonical.py`, Registrations D/L/M/N/O
and `data/fc_manifest_*.json` ("LEXICAL ARM: largest OPEN-CLASS faller and riser
per site").

**A FIRST VERSION OF THIS ANSWER USED `m05_sites.py` AND WAS WRONG.** M05's rule
is `top_changed` + availability: did the ARGMAX move, and was the substitute in
the base's top 20. That is a different construct -- it asks whether the model's
single best word changed, not which words lost and gained mass -- and it counts
a different population. It returned 6,345 sites where this returns far fewer,
and neither number is a correction of the other; they answer different
questions. RH caught it by asking which rule was used.

  CANONICAL   faller iff P >= 0.003 and Q < 0.5 P
              riser  iff not faller, max(P,Q) > 0.003, (Q-P) > 0.003,
                     and Q > null -- the RENORMALISATION NULL, which is the
                     whole point: without it every word "rises" a little when
                     a faller's mass is removed.
  ASYMMETRY   risers are null-tested, FALLERS ARE NOT. Nothing here may
              describe a faller as "beyond renormalisation".

THE POPULATION mirrors N section 3 exactly, as the census does: distinct
catalogue stimuli, sentinels out, CJK out -- 2,199 prompts. All six pairs cover
all 2,199, so no pair is measured on a different denominator.

OPEN CLASS is `content(w)`: not in f13's FUNC list, and contains a letter.
Applied to BOTH roles, as `build_population.py` does with `both_content`.

WHAT THIS IS NOT. It is not a wave-3 work order. Wave 3's 49-61 sites per pair
is what survives the r_population_k2 (k>=2, minimal-pair stems) and
beam_sample_105 SAMPLING on top of this rule. This counts the sites; the sample
is drawn from them and is much smaller. Reporting this number as a wave cost
would repeat, in the other direction, the 4x error that came of estimating it
by hand.
"""
import argparse
import collections
import json
import os
import re
import sys

ROOT = os.path.expanduser("~/github/malign-logits")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from malign_logits.checkpoint import Checkpoint
from malign_logits.movement import CANONICAL
from malign_logits.prompts import Prompts
from malign_logits.step import Step

SENTINEL = re.compile(r"^<<<[A-Z]+:")
CJK = re.compile(r"[一-鿿㐀-䶿]")
REG = os.path.join(ROOT, "data", "model_registry.json")

#: registry `family` values, read from it rather than guessed from model ids.
NEW_FAMILIES = ["granite", "llm-jp-3", "salamandra", "lucie", "gemma2", "jais"]
#: EXCLUDED BY NAME: 2,269 base / 1,233 aligned cells against 2,583 for a
#: complete arm. A different denominator, not a smaller sample of the same one.
INCOMPLETE = ["openGPT-X/Teuken-7B-base-v0.6",
              "openGPT-X/Teuken-7B-instruct-commercial-v0.4"]
#: a booked pair from the grid export, used as the known-answer column.
KNOWN = ("ibm-granite/granite-3.0-8b-base",
         "ibm-granite/granite-3.0-8b-instruct")


def func_words():
    """f13's FUNC, imported from the file that declares it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "draw", os.path.join(ROOT, "scripts/f13_draw_relation_items.py"))
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    return D.FUNC


def population():
    """N section 3: distinct catalogue stimuli, sentinels out, CJK out."""
    out = set()
    for p in Prompts().all():
        t = p if isinstance(p, str) else (getattr(p, "text", None) or str(p))
        if SENTINEL.match(t) or CJK.search(t):
            continue
        out.add(t)
    return out


def new_pairs(reg):
    fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for mid, r in reg.items():
        f = (r.get("family") or "").lower()
        if f in NEW_FAMILIES and mid not in INCOMPLETE:
            fam[f][(r.get("position") or "").lower()].append(mid)
    pairs, refused = [], []
    for f in NEW_FAMILIES:
        b = fam.get(f, {}).get("base", [])
        a = fam.get(f, {}).get("superego", [])
        if len(b) == 1 and len(a) == 1:
            pairs.append((b[0], a[0]))
        else:
            refused.append((f, "base=%d superego=%d" % (len(b), len(a))))
    return pairs, refused


def sites_for(base, aligned, pop, FUNC, keep=False):
    """(rows, tally). One site per prompt where BOTH roles have an open-class word."""
    content = lambda w: w.lower() not in FUNC and any(c.isalpha() for c in w)
    step = Step(Checkpoint(base), Checkpoint(aligned))
    shared = [p for p in step.prompts if p in pop]
    tally = collections.Counter(shared=len(shared))
    rows = []
    for t in shared:
        c = step.cell(t)
        if not c.is_present:
            tally["absent"] += 1
            continue
        m = c.movement(CANONICAL)
        if m is None:
            tally["absent"] += 1
            continue
        fall = [w for w in m.fallers if content(w)]
        rise = [w for w in m.risers if content(w)]
        if not fall:
            tally["no_open_faller"] += 1
        if not rise:
            tally["no_open_riser"] += 1
        if not (fall and rise):
            tally["no_site"] += 1
            continue
        #: LARGEST BY THE RULE'S OWN RANKING -- fallers by drop, risers by
        #: EXCESS. Ranking risers by delta re-introduces exactly what the null
        #: removes, which `top_riser()` documents; this mirrors it restricted
        #: to the open class rather than calling it and hoping the winner is
        #: open-class.
        f = max(fall, key=lambda w: -m.delta.get(w, 0.0))
        r = max(rise, key=lambda w: m.excess.get(w, 0.0))
        tally["site"] += 1
        if keep:
            rows.append({"prompt": t, "faller": f, "riser": r,
                         "faller_drop": round(-m.delta.get(f, 0.0), 6),
                         "riser_excess": round(m.excess.get(r, 0.0), 6),
                         "n_fallers": len(m.fallers), "n_risers": len(m.risers),
                         "residual_share": round(
                             m.diagnostics.get("residual_share", 0.0), 4)})
    return rows, tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", metavar="PATH")
    ap.add_argument("--known-answer", action="store_true")
    a = ap.parse_args()
    if a.emit and os.path.exists(a.emit):
        sys.exit("REFUSING: %s exists. This script never overwrites." % a.emit)

    reg = {m["model_id"]: m for m in json.load(open(REG))["models"]}
    FUNC = func_words()
    pop = population()
    print("population (N sec3: catalogue, no sentinels, no CJK): %d prompts"
          % len(pop))
    print("rule: CANONICAL  min_prob %.3f  fall_ratio %.1f  delta %.3f  "
          "null_test %s" % (CANONICAL.min_prob, CANONICAL.fall_ratio,
                            CANONICAL.delta, CANONICAL.null_test))

    pairs, refused = new_pairs(reg)
    for f, why in refused:
        print("REFUSED family %s: %s" % (f, why))

    print("\n%-46s %7s %7s %7s %7s"
          % ("pair", "shared", "sites", "no_fall", "no_rise"))
    print("-" * 78)
    out, total = [], collections.Counter()
    for b, al in pairs:
        rows, tally = sites_for(b, al, pop, FUNC, keep=bool(a.emit))
        total.update(tally)
        out.append({"base": b, "aligned": al, "n_sites": tally["site"],
                    "shared": tally["shared"], "sites": rows})
        print("%-46s %7d %7d %7d %7d"
              % ("%s > %s" % (b.split("/")[-1][:20], al.split("/")[-1][:22]),
                 tally["shared"], tally["site"],
                 tally["no_open_faller"], tally["no_open_riser"]))
    print("-" * 78)
    print("%-46s %7d %7d %7d %7d"
          % ("TOTAL (%d pairs)" % len(pairs), total["shared"], total["site"],
             total["no_open_faller"], total["no_open_riser"]))
    print("\nmean sites/pair: %.1f" % (total["site"] / max(1, len(pairs))))
    print("Teuken excluded by name: %s"
          % ", ".join(m.split("/")[-1] for m in INCOMPLETE))
    print("\nThis is the SITE COUNT, not a wave-3 work order: wave 3's 49-61 "
          "sites/pair\nis what survives r_population_k2 (k>=2) and "
          "beam_sample_105 sampling ON TOP of this.")

    if a.emit:
        doc = {"producer": "meta/M01_displacement/scripts/"
                           "canonical_sites_newlineages.py",
               "rule": "Cell.movement(CANONICAL); largest OPEN-CLASS faller "
                       "(by drop) and riser (by excess) per prompt",
               "population": "N sec3: catalogue stimuli, sentinels out, CJK "
                             "out (%d prompts)" % len(pop),
               "excluded_incomplete": INCOMPLETE,
               "pairs": out}
        with open(a.emit, "w") as fh:
            json.dump(doc, fh, indent=1)
        print("\nwrote %s (%d pairs, %d sites)"
              % (a.emit, len(out), sum(p["n_sites"] for p in out)))


if __name__ == "__main__":
    main()
