#!/usr/bin/env python
"""Z: human literary prose as the reference axis for model output.

    python z_human_baseline.py
    python z_human_baseline.py --per-corpus 400

WHAT THIS ADDS THAT A BASE/ALIGNED CONTRAST CANNOT. Every Y and Z result so far
is a DIFFERENCE between two checkpoints. A difference has a direction but no
location: "aligned uses more emotion vocabulary" does not say whether aligned
has moved toward human prose, past it, or away from it. Human corpora put the
two arms on an axis with a third point on it.

Human text is NOT prompt-matched to the models and cannot be. It is a reference
distribution, not a paired contrast, and every number here is a LOCATION on a
shared axis rather than a tested difference. No p-values on the human side.

CORPORA, each its own baseline rather than pooled -- they span 1500-2000 and
pooling them would produce a "human" mean that no human wrote:

    passages_eebo_tcp     EEBO-TCP, early modern print
    passages_ecco_tcp     ECCO-TCP, eighteenth century
    passages_earlyprint   EarlyPrint
    passages_c19          nineteenth century
    passages_markmark     modern literary fiction
    passages              mixed / default sample

THE COMPARISON IS ONE-SIDED IN A WAY WORTH STATING. Model passages are
continuations of a transgressive prompt battery; human passages are sampled from
whole works with no prompt at all. So a gap between them mixes (a) what the
models write, (b) what the prompts asked for, and (c) what literary sampling
selects. Only the BASE-TO-ALIGNED movement along the axis is interpretable as
an alignment effect; the absolute gap to human prose is not.
"""
import argparse
import collections
import glob
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")
sys.path.insert(0, HERE)

from malign_logits.cache import CacheManager   # noqa: E402
from malign_logits import fields                # noqa: E402

LDATA = os.environ["LITMOD_DATA_DIR"]
CORPORA = ["passages_eebo_tcp", "passages_ecco_tcp", "passages_earlyprint",
           "passages_c19", "passages_markmark", "passages"]
SEED = 20260808

#: The measures that carried Y and Z. Kept to a declared list rather than
#: everything, because the point here is LOCATING known effects on a human
#: axis, not discovering new ones in a population with no contrast in it.
WATCH = ["F:emotion_and_arousal", "F:sensory_perception",
         "F:physical_appearance_and_properties",
         "F:logical_modal_and_discourse_operators",
         "F:language_and_communication", "F:kinship_and_relationships",
         "N:dominance=dominant", "N:concreteness=concrete",
         "N:valence_extremity=extreme", "R:sensation", "R:need",
         "R:temporal_references"]


def profile(text):
    out = {}
    r = fields.count(text)
    n = r["n_counted"] or 0
    if n < 5:
        return None
    for g, c in r["counts"].items():
        out["F:" + g] = c / n
    for dim, x in fields.norms(text).items():
        t = sum(x["counts"].values())
        if t:
            for b, c in x["counts"].items():
                out["N:%s=%s" % (dim, b)] = c / t
    rd = fields.count(text, "rid")["counts"]
    rt = sum(rd.values())
    if rt >= 3:
        for g, c in rd.items():
            k = "R:" + g.split(":")[0]
            out[k] = out.get(k, 0) + c / rt
    return out


def human(per_corpus, rng):
    """corpus -> {measure: mean share}, plus a year census."""
    out, years = {}, {}
    for corp in CORPORA:
        files = sorted(glob.glob(os.path.join(LDATA, corp, "*.jsonl")))
        if not files:
            continue
        rng.shuffle(files)
        acc = collections.defaultdict(list)
        yrs = []
        n = 0
        for f in files:
            if n >= per_corpus:
                break
            try:
                lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
            except Exception:
                continue
            if not lines:
                continue
            try:
                head = json.loads(lines[0])
                if head.get("year"):
                    yrs.append(int(head["year"]))
            except Exception:
                pass
            #: one passage per work, so a long work cannot dominate the corpus
            body = [l for l in lines[1:] if l.strip()]
            if not body:
                continue
            try:
                rec = json.loads(rng.choice(body))
            except Exception:
                continue
            txt = rec.get("text") or ""
            if len(txt) < 400:
                continue
            p = profile(txt)
            if not p:
                continue
            n += 1
            for g, v in p.items():
                acc[g].append(v)
        if n >= 20:
            out[corp] = ({g: statistics.mean(v) for g, v in acc.items()}, n)
            years[corp] = (min(yrs), statistics.median(yrs), max(yrs)) if yrs else None
    return out, years


def models(rng, per_model=250):
    """base / aligned mean profiles over the roster, temp=1.0."""
    st = CacheManager()._stash("generations")
    roster = [(p["base"], p["aligned"]) for p in
              json.load(open(os.path.join(ROOT, "data", "base_aligned_pairs.json")))]
    role_of = {}
    for b, a in roster:
        role_of.setdefault(b, "base")
        role_of[a] = "aligned"
    keys = collections.defaultdict(list)
    for k in st.keys():
        if k.get("temp") != 1.0:
            continue
        m = k.get("model") or ""
        if ":" in m or m not in role_of:
            continue
        keys[role_of[m]].append(k)
    out = {}
    for role, ks in keys.items():
        rng.shuffle(ks)
        acc = collections.defaultdict(list)
        n = 0
        for k in ks:
            if n >= per_model * 12:
                break
            try:
                txt = st.get(k)
            except Exception:
                continue
            if not isinstance(txt, str) or len(txt) < 400:
                continue
            p = profile(txt)
            if not p:
                continue
            n += 1
            for g, v in p.items():
                acc[g].append(v)
        out[role] = ({g: statistics.mean(v) for g, v in acc.items()}, n)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-corpus", type=int, default=250)
    ap.add_argument("--per-model", type=int, default=250)
    a = ap.parse_args(argv)
    rng = random.Random(SEED)

    H, yrs = human(a.per_corpus, rng)
    M = models(rng, a.per_model)
    print("human corpora: %d   model passages: base %s / aligned %s\n"
          % (len(H), format(M.get("base", ({}, 0))[1], ","),
             format(M.get("aligned", ({}, 0))[1], ",")))
    print("  %-22s %7s  %s" % ("corpus", "n", "years (min/med/max)"))
    for c, (_, n) in H.items():
        y = yrs.get(c)
        print("  %-22s %7d  %s" % (c, n, ("%d / %d / %d" % y) if y else "-"))
    print()

    cols = list(H) + ["base", "aligned"]
    print("LOCATION ON EACH AXIS -- share of counted tags, %")
    print("  %-34s %s" % ("measure", " ".join("%9s" % c.replace("passages_", "")[:9] for c in cols)))
    print("  " + "-" * (36 + 10 * len(cols)))
    for g in WATCH:
        vals = []
        for c in cols:
            src = H[c][0] if c in H else M.get(c, ({}, 0))[0]
            vals.append(src.get(g))
        if all(v is None for v in vals):
            continue
        print("  %-34s %s" % (g, " ".join(("%9.2f" % (100 * v)) if v is not None else "%9s" % "-"
                                          for v in vals)))
    print()
    #: THE ONLY INTERPRETABLE QUANTITY: does base->aligned move toward the
    #: human range or away from it? Human range across corpora, not a mean.
    print("DOES ALIGNMENT MOVE TOWARD HUMAN PROSE?")
    print("  human range = min..max across the %d corpora, each its own baseline" % len(H))
    print("  %-34s %16s %9s %9s %s" % ("measure", "human range", "base", "aligned", "movement"))
    print("  " + "-" * 92)
    for g in WATCH:
        hv = [H[c][0].get(g) for c in H if H[c][0].get(g) is not None]
        b = M.get("base", ({}, 0))[0].get(g)
        x = M.get("aligned", ({}, 0))[0].get(g)
        if not hv or b is None or x is None:
            continue
        lo, hi = min(hv), max(hv)
        mid = (lo + hi) / 2

        def dist(v):
            return 0.0 if lo <= v <= hi else (lo - v if v < lo else v - hi)
        db, dx = dist(b), dist(x)
        if db == 0 and dx == 0:
            mv = "both inside human range"
        elif dx < db:
            mv = "TOWARD human (%.2f -> %.2f)" % (100 * db, 100 * dx)
        elif dx > db:
            mv = "away from human (%.2f -> %.2f)" % (100 * db, 100 * dx)
        else:
            mv = "no change"
        print("  %-34s %7.2f..%-7.2f %9.2f %9.2f %s"
              % (g, 100 * lo, 100 * hi, 100 * b, 100 * x, mv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
