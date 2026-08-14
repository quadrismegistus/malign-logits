#!/usr/bin/env python
"""f11_l1_masses.py — N3 §2.3 masses, §3 excess, §4 readout, from a CLASS MAP.

    scripts/f11_l1_masses.py --selftest
    scripts/f11_l1_masses.py --classes <file> [--baseline poles|controls]

**THE CLASSIFIER IS AN INPUT AND THIS SCRIPT WILL NOT SUPPLY ONE.** Which
instrument assigns POLE1/POLE2/IN-FRAME/OFF-FRAME to a (surface, group) is a
CONSTRUCT decision -- LLM coder, geometric axis, or something else -- and it
belongs in an amendment, not in a producer. What is NOT a construct decision is
the arithmetic from classes to the declared statistics: N3 §2.3's masses, §3's
excess and classification, §4's readout and the redo registration's contrast
hierarchy are all written down and none of them exists in code. That gap is
what this fills, so whichever fork is ruled, the analysis is one file away.

    --classes    JSON or JSONL, records carrying `surface`, `group`, `class`.
                 class in POLE1 / POLE2 / IN_FRAME / OFF_FRAME (case- and
                 hyphen-insensitive). REQUIRED. No default, no fallback.

**THE COMPUTATION IS PURE FUNCTIONS OVER DICTS** so the self-test can hand them
made-up numbers with an answer worked out by hand. A producer validated only
against its own inputs certifies nothing; `--selftest` runs first and is the
known-answer column.

## Two specification gaps this producer resolves BY EXISTING, both reported

**1. N3 §3's four classes are not disjoint as written.** `EXIT` fires on
`excess_off_frame > t` and `ENGAGE` on `excess_in_frame > t`, and both can hold
at once -- they are not complements, since `in_frame + off_frame + unresolved`
is 1 and unresolved can fall by more than 2t. The registration lists them in
order, so this takes FIRST MATCH IN THE DECLARED ORDER and **counts the
ambiguous cells on the face**. A silent tie-break would make the order do
invisible work.

**2. `unresolved` is recomputed here, not read.** N3 defines it as
`1 - (in_frame + off_frame)` -- mass the CODED vocabulary does not reach --
which after [5170] is θ-truncation plus anything the class map omits. A surface
present in the cell and absent from the class map is therefore UNRESOLVED, not
dropped, and the count of such surfaces is reported: a class map that silently
covers 60% of the mass would otherwise look like a model that hedges.
"""
import argparse, json, os, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

THETA = 0.001
DEMOTE = 0.50
T_DECLARED = 0.05
T_CURVE = (0.02, 0.03, 0.05, 0.08, 0.10)
CLASSES = ("POLE1", "POLE2", "IN_FRAME", "OFF_FRAME")
#: the mass keys, in the order N3 §2.3 defines them
MASSES = ("pole1", "pole2", "in_frame", "off_frame", "unresolved")


def norm_class(c):
    c = str(c).strip().upper().replace("-", "_").replace(" ", "_")
    return c if c in CLASSES else None


# ── the arithmetic, pure ───────────────────────────────────────────────────

def cell_masses(words, classes):
    """N3 §2.3, for one cell. `words` {surface: p}; `classes` {surface: class}.

    `in_frame` INCLUDES the poles -- it is pole1 + pole2 + IN_FRAME, not the
    IN_FRAME class alone. Getting that wrong makes ENGAGE and RESOLVE rivals
    when the registration has them nested.
    """
    m = {c: 0.0 for c in CLASSES}
    uncovered, uncovered_mass = 0, 0.0
    for w, p in words.items():
        if p < THETA:
            continue
        c = classes.get(w)
        if c is None:
            uncovered += 1; uncovered_mass += p
            continue
        m[c] += p
    in_frame = m["POLE1"] + m["POLE2"] + m["IN_FRAME"]
    off_frame = m["OFF_FRAME"]
    return dict(pole1=m["POLE1"], pole2=m["POLE2"], in_frame=in_frame,
                off_frame=off_frame,
                unresolved=1.0 - (in_frame + off_frame),
                uncovered_surfaces=uncovered, uncovered_mass=uncovered_mass)


def excess(ab, a, b):
    """N3 §3: excess_M = M(AB) - mean(M(A), M(B)), for every mass."""
    return {k: ab[k] - (a[k] + b[k]) / 2.0 for k in MASSES}


def classify(exc, t=T_DECLARED):
    """N3 §3's partition. Returns (label, ambiguous) -- see gap 1 in the docstring."""
    hits = []
    if exc["off_frame"] > t:
        hits.append("EXIT")
    if exc["in_frame"] > t:
        hits.append("ENGAGE")
    if max(exc["pole1"], exc["pole2"]) > t and abs(exc["in_frame"]) <= t:
        hits.append("RESOLVE")
    return (hits[0] if hits else "NULL"), len(hits) > 1


# ── the readout ────────────────────────────────────────────────────────────

def wilcoxon(vals):
    try:
        from scipy.stats import wilcoxon as w
        st, p = w(vals)
        return float(st), float(p)
    except Exception:
        return None, None


def paired(per_unit, label, a_roles, b_roles):
    """A declared contrast: mean(a_roles) - mean(b_roles), unit as keyed."""
    out = []
    for _u, cells in per_unit.items():
        d = []
        for _g, roles in cells.items():
            if not all(r in roles for r in a_roles + b_roles):
                continue
            ma = statistics.mean(roles[r] for r in a_roles)
            mb = statistics.mean(roles[r] for r in b_roles)
            d.append(ma - mb)
        if d:
            out.append(statistics.mean(d))
    if not out:
        print("  %-34s no complete pairs" % label); return
    st, p = wilcoxon(out)
    print("  %-34s n=%-4d mean %+.4f  median %+.4f  pos %d/%d  p=%s"
          % (label, len(out), statistics.mean(out), statistics.median(out),
             sum(1 for v in out if v > 0), len(out),
             "%.4g" % p if p is not None else "scipy?"))


def selftest():
    """KNOWN ANSWERS, WORKED BY HAND. Runs before anything reads the store."""
    ok = True
    def eq(name, got, want, tol=1e-12):
        nonlocal ok
        good = abs(got - want) < tol if isinstance(want, float) else got == want
        print("  [%s] %-38s got %s want %s"
              % ("PASS" if good else "FAIL", name, got, want))
        ok = good and ok

    cls = {"love": "POLE1", "hate": "POLE2", "him": "IN_FRAME",
           "However": "OFF_FRAME"}
    #: 0.4 + 0.2 + 0.1 + 0.2 = 0.9 covered; 0.05 uncovered; 0.05 below theta
    w = {"love": 0.4, "hate": 0.2, "him": 0.1, "However": 0.2,
         "zzz": 0.05, "tiny": 0.0005}
    m = cell_masses(w, cls)
    eq("pole1", m["pole1"], 0.4)
    eq("in_frame includes poles", m["in_frame"], 0.7)
    eq("off_frame", m["off_frame"], 0.2)
    eq("unresolved = 1-(in+off)", m["unresolved"], 0.1)
    eq("uncovered surface counted", m["uncovered_surfaces"], 1)
    eq("uncovered mass", m["uncovered_mass"], 0.05)
    eq("sub-theta not uncovered", m["uncovered_mass"] < 0.0505, True)

    #: excess: AB off_frame 0.30 against poles 0.10 and 0.10 -> +0.20 -> EXIT
    a = cell_masses({"love": 0.8, "However": 0.1}, cls)
    b = cell_masses({"hate": 0.8, "However": 0.1}, cls)
    ab = cell_masses({"love": 0.3, "hate": 0.3, "However": 0.3}, cls)
    e = excess(ab, a, b)
    eq("excess_off_frame", e["off_frame"], 0.2)
    eq("excess_in_frame", e["in_frame"], -0.2)
    eq("classify -> EXIT", classify(e)[0], "EXIT")

    #: the AMBIGUOUS case gap 1 exists for: off_frame and in_frame BOTH rise,
    #: paid for out of unresolved. First match in declared order wins, and the
    #: cell is counted.
    a2 = cell_masses({"love": 0.2, "However": 0.1}, cls)
    b2 = cell_masses({"hate": 0.2, "However": 0.1}, cls)
    ab2 = cell_masses({"love": 0.4, "hate": 0.2, "However": 0.3}, cls)
    e2 = excess(ab2, a2, b2)
    lab, amb = classify(e2)
    eq("both rise -> ambiguous flagged", amb, True)
    eq("first match in declared order", lab, "EXIT")
    print("\n%s" % ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--classes", help="JSON/JSONL of {surface, group, class}")
    ap.add_argument("--baseline", choices=("poles", "controls"),
                    default="poles",
                    help="poles: N3 §3 (AB vs pole_a/pole_b). "
                         "controls: the redo §4 primary (both vs controls)")
    ap.add_argument("--lang", choices=("en", "zh", "all"), default="en",
                    help="en is confirmatory; zh is descriptive-only under the "
                         "N3 §4 coverage gate (54.8%% of zh cells demoted)")
    ap.add_argument("--csv", help="write per-cell masses here")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.classes:
        print(__doc__.strip().split("\n\n")[0])
        print("\nrefusing to run: --classes is required and has no default.\n"
              "Which instrument assigns the four classes is a construct "
              "decision, not this producer's.")
        return 2
    if selftest():
        print("known-answer column failed; not reading the store."); return 1

    from f11_quintuplet_spec import PROMPT_ROLES
    from malign_logits.cache import get_cache
    from malign_logits.registry import Registry
    cm = get_cache()

    raw = open(a.classes).read().strip()
    recs = (json.loads(raw) if raw.startswith("[") else
            [json.loads(l) for l in raw.splitlines() if l.strip()])
    if isinstance(recs, dict):
        recs = recs.get("units") or recs.get("classes") or []
    classes = defaultdict(dict)
    bad = 0
    for r in recs:
        c = norm_class(r.get("class"))
        if c is None:
            bad += 1; continue
        classes[r["group"]][r["surface"]] = c
    print("class map: %d groups, %d assignments, %d unusable"
          % (len(classes), sum(len(v) for v in classes.values()), bad))

    q = json.load(open(os.path.join(ROOT, "data", "f11_quintuplets.json")))
    q = q["quintuplets"]
    items = q.items() if isinstance(q, dict) else [(e.get("group"), e)
                                                   for e in q]
    groups = {}
    for gid, v in items:
        if not isinstance(v, dict) or "RETIRED" in (v.get("status") or "").upper():
            continue
        name = v.get("group", gid)
        if name.startswith("f11_reason"):
            continue                       # negative control, held beside
        if a.lang != "all" and (name.endswith("_zh")) != (a.lang == "zh"):
            continue
        groups[name] = {r: v.get(r) for r in PROMPT_ROLES
                        if isinstance(v.get(r), str) and v.get(r)}

    ab_role = "both"
    base = ("pole_a", "pole_b") if a.baseline == "poles" else \
           ("control_a", "control_b")

    ckpts = sorted({m for p in Registry().base_aligned_pairs()
                    for m in (p["base"], p["aligned"])})
    cells, per_model = [], defaultdict(lambda: defaultdict(dict))
    for mid in ckpts:
        for g, roles in groups.items():
            cmap = classes.get(g, {})
            for role, prompt in roles.items():
                v = cm.get_true_word_probs(mid, prompt, theta=THETA)
                if not v or not v.get("rows"):
                    continue
                words = {}
                for r in v["rows"]:
                    words[r["word"]] = words.get(r["word"], 0.0) + r.get("p", 0)
                m = cell_masses(words, cmap)
                m.update(model=mid, group=g, role=role,
                         demoted=m["unresolved"] > DEMOTE)
                cells.append(m)
                per_model[mid][g][role] = m
    if not cells:
        print("no cells read"); return 1

    dem = sum(1 for c in cells if c["demoted"])
    unc = statistics.mean(c["uncovered_mass"] for c in cells)
    print("\ncells %d | demoted %d = %.1f%% | mass the CLASS MAP misses %.3f"
          % (len(cells), dem, 100.0 * dem / len(cells), unc))

    #: §3 excess and the classification, per (checkpoint, group)
    lin = json.load(open(os.path.join(ROOT, "data",
                                      "lineage_map_models.json")))["model_to_lineage"]
    exc_rows, ambiguous = [], 0
    for mid, gs in per_model.items():
        for g, roles in gs.items():
            if not all(r in roles for r in (ab_role,) + base):
                continue
            e = excess(roles[ab_role], roles[base[0]], roles[base[1]])
            lab, amb = classify(e)
            ambiguous += amb
            exc_rows.append(dict(model=mid, lineage=lin.get(mid, mid), group=g,
                                 label=lab, **{"excess_" + k: v
                                               for k, v in e.items()}))
    print("triples %d | AMBIGUOUS (EXIT and ENGAGE both fire) %d = %.1f%%"
          % (len(exc_rows), ambiguous,
             100.0 * ambiguous / max(1, len(exc_rows))))

    print("\nMECHANISM, t = %.2f  (baseline: %s)" % (T_DECLARED, a.baseline))
    tot = defaultdict(int)
    for r in exc_rows:
        tot[r["label"]] += 1
    for k in ("EXIT", "ENGAGE", "RESOLVE", "NULL"):
        print("  %-9s %5d  %5.1f%%" % (k, tot[k],
                                       100.0 * tot[k] / max(1, len(exc_rows))))

    print("\nt-SENSITIVITY (N3 §3 requires the whole curve, never one point)")
    print("  %-6s %7s %7s %7s %7s" % ("t", "EXIT", "ENGAGE", "RESOLVE", "NULL"))
    for t in T_CURVE:
        c = defaultdict(int)
        for r in exc_rows:
            c[classify({k[7:]: v for k, v in r.items()
                        if k.startswith("excess_")}, t)[0]] += 1
        print("  %-6.2f %6.1f%% %6.1f%% %6.1f%% %6.1f%%"
              % (t, *[100.0 * c[k] / max(1, len(exc_rows))
                      for k in ("EXIT", "ENGAGE", "RESOLVE", "NULL")]))

    #: N3 §4 PRIMARY: is EXIT the modal mechanism across LINEAGES?
    by_lin = defaultdict(list)
    for r in exc_rows:
        by_lin[r["lineage"]].append(r["label"])
    modal = {}
    for L, labs in by_lin.items():
        real = [x for x in labs if x != "NULL"]
        if real:
            modal[L] = max(set(real), key=real.count)
    n_exit = sum(1 for v in modal.values() if v == "EXIT")
    print("\nN3 §4 PRIMARY  EXIT modal in %d of %d lineages"
          % (n_exit, len(modal)))
    try:
        from scipy.stats import binomtest
        p = binomtest(n_exit, len(modal), 1/3, alternative="greater").pvalue
        print("  one-sided binomial vs p=1/3 (NULL is NOT a fourth category): "
              "p=%.4g" % p)
    except Exception:
        pass

    print("\nDECLARED CONTRASTS  (checkpoint as unit, roster Wilcoxon)")
    for key in ("off_frame", "in_frame", "pole1", "pole2"):
        sub = defaultdict(lambda: defaultdict(dict))
        for mid, gs in per_model.items():
            for g, roles in gs.items():
                for role, m in roles.items():
                    sub[mid][g][role] = m[key]
        print(" [%s]" % key)
        paired(sub, "  BOTH vs mean(CONTROLS)", ["both"],
               ["control_a", "control_b"])
        paired(sub, "  CONTROL_A vs CONTROL_B", ["control_a"], ["control_b"])
        paired(sub, "  mean(CONTROLS) vs mean(POLES)",
               ["control_a", "control_b"], ["pole_a", "pole_b"])
        paired(sub, "  BOTH vs BOTH_MATCHED", ["both"], ["both_matched"])

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cells[0]))
            w.writeheader(); w.writerows(cells)
        print("\ncells -> %s" % a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
