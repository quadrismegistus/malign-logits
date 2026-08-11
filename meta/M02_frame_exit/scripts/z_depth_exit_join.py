#!/usr/bin/env python
"""Does the DEPTH at which the arms diverge predict whether the passage exits?

    uv run python z_depth_exit_join.py --sweep     # build the marker cells (once)
    uv run python z_depth_exit_join.py             # the join, en
    uv run python z_depth_exit_join.py --zh        # the Chinese half, apart

Specified in `meta/M02_frame_exit/TODO.md`. Two instruments have never been put
in the same room:

    the LENS   `lens_group_layer.jsonl` -- at which DEPTH the base and aligned
               arms of one lineage stop agreeing on the BOTH prompt
    the MARKERS `y_exit_typology.TYPES` over the f11_l2 generations -- whether
               the continuation LEAVES the frame, and into what

The JS ratio licenses "the continuation is unrelated to the poles" and says
nothing about the destination. Only the markers can speak to destination, and
only the lens can speak to depth. The question is whether they are the same
event:

    a LATE GATE is cheap, reversible, cosmetic -- a mask over a computation
    that still ran. If the gate is what produces the exit, then the depth of
    divergence should predict the exit rate.

    an EARLY RE-ROUTING is a changed computation. If depth predicts nothing,
    the depth story is about the READOUT, not about what the model does, and
    the cheap-reversible-cosmetic reading loses its evidence.

Either answer is worth having, and the null is not a failure: it constrains
what may be written about alignment being a surface phenomenon.

THE INSTRUMENTS ARE IMPORTED, NEVER RE-AUTHORED. `TYPES` comes from
`y_exit_typology` by import. `REFUSAL` is lifted out of `exit_contradiction.py`
by parsing its source (importing that module would re-run its sweep), and the
lift is checked: if the literal there ever changes, this file raises rather
than silently measuring something else. REFUSAL is reported BESIDE the exit
types and never pooled into them -- the Y dissociation (refusal is not frame
exit) is the reason the typology has a REFUSAL slot at all.

FOUR THINGS THE TODO SAYS TO GET RIGHT, ENFORCED HERE RATHER THAN REMEMBERED:

  **THE UNIT IS THE LINEAGE.** A cell is (model, group); a lineage contributes
  one number. The ICC of a paired contrast across rungs on comparable material
  is 0.85, so cells are not observations.

  **THE MARKER CONTRAST IS `excess(BOTH) - mean(POLE_A, POLE_B)`**, the form
  declared in `exit_contradiction.py` before reading, not a bare BOTH rate.

  **ROLE COMES FROM `prompt_categorisation.json`, MATCHED ON PROMPT TEXT.** The
  `role` column of `gen_sequences` is EMPTY for every one of the 228,520
  f11_l2 rows, and `source='QUINTUPLETS'` returns nothing from the DB catalogue
  while the JSON lists 42 texts under it. The JSON is the source of truth for
  what a prompt IS; the DB is the source of truth for what was generated.

  **ZH APART.** The lens is 47% Chinese on an English-heavy roster.

AND ONE CORRECTION TO THE TODO THAT COMMISSIONED THIS. It records "50 models
shared with the lens", which is true and is not the sample size: a lineage
needs BOTH arms present in BOTH substrates, and that leaves 25 pairs of the 46.
The n is 25, and the TODO's "2,150 triples / 129,000 passages" counts cells,
which are not observations.
"""
import argparse
import ast
import collections
import csv
import json
import math
import os
import re
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

CH = "/opt/homebrew/bin/clickhouse"
CORPUS = "f11_l2"
CAT = os.path.join(ROOT, "data", "prompt_categorisation.json")
PAIRS = os.path.join(ROOT, "data", "lineage_representative_pairs.txt")
CELLS = os.path.join(CAMP, "results", "z_exit_f11l2_cells.csv")
OUT = os.path.join(CAMP, "results", "z_depth_exit_join.csv")

MIN_N = 10          #: every cell holds 20 generations; this only guards gaps
TOP = 0.875         #: "the top eighth", the grid point lens_analysis reports


# ----------------------------------------------------------------- instruments

def refusal_pattern():
    """Lift REFUSAL out of exit_contradiction.py WITHOUT importing it.

    That module runs its sweep at import time. Parsing the source gets the
    declared literal with no side effect, and asserting we found exactly one
    means a future edit there breaks this loudly instead of quietly.
    """
    src = open(os.path.join(HERE, "exit_contradiction.py")).read()
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple) or len(node.elts) != 2:
            continue
        head = node.elts[0]
        if not (isinstance(head, ast.Constant) and head.value == "REFUSAL"):
            continue
        call = node.elts[1]
        if isinstance(call, ast.Call) and call.args:
            found.append(ast.literal_eval(call.args[0]))
    if len(found) != 1:
        raise SystemExit("expected exactly one REFUSAL literal, found %d" % len(found))
    return re.compile(found[0], re.I)


from y_exit_typology import TYPES as EXIT_TYPES     # noqa: E402  the six, verbatim

TYPES = list(EXIT_TYPES) + [("REFUSAL", refusal_pattern())]
EXITS = [t for t, _ in EXIT_TYPES]

#: ANY-EXIT is the UNION over the six exit types, counted per passage at sweep
#: time. It is NOT the sum of the per-type counts and cannot be recovered from
#: them -- a passage may fire two types, which is why `exit_contradiction.py`
#: declined to report a pooled figure from its per-type table. REFUSAL is
#: excluded from the union by construction: the Y dissociation is that refusal
#: is not frame exit, and pooling them would erase the only result that says so.
ANY = "ANY-EXIT"
TNAMES = [t for t, _ in TYPES] + [ANY]


def prompt_map():
    """{prompt text -> [(group, role, language), ...]} from the categorisation.

    Only groups carrying all three of POLE_A / POLE_B / BOTH, since the excess
    is undefined without them.

    ONE TEXT CAN OCCUPY SEVERAL SLOTS AND THE MAP MUST BE ONE-TO-MANY. Five of
    the 132 slots share a text with another:

        'He was beautiful and she wanted to'   POLE_A of f11_beauty AND of
                                               f11_beauty_ugly, and the same
                                               pair in zh
        'In a place both holy and filthy...'   BOTH of f11_holy AND f11_holy_b,
                                               and the same pair in zh
        'The human stood in the clearing...'   POLE_A of f11_species and
                                               POLE_B of f11_species_wolf

    A dict keyed on text keeps the last writer, which quietly starves four
    groups of a role -- and the fifth case is worse than a loss, because the
    same text is a DIFFERENT ROLE in the two groups, so one group would have
    silently received the other's arm. The generation is a fact about the
    prompt; every slot that uses that prompt is entitled to it.
    """
    cat = json.load(open(CAT))["prompts"]
    groups = collections.defaultdict(dict)
    for p in cat:
        if p.get("domain") == "contradiction" and p.get("group_id"):
            groups[p["group_id"]][p.get("group_role")] = p
    out = collections.defaultdict(list)
    slots = 0
    for gid, g in groups.items():
        if not gid.startswith("f11_") or not {"POLE_A", "POLE_B", "BOTH"} <= set(g):
            continue
        for role in ("POLE_A", "POLE_B", "BOTH"):
            p = g[role]
            out[p["prompt"].strip()].append((gid, role, p.get("language", "en")))
            slots += 1
    shared = sum(1 for v in out.values() if len(v) > 1)
    print("slots %d over %d distinct texts (%d texts serve >1 slot)"
          % (slots, len(out), shared))
    return dict(out)


# --------------------------------------------------------------------- sweep

def sweep():
    """Type every f11_l2 generation on a triplet prompt. Writes CELLS."""
    pm = prompt_map()
    print("triplet prompts: %d over %d groups"
          % (len(pm), len({s[0] for v in pm.values() for s in v})))
    q = ("SELECT model, prompt, text FROM malign_logits.gen_sequences "
         "WHERE corpus='%s' FORMAT JSONEachRow" % CORPUS)
    proc = subprocess.Popen([CH, "client", "--max_memory_usage", "8000000000",
                             "-q", q], stdout=subprocess.PIPE, text=True,
                            bufsize=1 << 20)
    agg = collections.defaultdict(lambda: [0] + [0] * len(TNAMES))
    seen = unmapped = 0
    for line in proc.stdout:
        try:
            r = json.loads(line)
        except Exception:
            continue
        seen += 1
        slots = pm.get((r["prompt"] or "").strip())
        if not slots:
            unmapped += 1
            continue
        txt = r["text"] or ""
        #: typed ONCE per passage, then credited to every slot the text fills
        fired = [bool(rx.search(txt)) for _, rx in TYPES]
        union = any(f for f, (n, _) in zip(fired, TYPES) if n != "REFUSAL")
        for gid, role, lang in slots:
            row = agg[(r["model"], gid, role, lang)]
            row[0] += 1
            for i, f in enumerate(fired):
                if f:
                    row[1 + i] += 1
            if union:
                row[1 + len(TYPES)] += 1
        if seen % 50000 == 0:
            print("  ... %s rows" % format(seen, ","))
    proc.wait()
    print("swept %s rows; %s on non-triplet prompts; %d cells"
          % (format(seen, ","), format(unmapped, ","), len(agg)))
    with open(CELLS, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "group", "role", "language", "n_gens", *TNAMES])
        for (m, g, role, lang), v in sorted(agg.items()):
            w.writerow([m, g, role, lang, *v])
    print("wrote %s" % os.path.relpath(CELLS, ROOT))


# ------------------------------------------------------------------- the join

def load_cells(lang):
    cells = {}
    for r in csv.DictReader(open(CELLS)):
        if r["language"] != lang:
            continue
        cells[(r["model"], r["group"], r["role"])] = (
            int(r["n_gens"]), {t: int(r[t]) for t in TNAMES})
    return cells


def complete(cells, model, group):
    return all((model, group, r) in cells and cells[(model, group, r)][0] >= MIN_N
               for r in ("POLE_A", "POLE_B", "BOTH"))


def excess(cells, model, groups, tname):
    """excess in POINTS, from counts POOLED over `groups`. None if short.

    POOLED AT THE COUNT LEVEL, AND THIS IS THE WHOLE ESTIMATOR QUESTION. A cell
    holds exactly 20 generations, so a cell rate moves in 5-point steps and a
    cell EXCESS in 2.5-point steps. The first version of this file took the
    MEDIAN of those per-cell excesses per lineage and got exactly +0.000 for
    every marker type, which reads as a null and is an artefact of snapping a
    coarse quantity to its modal value. Pooling the counts first gives each
    lineage arm ~400 passages per role and a rate that can actually move.

    The pooling is only legitimate because `groups` is fixed across roles and
    across arms by the caller: every role of every arm is summed over the SAME
    group set, so group composition cannot differ between the things being
    subtracted. It is a balanced pool, not a convenience one.
    """
    v = {}
    for role in ("POLE_A", "POLE_B", "BOTH"):
        n = k = 0
        for g in groups:
            c = cells.get((model, g, role))
            if not c or c[0] < MIN_N:
                return None
            n += c[0]
            k += c[1][tname]
        if n == 0:
            return None
        v[role] = 100.0 * k / n
    return v["BOTH"] - (v["POLE_A"] + v["POLE_B"]) / 2


def spearman(xs, ys):
    n = len(xs)
    if n < 4:
        return float("nan"), float("nan"), n

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return float("nan"), float("nan"), n
    rho = num / den
    if abs(rho) >= 1.0 or n < 5:
        return rho, float("nan"), n
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), n - 2)
    except Exception:
        p = float("nan")
    return rho, p, n


def sign_test(vals):
    v = [x for x in vals if x != 0.0]
    n, k = len(v), sum(1 for x in v if x > 0)
    if n == 0:
        return 0, 0, float("nan")
    t = min(k, n - k)
    return n, k, min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n)


def depth_summaries(lang):
    """{(lineage, group): dict} -- the lens side, from lens_analysis's own code.

      top_share   fraction of the total |aligned-base| gap sitting at depth
                  >= 0.875. LATE gate -> high.
      argmax      depth carrying the single largest gap.
      gap_total   the size of the divergence, which is a DIFFERENT quantity
                  from where it sits and has to be controlled for: a lineage
                  whose arms barely differ has a meaningless top_share.
      d_final     aligned - base ratio at depth 1.0. This is the OUTPUT-level
                  JS contrast, i.e. the instrument the markers were always
                  going to be compared against; it is the baseline.
    """
    import lens_analysis as LA
    traj, _, _ = LA.load(lang)
    P = LA.paired(traj)
    out = {}
    for (lin, g), (b, a) in P.items():
        gaps = [(LA.GRID[i], abs(a[i] - b[i])) for i in range(len(LA.GRID))
                if b[i] is not None and a[i] is not None]
        if len(gaps) < 3:
            continue
        tot = sum(v for _, v in gaps)
        if tot <= 0:
            continue
        fin = None
        if b[-1] is not None and a[-1] is not None:
            fin = a[-1] - b[-1]
        out[(lin, g)] = {
            "top_share": sum(v for d, v in gaps if d >= TOP) / tot,
            "argmax": max(gaps, key=lambda t: t[1])[0],
            "gap_total": tot,
            "d_final": fin,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--zh", action="store_true")
    a = ap.parse_args()
    if a.sweep or not os.path.exists(CELLS):
        sweep()
        if a.sweep:
            return
    lang = "zh" if a.zh else "en"

    cells = load_cells(lang)
    dep = depth_summaries(lang)
    pairs = [l.strip().split(">") for l in open(PAIRS) if l.strip()]
    lin_of = {b: (b, al) for b, al in pairs}

    print("\n" + "=" * 78)
    print("DEPTH x EXIT   (%s)" % lang)
    print("=" * 78)
    print("  marker cells (%s): %d over %d models, %d groups"
          % (lang, len(cells), len({m for m, _, _ in cells}),
             len({g for _, g, _ in cells})))
    print("  lens (lineage, group) cells: %d over %d lineages"
          % (len(dep), len({l for l, _ in dep})))

    #: A LINEAGE IS ITS OWN GROUP SET. Only groups both instruments see AND
    #: where both arms carry all three roles -- fixed once per lineage and used
    #: for every pooled rate, so nothing being subtracted differs in
    #: composition from anything else.
    L = {}
    for lin, (base, aligned) in sorted(lin_of.items()):
        gs = sorted(g for (l, g) in dep if l == lin
                    and complete(cells, base, g) and complete(cells, aligned, g))
        if len(gs) < 4:
            continue
        d = [dep[(lin, g)] for g in gs]
        rec = {"lineage": lin, "aligned": aligned, "n_groups": len(gs),
               "n_passages": sum(cells[(m, g, r)][0] for m in (base, aligned)
                                 for g in gs
                                 for r in ("POLE_A", "POLE_B", "BOTH")),
               "top_share": st.median([x["top_share"] for x in d]),
               "argmax": st.median([x["argmax"] for x in d]),
               "gap_total": st.median([x["gap_total"] for x in d]),
               "d_final": st.median([x["d_final"] for x in d
                                     if x["d_final"] is not None] or [float("nan")])}
        for t in TNAMES:
            eb = excess(cells, base, gs, t)
            ea = excess(cells, aligned, gs, t)
            rec["base_" + t] = eb
            rec["aligned_" + t] = ea
            rec["d_" + t] = None if (eb is None or ea is None) else ea - eb
        L[lin] = (rec, gs)
    lins = sorted(L)
    print("  JOINED: %d lineages; groups per lineage %s; %s passages"
          % (len(lins),
             "-".join(str(x) for x in (min(L[l][0]["n_groups"] for l in lins),
                                       max(L[l][0]["n_groups"] for l in lins))),
             format(sum(L[l][0]["n_passages"] for l in lins), ",")))
    if len(lins) < 5:
        print("  too few lineages to test")
        return

    with open(OUT, "w", newline="") as fh:
        recs = [L[l][0] for l in lins]
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        for r in recs:
            w.writerow({k: ("" if v is None else
                            ("%.6g" % v if isinstance(v, float) else v))
                        for k, v in r.items()})
    print("  wrote %s" % os.path.relpath(OUT, ROOT))

    # ---- 0. is there anything to predict?
    print("\n--- 0. THE EXIT CONTRAST ITSELF ---")
    print("  excess = rate(BOTH) - mean(rate(POLE_A), rate(POLE_B)), points,")
    print("  from POOLED counts per lineage arm. Two questions, not one:")
    print("    'exc' -- does CONTRADICTION exit more than a single pole? (level)")
    print("    'd'   -- does ALIGNMENT change that? (the paired contrast)")
    print("  %-10s %9s %9s %9s %8s %9s"
          % ("type", "base exc", "algn exc", "d", "lins+", "p"))
    keep = []
    for t in TNAMES:
        ds = [L[l][0]["d_" + t] for l in lins if L[l][0]["d_" + t] is not None]
        bs = [L[l][0]["base_" + t] for l in lins if L[l][0]["base_" + t] is not None]
        as_ = [L[l][0]["aligned_" + t] for l in lins if L[l][0]["aligned_" + t] is not None]
        if not ds:
            continue
        n, k, p = sign_test(ds)
        star = " ***" if p < 0.01 else (" *" if p < 0.05 else "")
        print("  %-10s %+9.3f %+9.3f %+9.3f  %3d/%-3d %9.3g%s"
              % (t, st.median(bs), st.median(as_), st.median(ds), k, n, p, star))
        if len(ds) >= 15:
            keep.append(t)
    print("  types on >= 15 lineages, carried forward: %s" % (", ".join(keep) or "none"))
    print("\n  and the LEVEL, over every %s passage regardless of lineage." % lang)
    print("  THIS IS A POOL AND POOLS LIE HERE: on en the pooled ANY-EXIT excess")
    print("  is -2.02 pp while the MEDIAN MODEL is at -0.48, because three Qwen")
    print("  models sit at -19.5, -15.0 and -14.5. Read the per-unit tests above,")
    print("  not this row.")
    for t in ("ANY-EXIT", "E-QUIZ", "E-QA", "REFUSAL"):
        tot = collections.Counter()
        for (m, g, r), (n, c) in cells.items():
            tot[r + "_n"] += n
            tot[r] += c[t]
        rates = {r: 100.0 * tot[r] / tot[r + "_n"] for r in ("POLE_A", "POLE_B", "BOTH")}
        print("    %-9s  A %5.2f%%  B %5.2f%%  BOTH %5.2f%%   excess %+6.3f pp"
              % (t, rates["POLE_A"], rates["POLE_B"], rates["BOTH"],
                 rates["BOTH"] - (rates["POLE_A"] + rates["POLE_B"]) / 2))
    print("    (n = %s / %s / %s passages)"
          % tuple(format(sum(cells[k][0] for k in cells if k[2] == r), ",")
                  for r in ("POLE_A", "POLE_B", "BOTH")))

    # ---- 1. the primary
    print("\n--- 1. PRIMARY: does the DEPTH of divergence predict the exit change? ---")
    print("  Spearman over %d LINEAGES. top_share high = the arms part LATE (a" % len(lins))
    print("  gate); low = they part early (re-routing). A late gate that causes")
    print("  the exit predicts POSITIVE rho.")
    print("  %-10s %-12s %7s %9s %5s" % ("type", "depth stat", "rho", "p", "n"))
    for t in keep:
        for stat in ("top_share", "argmax", "gap_total"):
            xy = [(L[l][0][stat], L[l][0]["d_" + t]) for l in lins
                  if L[l][0]["d_" + t] is not None]
            rho, p, n = spearman([a for a, _ in xy], [b for _, b in xy])
            star = " ***" if p == p and p < 0.01 else (" *" if p == p and p < 0.05 else "")
            print("  %-10s %-12s %+7.3f %9.3g %5d%s" % (t, stat, rho, p, n, star))

    # ---- 2. the baseline
    print("\n--- 2. BASELINE: does the OUTPUT-level JS contrast predict it? ---")
    print("  d_final = ratio_aligned - ratio_base at depth 1.0. If the ratio and")
    print("  the markers do not agree HERE, they are not two views of one event")
    print("  and 1 was comparing unrelated things.")
    print("  %-10s %7s %9s %5s" % ("type", "rho", "p", "n"))
    for t in keep:
        xy = [(L[l][0]["d_final"], L[l][0]["d_" + t]) for l in lins
              if L[l][0]["d_" + t] is not None and L[l][0]["d_final"] == L[l][0]["d_final"]]
        rho, p, n = spearman([a for a, _ in xy], [b for _, b in xy])
        star = " ***" if p == p and p < 0.01 else (" *" if p == p and p < 0.05 else "")
        print("  %-10s %+7.3f %9.3g %5d%s" % (t, rho, p, n, star))

    # ---- 3. within lineage: split the GROUPS by how late the arms part
    print("\n--- 3. WITHIN LINEAGE: the LATE-parting groups against the EARLY ---")
    print("  Per lineage, split its groups at the median top_share and pool the")
    print("  counts within each half. Contrast = d(excess) in the LATE half minus")
    print("  d(excess) in the EARLY half. Everything model-level is held fixed by")
    print("  construction; only the depth profile of the group varies.")
    print("  %-10s %10s %10s %9s %8s %9s"
          % ("type", "late d", "early d", "late-early", "lins+", "p"))
    for t in keep:
        diffs = []
        for lin in lins:
            rec, gs = L[lin]
            base, aligned = lin_of[lin]
            sh = sorted(gs, key=lambda g: dep[(lin, g)]["top_share"])
            h = len(sh) // 2
            early, late = sh[:h], sh[len(sh) - h:]
            if len(early) < 2 or len(late) < 2:
                continue
            vals = []
            for half in (late, early):
                eb, ea = excess(cells, base, half, t), excess(cells, aligned, half, t)
                vals.append(None if eb is None or ea is None else ea - eb)
            if None in vals:
                continue
            diffs.append((vals[0], vals[1]))
        if len(diffs) < 5:
            print("  %-10s %10s" % (t, "too few"))
            continue
        d = [a - b for a, b in diffs]
        n, k, p = sign_test(d)
        star = " ***" if p < 0.01 else (" *" if p < 0.05 else "")
        print("  %-10s %+10.3f %+10.3f %+9.3f  %3d/%-3d %9.3g%s"
              % (t, st.median([a for a, _ in diffs]),
                 st.median([b for _, b in diffs]), st.median(d), k, n, p, star))


if __name__ == "__main__":
    main()
