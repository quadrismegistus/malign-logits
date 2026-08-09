"""L3 GEOMETRY: is the interior excursion contradiction-specific or conjunction-general?

    uv run python l3_geometry.py

Plan: `meta/M02_frame_exit/plans/l3_geometry.md`, posted [5153], population
verified by the pen at [5154]. Dispatched at [5152].

THE MEASURE, per (pair, group, arm, layer), with h_A = pole_a, h_B = pole_b:

    t(X)     = (h_X - h_B) . (h_A - h_B) / |h_A - h_B|^2
    resid(X) = |off-axis part| / |h_A - h_B|

for X in {both, control_a, control_b, both_matched}. t is where X sits on the
pole axis -- 0.5 between, 1 at A, 0 at B -- and resid is how far off that axis,
which is what frame exit looks like geometrically.

THE PRIMARY IS WITHIN-LAYER AND THAT IS DELIBERATE. `hidden_states[-1]` is
post-norm and every other entry is pre-norm; RMSNorm's learned weight is a
DIAGONAL MAP, not a scalar, so t is invariant within a layer and NOT across that
seam. Comparing an interior t to the final-layer t compares two spaces. The
primary contrast is roles at ONE layer of ONE model, which needs no depth
alignment and is immune. Cross-layer reads, including "the output repairs the
excursion", are secondary and carry the caveat.

WHAT THE PILOT FOUND AND COULD NOT ASK. Two checkpoints, one triplet, no
controls: base holds t ~ 0.42-0.46 through the stack, DPO hauls to 0.18 at layer
7 and reconverges by the top. The BOTH prompt lexically contains both pole words,
so t ~ 0.5 may be a fact about two-adjective sentences. The controls -- same-side
near-synonym conjunctions -- are what turns that into a question with an answer.

THE PRIOR, BOTH BRANCHES, FIXED BEFORE READING: if the excursion is about
contradiction, BOTH excurses and the controls do not. If the controls excurse as
much, the pole-pull is about CONJUNCTION, the pilot measured grammar, and M02's
interior reading changes. And `f11_reason`/`_zh` runs BESIDE as the negative
control on poles known not to separate: if the effect appears there too it is not
about contradiction whatever the controls say. Its result outranks the primary's.
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))

SRC = os.path.join(ROOT, "data", "f11_quintuplets.json")
ROLES = ("both", "control_a", "control_b", "both_matched")
POLES = ("pole_a", "pole_b")
NEGATIVE_CONTROL = {"f11_reason", "f11_reason_zh"}


def index_residuals():
    """model -> {prompt: (path, hidden_row, shape)}, over every residual dir."""
    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "data", "f11_twp*"))
                  if os.path.isdir(d))
    idx = collections.defaultdict(dict)
    for d in dirs:
        for p in sorted(glob.glob(d + "/*.jsonl")):
            h = p[:-len(".jsonl")] + ".hidden.f32"
            if not (os.path.exists(h) and os.path.getsize(h)):
                continue
            for line in open(p):
                r = json.loads(line)
                if r.get("hidden_row") is None:
                    continue
                #: FIRST WRITER WINS, and the dirs are sorted, so a prompt present
                #: in both the first fleet and the delta is read from one place.
                idx[r["model"]].setdefault(
                    r["prompt"], (h, r["hidden_row"], tuple(r["hidden_shape"])))
    return dirs, idx


def main():
    import numpy as np
    import pandas as pd

    dirs, idx = index_residuals()
    Q = json.load(open(SRC))
    live = [q for q in Q["quintuplets"] if q["status"] != "RETIRED"]
    pairs = json.load(open(os.path.join(ROOT, "data", "base_aligned_pairs.json")))

    print("residual directories scanned: %s"
          % ", ".join(os.path.basename(d) for d in dirs))
    print("models with residuals: %d   live groups: %d   roster pairs: %d\n"
          % (len(idx), len(live), len(pairs)))

    rows, roster = [], []
    #: SPEC-TIME counts, so the coverage drops are visible and not inferred
    #: (pen's request at [5154]). spec = what the source of record defines;
    #: read = what was actually on disk.
    spec = collections.Counter()
    read = collections.Counter()

    for pr in pairs:
        b, a = pr["base"], pr["aligned"]
        if b not in idx or a not in idx:
            continue
        got = collections.Counter()
        cache = {}
        for arm, mid in (("base", b), ("aligned", a)):
            for q in live:
                #: the pole axis must exist in THIS arm or the cell is not readable
                need = [q.get(k) for k in POLES]
                if not all(need) or not all(x in idx[mid] for x in need):
                    continue
                vec = {}
                ok = True
                for k in POLES:
                    p, n, sh = idx[mid][q[k]]
                    w = int(np.prod(sh))
                    v = np.fromfile(p, dtype=np.float32, count=w, offset=n * w * 4)
                    if v.size != w:
                        ok = False
                        break
                    vec[k] = v.reshape(sh)
                if not ok:
                    continue
                ax = vec["pole_a"] - vec["pole_b"]            # (n_layers, d)
                n2 = (ax * ax).sum(1)                          # per layer
                #: THE DEGENERACY GUARD'S RAW MATERIAL, stored not applied here.
                #: t divides by |h_A - h_B|^2, so when the two poles are nearly
                #: identical at a layer the ratio explodes: croissant/f11_captive
                #: reaches t = -40. Trimming on |t| would select on the outcome,
                #: so what is stored is the DENOMINATOR's scale relative to the
                #: states, and the guard is applied at analysis time on that.
                sep = np.sqrt(n2) / (
                    0.5 * (np.linalg.norm(vec["pole_a"], axis=1)
                           + np.linalg.norm(vec["pole_b"], axis=1)))
                for role in ROLES:
                    s = q.get(role)
                    if not s or s not in idx[mid]:
                        continue
                    p, n, sh = idx[mid][s]
                    w = int(np.prod(sh))
                    v = np.fromfile(p, dtype=np.float32, count=w, offset=n * w * 4)
                    if v.size != w:
                        continue
                    hx = v.reshape(sh) - vec["pole_b"]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        t = (hx * ax).sum(1) / n2
                        off = hx - t[:, None] * ax
                        rs = np.linalg.norm(off, axis=1) / np.sqrt(n2)
                    got[role] += 1
                    for L in range(sh[0]):
                        rows.append((pr["family"], b, a, arm, q["group"],
                                     q["language"], role, L, sh[0],
                                     float(t[L]), float(rs[L]), float(sep[L]),
                                     q["group"] in NEGATIVE_CONTROL))
                    cache[(arm, q["group"], role)] = True
        for role in ROLES:
            spec[role] += sum(1 for q in live if q.get(role))
            read[role] += got[role] // 2 if got[role] else 0
        roster.append((pr["family"], b, a, dict(got)))

    D = pd.DataFrame(rows, columns=["family", "base", "aligned", "arm", "group",
                                    "language", "role", "layer", "n_layers",
                                    "t", "resid", "pole_sep", "negative_control"])
    out = os.path.join(CAMP, "results", "l3_geometry.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    D.to_parquet(out, index=False)

    print("=" * 84)
    print("COVERAGE: SPEC-TIME AGAINST WHAT WAS ACTUALLY READ  (pen's request, [5154])")
    print("=" * 84)
    print("   %-14s %14s %14s" % ("role", "spec (x pairs)", "cells read"))
    for role in ROLES:
        print("   %-14s %14d %14d" % (role, spec[role], read[role]))
    print("\n   rows written: %d   pairs read: %d   models touched: %d"
          % (len(D), len(roster), D[["base", "aligned"]].stack().nunique()))
    print("   layer depths present: %s"
          % sorted(set(D.n_layers.tolist()))[:12])
    print("   wrote %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
