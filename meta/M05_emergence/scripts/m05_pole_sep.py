"""Secondary 4, the geometry half: pole separation along the M05 ladder.

    uv run python m05_pole_sep.py --validate    # prove the arithmetic first
    uv run python m05_pole_sep.py --run

Commissioned by registrar [5422]. The ratio half is the pen's (m05_ratio.py);
this is the column it joins to.

## WHY THIS COLUMN EXISTS

[5378]: F24 reads ratio ABOVE 1.0 on early pretraining checkpoints as NOISE.
The calibration in `contradiction_ratio_has_no_null.md` puts 1.006 at
NEUTRALIZATION. **Those are the same number meaning opposite things**, and the
ratio cannot tell them apart:

    early pretraining   cannot represent the poles yet  -> AB unrelated -> ~1
    aligned             can represent them, exits frame -> AB unrelated -> ~1

`pole_sep` separates them, because the two mechanisms differ exactly there: if
the poles are not yet distinct it is SMALL; if they are well separated and AB
is simply elsewhere it is LARGE.

## THE PREDICTIONS, REGISTERED BEFORE THIS RAN ([5378], verbatim)

    on the base ladder, ratio should approach 1.0 from ABOVE at early rungs
    WITH pole_sep near its floor, and any late-rung or SFT-arm approach to 1.0
    should come with pole_sep AT OR ABOVE its endpoint value. If early rungs
    show ratio ~1 with pole_sep already large, the noise reading is wrong and
    F24's developmental story needs revisiting. If late rungs show ratio ~1
    with pole_sep small, the frame-exit reading is wrong and mine does.

The ratio column was posted BEFORE this column existed ([5423]: step0 1.267,
stage1-final 0.754, SFT 0.808 -> 0.877, DPO 0.863, RLVR 0.869), so the test is
a test and not a fit. step0 at 1.267 is the cell my reading is staked on.

## THE ARITHMETIC IS `l3_geometry.py`'s, AND IS PROVED SO RATHER THAN CLAIMED

    ax       = h_A - h_B                          per layer
    pole_sep = |ax| / (0.5 * (|h_A| + |h_B|))
    t        = ((h_X - h_B) . ax) / |ax|^2
    resid    = |off-axis part of (h_X - h_B)| / |ax|

That code is inline in `l3_geometry.main()` and not callable, so it is copied
here -- and **a transcription is not an implementation**, so `--validate`
recomputes the base main from these bytes and diffs every value against
`results/l3_geometry.parquet`, which was produced by the original. The copy is
not used until that diff is zero to floating point.

## THREE THINGS FROM MY OWN [5373] CORRECTION, WHICH IS WHY THIS COLUMN IS
## TRUSTWORTHY AND `t` ALONE IS NOT

  `t` IS A ONE-DIMENSIONAL SHADOW. resid runs 0.954 for BOTH and exceeds 1.0
  in 31.5% of cells -- BOTH is not between the poles, it is off in another
  direction whose projection lands near the middle. **Both and neither cast
  the same shadow.** So resid travels beside t on every row, never behind it.

  THE DEGENERACY GUARD GOES ON THE DENOMINATOR, NEVER ON `t`. t divides by
  |h_A - h_B|^2, so near-identical poles make it explode (croissant/f11_captive
  reaches t = -40). Trimming on |t| would select on the outcome. The guard is
  `pole_sep >= 0.02`, applied at analysis time, on the denominator's scale.

  A LADDER IS THE THING THAT SETTLES THE ARROW. [5373] found pole_sep change
  and superposition change correlated at rho -0.420 (p=0.0041, 45 lineages)
  across a base/aligned cross-section, and said plainly that the arrow was NOT
  established because both may track how much alignment happened. 95 rungs is
  what separates those.
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)

POP = os.path.join(ROOT, "data", "m05_checkpoint_population.json")
QUINT = os.path.join(ROOT, "data", "f11_quintuplets.json")
L3 = os.path.join(ROOT, "meta", "M02_frame_exit", "results", "l3_geometry.parquet")
OUT = os.path.join(CAMP, "results", "m05_pole_sep.csv")

ROLES = ("both", "control_a", "control_b")
SEP_FLOOR = 0.02          #: the degeneracy guard, on the DENOMINATOR


def geometry(h_a, h_b, h_x):
    """l3_geometry.main()'s arithmetic, verbatim. Validated, not trusted."""
    ax = h_a - h_b
    n2 = (ax * ax).sum(1)
    sep = np.sqrt(n2) / (0.5 * (np.linalg.norm(h_a, axis=1)
                                + np.linalg.norm(h_b, axis=1)))
    hx = h_x - h_b
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (hx * ax).sum(1) / n2
        off = hx - t[:, None] * ax
        rs = np.linalg.norm(off, axis=1) / np.sqrt(n2)
    return t, rs, sep


def index_hidden(prefer="fleet"):
    """{model: {prompt: (path, row, shape)}} over the M05 sidecars AND the
    wider store.

    THE BASE MAIN IS SHORT IN THE FLEET'S OWN OUTPUT and that is not a defect
    of the fleet: malign re-scored only cells whose hidden sidecar was absent,
    so `allenai/Olmo-3-1025-7B` holds 59 of the 90 QUINT_EN texts under
    data/raw/m05 and the other 31 live in the pre-existing sidecars. 8,519
    distinct cells in the fleet output + 31 = the declared 8,550. Reading only
    m05 would silently drop a third of the LADDER'S OWN REFERENCE POINT.
    """
    from meta.M02_frame_exit.scripts import lens_ratio_by_layer as LENS  # noqa
    idx = LENS.scan_hidden()
    n_wide = sum(len(v) for v in idx.values())
    m05 = collections.defaultdict(dict)
    for jf in sorted(glob.glob(os.path.join(ROOT, "data/raw/m05/*/*.jsonl"))):
        base = jf[:-6] + ".hidden.f32"
        for line in open(jf):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("hidden_row") is None:
                continue
            m05[d["model"]][d["prompt"]] = (base, d["hidden_row"],
                                            tuple(d["hidden_shape"]))
    #: TWO STORES HOLD THE SAME 59 CELLS, SCORED ON DIFFERENT HARDWARE, AND
    #: WHICH ONE YOU READ IS A DESIGN CHOICE RATHER THAN A DETAIL.
    #:
    #: The fleet re-scored the three `main` checkpoints on a Quadro RTX 8000
    #: where the originals were CUDA. Measured on the base main, that moves
    #: pole_sep by up to 6.8e-4 and t by up to 1.2e-2 -- 0.2% of pole_sep's
    #: ~0.31, against the 8.6% base->aligned move the cross-section reports.
    #: Small, real, and NOT arithmetic: reading the wider store alone
    #: reproduces `l3_geometry.parquet` to EXACTLY zero.
    #:
    #: `prefer` decides. FLEET is the default because the unit here is a
    #: LADDER: 94 of 95 rungs are Quadro-scored, so a CUDA-scored base main
    #: would differ from its own neighbours by hardware rather than by
    #: training, and that difference sits at the anchor every other rung is
    #: read against. WIDER exists so the arithmetic can be validated against
    #: the instrument that produced the committed file.
    #:
    #: THE COST IS STATED RATHER THAN AVOIDED: under FLEET the base main rung
    #: is 59 Quadro cells and 31 CUDA cells, the only mixed-provenance rung on
    #: the ladder. Under WIDER it is 90 CUDA cells among 94 Quadro rungs.
    #: There is no all-one-hardware option and pretending otherwise would be
    #: the choice that hides.
    if prefer == "fleet":
        for m, v in m05.items():
            idx.setdefault(m, {}).update(v)
    print("hidden index: %d models, %d prompt-entries from the wider store, "
          "%d checkpoints from the fleet, preference=%s"
          % (len(idx), n_wide, len(m05), prefer))
    return idx


def read(idx, model, prompt):
    e = idx.get(model, {}).get(prompt)
    if e is None:
        return None
    p, n, sh = e
    w = int(np.prod(sh))
    v = np.fromfile(p, dtype=np.float32, count=w, offset=n * w * 4)
    return v.reshape(sh) if v.size == w else None


def quints(lang="en"):
    Q = json.load(open(QUINT))["quintuplets"]
    return [g for g in Q if g["status"] != "RETIRED" and g["language"] == lang]


def validate(idx):
    """Prove the copied arithmetic equals the committed instrument.

    Recomputes the BASE MAIN from these bytes and diffs against
    `l3_geometry.parquet`. Anything but zero to floating point means the copy
    is a different instrument wearing the same column names, and the run does
    not proceed.
    """
    import pandas as pd
    D = pd.read_parquet(L3)
    D = D[(D.base == "allenai/Olmo-3-1025-7B") & (D.arm == "base")
          & (D.role == "both")]
    print("committed rows for the base main, role=both: %d over %d groups"
          % (len(D), D.group.nunique()))
    worst = {"t": 0.0, "resid": 0.0, "pole_sep": 0.0}
    n = 0
    for g in quints():
        sub = D[D.group == g["group"]].sort_values("layer")
        if sub.empty:
            continue
        H = {r: read(idx, "allenai/Olmo-3-1025-7B", g[r])
             for r in ("pole_a", "pole_b", "both")}
        if any(v is None for v in H.values()):
            continue
        t, rs, sep = geometry(H["pole_a"], H["pole_b"], H["both"])
        if len(t) != len(sub):
            print("  layer count differs for %s: %d vs %d"
                  % (g["group"], len(t), len(sub)))
            continue
        for col, mine in (("t", t), ("resid", rs), ("pole_sep", sep)):
            d = np.nanmax(np.abs(np.asarray(sub[col], float) - mine))
            worst[col] = max(worst[col], float(d))
        n += 1
    print("groups compared: %d" % n)
    for k, v in worst.items():
        print("  max |mine - committed| on %-9s %.3g" % (k, v))
    ok = n > 0 and all(v < 1e-5 for v in worst.values())
    print("VALIDATION %s" % ("PASSES" if ok else "FAILS -- not running"))
    return ok


def run(idx):
    ck = json.load(open(POP))["checkpoints"]
    G = quints()
    print("checkpoints %d, en quintuplet groups %d" % (len(ck), len(G)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    import csv
    n = miss = 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model_id", "revision", "role_ck", "step", "checkpoint",
                    "group", "role", "layer", "n_layers", "t", "resid",
                    "pole_sep"])
        for c in ck:
            rev = c.get("revision")
            key = (c["model_id"] if (not rev or rev == "main")
                   else "%s@%s" % (c["model_id"], rev))
            for g in G:
                ha = read(idx, key, g["pole_a"])
                hb = read(idx, key, g["pole_b"])
                if ha is None or hb is None:
                    miss += 1
                    continue
                for role in ROLES:
                    txt = g.get(role)
                    if not txt:
                        continue
                    hx = read(idx, key, txt)
                    if hx is None:
                        miss += 1
                        continue
                    t, rs, sep = geometry(ha, hb, hx)
                    for L in range(len(t)):
                        w.writerow([c["model_id"], rev or "main", c["role"],
                                    c.get("step"), key, g["group"], role, L,
                                    len(t), "%.6g" % t[L], "%.6g" % rs[L],
                                    "%.6g" % sep[L]])
                        n += 1
            print("  %-46s %s" % (key.split("/")[-1][:44], c["role"]), flush=True)
    print("\nwrote %d rows -> %s   (cells with no hidden state: %d)"
          % (n, os.path.relpath(OUT, ROOT), miss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    #: VALIDATION READS THE WIDER STORE, because it is proving the ARITHMETIC
    #: against the file the original produced, and the fleet's re-scores would
    #: make a provenance difference look like a formula difference -- which is
    #: exactly what they did on the first attempt.
    if a.validate:
        idx = index_hidden(prefer="wider")
        return 0 if validate(idx) else 1
    if a.run:
        if not validate(index_hidden(prefer="wider")):
            raise SystemExit("arithmetic validation failed; refusing to run")
        return run(index_hidden(prefer="fleet"))
    ap.print_help()


if __name__ == "__main__":
    sys.exit(main() or 0)
