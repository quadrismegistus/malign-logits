#!/usr/bin/env python
"""h2_tables.py — the six H2 tables, built FROM the shards.

    scripts/h2_tables.py              build data/h2_depth/tables/*.parquet
    scripts/h2_tables.py --summary    read them back and describe

Downstream of the sweep on purpose. The forward passes are the expensive,
unrepeatable part; a schema that lives below them can change as often as it
needs to without re-running six hours of compute. Nothing here loads a model.

## SIX GRAINS, LONG NOT WIDE

    depth_pairs        (pair)                        weights + head survey
    depth_blocks       (pair, layer)                 ||dW||
    depth_heads        (pair, layer, head)           ||dW||
    depth_cells        (pair, prompt, rule)          <- THE UNIT OF ANALYSIS
    depth_cell_layers  (pair, prompt, rule, layer)   ||dh||, repr recovery
    depth_patch        (pair, prompt, rule, dir, k)  weight-patch recovery

A new measure is a ROW, never a column.

## `d` IS THE RAW DIFFERENCE

`d = recovery(bottom N-2) - recovery(top 2)`, both raw. **The ceiling is a
COLUMN and never a divisor**: it is shared by both terms, and a shared
denominator CANCELS IN A RATIO BUT SCALES IN A DIFFERENCE --
`d_norm = (A-B)/C = d_raw/C` -- so dividing by a ceiling that passes through
zero is not normalisation, it is a variance amplifier with a sign-flip
singularity (registrar [5242]). Declared at [5241], after one pair had been
seen, with the direction of the bias stated: raw is the conservative branch.

## `ceiling_class` IS FOUR LEVELS, NOT ONE CONTINUOUS COLUMN

lacan [5243].3: **a near-zero ceiling and a NEGATIVE ceiling are different
objects and must not share a column semantics.**

    failed    ceiling <= 0    the all-blocks-aligned construction scored WORSE
                              than the thing it is meant to bound. Not a small
                              ceiling -- the construction failing on that cell.
    low       0 < c < 0.5     little of the effect is reachable through blocks
                              alone; plausibly it lives in the HEAD, which is
                              held at BASE by design. A HYPOTHESIS, not a
                              reading.
    normal    0.5 <= c <= 1.2
    over      c > 1.2         blocks alone overshoot the aligned model.

Averaging a `failed` cell together with a `normal` one is the error this column
exists to prevent, and it cannot be prevented after the fact by a filter,
because the filter would have to know which was which.
"""
import argparse, glob, hashlib, json, os, platform, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "data", "h2_depth", "tables")
SH = os.path.join(ROOT, "data", "h2_depth")


def rows_of(path):
    if not os.path.exists(path): return []
    out, lines = [], open(path, "r", errors="replace").read().splitlines()
    for i, ln in enumerate(lines):
        if not ln.strip(): continue
        try:
            out.append(json.loads(ln))
        except Exception:
            if i == len(lines) - 1: break     # truncated tail from a kill
            raise SystemExit("CORRUPT %s line %d" % (path, i + 1))
    return out


def ceiling_class(c):
    """**FOUR LEVELS, NOT A THRESHOLD ON A CONTINUUM.** See the module docstring:
    a negative ceiling is the construction failing, not a small ceiling."""
    if c is None: return "missing"
    if c <= 0: return "failed"
    if c < 0.5: return "low"
    if c <= 1.2: return "normal"
    return "over"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)

    if a.summary:
        for f in sorted(glob.glob(os.path.join(OUT, "*.parquet"))):
            df = pd.read_parquet(f)
            print("\n=== %s  %d rows x %d cols" % (os.path.basename(f), len(df), df.shape[1]))
            print("   ", ", ".join(df.columns[:14]))
            if "d" in df:
                ok = df[df.ceiling_class != "failed"]
                print("    d (all):          med %+.3f  n %d  d>0 %d"
                      % (df.d.median(), len(df), int((df.d > 0).sum())))
                print("    d (ex-failed):    med %+.3f  n %d  d>0 %d"
                      % (ok.d.median(), len(ok), int((ok.d > 0).sum())))
                print("    ceiling_class:", dict(df.ceiling_class.value_counts()))
                print("    lens permitted:  %d of %d" % (int(df.lens_permitted.sum()), len(df)))
        return 0

    pop = json.load(open(os.path.join(ROOT, "data", "h2_depth_population.json")))
    meta = {p["prompt"]: p for p in pop["prompts"]}
    pair_meta = {p["aligned"]: p for p in pop["pairs"]}
    import torch, transformers
    stamp = {"prompt_sha": pop["prompt_list_sha256_16"],
             "pair_sha": pop["pair_list_sha256_16"],
             "torch": torch.__version__, "transformers": transformers.__version__,
             "platform": platform.platform()}

    # ---- the three weight tables, straight through
    wp = rows_of(os.path.join(SH, "weights.pairs.jsonl"))
    pairs_df = pd.DataFrame([{**{k: v for k, v in r.items() if k != "group"},
                              **{"wdelta_" + g: x for g, x in (r.get("group") or {}).items()},
                              **stamp} for r in wp])
    blocks_df = pd.DataFrame(rows_of(os.path.join(SH, "weights.blocks.jsonl")))
    heads_df = pd.DataFrame(rows_of(os.path.join(SH, "weights.heads.jsonl")))

    # ---- the three cell tables, from the sweep shards
    cells, cell_layers, patch = [], [], []
    for f in sorted(glob.glob(os.path.join(SH, "*.canonical.jsonl"))
                    + glob.glob(os.path.join(SH, "*.lens.jsonl"))):
        if os.path.basename(f).startswith("weights."): continue
        for r in rows_of(f):
            al, pr, rule, N = r["aligned"], r["prompt"], r["rule"], r["N"]
            pm = meta.get(pr, {})
            top, bot = r.get("top") or {}, r.get("bottom") or {}
            t2, bN2 = top.get("2"), bot.get(str(N - 2))
            c = r.get("ceiling")
            lens = r.get("lens") or {}
            cells.append({
                "aligned": al, "base": r["base"], "prompt": pr, "rule": rule,
                "prompt_id": pm.get("prompt_id"), "set": pm.get("set"),
                "stem": pm.get("stem"), "member": pm.get("member"),
                "domain": pm.get("domain"), "subdomain": pm.get("subdomain"),
                "stratum": pm.get("stratum"), "lang": pm.get("lang"),
                "N": N, "n_fallers": r["n_fallers"], "n_risers": r["n_risers"],
                "n_movers": r["n_fallers"] + r["n_risers"], "n_tokens": r["n_tokens"],
                #: THE PRIMARY. Raw difference; the ceiling is beside it, never under it.
                "recovery_top2": t2, "recovery_bottom_N2": bN2,
                "d": (bN2 - t2) if (t2 is not None and bN2 is not None) else None,
                "ceiling": c, "ceiling_class": ceiling_class(c),
                "repr_L50": r.get("repr_L50"),
                "repr_L50_frac": (r["repr_L50"] / N) if r.get("repr_L50") is not None else None,
                "lens_permitted": bool(lens.get("permitted")),
                "lens_worst_ratio": lens.get("worst_ratio"),
                "lens_first_outside": lens.get("first_outside"),
                "lens_n_flats": lens.get("n_flats"),
                "spec_sha": r.get("spec_sha"), **stamp})
            dh, rr = r.get("dh") or {}, r.get("repr_recovery") or {}
            for L in range(N + 1):
                k = str(L)
                if k not in dh and k not in rr: continue
                cell_layers.append({"aligned": al, "prompt": pr, "rule": rule,
                                    "layer": L, "layer_frac": L / N,
                                    "dh": dh.get(k), "repr_recovery": rr.get(k)})
            for direction, dd in (("top", top), ("bottom", bot)):
                for k, v in dd.items():
                    patch.append({"aligned": al, "prompt": pr, "rule": rule,
                                  "direction": direction, "k": int(k),
                                  "k_frac": int(k) / N, "recovery": v})

    cells_df = pd.DataFrame(cells)
    cl_df = pd.DataFrame(cell_layers)
    patch_df = pd.DataFrame(patch)

    tables = {"depth_pairs": pairs_df, "depth_blocks": blocks_df,
              "depth_heads": heads_df, "depth_cells": cells_df,
              "depth_cell_layers": cl_df, "depth_patch": patch_df}
    for name, df in tables.items():
        p = os.path.join(OUT, name + ".parquet")
        if len(df):
            df.to_parquet(p, index=False)
        print("  %-20s %7d rows  %s" % (name, len(df), "->" if len(df) else "(empty, not written)"))

    if len(cells_df):
        print("\n  PRIMARY, as it stands with %d pairs present:" % cells_df.aligned.nunique())
        ok = cells_df[(cells_df.ceiling_class != "failed") & cells_df.d.notna()]
        print("    d = recovery(bottom N-2) - recovery(top 2), RAW, per cell")
        print("    all cells     n %5d  med %+.3f  d>0 %d" %
              (int(cells_df.d.notna().sum()), cells_df.d.median(),
               int((cells_df.d > 0).sum())))
        print("    ex-'failed'   n %5d  med %+.3f  d>0 %d" %
              (len(ok), ok.d.median(), int((ok.d > 0).sum())))
        print("    ceiling_class %s" % dict(cells_df.ceiling_class.value_counts()))
        rev = ok[ok.d <= 0]
        print("    REVERSING cells (d<=0, excluding failed): %d" % len(rev))
        for _, x in rev.head(10).iterrows():
            print("       %-30s %-40s %+.3f"
                  % (x.aligned.split('/')[-1][:30], str(x.prompt)[:40], x.d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
