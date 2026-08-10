#!/usr/bin/env python
"""h2_weight_tables.py — the WEIGHT half of the H2 depth sweep.

    scripts/h2_weight_tables.py            run (resumes; skips pairs already done)
    scripts/h2_weight_tables.py --pairs Llama,gemma
    scripts/h2_weight_tables.py --report   read back what exists

Emits three of the six H2 tables as jsonl, one row per grain:

    data/h2_depth/weights.pairs.jsonl    1 row / pair    group deltas + head survey
    data/h2_depth/weights.blocks.jsonl   1 row / (pair, layer)
    data/h2_depth/weights.heads.jsonl    1 row / (pair, layer, head, proj)

## WHY THIS IS A SEPARATE PRODUCER FROM THE SWEEP

**A weight delta has no prompt in it.** It is a fact about two checkpoints, so
its grain is the PAIR and not the (pair, prompt) cell. Folding it into the
per-prompt loop would recompute one number 231 times per pair and store it
231 times, and -- the actual defect this script exists to fix -- the battery
did compute the per-block vector and then collapse it to four summary numbers
(mean, first third, last third, n_blocks), **discarding the vector it had just
built**. `depth_blocks` and `depth_heads` would have come back empty from a
six-hour run that had the numbers in memory and dropped them.

It also needs NO forward passes, NO model load and NO GPU: `safetensors` opens
a shard and returns one tensor. So it runs beside the sweep rather than inside
it, and a failure here costs minutes.

## USES `weightdelta.weight_delta`, NOT A PRIVATE COPY

The battery carried its own reimplementation whose regex `layers?\\.(\\d+)\\.`
matched only in-block tensors, so `lm_head`, `embed_tokens` and the final norm
were **not measured at all** -- and those are exactly the tensors the "alignment
only changes the readout" claim is about. `weightdelta._group_of` also carries a
fix this copy never had: `lm_head.weight` fell into a catch-all labelled `norm`
and read 0.0504 where the true value is 0.0064.

## THE HEAD SURVEY IS A NECESSARY-NOT-SUFFICIENT COLUMN

`head_rel_diff` is reported per pair because the lens gate needs it, but a small
head difference does NOT license a cross-read: Amber's head moved LESS than
Llama's (3.5e-2 vs 6.6e-2) and Amber's cross-read was the one that blew up, at
5x out of distribution, because the two arms' STATES were far apart rather than
their heads. The distributional gate is computed in the sweep, per cell, and
this column never substitutes for it.
"""
import argparse, hashlib, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

OUTDIR = os.path.join(ROOT, "data", "h2_depth")
POP = os.path.join(ROOT, "data", "h2_depth_population.json")
T_PAIRS = os.path.join(OUTDIR, "weights.pairs.jsonl")
T_BLOCKS = os.path.join(OUTDIR, "weights.blocks.jsonl")
T_HEADS = os.path.join(OUTDIR, "weights.heads.jsonl")


def read_jsonl(p):
    if not os.path.exists(p): return []
    out, lines = [], open(p, "r", errors="replace").read().splitlines()
    for i, ln in enumerate(lines):
        if not ln.strip(): continue
        try:
            out.append(json.loads(ln))
        except Exception:
            if i == len(lines) - 1:
                #: a killed write leaves a partial final line with no newline;
                #: it must be CUT, not merely skipped, or the next append
                #: concatenates onto it and corrupts the middle of the file
                _rewrite(p, out)
                return out
            raise SystemExit("CORRUPT %s at line %d of %d (not the last line)"
                             % (p, i + 1, len(lines)))
    return out


def _rewrite(p, rows):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def append(p, rows):
    with open(p, "a") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
        f.flush(); os.fsync(f.fileno())


def head_numbers(base, aligned):
    """(rel_diff, rows_moved_frac, key, tied) or (None, ...) with a reason.

    `rows_moved_frac` is the share of vocabulary ROWS that changed at all --
    the question "did the fine-tune touch the readout everywhere or in a few
    places", which a single norm cannot answer. **Tied embeddings are the
    common case and must not read as a missing head**: where `lm_head.weight`
    is absent the model ties input and output embeddings.
    """
    import torch
    from head_frozen_survey import head_tensor
    tb, kb, tie = head_tensor(base)
    ta, ka, _ = head_tensor(aligned)
    if tb is None or ta is None:
        return None, None, (kb if tb is None else ka), None
    if tb.shape != ta.shape:
        return None, None, "shape %s vs %s" % (tuple(tb.shape), tuple(ta.shape)), None
    rel = float((ta - tb).norm() / tb.norm())
    moved = float((( ta - tb).abs().sum(dim=1) > 0).float().mean())
    return rel, moved, kb, bool(tie)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help="comma-separated aligned-name substrings")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-heads", action="store_true",
                    help="skip the per-head table (the slow slice)")
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(POP):
        raise SystemExit("run scripts/h2_depth_run.py --plan first (writes %s)"
                         % os.path.relpath(POP, ROOT))
    pop = json.load(open(POP))
    pairs = pop["pairs"]
    if a.pairs:
        keys = [k.strip().lower() for k in a.pairs.split(",")]
        pairs = [p for p in pairs if any(k in p["aligned"].lower() for k in keys)]

    have_pairs = read_jsonl(T_PAIRS)
    have_blocks = read_jsonl(T_BLOCKS)
    have_heads = read_jsonl(T_HEADS)
    done = {r["aligned"] for r in have_pairs}

    if a.report:
        print("weights.pairs  %4d rows" % len(have_pairs))
        print("weights.blocks %4d rows" % len(have_blocks))
        print("weights.heads  %4d rows" % len(have_heads))
        print("pairs done: %d of %d" % (len(done), len(pairs)))
        #: **A TIED MODEL HAS NO `head` GROUP AND THAT IS A STRUCTURE, NOT A
        #: FAILED MEASUREMENT.** Where `lm_head.weight` is absent the model ties
        #: input and output embeddings, so the readout IS `embed`. Printing
        #: `nan` there reads as "we could not measure it", which is the opposite
        #: of the truth -- it was measured, under the name the model uses.
        print("  %-42s %9s %8s %8s %7s" % ("aligned", "head/emb", "attn", "mlp", "rows"))
        for r in sorted(have_pairs, key=lambda x: -((x.get("group") or {}).get("head")
                                                    or (x.get("group") or {}).get("embed") or 0)):
            g = r.get("group") or {}
            tied = r.get("head_tied")
            hv = g.get("head") if g.get("head") is not None else g.get("embed")
            print("  %-42s %9s %8.4f %8.4f %7s%s"
                  % (r["aligned"][:42],
                     ("%.4f" % hv) if hv is not None else "-",
                     g.get("attn", float('nan')), g.get("mlp", float('nan')),
                     ("%.3f" % r["head_rows_moved_frac"])
                     if r.get("head_rows_moved_frac") is not None else "-",
                     "  (tied)" if tied else ""))
        return 0

    from malign_logits.weightdelta import weight_delta
    todo = [p for p in pairs if p["aligned"] not in done]
    print("H2 WEIGHT TABLES")
    print("  pairs %d, %d already done, %d owed" % (len(pairs), len(done), len(todo)))
    print("  NO forward passes, NO model load, NO GPU -- safetensors reads only\n")

    t0 = time.time()
    for i, p in enumerate(todo, 1):
        b, al = p["base"], p["aligned"]
        t1 = time.time()
        try:
            #: **`x or {}` DESTROYS THE TYPE ON AN EMPTY RESULT.** An empty
            #: Delta is falsy, so the idiom silently swaps it for a plain dict
            #: and `.skipped` disappears -- which is how a metadata-only
            #: checkpoint surfaced as an AttributeError 11 pairs into a run
            #: instead of as the missing-weights fact it actually was.
            grp = weight_delta(b, al, by="group")
            blk = weight_delta(b, al, by="block")
            hd = None if a.no_heads else weight_delta(b, al, by="head")
            #: **AN EMPTY DELTA IS A REFUSAL, NOT A ZERO.** No tensor matched,
            #: so nothing was measured. Writing a row here would put a pair in
            #: the table with block_mean None and no blocks, and it would read
            #: downstream as a pair whose weights did not move.
            if blk is None or not len(blk):
                print("  %-44s NO TENSORS MEASURED -- weights absent or "
                      "unmatched. Not writing a row." % al[:44])
                continue
            grp = grp if grp is not None else {}
            rel, moved, key, tied = head_numbers(b, al)
        except Exception as e:
            print("  %-44s FAILED %s: %s" % (al[:44], type(e).__name__, e))
            continue
        v = [blk[L] for L in sorted(blk)]
        n = len(v)
        row = {"base": b, "aligned": al, "arch": p.get("arch"),
               "n_blocks": p.get("n_blocks"), "family": p.get("family"),
               "group": {str(k): float(x) for k, x in grp.items()},
               "block_mean": (sum(v) / n) if n else None,
               "block_first_third": (sum(v[:n // 3]) / (n // 3)) if n >= 3 else None,
               "block_last_third": (sum(v[-(n // 3):]) / (n // 3)) if n >= 3 else None,
               "head_rel_diff": rel, "head_rows_moved_frac": moved,
               "head_key": key, "head_tied": tied,
               #: **A SKIPPED KEY IS AN ABSENT OBSERVATION, NOT A ZERO.** Tensors
               #: present in one arm and not the other, or of different shape,
               #: are counted here so a pair whose delta looks small because
               #: half its tensors were skipped cannot read as a pair whose
               #: weights barely moved.
               "skipped_group": getattr(grp, "skipped", None),
               "skipped_block": getattr(blk, "skipped", None),
               "skipped_head": getattr(hd, "skipped", None)}
        brows = [{"aligned": al, "base": b, "layer": int(L), "wdelta_block": float(x)}
                 for L, x in sorted(blk.items())]
        hrows = []
        for k, x in sorted((hd or {}).items()):
            L, h = (k if isinstance(k, (tuple, list)) else (None, None))
            hrows.append({"aligned": al, "base": b, "layer": int(L), "head": int(h),
                          "wdelta_head": float(x)})
        append(T_BLOCKS, brows)
        if hrows: append(T_HEADS, hrows)
        append(T_PAIRS, [row])
        el = time.time() - t0
        print("  [%2d/%2d] %-40s %2d blocks, %d heads  %.1fs  (~%.0f min left)"
              % (i, len(todo), al[:40], len(brows), len(hrows), time.time() - t1,
                 (el / i) * (len(todo) - i) / 60))
    print("\nDONE in %.1f min" % ((time.time() - t0) / 60))
    print("  pairs %s\n  blocks %s\n  heads %s"
          % (os.path.relpath(T_PAIRS, ROOT), os.path.relpath(T_BLOCKS, ROOT),
             os.path.relpath(T_HEADS, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
