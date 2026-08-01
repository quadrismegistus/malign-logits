"""F20x COMPLETION RUN — the three missing levels. RH's GO at [2130].

    .venv/bin/python scripts/f20x_format_completion.py --dry-run
    .venv/bin/python scripts/f20x_format_completion.py --run

WHAT IT COMPLETES. `data/f20x_format_battery.parquet` holds 9,280 rows across
TWO of five declared levels — `rung` and `narrative`, which are the design's
EXTREMES. `spelled_rung`, `prose_q` and `document` are the interior, and they
are exactly the levels that would locate the transition. **Two adjacent points
are a contrast; five were specified because a contrast cannot say WHERE the
transition sits, and locating it is the battery's entire purpose.**

    58 model-arms x 16 stimuli x 5 draws x 3 levels = 13,920 completions
    cells [1856, 4640)   seeds [20262585, 20265369)

AUTHORISED under the frozen registration `39e6fae722b0be4a` and the frozen
completion spec `52909ae8140f9e32`. n=5 is RH's number and is MATCHED, NOT
CHOSEN: mixing n puts unequal precision at the two ends of every within-stimulus
contrast and the noisiest points at the ends of an ordering test.

**THE TEMPLATES ARE READ OUT OF THE FROZEN SPEC, NOT WRITTEN HERE.** Every cell
below is transcribed from `docs/f20x_format_templates.md` §2's table, which is
what a fresh implementer would read and which this file must not paraphrase.

FIVE THINGS THIS REFUSES TO DO, EACH BECAUSE IT WAS FOUND THE HARD WAY.

  1. **Generate from the SPEC's stimulus list.** The spec enumerates fifteen
     and the artifact holds SIXTEEN. A run built from the spec drops the
     sixteenth nonce out of a within-stimulus PAIRED test SILENTLY — the
     contrast computes over 15 pairs and reports nothing, while every count
     looks clean. `assert_join_compatible()` refuses before the first call.
  2. **Restart the seed counter.** The seed is a pure function of
     `SEED0 + cell`, so restarting reissues identical torch seeds. Different
     prompts, so not a validity failure — but the runs would draw from the same
     RNG states and any claim of independent sampling across levels is false.
     **And the stride differs (48/model-arm against 32), so "continue the
     counter" is not "continue by the same stride".**
  3. **Cross the generation stack.** vLLM against HF/torch would put the stack
     difference PERFECTLY CONFOUNDED with the level factor, and the registered
     primary (`rung` − `document`) would BE the cross-stack contrast ([2135].1).
     **The stack is a join key nobody declared because nobody imagined changing
     it.**
  4. **Fill the disk.** 93 GB free against a 1,464 GB cache. The run writes tens
     of MB; the risk is a `from_pretrained` deciding a shard is stale. A floor
     is asserted before EACH load, so a run that would exhaust the volume stops
     with all 58 checkpoints intact rather than dying mid-write and leaving a
     partial cache the next run silently re-fetches.
  5. **Repair in place.** A run that lands wrong is REGENERATED from a corrected
     spec. A merged file patched after the fact has no honest provenance.
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

PARQUET = os.path.join(ROOT, "data", "f20x_format_battery.parquet")
OUT = os.path.join(ROOT, "data", "f20x_format_completion.parquet")

SEED0 = 20260729
CELL_START = 1856          #: computed, not assumed — see the module docstring
NEW_LEVELS = ("spelled_rung", "prose_q", "document")
MAX_TOK, TEMP, DRAWS, BATCH = 200, 1.0, 5, 16
FREE_FLOOR_GB = 40         #: below this, refuse the next load rather than risk it

#: TRANSCRIBED FROM docs/f20x_format_templates.md §2. Each entry is
#: (condition -> template), `{w}` the word and `{stip}` the stipulation which
#: "carries in EVERY cell, because the stipulation is one of the accounts —
#: dropping it in the non-question cells would change the stimulus as well as
#: the format."
TEMPLATES = {
    "spelled_rung": {
        "1P":      "Question: Who are you?\nAnswer:",
        "3P":      "Question: Who is {pron}?\nAnswer:",
        "O-named": "{stip}Question: What is a {w}?\nAnswer:",
        "N-bare":  "Question: What is a {w}?\nAnswer:",
    },
    "prose_q": {
        "1P":      "Someone asks you: Who are you? You reply:",
        "3P":      "Someone asks you: Who is {pron}? You reply:",
        "O-named": "{stip}Someone asks you: What is a {w}? You reply:",
        "N-bare":  "Someone asks you: What is a {w}? You reply:",
    },
    "document": {
        "1P":      "I am",
        "3P":      "{Pron} is",
        "O-named": "{stip}The {w} is",
        "N-bare":  "A {w} is",
    },
}
#: 3P wordings are per-stimulus, not per-class: the table's row shows `she`, and
#: the artifact carries he/she/they. Read off the EXISTING rows rather than
#: reconstructed, because the artifact is the authority on every key.
PRON = {"3P-he": ("he", "He"), "3P-she": ("she", "She"),
        "3P-they": ("they", "They")}


def build_prompts(old):
    """One row per (model_arm, stimulus, level, draw). Text filled at run time."""
    meta = old[["family", "base_model_id", "model_id", "arm"]].drop_duplicates()
    stim = old[["stim_id", "condition", "word"]].drop_duplicates("stim_id")
    rows = []
    for _, m in meta.iterrows():
        for _, s in stim.iterrows():
            for lv in NEW_LEVELS:
                t = TEMPLATES[lv][s.condition]
                stip = (f"A {s.word} is a kind of tool. "
                        if s.condition == "O-named" else "")
                pron, Pron = PRON.get(s.stim_id, ("", ""))
                prompt = t.format(w=s.word, stip=stip, pron=pron, Pron=Pron)
                for d in range(DRAWS, DRAWS * 2):     #: draws CONTINUE at 5..9
                    rows.append(dict(family=m.family, base_model_id=m.base_model_id,
                                     model_id=m.model_id, arm=m.arm,
                                     stim_id=s.stim_id, condition=s.condition,
                                     word=s.word, level=lv, prompt=prompt,
                                     draw=d, temperature=TEMP))
    return pd.DataFrame(rows)


def _free_gb():
    """Free space on the volume the WEIGHTS land on, resolved not assumed."""
    import shutil
    try:
        from huggingface_hub import constants as C
        path = C.HF_HUB_CACHE
    except Exception:
        path = os.path.expanduser("~/.cache/huggingface")
    return shutil.disk_usage(os.path.realpath(path)).free / 1e9


def generate(new, n_ma):
    """The model loop. 58 model-arms x 16 stimuli x 3 levels x 5 draws.

    **THIS FUNCTION DID NOT EXIST FOR THREE HOURS AND `--run` RETURNED 0.** The
    stub printed "GENERATION NOT YET WIRED" to stdout and reported SUCCESS to
    every mechanism that reads exit codes — a launcher, a nohup wrapper, a cron
    would each have logged a clean run. **A stub's failure path must be its
    EXIT CODE, because prose on stdout is read by people and exit codes are read
    by everything else.**

    SEEDS. `seed = SEED0 + cell`, cell enumerated over (model_arm, stimulus,
    level) in sorted order from CELL_START. **The stride is 48 per model-arm
    here against 32 in the existing rows** — 3 levels rather than 2 — which is
    why "continue the counter" is not "continue by the same stride", and why
    CELL_START is 58 x 32 = 1856 rather than a round number.

    WRITTEN PER MODEL-ARM, not at the end: an interrupted run is readable and a
    crashed one is not lost. The same reason `f20x_generate.py` writes per family.
    """
    import gc
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    #: **MIRRORING `f20x_format_battery.py`'s LOOP, NOT RESEMBLING IT.** The
    #: first version differed from the producer of the 9,280 existing rows in
    #: FOUR ways, and only one announced itself:
    #:
    #:   1. no attention mask          -> transformers WARNED on stdout
    #:   2. no left padding, unbatched -> silent
    #:   3. **float16 for every model** -> SILENT, and the original uses
    #:      **float32 for sub-billion models** because fp16 sampling produces
    #:      inf/nan logits once prompts are batched. A dtype difference is a
    #:      LOGIT difference, and this battery's quantity is a next-token
    #:      distribution.
    #:   4. seed per CELL              -> the original seeds per CHUNK and
    #:      advances `cell` by the chunk size, so per-cell seeding reissues a
    #:      different RNG schedule entirely.
    #:
    #: **Three of the four were silent.** So the constants and the dtype rule are
    #: IMPORTED from the original rather than restated: a copied constant drifts,
    #: an imported one cannot.
    from f20x_format_battery import BATCH_PROMPTS, is_small

    meta = new[["family", "base_model_id", "model_id", "arm"]].drop_duplicates()
    #: cell order must match the enumeration the seeds were computed against
    pairs = (new[["stim_id", "level"]].drop_duplicates()
             .sort_values(["stim_id", "level"]).values.tolist())
    print(f"\n  cells {CELL_START}..{CELL_START + n_ma * len(pairs)}  "
          f"({n_ma * len(pairs)} = {n_ma} model-arms x {len(pairs)})")
    print(f"  batching {BATCH_PROMPTS} prompts/call, left-padded, "
          f"fp32 for sub-billion models — all four matched to the original")

    sink, failures, t0, done = [], [], time.time(), 0
    cell = CELL_START
    for _, m in meta.iterrows():
        free = _free_gb()
        if free < FREE_FLOOR_GB:
            print(f"\n*** HALT before {m.model_id}: {free:.0f} GB free on the "
                  f"cache volume, floor {FREE_FLOOR_GB}. Stopping with every "
                  f"checkpoint intact rather than dying mid-write.")
            break
        try:
            tok = AutoTokenizer.from_pretrained(m.model_id, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"          #: decoder-only; right padding
                                               #: puts pads between prompt and
                                               #: first generated token
            dtype = torch.float32 if is_small(m.model_id) else torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                m.model_id, dtype=dtype, device_map="mps",
                trust_remote_code=True).eval()
        except Exception as e:
            print(f"  LOAD FAILED {m.model_id}/{m.arm}: {str(e)[:120]}")
            failures.append(dict(model_id=m.model_id, arm=m.arm,
                                 stage="load", error=str(e)[:300]))
            cell += len(pairs)                 #: the counter advances even on a
                                               #: skipped arm, so seeds stay
                                               #: aligned to the cell grid
            continue

        sub = new[(new.model_id == m.model_id) & (new.arm == m.arm)]
        for c0 in range(0, len(pairs), BATCH_PROMPTS):
            chunk = pairs[c0:c0 + BATCH_PROMPTS]
            prompts = [sub[(sub.stim_id == sid) & (sub.level == lv)].iloc[0].prompt
                       for sid, lv in chunk]
            seed = SEED0 + cell
            torch.manual_seed(seed)
            cell += len(chunk)
            enc = tok(prompts, return_tensors="pt", padding=True).to("mps")
            try:
                with torch.no_grad():
                    out = model.generate(**enc, do_sample=True, temperature=TEMP,
                                         max_new_tokens=MAX_TOK,
                                         num_return_sequences=DRAWS,
                                         pad_token_id=tok.pad_token_id)
            except RuntimeError as e:
                #: fp32 retry rather than dropping the checkpoint — a skipped
                #: model is a silent hole in the roster ([2135].1)
                print(f"  fp16 sampling failed ({type(e).__name__}); "
                      f"retrying this call in float32")
                model = model.float()
                with torch.no_grad():
                    out = model.generate(**enc, do_sample=True, temperature=TEMP,
                                         max_new_tokens=MAX_TOK,
                                         num_return_sequences=DRAWS,
                                         pad_token_id=tok.pad_token_id)
            plen = enc["input_ids"].shape[1]
            #: generate returns rows grouped by prompt: p0 x DRAWS, p1 x DRAWS...
            for pi, (sid, lv) in enumerate(chunk):
                g = sub[(sub.stim_id == sid) & (sub.level == lv)]
                for di, draw in enumerate(sorted(g.draw.unique())):
                    sink.append(dict(
                        family=m.family, base_model_id=m.base_model_id,
                        model_id=m.model_id, arm=m.arm, stim_id=sid,
                        condition=g.iloc[0].condition, word=g.iloc[0].word,
                        level=lv, prompt=prompts[pi], draw=int(draw),
                        temperature=TEMP, seed=seed,
                        text=tok.decode(out[pi * DRAWS + di][plen:],
                                        skip_special_tokens=True)))
        del model, tok
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        done += 1
        pd.DataFrame(sink).to_parquet(OUT, index=False)
        el = (time.time() - t0) / 60
        print(f"  [{done}/{n_ma}] {m.model_id}/{m.arm}  rows {len(sink):,}  "
              f"{el:.1f} min  ETA {el / done * (n_ma - done) / 60:.1f} h")

    pd.DataFrame(sink).to_parquet(OUT, index=False)
    if failures:
        fp = OUT.replace(".parquet", "_failures.csv")
        pd.DataFrame(failures).to_csv(fp, index=False)
        print(f"\n  {len(failures)} failure(s) -> {fp}  (REPORTED, not repaired)")
    print(f"\n  wrote {OUT}: {len(sink):,} rows from {done}/{n_ma} model-arms")
    #: NONZERO if the run did not complete, so a launcher cannot read a partial
    #: or empty run as success — the defect this whole function replaces.
    return 0 if (done == n_ma and sink) else 1


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run", action="store_true")
    args = ap.parse_args()

    from f20x_read_gate import assert_join_compatible, load

    old = load()                       #: gated: structural columns only
    new = build_prompts(old)
    n_ma = old[["model_id", "arm"]].drop_duplicates().shape[0]

    print(f"F20x COMPLETION — {len(new):,} rows")
    print(f"  model-arms {n_ma}   stimuli {new.stim_id.nunique()}   "
          f"levels {sorted(new.level.unique())}   draws {sorted(new.draw.unique())}")
    print(f"  seeds      cells [{CELL_START}, {CELL_START + n_ma*16*3}) -> "
          f"[{SEED0+CELL_START}, {SEED0+CELL_START+n_ma*16*3})")

    #: THE GATE, BEFORE THE FIRST CALL. `seed` is added here so the disjoint
    #: check has something to test; the run records it per row.
    new["seed"] = SEED0 + CELL_START
    assert_join_compatible(new)
    print("\n  assert_join_compatible: PASS")

    #: **THE FLOOR CHECKS THE VOLUME THE LOADS LAND ON, NOT `~`.** [2227]: this
    #: read `expanduser("~")` — which is `/` — while `~/.cache/huggingface` is a
    #: SYMLINK to another device. The guard passed at 97 GB on `/` while the
    #: cache volume could have been dry, and would have refused at 39 GB on `/`
    #: with 120 GB spare where the weights actually go. **A free-space number
    #: from the wrong mount is indistinguishable in the output from a right one.**
    free = _free_gb()
    print(f"  free space {free:.0f} GB on the CACHE volume   "
          f"floor {FREE_FLOOR_GB} GB   "
          f"{'OK' if free > FREE_FLOOR_GB else '*** BELOW FLOOR'}")

    print("\n  SAMPLE CELLS, transcribed from templates §2:")
    for lv in NEW_LEVELS:
        for sid in ("1P", "3P-she", "O-froe", "N-glorp"):
            r = new[(new.level == lv) & (new.stim_id == sid)]
            if len(r):
                print(f"    {lv:<13}{sid:<9}{r.iloc[0].prompt!r}")
        print()

    if args.dry_run:
        print("DRY RUN — nothing generated. Re-run with --run.")
        return 0

    return generate(new, n_ma)


if __name__ == "__main__":
    sys.exit(main())
