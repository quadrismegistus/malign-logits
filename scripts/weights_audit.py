"""Which weight formats each model publishes, and whether it can be loaded.

    uv run .venv/bin/python scripts/weights_audit.py

Writes `data/weights_audit.csv` — one row per model in the frozen grid spec.

WHY THIS IS A SWEEP AND NOT AN AD-HOC READ. It began as a live query during the
v3 grid run, which is exactly the shape the canonical model file forbids: a
measured field whose producer is a shell history. Measured facts name the script
and commit that produced them, or the file documents a different object than the
one that ran.

WHAT IT MEASURES, AND WHY THE THIRD CASE EXISTS. `transformers` refuses `.bin`
weights unless torch >= 2.6 (`check_torch_load_is_safe`), so a model's publishing
format decides whether a given box can load it at all. Two values are not enough:

    safetensors     loadable anywhere
    bin             needs torch >= 2.6
    mixed           BOTH published -- and loadability then turns on WHICH INDEX
                    EXISTS, not on which weights do

The third case is not hypothetical. `HuggingFaceH4/mistral-7b-sft-beta` ships
both formats and only `pytorch_model.bin.index.json`; a sharded checkpoint needs
its index to map tensors to shards, so transformers falls back to the `.bin`
index and refuses. The safetensors sit there unreachable and
`use_safetensors=True` does not help, because the flag selects a format rather
than synthesising the map. `index_present` is therefore a separate column: for
sharded models it is the field that decides, and for single-file models it is
vacuously true.

THE TORCH FLOOR IS RECORDED AS A NUMBER, NOT A VERDICT. `needs_torch` says what
the checkpoint requires; whether a particular box satisfies it is a fact about
that box and belongs with the run, not with the model.

NO WEIGHTS ARE DOWNLOADED. This reads the repository file listing only.
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402

OUT = os.path.join(PATH_DATA, "weights_audit.csv")
# THE ROSTER, not the last execution plan. grid_spec.json was narrowed twice on
# 2026-07-31 and now holds 82 entries describing what ran last; the audit is a
# question about the 103-model object.
SPEC = os.path.join(PATH_DATA, "grid_roster.json")
ST_INDEX = "model.safetensors.index.json"
BIN_INDEX = "pytorch_model.bin.index.json"


def audit(model_id):
    from huggingface_hub import list_repo_files
    try:
        fs = list_repo_files(model_id)
    except Exception as e:
        return dict(model=model_id, weights_format="unknown", index_present="",
                    n_safetensors="", n_bin="", needs_torch="",
                    note=f"listing failed: {type(e).__name__}")
    st = [f for f in fs if f.endswith(".safetensors")]
    bn = [f for f in fs if f.endswith(".bin")]
    st_idx, bin_idx = ST_INDEX in fs, BIN_INDEX in fs

    if st and bn:
        fmt = "mixed"
    elif st:
        fmt = "safetensors"
    elif bn:
        fmt = "bin"
    else:
        fmt = "none"

    # A SINGLE-FILE CHECKPOINT NEEDS NO INDEX, so absence is not a defect there.
    # Only a sharded one can be broken this way, which is why the column is not
    # simply `ST_INDEX in fs`.
    if fmt in ("safetensors", "mixed"):
        idx = st_idx if len(st) > 1 else True
    else:
        idx = ""

    # usable-as-safetensors is what decides the torch floor, not the mere
    # presence of a .safetensors file
    usable_st = bool(st) and (idx is True)
    needs = "" if usable_st else ("2.6" if bn else "")

    note = ""
    if fmt == "mixed" and not usable_st:
        note = ("safetensors shards present, index absent -- falls back to the "
                ".bin index and is refused below torch 2.6")
    elif fmt == "bin":
        note = "bin-only; refused below torch 2.6"
    return dict(model=model_id, weights_format=fmt,
                index_present=("" if idx == "" else str(bool(idx)).lower()),
                n_safetensors=len(st), n_bin=len(bn),
                needs_torch=needs, note=note)


def main(a):
    spec = json.load(open(SPEC))
    rows_in = spec["spec"] if isinstance(spec, dict) else spec
    models = [r["model"] for r in rows_in]
    print(f"auditing {len(models)} models from the frozen spec (no weights fetched)")
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(audit, models))
    rows.sort(key=lambda r: r["model"])
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import collections
    c = collections.Counter(r["weights_format"] for r in rows)
    print(f"\nformats: {dict(c)}")
    blocked = [r for r in rows if r["needs_torch"] == "2.6"]
    print(f"REQUIRE torch >= 2.6: {len(blocked)}")
    for r in blocked:
        print(f"  {r['model']:<52}{r['weights_format']}"
              f"{'  (index absent)' if r['weights_format'] == 'mixed' else ''}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    main(ap.parse_args())
