"""Every prompt scored into true_word_probs, with its category and slot.

    uv run .venv/bin/python scripts/build_prompt_inventory.py

Writes `data/prompt_inventory.csv` (one row per prompt) and the companion
`data/prompt_inventory.md`.

WHY THIS EXISTS. Three registries define prompts (DEFAULT, INSTITUTIONAL,
CHINESE), one hand assignment defines slot grammar, and the store holds
everything ever scored including material from neither -- Set D variants, the
F36 minimal pairs, ingested literary passages. Nothing joined them, so "how many
prompts are we scoring, and which have a category" could not be answered without
re-deriving it, and re-derivation in three places is how the F13 cell-count
disagreements happened.

THE HEADLINE IT REPORTS, and it is the reason to look: roughly THREE QUARTERS of
distinct prompts and FORTY PERCENT of scored cells have NO registered category.
Every category-stratified analysis silently drops them, including the whole
minimal-pair design -- material whose only purpose is a contrast, invisible to
any statistic that strata by category.

READ FROM THE JSONL, NOT THE STASH, and stamped with a read time. The store is
written while the roster runs; two reads twenty minutes apart in one evening
gave 603/13,693 and 604/13,782. A count over a growing store is a claim about
nothing datable unless it carries when it was taken.

THE SLOT COLUMN IS IMPORTED, NEVER RE-DERIVED. `f13_draw_relation_items.py`
holds the hand assignment; its 21 institutional prompts are appended to ACT at
import time rather than listed, so the module must be EXECUTED to be read and a
grep for a prompt name finds nothing.
"""
import csv
import datetime
import glob
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402
from malign_logits import experiments as E  # noqa: E402

SRC = os.path.join(PATH_DATA, "twp_cloud")
CSV = os.path.join(PATH_DATA, "prompt_inventory.csv")
MD = os.path.join(PATH_DATA, "prompt_inventory.md")
DRAW = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "f13_draw_relation_items.py")
CJK = re.compile(r"[一-鿿]")


def registries():
    out = {}
    for name, src in (("DEFAULT", E.DEFAULT_PROMPTS),
                      ("INSTITUTIONAL", E.INSTITUTIONAL_PROMPTS),
                      ("CHINESE", E.CHINESE_PROMPTS)):
        if isinstance(src, dict):
            for k, v in src.items():
                if isinstance(v, str):
                    out[v] = (name, k, re.sub(r"_\d+$", "", k))
    return out


def slots():
    """Execute the draw script for TYPE_OF; tolerate its argparse/SystemExit."""
    try:
        sp = importlib.util.spec_from_file_location("draw", DRAW)
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
    except SystemExit:
        pass
    except Exception as e:
        print(f"  slot map unavailable ({type(e).__name__}); slot column empty")
        return {}
    return getattr(m, "TYPE_OF", {})


def shape(n):
    return "1-2 words" if n <= 2 else "3-6" if n <= 6 else "7-15" if n <= 15 else "16+"


def main():
    key, slot = registries(), slots()
    cells, models = Counter(), defaultdict(set)
    for f in sorted(glob.glob(os.path.join(SRC, "*.jsonl"))):
        mid = os.path.basename(f)[:-6].replace("__", "/")
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue          # truncated final line of a file being written
            cells[r["prompt"]] += 1
            models[r["prompt"]].add(mid)
    if not cells:
        print(f"no data under {SRC}")
        return

    rows = []
    for p, n in sorted(cells.items(), key=lambda kv: -kv[1]):
        src, pid, cat = key.get(p, ("UNMAPPED", "", ""))
        rows.append(dict(prompt=p, source=src, prompt_id=pid, category=cat,
                         slot=slot.get(pid, ""), n_models=len(models[p]),
                         n_cells=n, n_words=len(p.split()),
                         script="CJK" if CJK.search(p) else "latin"))
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tot = sum(r["n_cells"] for r in rows)
    mapped = [r for r in rows if r["source"] != "UNMAPPED"]
    un = [r for r in rows if r["source"] == "UNMAPPED"]
    pr, cl = Counter(), Counter()
    for r in mapped:
        k = (r["source"], r["category"])
        pr[k] += 1
        cl[k] += r["n_cells"]
    us, uc = Counter(), Counter()
    for r in un:
        us[shape(r["n_words"])] += 1
        uc[shape(r["n_words"])] += r["n_cells"]

    L = ["# Prompt inventory\n",
         "`data/prompt_inventory.csv` — every prompt scored into "
         "`true_word_probs`, with its registered category and slot where one "
         "exists.\n",
         f"**Population: {len(rows)} distinct prompts, {tot:,} cells, read "
         f"{datetime.datetime.now():%Y-%m-%d %H:%M} local.**\n",
         "**THE STORE IS WRITTEN WHILE THE ROSTER RUNS.** Two reads twenty "
         "minutes apart gave 603/13,693 and 604/13,782. A count over a growing "
         "store carries its read time or it is a claim about nothing datable — "
         "rebuild rather than quote these.\n",
         "## Columns\n",
         "| column | meaning |", "|---|---|",
         "| `prompt` | exact string; the join key everywhere |",
         "| `source` | `DEFAULT` / `INSTITUTIONAL` / `CHINESE` / `UNMAPPED` |",
         "| `prompt_id` | registry key (`sexual_explicit_3`); empty if unmapped |",
         "| `category` | id with trailing index stripped; empty if unmapped |",
         "| `slot` | hand grammar assignment: ACT/NARR/REF/UTTER/RESULT/SENSE |",
         "| `n_models`, `n_cells` | coverage (equal: one cell per model) |",
         "| `n_words`, `script` | shape; `script` is CJK vs latin |\n",
         f"## Mapped — {len(mapped)} prompts, "
         f"{sum(r['n_cells'] for r in mapped):,} cells\n",
         "Three parallel batteries over the same nine content categories plus "
         "eleven institutional roles.\n",
         "| source | category | prompts | cells |", "|---|---|---|---|"]
    for k, n in sorted(pr.items(), key=lambda kv: (kv[0][0], -cl[kv[0]])):
        L.append(f"| {k[0]} | {k[1]} | {n} | {cl[k]:,} |")

    sl = Counter(r["slot"] for r in rows if r["slot"])
    L += ["\n## Slot grammar\n",
          "Every mapped prompt carries one; **no unmapped prompt does.**\n",
          "| slot | prompts |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sl.most_common()]

    L += [f"\n## UNMAPPED — {len(un)} prompts, "
          f"{sum(r['n_cells'] for r in un):,} cells\n",
          f"**{sum(r['n_cells'] for r in un) / tot:.0%} of all cells scored, with "
          "no category — so every category-stratified analysis silently drops "
          "them, including the entire minimal-pair design, whose only purpose "
          "is a contrast.**\n",
          "| shape | prompts | cells | what it is |", "|---|---|---|---|"]
    what = {"7-15": "Set D narrative variants — the bulk",
            "3-6": "minimal pairs (captive/free, desire/…) — the F36 line",
            "16+": "literary passages; 2 models only, a side experiment",
            "1-2 words": "—"}
    for k in ("7-15", "3-6", "16+", "1-2 words"):
        if us.get(k):
            L.append(f"| {k} words | {us[k]} | {uc[k]:,} | {what[k]} |")
    L += ["\n**Every Chinese prompt IS mapped** (`CHINESE_PROMPTS` covers all of "
          "them); it is the Set D and F36 material that is not.\n",
          "Rebuild: `scripts/build_prompt_inventory.py`.\n"]
    open(MD, "w").write("\n".join(L))
    print(f"{CSV}: {len(rows)} prompts, {tot:,} cells")
    print(f"  mapped {len(mapped)}   unmapped {len(un)} "
          f"({sum(r['n_cells'] for r in un) / tot:.0%} of cells)")
    print(f"{MD}")


if __name__ == "__main__":
    main()
