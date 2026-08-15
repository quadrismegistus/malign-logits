#!/usr/bin/env python
"""twp_coverage_matrix.py — twp coverage for every checkpoint, split by language.

    scripts/twp_coverage_matrix.py            report + write data/twp_coverage.csv
    scripts/twp_coverage_matrix.py --by finding

**"DOES MODEL X HAVE twp?" HAS NO ANSWER WITHOUT NAMING THE BATTERY.** The store
keys on (model, prompt), and this campaign holds at least four disjoint prompt
sets that have all been called "the battery": the categorisation file (2,579
ACTIVE distinct strings), the F11 contradiction subset (115), the delta subset
(84), and the 105 minimal-pair stems (210, ZERO overlap with any of the others). Coverage read as
a model property instead of a (model, battery) property is what put nine pairs
in an "unscored" bucket that mixed never-run, refused-for-cause, and
wrong-battery. This script answers the question at the grain it actually has.

**LANGUAGE IS A COLUMN FROM THE SOURCE, NOT INFERRED FROM THE PROMPT ID.** The
categorisation file declares `language` (2,200 en / 379 zh, ACTIVE). Guessing from a
`_zh` suffix would have missed every Chinese prompt whose id does not carry one,
and a coverage table that silently drops a language is worse than no table --
the tokenizers differ in tokens-per-word, which is the quantity every downstream
word-level measure reads.

**DEDUPLICATED ON THE STRING, AND FILTERED TO ACTIVE.** 2,809 catalogue records
hold 2,579 distinct ACTIVE strings; the store keys on the string, so two ids for
one string are ONE cell. See `load_prompts` for why the status filter is not
optional.
"""
import argparse, collections, csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)


def load_prompts(status="ACTIVE"):
    """**GOES THROUGH `Prompts.where()`, NOT THE RAW JSON.** The first version of
    this script read `prompt_categorisation.json` directly and so counted RETIRED
    rows: 2,809 catalogue rows are 2,590 ACTIVE, 215 RETIRED and 4 DISPUTED, with
    163 texts that exist ONLY under retired ids. Every coverage figure it produced
    divided by a denominator containing prompts no active design consumes, which
    inflates the apparent gap -- and a fleet sized from it would have scored 163
    retired prompts on every checkpoint. `prompts.py` says so in its own
    docstring ("status defaults to ACTIVE because nearly every count wants it and
    forgetting it once is what turned 4 into 55") and I read past it.

    **AND `<<<LOGICAL:BOS>>>` IS NOT A TEXT.** It resolves per family in
    `twp._prompt_ids` and is already scored on 117 checkpoints; the four literal
    BOS strings it superseded are RETIRED for that reason. Feeding it as a
    literal, or feeding those four, scores a foreign family's special token as
    prose. The status filter removes them, which is the point of using it."""
    from malign_logits.prompts import Prompts
    by_str = {}
    for pr in Prompts.where(status=status):
        row = pr.row
        s = row.get("prompt")
        if not s or s in by_str:
            continue
        by_str[s] = {"lang": row.get("language") or "?", "finding": row.get("finding"),
                     "domain": row.get("domain"), "prompt_id": row.get("prompt_id"),
                     "status": row.get("status")}
    return by_str


def checkpoints():
    """Every checkpoint we know about: the registry UNION the declared pairs.

    The registry is the wider set (146 vs 104); a pair member absent from the
    registry would otherwise vanish from a table titled "all checkpoints"."""
    #: THROUGH `Checkpoint`, NOT THE RAW FILE. This block hand-rolled the
    #: list-or-dict shape of `model_registry.json` -- 16 of the 18 consumers
    #: that opened it did, each a place a schema change breaks silently and
    #: each re-implementing what `Registry` already does once.
    ids = set()
    from malign_logits.checkpoint import Checkpoint
    ids |= {cp.id for cp in Checkpoint.all()}
    bap = json.load(open(os.path.join(ROOT, "data", "base_aligned_pairs.json")))
    rows = bap["pairs"] if isinstance(bap, dict) and "pairs" in bap else bap
    role = {}
    for x in rows:
        ids.add(x["base"]); ids.add(x["aligned"])
        role[x["base"]] = "base"; role[x["aligned"]] = "aligned"
    return sorted(i for i in ids if i), role


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/twp_coverage.csv")
    ap.add_argument("--by", choices=("lang", "finding"), default="lang")
    a = ap.parse_args()

    from malign_logits.cache import get_cache
    cm = get_cache()
    by_str = load_prompts()
    ids, role = checkpoints()
    langs = collections.Counter(v["lang"] for v in by_str.values())
    en = [s for s, v in by_str.items() if v["lang"] == "en"]
    zh = [s for s, v in by_str.items() if v["lang"] == "zh"]
    other = [s for s, v in by_str.items() if v["lang"] not in ("en", "zh")]

    print("PROMPTS  %d distinct ACTIVE strings from %d catalogue records   en %d / zh %d%s"
          % (len(by_str), 2809, len(en), len(zh),
             ("  / other %d" % len(other)) if other else ""))
    print("CHECKPOINTS  %d (model_registry UNION base_aligned_pairs)\n" % len(ids))

    rows = []
    for m in ids:
        e = sum(1 for s in en if cm.has_true_word_probs(m, s))
        z = sum(1 for s in zh if cm.has_true_word_probs(m, s))
        rows.append({"model": m, "role": role.get(m, ""),
                     "en_covered": e, "en_total": len(en),
                     "zh_covered": z, "zh_total": len(zh),
                     "en_pct": round(100 * e / len(en), 1) if en else 0,
                     "zh_pct": round(100 * z / len(zh), 1) if zh else 0,
                     "any": e + z})

    rows.sort(key=lambda r: (-r["any"], r["model"]))
    p = os.path.join(ROOT, a.out)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    #: **BANDS, NOT A MEAN.** A mean over checkpoints would report the roster as
    #: "62% covered" and hide that the distribution is bimodal -- most are at 0
    #: or near-complete, and the middle is nearly empty. The bands are the shape.
    band = collections.Counter()
    for r in rows:
        f = r["any"] / (len(en) + len(zh))
        band["none (0)" if f == 0 else
             "sliver (<5%)" if f < .05 else
             "partial (5-90%)" if f < .90 else "near-complete (>=90%)"] += 1
    print("COVERAGE BANDS over %d checkpoints, all prompts:" % len(rows))
    for k in ("near-complete (>=90%)", "partial (5-90%)", "sliver (<5%)", "none (0)"):
        print("   %-24s %3d" % (k, band[k]))

    print("\n%-52s %5s %13s %13s" % ("checkpoint", "role", "EN of %d" % len(en),
                                     "ZH of %d" % len(zh)))
    for r in rows:
        if not r["any"]:
            continue
        print("  %-50s %-7s %6d %5.1f%% %6d %5.1f%%"
              % (r["model"][:50], r["role"], r["en_covered"], r["en_pct"],
                 r["zh_covered"], r["zh_pct"]))
    zero = [r for r in rows if not r["any"]]
    print("\nZERO COVERAGE ON EVERY PROMPT IN THIS FILE: %d checkpoints" % len(zero))
    for r in zero:
        print("   %-50s %s" % (r["model"][:50], r["role"]))
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
