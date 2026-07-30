"""M01 clause 8: which stage does the displacing, at word level, across families.

    uv run .venv/bin/python scripts/m01_clause8_stage_share.py --dry-run
    uv run .venv/bin/python scripts/m01_clause8_stage_share.py

WHY THIS EXISTS. Clause 8 is VERIFIED by two seats and reads as a roster-grade
claim -- "base->SFT carries a median 72% of word-level distributional movement
(2.58x DPO), uniformly across content categories." It cites dockets, not a
script. There is no producer in `scripts/` and `meta/M01_displacement/scripts/`
is empty, so a clause the paper leans on cannot presently be re-run at all.

AND THE NUMBER IS PROBABLY ONE FAMILY. `f13_code_amber_stages.py` opens: "Amber
is the only family in `true_word_probs` with all three arms present ... So it is
THE ONE PLACE the annotation can ask WHICH STAGE DOES THE DISPLACING." A staged
base->SFT->DPO decomposition needs three arms; if amber had the only three, then
"median 72%" is a median over amber's prompts and "uniformly across content
categories" is a WITHIN-AMBER uniformity stated as a cross-cutting one.

Two seats reproducing a number exactly confirms the ARITHMETIC and says nothing
about the SCOPE. This is not a wrong number -- it is a correct number carrying
an unstated denominator.

THE v3 GRID MAKES IT A FIRST TEST RATHER THAN A RECHECK: 21 families carry
base+ego+superego, six of them a fourth arm. So the question "which stage does
the displacing" becomes answerable across families, on identical support, for
the first time.

ABSOLUTE JS IS REPORTED BESIDE EVERY SHARE, ALWAYS (lacan's addition, and it is
the point). A share of 72% can mean a large SFT movement or a SMALL DPO one, and
the ratio cannot distinguish them. If a family's DPO stage barely moves, "the
operation installs almost entirely at SFT" is really "this particular DPO did
little" -- the same sentence, the opposite finding. A ratio whose terms are not
published is a number whose meaning lives in a computation nobody can re-run,
which is how clause 8 got here.

THE ARCHANGEL FOUR ARE THE DECISIVE CELL. DPO/KTO/PPO/SLiC share a base and one
SFT checkpoint and differ only in the preference-optimisation METHOD:

    share ~equal in all four   -> a fact about the SFT STAGE: the operation is
                                  installed by imitation of curated
                                  demonstrations, whatever method follows
    share varies by method     -> the sentence measures THE SECOND STAGE'S
                                  WEAKNESS, not the first stage's strength

With one DPO variant those readings are observationally identical.

DECLARED BEFORE THE DATA EXISTS: clause 8's 72% either replicates across the
three-arm families or is revealed as amber-shaped. Both are results. Only the
current state -- verified, general-sounding, single-family, no producer -- is
not.

NEVER POOLED-ONLY. Per-family rows are the output; any aggregate carries its
per-family decomposition and its n, per M01's own Figures rule.
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import MODEL_FAMILIES, PATH_DATA  # noqa: E402
from malign_logits.cache import get_cache  # noqa: E402

OUT = os.path.join(PATH_DATA, "m01_clause8_stage_share.csv")
ARMS = ("base", "ego", "superego")


def word_dist(cell):
    """{surface: prob} from a stored cell, summed over duplicate surfaces.

    Rows are (surface, ids) keyed, so one surface can appear more than once --
    the same word reached by different token paths. Summing is what makes the
    comparison a distribution over WORDS rather than over paths.
    """
    d = defaultdict(float)
    for r in cell["rows"]:
        d[r["word"]] += r["p"]
    return d


def js(a, b):
    """Jensen-Shannon (base 2) over the union of surfaces, each renormalised.

    Renormalisation is deliberate and must be stated with any number this
    produces: the residual (unresolved mass) is DROPPED, so this measures
    movement within the resolved vocabulary, not movement including the part
    the instrument could not name. Two cells with different residuals are
    still comparable here, but the metric is blind to a shift INTO residual.
    """
    keys = sorted(set(a) | set(b))
    if not keys:
        return float("nan")
    p = np.array([a.get(k, 0.0) for k in keys])
    q = np.array([b.get(k, 0.0) for k in keys])
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(x, y):
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / y[nz])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def staged_families():
    """Families with all three arms registered. Membership, not availability."""
    out = {}
    for key, fam in MODEL_FAMILIES.items():
        ids = {a: getattr(fam, a, None) for a in ARMS}
        if all(ids.values()):
            out[key] = ids
    return out


def categorise(prompt, index):
    return index.get(prompt, "")


def load_category_index():
    """prompt string -> category, from the registries. Empty when unmapped."""
    from malign_logits import experiments as E
    import re
    idx = {}
    for src in (getattr(E, n, None) for n in
                ("DEFAULT_PROMPTS", "INSTITUTIONAL_PROMPTS", "CHINESE_PROMPTS")):
        if isinstance(src, dict):
            for k, v in src.items():
                if isinstance(v, str):
                    idx[v] = re.sub(r"_\d+$", "", k)
    return idx


def main(a):
    cm = get_cache()
    fams = staged_families()
    cat_of = load_category_index()
    print(f"{len(fams)} families carry base+ego+superego in MODEL_FAMILIES\n")

    prompts = sorted(cat_of) if not a.prompts else a.prompts
    rows, versions, skipped = [], Counter(), Counter()

    for key, ids in sorted(fams.items()):
        got = 0
        for p in prompts:
            cells = {}
            for arm in ARMS:
                if not cm.has_true_word_probs(ids[arm], p, theta=a.theta):
                    break
                cells[arm] = cm.get_true_word_probs(ids[arm], p, theta=a.theta)
            if len(cells) < 3:
                skipped[key] += 1
                continue
            # THE STAMP IS CHECKED, NOT ASSUMED. Three cells coded under two
            # boundary rules are not a staged decomposition of anything -- v3
            # changed what a word IS, so a v1 base against a v3 superego would
            # book an instrument change as alignment movement.
            vs = {c.get("rule_version", 1) for c in cells.values()}
            versions[tuple(sorted(vs))] += 1
            if len(vs) > 1:
                skipped[f"{key}:mixed_rule_version"] += 1
                continue
            d = {arm: word_dist(c) for arm, c in cells.items()}
            js_sft = js(d["base"], d["ego"])
            js_dpo = js(d["ego"], d["superego"])
            tot = js_sft + js_dpo
            if not np.isfinite(tot) or tot <= 0:
                skipped[f"{key}:degenerate"] += 1
                continue
            rows.append(dict(
                family=key, prompt=p, category=cat_of.get(p, ""),
                rule_version=vs.pop(),
                # ABSOLUTES FIRST, share second -- deliberately, so a reader
                # cannot take the ratio without seeing what it is a ratio of.
                js_base_ego=round(js_sft, 6),
                js_ego_superego=round(js_dpo, 6),
                js_total=round(tot, 6),
                sft_share=round(js_sft / tot, 6),
                sft_over_dpo=round(js_sft / js_dpo, 6) if js_dpo > 0 else float("inf"),
            ))
            got += 1
        print(f"  {key:<20}{got:>5} prompts with all three arms"
              f"{'   (none)' if not got else ''}")

    if not rows:
        print("\nNO FAMILY HAS THREE ARMS IN THE STORE YET.")
        print("Expected before the v3 ingest -- this producer is written to run "
              "AFTER it, and running it now is a plumbing check, not a result.")
        if skipped:
            print(f"skipped: {dict(list(skipped.items())[:6])}")
        return

    print(f"\nrule_version combinations seen: {dict(versions)}")
    if any(len(k) > 1 for k in versions):
        print("!! MIXED-VERSION CELLS WERE SKIPPED. A staged decomposition "
              "across two boundary rules books an instrument change as "
              "alignment movement.")

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fam_of = defaultdict(list)
    for r in rows:
        fam_of[r["family"]].append(r)
    print(f"\n{'family':<20}{'n':>6}{'med share':>11}{'med JS b>e':>12}"
          f"{'med JS e>s':>12}{'med ratio':>11}")
    for k, rs in sorted(fam_of.items()):
        print(f"{k:<20}{len(rs):>6}"
              f"{np.median([r['sft_share'] for r in rs]):>11.3f}"
              f"{np.median([r['js_base_ego'] for r in rs]):>12.4f}"
              f"{np.median([r['js_ego_superego'] for r in rs]):>12.4f}"
              f"{np.median([r['sft_over_dpo'] for r in rs]):>11.2f}")

    shares = [np.median([r["sft_share"] for r in rs]) for rs in fam_of.values()]
    print(f"\nACROSS {len(shares)} FAMILIES, family as unit: "
          f"median {np.median(shares):.3f}, range {min(shares):.3f}-{max(shares):.3f}")
    print("Clause 8 books 0.72 (2.58x). It is a ONE-FAMILY figure until this "
          "table says otherwise; the spread above is the test.")

    arch = {k: v for k, v in fam_of.items() if k.startswith("archangel")}
    if len(arch) > 1:
        print(f"\nARCHANGEL: {len(arch)} preference methods on one base+SFT")
        for k, rs in sorted(arch.items()):
            print(f"  {k:<20}{np.median([r['sft_share'] for r in rs]):.3f}  "
                  f"(n={len(rs)})")
        sp = max(np.median([r["sft_share"] for r in rs]) for rs in arch.values()) - \
            min(np.median([r["sft_share"] for r in rs]) for rs in arch.values())
        print(f"  spread {sp:.3f} -- near zero means the share is a fact about "
              f"the SFT stage; wide means it measures the second stage's weakness.")
    print(f"\nwrote {OUT}   ({len(rows):,} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.001)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report arm availability without computing")
    main(ap.parse_args())
