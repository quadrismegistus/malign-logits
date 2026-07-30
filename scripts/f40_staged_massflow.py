"""Stage-decompose alignment on the archangel cell: JS beside F40 CATEGORY MASS FLOW.

    uv run .venv/bin/python scripts/f40_staged_massflow.py
    uv run .venv/bin/python scripts/f40_staged_massflow.py --domain violence

THE QUESTION, which is RH's and which no existing measure answers.

Clause 8 reads "the operation installs almost entirely at SFT: base->SFT carries a median
72% of word-level distributional movement." That statistic is JS over the WHOLE scored
word set. **JS measures TOTAL movement; the claim is about DE-TRANSGRESSING, and the two
come apart badly.** SFT is where a model learns chat format, register, "As an AI
assistant", refusal templates -- an enormous distributional shift on nearly every prompt
with nothing to do with transgression. DPO may be a much smaller nudge aimed precisely at
the transgressive words. On that reading "SFT carries 72%" is a fact about FORMAT WORK and
the de-transgressing happens at the second stage, which inverts both the sentence and its
mapping onto the book's chapters (ch04 socialization / ch05 legislation).

**F40 supplies the control rather than an argument.** Its 347 discovered words are
blind-tagged twice into seven categories, and PROCEDURAL is exactly the format channel the
confound runs through, while TRANSGRESSIVE is the channel the claim is about. So the
confound stops being something to reason about and becomes a COLUMN.

WHY THE ARCHANGEL CELL. archangel-{dpo,kto,ppo,slic} share base (pythia-2.8b) AND ego
(archangel_sft) and differ ONLY in the preference method. So the SFT stage is held
constant BY CONSTRUCTION -- `JS(base,ego)` is the same number for all four, and any spread
is attributable to the second stage alone. One family cannot separate "the first stage did
the work" from "the second stage was weak"; four methods on one base can. The cell is
complete in the v3 grid already.

THREE THINGS THIS PRINTS AND WHY EACH IS THERE

  1. JS PER STAGE, and the share -- the clause-8 statistic, reproduced on v3 cells so the
     new number has an old one to sit beside.
  2. F40 CATEGORY MASS FLOW PER STAGE -- signed sum of dp over each tag's words. Negative
     means mass LEFT that category at that stage. This is the measure the claim needs.
  3. THE PINNED-NUMERATOR CHECK -- JS(base,ego) must be identical across the four arms.
     It is a free integrity test: if v3 breaks it, something upstream is wrong, because
     those two checkpoints do not know which preference method comes later.

DECISIONS THAT COULD HAVE HIDDEN THE EFFECT, MADE EXPLICITLY

  THE TAIL IS A BUCKET, NOT A RENORMALISATION. true_word_probs is truncated at theta, so
  the scored words sum to 1 - residual. Renormalising to 1 would DELETE exactly the
  movement of interest -- mass leaving the scored set entirely -- and report a
  redistribution among survivors instead. The residual enters as one `__TAIL__` mass so
  every distribution sums to 1 and mass that exits is visible as exit.

  BOTH TAGGINGS ARE REPORTED, NEVER AVERAGED. v1 and v2 agree on only 79.5% of primary
  tags. Averaging them would manufacture a precision neither has; printing both makes the
  disagreement part of the read. TRANSGRESSIVE holds 24 words in v1 and 27 in v2 against
  DEMOTIC's 119/133, so it is the noisiest column and is labelled with its n.

  WORDS MATCH CASE-INSENSITIVELY (the vocabulary has `Damn`, the cells have `damn`), and
  a word tagged differently by the two passes contributes to a different column in each --
  which is the point of printing both.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = os.path.join(ROOT, "data", "twp_grid_v3")
VOCAB = os.path.join(ROOT, "data", "f40_vocab")
CATS = os.path.join(ROOT, "data", "prompt_categorisation.json")

BASE = "EleutherAI/pythia-2.8b"
EGO = "ContextualAI/archangel_sft_pythia2-8b"
SUPS = {
    "dpo":  "ContextualAI/archangel_sft-dpo_pythia2-8b",
    "kto":  "ContextualAI/archangel_sft-kto_pythia2-8b",
    "ppo":  "ContextualAI/archangel_sft-ppo_pythia2-8b",
    "slic": "ContextualAI/archangel_sft-slic_pythia2-8b",
}
TAGS = ["TRANSGRESSIVE", "PROCEDURAL", "DEMOTIC", "NARRATIVE_CRAFT",
        "AFFECT", "CONTESTATION", "OTHER"]


def load_arm(model):
    path = os.path.join(GRID, model.replace("/", "__") + ".jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            dist = {r["word"]: r["p"] for r in d.get("rows") or []}
            # THE TAIL IS A BUCKET. See the docstring: renormalising would delete the
            # movement this script exists to measure.
            dist["__TAIL__"] = (d.get("residual") or {}).get("total", 0.0)
            out[d["prompt"]] = dist
    return out


def js(p, q):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1.0, sum(q.values()) or 1.0
    d = 0.0
    for k in keys:
        a, b = p.get(k, 0.0) / sp, q.get(k, 0.0) / sq
        m = 0.5 * (a + b)
        if m <= 0:
            continue
        if a > 0:
            d += 0.5 * a * math.log2(a / m)
        if b > 0:
            d += 0.5 * b * math.log2(b / m)
    return max(0.0, d)


def load_tags():
    out = {}
    for v in ("v1", "v2"):
        path = os.path.join(VOCAB, f"vocab_tagged_{v}.csv")
        m = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                w = (row.get("word") or "").strip().lower()
                if w:
                    m[w] = (row.get("primary") or "OTHER").strip()
        out[v] = m
    return out


def flow(p, q, tagmap):
    """Signed mass moved from p to q, in percentage points, per tag."""
    acc = collections.Counter()
    for w in set(p) | set(q):
        t = tagmap.get(w.strip().lower())
        if t is None:
            continue
        acc[t] += (q.get(w, 0.0) - p.get(w, 0.0)) * 100.0
    return acc


def main(domain, theta_note):
    tags = load_tags()
    for v in tags:
        n = collections.Counter(tags[v].values())
        print(f"  vocab {v}: " + "  ".join(f"{t}={n.get(t,0)}" for t in TAGS))

    dom = {}
    if os.path.exists(CATS):
        for r in json.load(open(CATS))["prompts"]:
            if r.get("status") == "ACTIVE" and r.get("prompt"):
                dom.setdefault(r["prompt"], r.get("domain"))

    base, ego = load_arm(BASE), load_arm(EGO)
    if not base or not ego:
        print("base or ego arm absent from the v3 grid")
        return
    sups = {k: load_arm(m) for k, m in SUPS.items()}
    sups = {k: v for k, v in sups.items() if v}
    if not sups:
        print("no superego arms present")
        return

    common = set(base) & set(ego)
    for s in sups.values():
        common &= set(s)
    if domain:
        common = {p for p in common if dom.get(p) == domain}
    prompts = sorted(common)
    print(f"\nprompts common to all {2+len(sups)} arms: {len(prompts)}"
          + (f"   (domain={domain})" if domain else ""))
    if not prompts:
        return

    print(f"\n{'='*86}\n1. JS PER STAGE  (the clause-8 statistic, on v3 cells)\n{'='*86}")
    print(f"{'method':<8}{'JS base>ego':>13}{'JS ego>sup':>13}{'share':>9}"
          f"{'n':>7}")
    js_be_all = []
    for k, sup in sups.items():
        be = [js(base[p], ego[p]) for p in prompts]
        es = [js(ego[p], sup[p]) for p in prompts]
        mbe = sorted(be)[len(be)//2]
        mes = sorted(es)[len(es)//2]
        js_be_all.append(round(mbe, 6))
        share = mbe / (mbe + mes) if (mbe + mes) else float("nan")
        print(f"{k:<8}{mbe:>13.6f}{mes:>13.6f}{share:>9.3f}{len(prompts):>7}")

    print(f"\n  PINNED-NUMERATOR CHECK: JS(base,ego) across the four arms -> "
          f"{sorted(set(js_be_all))}")
    print("  identical is REQUIRED -- same two checkpoints; they do not know which "
          "preference method follows." if len(set(js_be_all)) == 1 else
          "  *** NOT IDENTICAL. Something upstream is wrong; the arms share base and ego "
          "by construction. ***")

    for v in ("v1", "v2"):
        tm = tags[v]
        nt = collections.Counter(tm.values())
        print(f"\n{'='*86}\n2. F40 MASS FLOW BY CATEGORY, tagging {v}  "
              f"(percentage points; NEGATIVE = mass LEFT)\n{'='*86}")
        print(f"{'stage':<16}" + "".join(f"{t[:11]:>13}" for t in TAGS))
        print(f"{'':<16}" + "".join(f"{'n='+str(nt.get(t,0)):>13}" for t in TAGS))
        acc = collections.Counter()
        for p in prompts:
            acc.update(flow(base[p], ego[p], tm))
        print(f"{'base>ego (SFT)':<16}"
              + "".join(f"{acc.get(t,0.0)/len(prompts):>13.4f}" for t in TAGS))
        for k, sup in sups.items():
            a2 = collections.Counter()
            for p in prompts:
                a2.update(flow(ego[p], sup[p], tm))
            print(f"{'ego>sup ' + k:<16}"
                  + "".join(f"{a2.get(t,0.0)/len(prompts):>13.4f}" for t in TAGS))

    # THE CONTROL THAT DECIDES WHETHER SECTION 2 MEANS ANYTHING.
    # Section 2 shows SFT adding mass to EVERY category at once. That is what SHARPENING
    # looks like -- probability leaving the tail for common words generally -- and it is
    # indistinguishable from "SFT favours transgression" in absolute percentage points.
    # The test: track the tail, and express each category as a SHARE OF SCORED MASS. If
    # SFT lifts everything proportionally the shares stay flat; if it targets a category
    # the share moves. A raw flow read without this reports sharpening as targeting.
    print(f"\n{'='*86}\n2b. THE SHARPENING CONTROL\n{'='*86}")
    arms = [("base", base), ("ego", ego)] + [(f"sup {k}", s) for k, s in sups.items()]
    print(f"{'arm':<12}{'TAIL mass':>12}{'scored mass':>13}"
          + "".join(f"{t[:9]:>11}" for t in ("TRANSGRESSIVE", "PROCEDURAL", "DEMOTIC")))
    print(f"{'':<12}{'':>12}{'':>13}" + "".join(f"{'% of scored':>11}" for _ in range(3)))
    tm = tags["v2"]
    for name, arm in arms:
        tail = sum(arm[p].get("__TAIL__", 0.0) for p in prompts) / len(prompts)
        scored = sum(sum(v for k, v in arm[p].items() if k != "__TAIL__")
                     for p in prompts) / len(prompts)
        cells = []
        for t in ("TRANSGRESSIVE", "PROCEDURAL", "DEMOTIC"):
            m = sum(sum(v for k, v in arm[p].items()
                        if k != "__TAIL__" and tm.get(k.strip().lower()) == t)
                    for p in prompts) / len(prompts)
            cells.append(100.0 * m / scored if scored else float("nan"))
        print(f"{name:<12}{tail:>12.4f}{scored:>13.4f}"
              + "".join(f"{c:>11.3f}" for c in cells))
    print("\n  A category whose SHARE OF SCORED MASS is flat across arms was not targeted,")
    print("  however large its absolute flow. Read section 2 only against this table.")

    print(f"\n{'='*86}\n3. THE READ\n{'='*86}")
    print("  Compare row `base>ego (SFT)` against the `ego>sup` rows, column by column.")
    print("  PROCEDURAL large at SFT + TRANSGRESSIVE large at ego>sup  -> clause 8's JS")
    print("    was measuring FORMAT work and the de-transgressing is the second stage.")
    print("  TRANSGRESSIVE large at SFT                                -> clause 8 holds")
    print("    on an instrument that can tell the two apart.")
    print("  TRANSGRESSIVE n is 24-27 of 347 and the two taggings agree on 79.5% of")
    print("    primaries: treat a difference between v1 and v2 columns as the error bar,")
    print("    and do not quote a category whose sign flips between them.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=None,
                    help="restrict to prompts of this categorisation domain")
    a = ap.parse_args()
    main(a.domain, None)
