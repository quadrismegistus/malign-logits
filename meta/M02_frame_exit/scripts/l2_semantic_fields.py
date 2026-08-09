#!/usr/bin/env python
"""Semantic fields over the L2 passages: is there a CONTRADICTION-specific
alignment effect, over and above what alignment does to any prompt?

THE DESIGN, WHICH IS THE WHOLE POINT.

The quintuplets give a control matched on syntax rather than on nothing:

    both        He loved her and hated her and wanted to      <- contradiction
    control_a   He loved her and adored her and wanted to     <- one pole, doubled
    control_b   He hated her and despised her and wanted to   <- other pole, doubled

Same clause count, same conjunction, same "wanted to" hinge. What `both` has and
the controls lack is the contradiction. So the quantity of interest is a
difference in differences, per model pair:

    D_CONTRA  = rate(aligned, both)     - rate(base, both)
    D_CONTROL = rate(aligned, controls) - rate(base, controls)
    SPECIFIC  = D_CONTRA - D_CONTROL

`both` here is the ROLE NAME -- the prompt carrying both poles -- not "both
arms". Every one of these three quantities is an aligned-minus-base difference;
what changes between them is which prompt it was taken on.

D_CONTROL is what alignment does to a prompt of this shape in general. Only the
residual is about holding two things at once. Tested with Wilcoxon over the 26
pairs, which is the unit that can fail.

ALL THREE ARE TESTED, not only the residual. A SPECIFIC near zero is ambiguous
between "alignment does nothing here" and "alignment does the same thing to both
prompt types", and those are opposite readings. Only D_CONTRA and D_CONTROL
separate them. The JSON keys keep their original names (d_both, p_both) so
files written before the rename still load.

POLE WORDS ARE STRIPPED BEFORE COUNTING, AND THIS IS NOT OPTIONAL.

The three roles differ lexically BY CONSTRUCTION -- loved/hated against
loved/adored against hated/despised -- and those are exactly the high-affect
words a semantic-field count keys on. A continuation that echoes its prompt
therefore inherits its role's field profile with no model behaviour involved.
Measured on this corpus the same day: 82% of the cases where a continuation
brought both pole words back were the prompt restated. So the primary strips
each passage's own prompt content words; --keep-poles runs it unstripped and
the two should be read together.

RATES ARE SHARES OF CLASSIFIED WORDS, not of all words.

Denominator is `n_counted`. If it were the token count, a difference in how much
of a text the lexicon KNOWS would appear as a difference in fields, and the two
arms do not write the same vocabulary. Coverage is reported as its own row so a
coverage shift is visible as itself.

    l2_semantic_fields.py --source meta
    l2_semantic_fields.py --source usas_fine --top 12
    l2_semantic_fields.py --source norms
    l2_semantic_fields.py --source meta --keep-poles      # sensitivity
"""
import argparse
import difflib
import glob
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")

from malign_logits import fields as F                      # noqa: E402

ROLES = ("both", "control_a", "control_b")
TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")


def prompt_content(prompt):
    """Content words of the prompt, to strip from its own continuation."""
    return {w.lower() for w in TOKEN.findall(prompt) if F.is_content_word(w)}


def strip_words(text, drop):
    if not drop:
        return text
    return " ".join(w for w in text.split()
                    if TOKEN.sub(lambda m: m.group(0), w).strip(".,!?\"'").lower()
                    not in drop)


def load(keep_poles, limit_per_cell=None):
    receipt = json.load(open(os.path.join(ROOT, "data", "f11_l2_receipt.json")))
    pairs = [(c["base"], c["aligned"]) for c in receipt["complete"]]
    aligned = {a for _, a in pairs}
    pair_of = {}
    for b, a in pairs:
        pair_of[b] = pair_of[a] = a
    quints = {q["group"]: q for q in json.load(
        open(os.path.join(ROOT, "data", "f11_quintuplets.json")))["quintuplets"]}

    out = []
    seen = defaultdict(int)
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "f11_l2",
                                              "*.gen.jsonl"))):
        model = os.path.basename(path).rsplit(".gen.jsonl", 1)[0].replace("__", "/")
        if model not in pair_of:
            continue
        arm = "aligned" if model in aligned else "base"
        for line in open(path):
            r = json.loads(line)
            if r.get("lang") != "en":
                continue
            for c in (r.get("claims") or []):
                role, g = c.get("role"), c.get("group")
                if role not in ROLES or g not in quints:
                    continue
                cell = (pair_of[model], g, role, arm)
                if limit_per_cell and seen[cell] >= limit_per_cell:
                    continue
                seen[cell] += 1
                text = r["text"]
                if not keep_poles:
                    text = strip_words(text, prompt_content(r["prompt"]))
                out.append((pair_of[model], arm, role, g, text))
                break
    return out, pairs


def score(rows, source):
    """Passage-level share of classified words, accumulated by cell.

    THE PASSAGE IS THE UNIT, not the word. Pooling words across a cell would
    weight long passages more, and length differs by arm (aligned exits the
    frame more, and an exited passage is a different length). Mean of
    passage-level rates weights each passage once.
    """
    acc = defaultdict(lambda: defaultdict(list))
    cov = defaultdict(list)
    for pair, arm, role, g, text in rows:
        if source == "norms":
            n = F.norms(text)
            tot = 0
            d = {}
            for dim, v in n.items():
                s = sum(v["counts"].values())
                for lab, k in v["counts"].items():
                    d["%s:%s" % (dim, lab)] = k / s if s else 0.0
                tot += v["coverage"]
            cov[(pair, arm, role)].append(tot / max(len(n), 1))
        else:
            c = F.count(text, source=source)
            tot = c["n_counted"]
            d = {k: v / tot for k, v in c["counts"].items()} if tot else {}
            cov[(pair, arm, role)].append(c["coverage"])
        for k, v in d.items():
            acc[(pair, arm, role)][k].append(v)
        acc[(pair, arm, role)]["__n"].append(1)
    return acc, cov


def cell_mean(acc, key, field):
    d = acc.get(key)
    if not d:
        return None
    n = len(d["__n"])
    if not n:
        return None
    return sum(d.get(field, [])) / n            # absent field contributes zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="meta",
                    choices=["meta", "usas_fine", "usas", "gi", "wordnet",
                             "rid", "norms"])
    ap.add_argument("--keep-poles", action="store_true")
    ap.add_argument("--top", type=int, default=0,
                    help="show only the N largest |effect| rows")
    ap.add_argument("--limit-per-cell", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from scipy.stats import wilcoxon
    import numpy as np

    rows, pairs = load(a.keep_poles, a.limit_per_cell)
    print("source=%s   pole words %s"
          % (a.source, "KEPT (sensitivity run)" if a.keep_poles else "stripped"))
    print("%s passages over %d pairs, roles %s"
          % (format(len(rows), ","), len(pairs), "/".join(ROLES)))
    acc, cov = score(rows, a.source)

    allf = sorted({f for d in acc.values() for f in d if f != "__n"})
    print("%d fields\n" % len(allf))

    #: Coverage first. A field table read without it compares how much of each
    #: arm's text the lexicon happens to know.
    cv = {}
    for arm in ("base", "aligned"):
        v = [x for (p, ar, ro), xs in cov.items() if ar == arm for x in xs]
        cv[arm] = float(np.mean(v)) if v else float("nan")
    print("  lexicon coverage  base %.3f   aligned %.3f   diff %+.4f"
          % (cv["base"], cv["aligned"], cv["aligned"] - cv["base"]))

    res = []
    for f in allf:
        d_both, d_ctrl, eff = [], [], []
        for _, al in pairs:
            vb = cell_mean(acc, (al, "base", "both"), f)
            va = cell_mean(acc, (al, "aligned", "both"), f)
            cb = [cell_mean(acc, (al, "base", r), f) for r in ("control_a", "control_b")]
            ca = [cell_mean(acc, (al, "aligned", r), f) for r in ("control_a", "control_b")]
            if None in (vb, va) or None in cb or None in ca:
                continue
            db = va - vb
            dc = (sum(ca) / 2) - (sum(cb) / 2)
            d_both.append(db)
            d_ctrl.append(dc)
            eff.append(db - dc)
        if len(eff) < 10 or all(abs(x) < 1e-12 for x in eff):
            continue
        #: Test all three, not only the residual. A DiD near zero means one of
        #: two very different things -- alignment does nothing here, or it does
        #: the same thing to both prompt types -- and only D_both/D_ctrl can
        #: tell them apart. Reporting the residual alone would let a large,
        #: consistent, general alignment effect vanish into a null.
        def wp(v):
            try:
                return wilcoxon(v).pvalue
            except Exception:
                return float("nan")
        res.append(dict(field=f, n=len(eff),
                        d_both=float(np.median(d_both)), p_both=wp(d_both),
                        up_both=sum(x > 0 for x in d_both),
                        d_ctrl=float(np.median(d_ctrl)), p_ctrl=wp(d_ctrl),
                        effect=float(np.median(eff)), p=wp(eff),
                        up=sum(x > 0 for x in eff)))

    #: BH over the whole field family for this source.
    m = len(res)
    order = sorted(range(m), key=lambda i: res[i]["p"])
    crit = 0
    for rank, i in enumerate(order, 1):
        if res[i]["p"] <= rank / m * 0.05:
            crit = rank
    passing = {order[r - 1] for r in range(1, crit + 1)}
    for i, r in enumerate(res):
        r["bh"] = i in passing

    #: BH again, separately, over the GENERAL alignment effect.
    om = sorted(range(m), key=lambda i: res[i]["p_both"])
    cb = 0
    for rank, i in enumerate(om, 1):
        if res[i]["p_both"] <= rank / m * 0.05:
            cb = rank
    passb = {om[r - 1] for r in range(1, cb + 1)}
    for i, r in enumerate(res):
        r["bh_both"] = i in passb

    print("\n  D_CONTRA  = aligned minus base ON THE both-ROLE (contradiction) PROMPT")
    print("  D_CONTROL = aligned minus base ON THE SINGLE-POLE CONTROL PROMPTS")
    print("  SPECIFIC  = D_CONTRA - D_CONTROL, the contradiction-specific residual\n")
    show = sorted(res, key=lambda r: -abs(r["d_both"]))
    if a.top:
        show = show[:a.top]
    print("  ordered by |D_CONTRA|, the general alignment effect")
    print("  %-38s %19s %9s %19s"
          % ("field", "D_CONTRA  up/26  p", "D_CONTROL", "SPECIFIC  up/26  p"))
    for r in show:
        print("  %-38s %+8.4f %3d %8.4g%s %+9.4f %+8.4f %3d %8.4g%s"
              % (r["field"][:38], r["d_both"], r["up_both"], r["p_both"],
                 "*" if r["bh_both"] else " ", r["d_ctrl"],
                 r["effect"], r["up"], r["p"], "*" if r["bh"] else " "))
    print("\n  * survives Benjamini-Hochberg at FDR 0.05, computed separately")
    print("    over the %d fields for D_both and for EFFECT." % m)
    print("  Median over pairs; shares of classified words, so a rise in one")
    print("  field is partly a fall in another.")
    ns = sum(r["bh_both"] for r in res), sum(r["bh"] for r in res)
    print("  SURVIVING: %d of %d on the general effect, %d of %d on the "
          "contradiction-specific one." % (ns[0], m, ns[1], m))

    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        json.dump(dict(source=a.source, keep_poles=a.keep_poles,
                       coverage=cv, results=res), open(p, "w"), indent=1)
        print("  wrote %s" % p)


if __name__ == "__main__":
    main()
