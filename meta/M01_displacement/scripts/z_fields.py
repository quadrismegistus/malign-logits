#!/usr/bin/env python
"""Registration Z: semantic fields over the generations stash, prompt-matched.

    python z_fields.py                 # all prompts
    python z_fields.py --split         # by prompt domain
    python z_fields.py --min-prompts 40

WHAT THIS IS FOR. Y answers "what does alignment do to five explicit sexual
prompts". The generations stash holds 256,035 passages over 152 prompts and 127
models, spanning the whole F01 battery — violence, substance, authority,
neutral controls. So Z asks the question Y structurally cannot: **is the
register shift a property of sexual material, or of alignment?**

THE PROMPTS DIFFER TOO MUCH TO POOL, and that is the design constraint (RH).
Zero prompts are shared by all 127 models; models carry between 4 and 117 of
them. A field profile computed per model and compared across models would be
measuring which prompts that model happens to have.

So every contrast is **PROMPT-MATCHED INSIDE A PAIR**:

    for each pair, for each prompt BOTH arms carry:
        delta(prompt) = field share (aligned) - field share (base)
    per-pair value = median over that pair's shared prompts
    across pairs   = Wilcoxon + bootstrap CI, unit = pair

A prompt only ever compares to itself, and a pair only ever compares to itself.
Nothing crosses either boundary.

TEMPERATURE IS PINNED TO 1.0. The stash also holds 9,360 passages at 0.7 and
2,261 at 0.0. Mixing decoders across arms is the defect that cost the beam
corpus a day (docket [4994]); here it is avoidable by filtering, so it is
filtered rather than noted.

MODE. Keys carry an optional `:continue` suffix on the model name (1,098 of
256,035). Those are a different generation mode and are dropped, not folded in.
"""
import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")
sys.path.insert(0, HERE)

from malign_logits.cache import CacheManager           # noqa: E402
from malign_logits import fields                        # noqa: E402
from y_paired_tests import wilcoxon, boot_ci            # noqa: E402

PAIRS = os.path.join(ROOT, "data", "base_aligned_pairs.json")
CATS = os.path.join(ROOT, "data", "prompt_categorisation.json")
MIN_PER_CELL = 3          #: generations per (pair, arm, prompt) to use that prompt


def load_domains():
    """prompt text -> domain, if the categorisation file can supply it."""
    if not os.path.exists(CATS):
        return {}
    try:
        d = json.load(open(CATS))
    except Exception:
        return {}
    out = {}
    items = d if isinstance(d, list) else d.get("prompts", d.values() if isinstance(d, dict) else [])
    for it in items:
        if not isinstance(it, dict):
            continue
        t = it.get("prompt") or it.get("text") or it.get("stem")
        dom = it.get("domain") or it.get("category") or it.get("stratum")
        if t and dom:
            out[t] = dom
    return out


def profile(text):
    """Field shares for one passage. Shares, so passage length cancels."""
    out = {}
    r = fields.count(text)
    n = r["n_counted"] or 0
    if n < 5:
        return None
    for g, c in r["counts"].items():
        out["F:" + g] = c / n
    for dim, x in fields.norms(text).items():
        t = sum(x["counts"].values())
        if t:
            for b, c in x["counts"].items():
                out["N:%s=%s" % (dim, b)] = c / t
    rd = fields.count(text, "rid")["counts"]
    rt = sum(rd.values())
    if rt >= 3:
        for g, c in rd.items():
            out["R:" + g.split(":")[0]] = out.get("R:" + g.split(":")[0], 0) + c / rt
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prompts", type=int, default=20,
                    help="a pair needs this many shared prompts to contribute")
    ap.add_argument("--split", action="store_true", help="report by prompt domain")
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args(argv)

    st = CacheManager()._stash("generations")
    roster = [(p["base"], p["aligned"]) for p in json.load(open(PAIRS))]
    want = {m for b, x in roster for m in (b, x)}

    #: index keys first, read values only for cells we will actually use --
    #: the stash holds 256k passages and most belong to models outside the
    #: roster or to a temperature we are not using.
    idx = collections.defaultdict(list)
    for k in st.keys():
        if k.get("temp") != 1.0:
            continue
        m = k.get("model") or ""
        if ":" in m:
            continue
        if m not in want:
            continue
        idx[(m, k.get("prompt"))].append(k)
    print("indexed %s (model, prompt) cells at temp=1.0, roster models only"
          % format(len(idx), ","))

    doms = load_domains()
    print("prompt domains available: %s" % ("yes, %d prompts" % len(doms) if doms else "no"))

    #: pair -> measure -> list of per-prompt deltas
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    accd = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    npass = 0
    for b, x in roster:
        shared = [p for (m, p) in idx if m == b and (x, p) in idx]
        if len(shared) < a.min_prompts:
            continue
        for p in shared:
            prof = {}
            for role, m in (("base", b), ("aligned", x)):
                ks = idx[(m, p)]
                if len(ks) < MIN_PER_CELL:
                    prof = None
                    break
                acc_m = collections.defaultdict(list)
                for k in ks:
                    try:
                        txt = st.get(k)
                    except Exception:
                        continue
                    if not isinstance(txt, str) or len(txt) < 120:
                        continue
                    pr = profile(txt)
                    if not pr:
                        continue
                    npass += 1
                    for g, v in pr.items():
                        acc_m[g].append(v)
                if len(acc_m) == 0:
                    prof = None
                    break
                prof[role] = {g: statistics.mean(v) for g, v in acc_m.items()}
            if not prof or "base" not in prof or "aligned" not in prof:
                continue
            dom = doms.get(p, "?")
            for g in set(prof["base"]) & set(prof["aligned"]):
                d = prof["aligned"][g] - prof["base"][g]
                acc[(b, x)][g].append(d)
                accd[dom][(b, x)][g].append(d)
    print("passages scored: %s   pairs contributing: %d\n" % (format(npass, ","), len(acc)))

    def report(store, title, top):
        print(title)
        print("  %-34s %9s %8s %18s %5s" % ("", "med d", "WILCOX", "boot 95% CI", "pairs"))
        print("  " + "-" * 76)
        meas = collections.Counter()
        for pr, gs in store.items():
            for g in gs:
                meas[g] += 1
        res = []
        for g, _ in meas.most_common():
            d = [statistics.median(gs[g]) for gs in store.values() if len(gs.get(g, [])) >= 10]
            if len(d) < 8:
                continue
            wp, _ = wilcoxon(d)
            lo, hi = boot_ci(d)
            res.append((wp, g, statistics.median(d), lo, hi, len(d)))
        res.sort()
        for wp, g, md, lo, hi, n in res[:top]:
            cl = " <=" if (lo > 0 or hi < 0) else ""
            print("  %-34s %+9.3f %8.4f  [%+6.3f,%+6.3f] %5d%s"
                  % (g, 100 * md, wp, 100 * lo, 100 * hi, n, cl))
        nn = sum(1 for r in res if r[3] > 0 or r[4] < 0)
        print("  %d of %d with CI excluding zero\n" % (nn, len(res)))

    report(acc, "Z: ALL PROMPTS, prompt-matched within pair", a.top)

    if a.split and doms:
        for dom in sorted(accd, key=lambda d: -len(accd[d])):
            if dom == "?" or len(accd[dom]) < 8:
                continue
            report(accd[dom], "Z: domain = %s" % dom, 8)
    return 0


if __name__ == "__main__":
    sys.exit(main())
