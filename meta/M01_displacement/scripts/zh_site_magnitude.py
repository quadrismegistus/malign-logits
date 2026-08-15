"""F AND G AT CHINESE TRANSGRESSIVE SITES — the empty cell in M01's own 2x3.

    uv run python meta/M01_displacement/scripts/zh_site_magnitude.py

The README's axis 1 names F and G for "at transgressive sites" with no O beside
them: **the site question has never been asked in Chinese.** The data has been
collected the whole time. This asks it.

WHAT THIS IS NOT
================

**NOT a replication of F/G.** F and G run on `M01_PAIRS`, 684 pairs, and NONE of
them was ever translated -- the overlap between the M01 pair corpus and the
Chinese translation set is EXACTLY ZERO, checked by pair_id and by English
string. The Chinese transgressive minimal pairs come from SETE, SETD,
F36_MINIMAL_PAIRS and CENSUS. So language and corpus BOTH change, and a
difference from the English result cannot be attributed to language.

**The instrument is the same and that is the whole of what transfers.** Same
frozen site rule (imported, hash-checked, never reimplemented), same
`departed`/`concentration` quantities, same unit (the base checkpoint), same
sign-flip permutation. `build`, `pair_quantity` and `sign_flip_p` are IMPORTED
from `magnitude.py` so the definitions cannot drift.

POWER, COMPUTED BEFORE THE RUN AND NOT AFTER
============================================

    F, sign test        n=20 units   MDE P(positive) = 0.799
                        English realised 0.606 (a null); detecting THAT needs
                        n = 141 units.
    G, sign-flip perm   n=20 units   80% power at standardised d ~ 0.58

**So F IS NOT RUN AS A TEST HERE.** Its rate quantities are computed and
reported as description, because G's §6 admissibility depends on the two arms
firing at indistinguishable rates and that has to be checked rather than
inherited from an English null. A Chinese F verdict would be uninterpretable at
this n, and M01's own reading rule 2 -- no null without its MDE -- forbids
reporting one as though it meant something.

G is the registered-style test. Its p-value is reported with the MDE beside it.
"""
import collections
import hashlib
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

FROZEN_SITES = "b8fd9a52cd5c794b"
CJK_OK = {"FLUENT", "MARGINAL", "PARTIAL"}       #: ordinal, not a name list
TRANSGRESSIVE = "transgressive_swap"


def zh_pairs():
    """The Chinese transgressive minimal pairs, as {group: {MARKED, UNMARKED}}.

    Keyed on the CHINESE string, because that is what the store holds. Matching
    on `prompt_categorisation`'s `prompt` field returns zero rows -- that field
    carries the ENGLISH twin for these ids, and reading it as the prompt is how
    a first pass here reported 0 collected pairs out of 356.
    """
    cat = json.load(open(os.path.join(ROOT, "data/prompt_categorisation.json")))["prompts"]
    cat = list(cat.values()) if isinstance(cat, dict) else cat
    byid = {str(r.get("prompt_id")): r for r in cat}
    zh = {}
    for f in ("data/chinese_translations.json", "data/chinese_translations_2.json"):
        for p in json.load(open(os.path.join(ROOT, f)))["prompts"]:
            zh[p["prompt_id"]] = p
    grp = collections.defaultdict(dict)
    for pid, p in zh.items():
        if p.get("group_role") in ("MARKED", "UNMARKED"):
            grp[p["group"]][p["group_role"]] = (pid, p)
    out, meta = {}, {}
    for g, v in grp.items():
        if len(v) != 2:
            continue
        recs = [byid.get(v[r][0]) for r in ("MARKED", "UNMARKED")]
        if not all(recs) or {r.get("contrast_type") for r in recs} != {TRANSGRESSIVE}:
            continue
        out[g] = {r: v[r][1]["chinese"].strip() for r in ("MARKED", "UNMARKED")}
        meta[g] = {"domain": recs[0].get("domain"),
                   "pair_minimal": v["MARKED"][1].get("pair_minimal"),
                   "source": recs[0].get("source")}
    return out, meta


def cjk_edges():
    """{base: [aligned arms]} over pairs whose BOTH arms clear tier >= PARTIAL."""
    import x_bodypart_classes as B
    #: THROUGH `Checkpoint`, NOT THE RAW FILE. `.record` because the rows are
    #: read with `.get()` below and `__getattr__` raises where `.get()`
    #: returns None -- the absent-model path a byte-identical diff cannot show.
    from malign_logits.checkpoint import Checkpoint as _CP
    reg = {cp.id: cp.record for cp in _CP.all()}
    same, cross = B.roster()
    edges = collections.defaultdict(list)
    for b, a in same + cross:
        tb = str(reg.get(b, {}).get("cjk_tier"))
        ta = str(reg.get(a, {}).get("cjk_tier"))
        if tb in CJK_OK and ta in CJK_OK:
            edges[b].append(a)
    return dict(edges), reg


def load_grid(models, texts, S):
    """Bulk ClickHouse load. One query per model, not one per cell."""
    from malign_logits.ch_read import prefetch
    grid, resid = collections.defaultdict(dict), collections.defaultdict(dict)
    for m in models:
        try:
            cells = prefetch(m)
        except Exception as e:
            print("   prefetch failed %-46s %s" % (m, type(e).__name__))
            continue
        for t in texts:
            v = cells.get(t)
            rs = (v or {}).get("rows") or []
            if rs:
                grid[m][t] = S.prepare(rs)
                resid[m][t] = float(((v or {}).get("residual") or {}).get("total", 0.0))
    return dict(grid), dict(resid)


def main():
    import m05_sites as S
    h = hashlib.sha256(open(os.path.join(HERE, "m05_sites.py"), "rb").read()).hexdigest()[:16]
    print("site rule %s  %s" % (h, "OK" if h == FROZEN_SITES else "MISMATCH -> REFUSING"))
    if h != FROZEN_SITES:
        return 1
    import magnitude as G
    from within_pair import sign_test, mde

    pairs, meta = zh_pairs()
    edges, reg = cjk_edges()
    texts = {t for v in pairs.values() for t in v.values()}
    print("zh transgressive minimal pairs: %d   |   cjk base units: %d   arms: %d"
          % (len(pairs), len(edges), sum(len(v) for v in edges.values())))
    print("domains: %s" % dict(collections.Counter(m["domain"] for m in meta.values())))

    models = sorted(set(edges) | {a for v in edges.values() for a in v})
    print("\nloading %d models from ClickHouse ..." % len(models))
    grid, resid = load_grid(models, texts, S)
    print("models with any zh cell: %d" % len(grid))

    # ── F's rate quantities, DESCRIPTIVE (see the docstring on power) ──────
    rate = {}
    for b, arms in edges.items():
        per = []
        for a in arms:
            gb, ga = grid.get(b, {}), grid.get(a, {})
            atrisk = fm = fu = 0
            for g, mem in pairs.items():
                if all(t in gb and t in ga for t in mem.values()):
                    atrisk += 1
                    for role, acc in (("MARKED", "M"), ("UNMARKED", "U")):
                        if "FREE" in S.classify(gb[mem[role]], ga[mem[role]], frozenset()):
                            if acc == "M":
                                fm += 1
                            else:
                                fu += 1
            if atrisk:
                per.append((atrisk, fm / atrisk, fu / atrisk))
        if per:
            rate[b] = {"at_risk": st.median([p[0] for p in per]),
                       "rate_M": st.median([p[1] for p in per]),
                       "rate_U": st.median([p[2] for p in per])}
    deltas = [v["rate_M"] - v["rate_U"] for v in rate.values()]
    ft = sign_test(deltas)
    k, m_ = mde(ft["n"])
    print("\n" + "=" * 74)
    print("F -- RATE AT CHINESE SITES.  DESCRIPTIVE, NOT A VERDICT.")
    print("=" * 74)
    print("  units %d   median at-risk pairs/unit %.0f" %
          (len(rate), st.median([v["at_risk"] for v in rate.values()]) if rate else 0))
    print("  median rate_M %.4f   rate_U %.4f   median Delta %+.4f"
          % (st.median([v["rate_M"] for v in rate.values()]),
             st.median([v["rate_U"] for v in rate.values()]), st.median(deltas)))
    print("  positives %d/%d   p %.4f   MDE at this n: P(pos) >= %.3f"
          % (ft["positives"], ft["n"], ft["p_value"], m_))
    print("  ENGLISH F: 20/33 positive, p 0.1481 -- a null. Different corpus; not comparable.")

    # ── G's admissibility: does both-fire conditioning select differentially? ──
    units = G.build(edges, grid, resid, pairs, S)
    print("\nunits with a magnitude: %d" % len(units))
    skew = [v["skew"] for v in units.values() if v["skew"] is not None]
    print("\nCONDITIONING CHECK (G §6 -- both-fire is only safe if the arms fire alike)")
    print("  median onlyMARKED %.1f  onlyUNMARKED %.1f  median skew %+.4f  max |skew| %.4f"
          % (st.median([v["onlyM"] for v in units.values()]),
             st.median([v["onlyU"] for v in units.values()]),
             st.median(skew) if skew else float("nan"),
             max(abs(s) for s in skew) if skew else float("nan")))

    dep = [v["D_departed"] for v in units.values() if v["D_departed"] is not None]
    con = [v["D_concentration"] for v in units.values() if v["D_concentration"] is not None]
    print("\n" + "=" * 74)
    print("G -- MAGNITUDE AT CHINESE SITES.  sign-flip permutation, one-sided upper.")
    print("=" * 74)
    for name, vals in (("PRIMARY  departed", dep), ("SECONDARY concentration", con)):
        if not vals:
            print("  %-24s no units" % name)
            continue
        r = G.sign_flip_p(vals)          #: a DICT, not a float
        d = G.cohen_d(vals)
        print("  %-24s n %-3d median %+0.6f  d %s  p %.5f%s  %s"
              % (name, len(vals), st.median(vals),
                 ("%+.3f" % d) if d is not None else "n/a", r["p_value"],
                 " (AT RESOLUTION LIMIT)" if r["at_resolution_limit"] else "",
                 "REJECT" if r["reject"] else "no"))
    print("\n  positives: departed %d/%d, concentration %d/%d"
          % (sum(1 for v in dep if v > 0), len(dep),
             sum(1 for v in con if v > 0), len(con)))
    print("  80%% power at this n reaches standardised d ~ 0.58 (simulated, 20k draws).")
    print("  Base-side confound column (G §7): median base top-mass MARKED %.4f UNMARKED %.4f"
          % (st.median([v["base_top_M"] for v in units.values()]),
             st.median([v["base_top_U"] for v in units.values()])))

    #: PROVENANCE. The per-unit records already name every base and its arms,
    #: so the roster reconstructs -- but the TIER RULE, the prompt set and the
    #: commit do not, and those are what make the roster mean something later.
    import subprocess
    prov = {"git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
            "site_rule_sha16": FROZEN_SITES,
            "cjk_ok": sorted(CJK_OK),
            "edges": {b: sorted(a) for b, a in edges.items()},
            "models_with_cells": sorted(grid),
            "models_requested": models,
            "models_missing": sorted(set(models) - set(grid)),
            "pair_groups": sorted(pairs),
            "n_pairs": len(pairs), "n_units": len(edges)}
    out = os.path.join(CAMPAIGN, "results", "zh_site_magnitude.json")
    json.dump({"_provenance": prov, "pairs": {g: {"texts": pairs[g], **meta[g]} for g in pairs},
               "rate": rate, "units": units,
               "F": {**ft, "mde_P": m_},
               "G": {"departed": dep, "concentration": con,
                     "p_departed": G.sign_flip_p(dep) if dep else None,
                     "p_concentration": G.sign_flip_p(con) if con else None,
                     "power_note": "80% at standardised d ~0.58, n=20"}},
              open(out, "w"), indent=1, ensure_ascii=False, default=str)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
