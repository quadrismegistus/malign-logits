#!/usr/bin/env python
"""f11_k2_units.py — the k>=2 coding frame, ENUMERATED.

    scripts/f11_k2_units.py --write   -> data/f11_k2_units.json

**THE FRAME MUST BE A FILE, NOT A LOOP** (pen, [5179].2). `f11_l1_candidate_vocab
.json` carries surfaces and total cell counts and cannot answer *how many models
put this surface above theta IN THIS GROUP* — which is exactly what k>=2 reads.
So both seats derived the frame by re-scanning the store with their own loop,
each inheriting its own roster, population and language-key choices, and the two
frames disagreed by 16%. Populations enumerate.

**k IS THE LIST OF VOTING MODELS, NOT THE COUNT** ([5179].6). A count cannot be
audited; a list can. It also makes the roster a property of the file rather than
of whoever ran it, which is the whole disagreement this file exists to end.

DECLARED, per [5179].3:

    language key   GROUP language -- the coder sees the pair and the pair has a
                   language. 4.6% of units have a surface in one language and a
                   group in the other; under surface-language they would move.
    k-roster       every checkpoint with twp data on the cell. A model with no
                   data cannot put a surface above theta, so "104 roster" and
                   "92 with data" are the same k by construction.
    population     the 41 status-filtered groups; reason/_zh held BESIDE;
                   f11_species_wolf dropped as wholly RETIRED; all six roles.
    theta          0.001 (RH [5136], tied to N3)
    predicate      p >= theta, NO shape filter ([5170]) -- the frequency test
                   removed 0.0% of English mass and 53.2% of Chinese.
"""
import argparse, collections, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
THETA = 0.001
K = 2
OUT = os.path.join(ROOT, "data", "f11_k2_units.json")
ZH = lambda s: bool(re.search(r"[一-鿿]", s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    from f11_quintuplet_spec import PROMPT_ROLES
    from malign_logits.cache import get_cache
    from malign_logits.registry import Registry
    import subprocess
    cm = get_cache()

    q = json.load(open(os.path.join(ROOT, "data", "f11_quintuplets.json")))["quintuplets"]
    items = q.items() if isinstance(q, dict) else [(e.get("group"), e) for e in q]
    groups, beside, dropped = {}, {}, {}
    for gid, v in items:
        if not isinstance(v, dict):
            continue
        name, st = v.get("group", gid), (v.get("status") or "").upper()
        cells = {r: v.get(r) for r in PROMPT_ROLES
                 if isinstance(v.get(r), str) and v.get(r)}
        if "RETIRED" in st:
            dropped[name] = v.get("status")
        elif name.startswith("f11_reason"):
            beside[name] = cells
        else:
            groups[name] = cells

    roster = sorted({m for p in Registry().base_aligned_pairs()
                     for m in (p["base"], p["aligned"])})
    idx = {m: i for i, m in enumerate(roster)}
    votes = collections.defaultdict(set)          # (surface, group) -> model indices
    with_data = set()
    for mid in roster:
        for g, roles in groups.items():
            for role, prompt in roles.items():
                v = cm.get_true_word_probs(mid, prompt, theta=THETA)
                if not v or not v.get("rows"):
                    continue
                with_data.add(mid)
                for r in v["rows"]:
                    if r.get("p", 0.0) >= THETA and r.get("word"):
                        votes[(r["word"], g)].add(idx[mid])

    glang = {g: ("zh" if ZH(next(iter(r.values()))) else "en")
             for g, r in groups.items()}
    rows = []
    for (w, g), ms in votes.items():
        rows.append({"surface": w, "group": g, "lang": glang[g],
                     "k": len(ms), "models": sorted(ms),
                     "survives": len(ms) >= K})
    rows.sort(key=lambda r: (r["group"], r["surface"]))

    by = collections.Counter((r["lang"], r["survives"]) for r in rows)
    print("k>=%d FRAME, ENUMERATED" % K)
    print("  roster declared   %d checkpoints, %d with twp data" % (len(roster), len(with_data)))
    print("  groups            %d primary | %d beside | %d dropped"
          % (len(groups), len(beside), len(dropped)))
    print("  units total       %d" % len(rows))
    for lang in ("en", "zh"):
        s, d = by[(lang, True)], by[(lang, False)]
        print("  %-4s  %6d total   %6d survive (%.1f%%)   %6d dropped"
              % (lang, s + d, s, 100 * s / (s + d) if s + d else 0, d))
    tot_s = by[("en", True)] + by[("zh", True)]
    print("  ALL   %6d total   %6d survive (%.1f%%)" % (len(rows), tot_s, 100 * tot_s / len(rows)))

    if a.write:
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True, cwd=ROOT).stdout.strip()
        except Exception:
            commit = None
        json.dump({
            "_about": "The k>=2 coding frame for the M02 redo L1 arm, one row per "
                      "(surface, group). k is the LIST of voting models, not the "
                      "count, so the roster is a property of this file rather than "
                      "of whoever ran the loop.",
            "_producer": "scripts/f11_k2_units.py",
            "_rulings": "[5170] no shape filter; [5176] k>=2 (RH); [5179] group "
                        "language, k-roster = every checkpoint with data, "
                        "population = 41 status-filtered groups, six roles.",
            "theta": THETA, "k": K,
            "roster_declared": roster,
            "roster_with_data": sorted(with_data),
            "population_groups": sorted(groups),
            "held_beside": sorted(beside), "dropped_retired": dropped,
            "roles": list(PROMPT_ROLES),
            "extends_vocab_commit": commit,
            "n_units": len(rows),
            "n_survive": tot_s,
            "by_language": {l: {"total": by[(l, True)] + by[(l, False)],
                                "survive": by[(l, True)]} for l in ("en", "zh")},
            "units": rows,
        }, open(OUT, "w"), ensure_ascii=False)
        print("\nwrote %s (%.1f MB)"
              % (os.path.relpath(OUT, ROOT), os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
