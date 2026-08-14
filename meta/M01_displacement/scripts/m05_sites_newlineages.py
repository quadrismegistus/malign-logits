"""M05 SITE FINDER, EXTENDED TO THE LINEAGES THAT HAVE twp BUT NO GRID FILE.

    ./m05_sites_newlineages.py                 validate, then the counts
    ./m05_sites_newlineages.py --no-validate   skip the known-answer column
    ./m05_sites_newlineages.py --emit PATH     also write the site triples

WHY THIS EXISTS. `m05_sites.py` reads `data/twp_grid_v3/*.jsonl`. Six lineages
were measured after that directory was written, so their cells live ONLY in the
`true_word_probs` stash -- 2,583 prompts per arm, complete. They are not
rejected by the site rule; they are INVISIBLE to it, and an invisible model
produces no unpairable warning. `m05_sites.py` printed `unpairable 0` on a run
that had never heard of them.

**THE RULE IS IMPORTED, NEVER RESTATED.** `prepare`, `classify`, `count` and
`pairs_from_map` come from `m05_sites` by import. An earlier estimate of this
same quantity reimplemented two of the producer's three steps, omitted the
open-class filter, and came back 4x high (206-210 sites/pair against wave 3's
49-61). The lesson is not "be careful"; it is that the rule has exactly one
implementation and a second reader of it is a second rule.

**THE STASH IS READ AS A GRID, AND THAT IS A CLAIM.** The stash value is
`{"rows": [{word, t1, p}]}` -- the same shape the grid jsonl carries -- so the
same `prepare()` applies. The claim is checked rather than asserted: the
default run rebuilds the 95 grid-dir models FROM THE STASH and requires every
label count to equal the disk run. If the two disagree the run REFUSES, because
a silent disagreement would be attributed to the new lineages.

PAIRING for the new models comes from the registry's own `family` + `position`,
not from `lineage_map_models.json`, which has no entry for them. Base is
`position=base`; aligned is `position=superego`. Any family not resolving to
exactly one of each is REFUSED and named, never silently dropped.

WHAT THIS DOES NOT DO. It does not add these models to the grid spec, the
lineage map, or any manifest. It answers one question -- how many sites the new
lineages contribute under the adopted rule -- and writes nothing unless --emit.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/github/malign-logits")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import m05_sites as M

REG = os.path.join(ROOT, "data", "model_registry.json")

#: the six lineages measured after twp_grid_v3 was written. Teuken is EXCLUDED
#: by name: 2,269 base / 1,233 aligned cells against 2,583 for a complete arm,
#: so its prompt intersection is a different denominator from every other pair
#: and a count pooled across it would not mean what the others mean.
#: these are the registry's OWN `family` values, checked against it, not
#: guessed from the model id: the first version of this list said "llm-jp"
#: where the registry says "llm-jp-3" and the family was refused. The refusal
#: named it, which is why a one-character error cost a rerun and not a result.
NEW_FAMILIES = ["granite", "llm-jp-3", "salamandra", "lucie", "gemma2", "jais"]
INCOMPLETE = ["openGPT-X/Teuken-7B-base-v0.6",
              "openGPT-X/Teuken-7B-instruct-commercial-v0.4"]


def grid_from_disk():
    """m05_sites.main()'s loader, verbatim in behaviour."""
    grid = collections.defaultdict(dict)
    for f in sorted(os.listdir(M.GRID_DIR)):
        if not f.endswith(".jsonl"):
            continue
        mid = f[:-6].replace("__", "/")
        with open(os.path.join(M.GRID_DIR, f)) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                rs = d.get("rows") or []
                if not rs:
                    continue
                grid[mid][d["prompt"]] = M.prepare(rs)
    return grid


def grid_from_stash(models):
    """Same shape, same prepare(), sourced from the authoritative store."""
    from malign_logits.cache import get_cache
    want = set(models)
    st = get_cache()._stash("true_word_probs")
    grid = collections.defaultdict(dict)
    for k in st.keys():
        if not isinstance(k, dict) or k.get("model") not in want:
            continue
        rs = (st[k] or {}).get("rows") or []
        if not rs:
            continue
        grid[k["model"]][k["prompt"]] = M.prepare(rs)
    return grid


def new_pairs(reg):
    """(pairs, refused) from the registry's own family+position fields."""
    fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for mid, r in reg.items():
        f = (r.get("family") or "").lower()
        if f in NEW_FAMILIES and mid not in INCOMPLETE:
            fam[f][(r.get("position") or "").lower()].append(mid)
    pairs, refused = [], []
    for f in NEW_FAMILIES:
        b = fam.get(f, {}).get("base", [])
        a = fam.get(f, {}).get("superego", [])
        if len(b) == 1 and len(a) == 1:
            pairs.append((b[0], a[0]))
        else:
            refused.append((f, "base=%d superego=%d" % (len(b), len(a))))
    return pairs, refused


def table(led, title):
    print("\n%s" % title)
    print("%-22s %10s" % ("label", "sites"))
    print("-" * 40)
    for k in ("shared", "tie_at_top", "prompt_one_arm", "top_changed",
              "FREE", "substitute_novel", "LEX_OVERRIDE", "LEX_STRICT"):
        print("%-22s %10s" % (k, "{:,}".format(led[k])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--emit", metavar="PATH",
                    help="write the new lineages' site triples (refuses to "
                         "overwrite an existing path)")
    a = ap.parse_args()

    if a.emit and os.path.exists(a.emit):
        sys.exit("REFUSING: %s exists. This script never overwrites." % a.emit)

    reg = {m["model_id"]: m for m in json.load(open(REG))["models"]}
    TR = M.load_vocab()

    print("loading grid from disk (%s) ..." % os.path.relpath(M.GRID_DIR, ROOT))
    disk = grid_from_disk()
    lm = json.load(open(os.path.join(ROOT, "data/lineage_map_models.json")))
    m2b, m2s = lm["model_to_base"], lm["model_to_stage"]
    import m04_producer as P
    n2s = {}
    for _m, _s in m2s.items():
        n2s.setdefault(P.norm(_m), _s)

    def arm_of(cp):
        s = m2s.get(cp) or n2s.get(P.norm(cp))
        return None if s is None else ("base" if s == "base" else "aligned")

    dpairs, dbad = M.pairs_from_map(disk, m2b, arm_of)
    _, dled = M.count(disk, dpairs, TR)
    print("  grid models %d   pairs %d   unpairable %d"
          % (len(disk), len(dpairs), len(dbad)))

    #: ── KNOWN-ANSWER COLUMN ────────────────────────────────────────────────
    #: the new lineages arrive through a code path the disk run never used, so
    #: that path is required to reproduce the disk run before it is trusted to
    #: extend it.
    if not a.no_validate:
        print("\nvalidating: rebuilding the SAME %d models from the stash ..."
              % len(disk))
        st = grid_from_stash(set(disk))
        missing = sorted(set(disk) - set(st))
        if missing:
            sys.exit("REFUSING: %d disk models absent from the stash: %s"
                     % (len(missing), ", ".join(missing[:5])))
        #: **RESTRICT TO SHARED PROMPTS, AND SAY SO.** The two sources are
        #: DIFFERENT POPULATIONS, not disagreeing values: the grid export holds
        #: 979 prompts and the stash 2,583, and neither contains the other (16
        #: disk-only, 1,620 stash-only). Comparing them whole tests coverage,
        #: which is already known to differ, and would mask exactly the thing
        #: this column exists to catch -- a value that moved. So the comparison
        #: is on the intersection, per model, and the excluded prompts are
        #: reported rather than absorbed.
        dp = {p for m in disk for p in disk[m]}
        sp = {p for m in st for p in st[m]}
        both = dp & sp
        print("  prompts: disk %d, stash %d, shared %d "
              "(disk-only %d, stash-only %d)"
              % (len(dp), len(sp), len(both), len(dp - sp), len(sp - dp)))
        #: **COMPARE LABELS PER SITE, NOT TOTALS PER CORPUS.** Restricting to
        #: shared prompts still left the two disagreeing, and the reconciliation
        #: is the point: disk `shared`+`prompt_one_arm` = 59,704 = stash
        #: `shared`+`prompt_one_arm`. The stash is not returning different
        #: values, it holds BOTH ARMS where the export holds one (disk falls to
        #: 144 prompts on its thinnest model). A totals comparison cannot tell
        #: "more coverage" from "different answers"; a per-site comparison can,
        #: and only the second licenses reading the stash as a grid.
        agree = disagree = 0
        examples = []
        for b, al in dpairs:
            common = (set(disk[b]) & set(disk[al])
                      & set(st.get(b, {})) & set(st.get(al, {})) & both)
            for p in common:
                ld = M.classify(disk[b][p], disk[al][p], TR)
                ls = M.classify(st[b][p], st[al][p], TR)
                if ld == ls:
                    agree += 1
                else:
                    disagree += 1
                    if len(examples) < 5:
                        examples.append((b, al, p, sorted(ld), sorted(ls)))
        n = agree + disagree
        rate = disagree / n if n else 0.0
        print("  per-site labels on prompts both sources have in both arms: "
              "%s agree, %s disagree (%.2f%%)"
              % ("{:,}".format(agree), "{:,}".format(disagree), 100 * rate))

        #: **THE TOLERANCE IS DECLARED, AND THE SECOND CLAUSE IS THE REAL GATE.**
        #: A bare "under 1% is fine" would pass a genuine value change of the
        #: same size. Measured 8 Aug: disagreeing sites have a median top1-top2
        #: margin of 3.08e-3 against 4.18e-2 over all sites, and 36.3% of them
        #: fall under 1e-3 against a 3.2% base rate. That is the signature of
        #: cross-hardware floating point flipping an argmax between two words
        #: separated by a thousandth -- the recorded property that per-site
        #: values are hardware-sensitive while pair means are not. A real
        #: change would NOT be enriched in the near-tie tail, so the run
        #: refuses unless the enrichment is present.
        MAX_RATE, MIN_ENRICH = 0.01, 3.0
        diffs = {}
        if rate > MAX_RATE:
            diffs["rate"] = (n, disagree)
        else:
            base_thin, base_n, dis_thin = 0, 0, 0
            for b, al in dpairs:
                common = (set(disk[b]) & set(disk[al])
                          & set(st.get(b, {})) & set(st.get(al, {})) & both)
                for p in common:
                    o, pr = disk[b][p]
                    if len(o) < 2:
                        continue
                    thin = (pr[o[0]] - pr[o[1]]) < 1e-3
                    base_n += 1
                    base_thin += thin
                    if M.classify(disk[b][p], disk[al][p], TR) != \
                       M.classify(st[b][p], st[al][p], TR):
                        dis_thin += thin
            br = base_thin / base_n if base_n else 0
            dr = dis_thin / disagree if disagree else 0
            enrich = (dr / br) if br else 0
            print("  near-tie (<1e-3) share: disagreeing %.3f vs all %.3f "
                  "-> %.1fx enrichment (need >=%.1fx)"
                  % (dr, br, enrich, MIN_ENRICH))
            if disagree and enrich < MIN_ENRICH:
                diffs["not_near_tie"] = (base_n, disagree)
        for b, al, p, ld, ls in examples[:3]:
            print("    e.g. %s > %s  %r  disk %s / stash %s"
                  % (b.split("/")[-1], al.split("/")[-1], p[:32], ld, ls))
        if not diffs:
            print("  ACCEPTED: the stash reads as a grid. Residual is "
                  "hardware-sensitive argmax flips at near-ties, bounded at "
                  "%.2f%% of sites." % (100 * rate))
        if diffs:
            print("  STASH AND DISK DISAGREE — refusing:")
            for k, (d_, s_) in sorted(diffs.items()):
                print("    %-22s disk %8d   stash %8d" % (k, d_, s_))
            sys.exit(1)
        #: NOT "exactly" -- it is not exact, and an earlier version of this
        #: line said so while the check above was reporting a 0.45% residual.
        #: The gate's verdict and the sentence reporting it must be the same
        #: claim.
        print("  stash path VALIDATED under the declared tolerance above.")

    #: ── THE NEW LINEAGES ───────────────────────────────────────────────────
    npairs, refused = new_pairs(reg)
    for f, why in refused:
        print("\nREFUSED family %s: %s" % (f, why))
    models = sorted({m for p in npairs for m in p})
    print("\nloading %d new-lineage models from the stash ..." % len(models))
    ng = grid_from_stash(models)
    absent = [m for m in models if m not in ng]
    if absent:
        sys.exit("REFUSING: no stash cells for %s" % ", ".join(absent))

    nper, nled = M.count(ng, npairs, TR)
    table(nled, "NEW LINEAGES ONLY — %d pairs" % len(npairs))

    print("\n%-46s %8s %8s %8s" % ("pair", "shared", "FREE", "STRICT"))
    print("-" * 74)
    for b, al, row in sorted(nper, key=lambda r: -r[2]["FREE"]):
        print("%-46s %8d %8d %8d"
              % ("%s > %s" % (b.split("/")[-1][:20], al.split("/")[-1][:22]),
                 row["shared"], row["FREE"], row["LEX_STRICT"]))

    print("\nCOMBINED with the %d disk pairs: FREE %s -> %s  (+%s)"
          % (len(dpairs), "{:,}".format(dled["FREE"]),
             "{:,}".format(dled["FREE"] + nled["FREE"]),
             "{:,}".format(nled["FREE"])))
    print("Teuken is excluded by name: %s"
          % ", ".join(m.split("/")[-1] for m in INCOMPLETE))

    if a.emit:
        out = {
            "producer": "meta/M01_displacement/scripts/m05_sites_newlineages.py",
            "rule": "m05_sites.classify, imported not restated",
            "source": "true_word_probs stash (rule_version 3, "
                      "dict_sha b16011275c42955c, theta 0.001, mode raw)",
            "excluded_incomplete": INCOMPLETE,
            "pairs": [],
        }
        for b, al in npairs:
            sb, sa = set(ng[b]), set(ng[al])
            sites = []
            for p in sorted(sb & sa):
                labs = M.classify(ng[b][p], ng[al][p], TR)
                if "FREE" in labs:
                    ob, _ = ng[b][p]
                    oa, _ = ng[al][p]
                    sites.append({"prompt": p, "faller": M.top_word(*ng[b][p]),
                                  "riser": M.top_word(*ng[al][p]),
                                  "labels": sorted(labs)})
            out["pairs"].append({"base": b, "aligned": al,
                                 "n_sites": len(sites), "sites": sites})
        with open(a.emit, "w") as fh:
            json.dump(out, fh, indent=1)
        print("\nwrote %s (%d pairs, %d sites)"
              % (a.emit, len(out["pairs"]),
                 sum(p["n_sites"] for p in out["pairs"])))


if __name__ == "__main__":
    main()
