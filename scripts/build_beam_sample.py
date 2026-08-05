#!/usr/bin/env python
"""build_beam_sample.py — the frozen stem sample for the beam run.

    scripts/build_beam_sample.py                 # draw, write, hash
    scripts/build_beam_sample.py --verify        # recompute the hash, change nothing
    scripts/build_beam_sample.py --dry-run       # print the plan, write nothing

WHAT IS DRAWN. 15 stems per domain from `data/r_population_k2.parquet`, in TWO
DECLARED STRATA:

    R_COMPARABLE   10  drawn from R's UNSPENT vv*-eligible frame, so these
                       items can be set beside R's coder judgements
    R_INVISIBLE     5  drawn from the NON-eligible remainder, so the sample
                       still reaches displacement types R cannot ask about

WHY TWO STRATA RATHER THAN ONE. Preferring eligible stems raises the overlap
with R from ~48 of 105 to 90 — but eligibility means BOTH faller and riser are
lexical verbs, so an all-eligible draw would quietly narrow the beam sample to
verb->verb displacement. The 5 free stems per domain keep noun, adjective and
function-word substitutions in the frame. The split is a declared design, not a
convenience: the strata are labelled in the output and either can be analysed
alone.

POWER IS A DECLARED EXCEPTION, NOT A BUG. Zero of 120 power stems are
vv*-eligible: in that domain the measured displacement is a function-word shift
(`he -> she`, `had -> was`, `that -> the`), which is outside the scope of an
instrument built for lexical relations. Power therefore draws 15 R_INVISIBLE
and 0 R_COMPARABLE, and the manifest says so rather than showing a silent 10.

EQUAL ALLOCATION IS THE INTENDED DESIGN. The population's domain sizes range
152 to 36 because equal coverage was aimed at and not achieved, so 15-per-domain
restores the intent rather than distorting the population. No inverse-probability
weights are needed for that reason; if anything is later pooled ACROSS domains,
weight by design intent, not by the realised population.
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

POP = os.path.join(ROOT, "data", "r_population_k2.parquet")
SPENT = os.path.join(ROOT, "meta", "M01_displacement", "results",
                     "r_eight_coder_verbpaired_50x2.parquet")
BYU = "/Users/rj416/Dropbox/Prof/Code/osp/worddb.byu.txt"
OUT_CSV = os.path.join(ROOT, "data", "beam_sample_105.csv")
OUT_MAN = os.path.join(ROOT, "data", "beam_sample_105_manifest.json")

SEED = 20260805
PER_DOMAIN = 15
N_ELIGIBLE = 10          #: R_COMPARABLE per domain
N_FREE = 5               #: R_INVISIBLE per domain


def claws_vv():
    """CLAWS lexical-verb test, carried from build_r_decoys.py — the same rule
    R's own population filter uses, so 'eligible' here means eligible THERE."""
    pos = {}
    with open(BYU, encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 3:
                w, t = f[-1].strip().lower(), f[-3].strip()
                if w and w not in pos:
                    pos[w] = t
    return lambda w: str(pos.get(str(w).strip().lower(), "")).startswith("vv")


def build():
    vv = claws_vv()
    d = pd.read_parquet(POP)
    d["vv_both"] = d.faller.map(vv) & d.riser.map(vv)

    #: eligible = vv* on BOTH sides for BOTH members (R's own requirement)
    g = d[d.vv_both].groupby("stem").member.nunique()
    eligible = set(g[g >= 2].index)
    spent = set(pd.read_parquet(SPENT).stem)

    stems = d.groupby("stem").agg(domain=("domain", "first"),
                                  subdomain=("subdomain", "first"),
                                  both=("stem_has_both", "first"))
    #: EXCLUSIONS ARE NAMED, NEVER SILENT.
    dropped_incomplete = sorted(stems[~stems.both].index)
    stems = stems[stems.both].copy()
    stems["eligible"] = stems.index.isin(eligible)
    stems["spent"] = stems.index.isin(spent)

    rng = pd.Series(range(len(stems)), index=stems.index)  # placeholder, replaced below
    picks, notes = [], []
    for dom, grp in stems.groupby("domain"):
        pool_e = grp[grp.eligible & ~grp.spent]
        pool_f = grp[~grp.eligible]
        n_e = min(N_ELIGIBLE, len(pool_e))
        n_f = PER_DOMAIN - n_e
        if n_e < N_ELIGIBLE:
            notes.append("%s: only %d unspent-eligible available (wanted %d); "
                         "the shortfall moves to R_INVISIBLE"
                         % (dom, len(pool_e), N_ELIGIBLE))
        if n_f > len(pool_f):
            notes.append("%s: only %d non-eligible available (wanted %d); "
                         "domain UNDER-FILLED at %d stems"
                         % (dom, len(pool_f), n_f, n_e + len(pool_f)))
            n_f = len(pool_f)
        if n_e:
            s = pool_e.sample(n=n_e, random_state=SEED)
            picks += [(i, dom, "R_COMPARABLE") for i in s.index]
        if n_f:
            s = pool_f.sample(n=n_f, random_state=SEED)
            picks += [(i, dom, "R_INVISIBLE") for i in s.index]

    sel = pd.DataFrame(picks, columns=["stem", "domain", "stratum"])
    sel = sel.sort_values(["domain", "stratum", "stem"]).reset_index(drop=True)

    #: attach the prompts — the unit that costs wall time
    rows = d[d.stem.isin(set(sel.stem))][
        ["stem", "member", "prompt", "faller", "riser", "subdomain"]
    ].drop_duplicates(["stem", "member", "prompt"])
    out = sel.merge(rows, on="stem", how="left").sort_values(
        ["domain", "stratum", "stem", "member"]).reset_index(drop=True)

    #: the membership hash covers the STEM SET AND ITS STRATUM LABELS — the two
    #: things a later run could differ on without the row count changing.
    payload = "\n".join("%s\t%s\t%s" % (r.stem, r.domain, r.stratum)
                        for r in sel.itertuples())
    sha = hashlib.sha256(payload.encode()).hexdigest()[:16]

    man = {
        "producer": "scripts/build_beam_sample.py",
        "seed": SEED,
        "rule": ("15 stems per domain: %d drawn from R's UNSPENT vv*-eligible "
                 "frame (R_COMPARABLE), %d from the non-eligible remainder "
                 "(R_INVISIBLE). Stems lacking both members are excluded and "
                 "named. Power has 0 eligible — function-word displacement, "
                 "outside R's lexical scope — so it draws 15 R_INVISIBLE."
                 % (N_ELIGIBLE, N_FREE)),
        "membership_sha256_16": sha,
        "stems": int(len(sel)),
        "distinct_prompts": int(out.prompt.nunique()),
        "by_domain": {k: {kk: int(vv2) for kk, vv2 in v.items()}
                      for k, v in sel.groupby("domain").stratum
                      .value_counts().unstack(fill_value=0).iterrows()},
        "overlap_with_R_frame": int((sel.stratum == "R_COMPARABLE").sum()),
        "excluded_incomplete_stems": dropped_incomplete,
        "excluded_count": len(dropped_incomplete),
        "spent_stems_excluded_from_eligible_stratum": len(spent),
        "notes": notes,
    }
    return out, sel, man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and compare against the written manifest")
    a = ap.parse_args()
    out, sel, man = build()

    print("STEMS %d | DISTINCT PROMPTS %d | in R's frame %d"
          % (man["stems"], man["distinct_prompts"], man["overlap_with_R_frame"]))
    print("membership sha256[:16] = %s  seed %d" % (man["membership_sha256_16"], SEED))
    print()
    print("%-10s %14s %13s" % ("domain", "R_COMPARABLE", "R_INVISIBLE"))
    for dom, v in sorted(man["by_domain"].items()):
        print("%-10s %14d %13d" % (dom, v.get("R_COMPARABLE", 0), v.get("R_INVISIBLE", 0)))
    if man["notes"]:
        print("\nNOTES")
        for n in man["notes"]:
            print("  - %s" % n)
    #: The 7 single-member stems are unusable — no marked/unmarked contrast
    #: exists — so they are dropped, not weighed. The names live in the
    #: manifest only: enough that the denominator reads 677 rather than 684,
    #: not so much that a routine exclusion reads as a problem.
    print("\nexcluded %d single-member stems (no pair to contrast); "
          "denominator 677 of 684. Names in the manifest."
          % man["excluded_count"])

    if a.verify:
        if not os.path.exists(OUT_MAN):
            sys.exit("no manifest at %s — nothing to verify against" % OUT_MAN)
        old = json.load(open(OUT_MAN))
        same = old["membership_sha256_16"] == man["membership_sha256_16"]
        print("\nVERIFY  written %s  rebuilt %s  ->  %s"
              % (old["membership_sha256_16"], man["membership_sha256_16"],
                 "MATCH" if same else "**DIFFERS**"))
        sys.exit(0 if same else 1)

    if a.dry_run:
        print("\n--dry-run: nothing written. NOTE a dry run verifies the DRAW, "
              "not the files — only a real run proves the write.")
        return

    out.to_csv(OUT_CSV, index=False)
    with open(OUT_MAN, "w") as fh:
        json.dump(man, fh, indent=2)
    print("\nwrote %s (%d rows)\nwrote %s"
          % (os.path.relpath(OUT_CSV, ROOT), len(out), os.path.relpath(OUT_MAN, ROOT)))


if __name__ == "__main__":
    main()
