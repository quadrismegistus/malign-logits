"""Is any Tulu SFT ablation responsible for withdrawal of probability MASS at
transgressive sites? X 3e's question, re-asked on mass and on matched pairs.

    uv run python x_pair_ablation_split.py            # 684 declared pairs
    uv run python x_pair_ablation_split.py --metric js_fall
    uv run python x_pair_ablation_split.py --edge data_ablation_of

TWO THINGS CHANGE FROM X 3e AND BOTH MATTER.

**1. THE METRIC. 3e's table is faller JACCARD, which cannot see mass.**
`t_fans.jac` is set overlap of the faller words: two arms that drop the SAME
words while withdrawing very different amounts of probability score 1.0. 3e used
it deliberately, for the WHICH-WORDS question, with JS carrying HOW-MUCH -- its
own summary of `no-wildchat` is "magnitude normal, direction different", JS
0.0584 against Jaccard 0.340. A question about *withdrawal of probability mass*
is the magnitude question, so this runs on:

    departed   mass that left the base's support     <- DEFAULT, the quantity asked about
    js_fall    the fall component of JS              <- companion, --metric js_fall
    js_total   whole-distribution divergence

**2. THE DESIGN. 3e compared two independent CELLS of different prompts** --
118-122 neutral against 1,197-1,218 transgressive, differing in scene, length and
syntax, so any cell difference carried all of that. M01's pairs hold the scene
fixed and swap one word:

    UNMARKED  He pulled her into a hug and
    MARKED    He pulled her into a chokehold and

so the contrast is WITHIN a pair and prompt-level variance is differenced away.

THE STATISTIC IS A DIFFERENCE-IN-DIFFERENCES, because the question is comparative.
"Does this arm withdraw more at transgressive sites" is not answerable from one
arm: every arm withdraws more at MARKED sites if MARKED sites simply move more.
So, per pair:

    within(arm)  = departed(arm, MARKED) - departed(arm, UNMARKED)
    DiD(arm)     = within(arm) - within(full)

**DiD < 0 means the ablation withdrew LESS at the transgressive member than full
SFT did, relative to its own neutral member** -- i.e. that removing that corpus
cost the model some of its transgressive-specific withdrawal. That is the shape
the claim "the safety ablation is responsible for withdrawal at transgressive
sites" predicts, and `no-safety` is the arm it predicts it for.

READS THE STORE, MEASURES NOTHING. `movement_cells` already holds one row per
(base, aligned, prompt) with `departed`, `js_fall`, `js_rise`, `js_tail`. All 684
pairs are present on every arm at the `sft_of` edge, verified before this was
written, so there is no model loading and no `t_fans.measure` -- and therefore
none of `measure`'s silent `not c.is_present` drop. Coverage is asserted per arm
instead.

TWO EDGES ARE AVAILABLE AND THEY ASK DIFFERENT QUESTIONS.
    sft_of              meta-llama/Llama-3.1-8B -> each arm.  DEFAULT.
                        What each training run withdrew from the pretrained base.
                        This is U's and 3e's framing.
    data_ablation_of    full SFT -> each ablation directly. What the ablation
                        changed relative to the full mixture. Fewer assumptions,
                        but its `departed` is not comparable to the fan's.

POPULATION DECLARED, NOT DISCOVERED. 684 pairs: ACTIVE, contrast_type
`transgressive_swap`, both roles present, `source` starting `M01_PAIRS_`. That
last clause is what makes it 684 rather than 699 -- SETE (6), F13_CHINESE (6) and
SETD (3) carry the same contrast type from other provenance and are excluded BY
SOURCE rather than by hand. Asserted; the script refuses any other number.

ARMS COME FROM data/model_registry.json, never a literal, so the fan grows or
the assert fires rather than the fan quietly narrowing.

NO REPRODUCTION CHECK AGAINST U'S FAN IS AVAILABLE. `x_wildchat_split.py` could
assert its unpartitioned column against `t_fans_jaccard.csv` because it ran on
U's prompts and U's statistic. This runs on a different population AND a
different metric, so no such assert exists and none is faked.
"""
import argparse
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
CATEG = os.path.join(ROOT, "data", "prompt_categorisation.json")
REGISTRY = os.path.join(ROOT, "data", "model_registry.json")
OUT = os.path.join(CAMP, "results", "x_pair_ablation_split.csv")
OUT_JSON = os.path.join(CAMP, "results", "x_pair_ablation_split.json")

SOURCE_PREFIX = "M01_PAIRS_"
CONTRAST = "transgressive_swap"
N_PAIRS_EXPECTED = 684

FULL_SFT = "allenai/Llama-3.1-Tulu-3-8B-SFT"
ABLATION_MARK = "-SFT-no-"
PRE = "meta-llama/Llama-3.1-8B"


def q(sql):
    r = subprocess.run([CH, "client", "--query", sql], capture_output=True, text=True, timeout=1800)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[:400])
    return r.stdout


#: js_fall and js_rise are DECLARED IN THE SCHEMA AND NEVER POPULATED -- 0 of
#: 568,977 rows carry a non-zero value, checked across the whole table, not just
#: these edges. A first run printed them as +0.00000 for every arm, which reads
#: as "no effect" and is actually "no data". They are excluded here and the
#: refusal is loud: an empty column and a null result are the same number on a
#: screen, and only one of them is a finding.
CELL_METRICS = ["departed", "arrived", "js_total"]
EMPTY_COLUMNS = {"js_fall", "js_rise"}


def load_fallers(base, aligned, prompts):
    """Faller SETS per prompt, from the per-word table. movement_cells has no
    word sets, so 3e's Jaccard cannot be computed from it at all."""
    sql = ("SELECT prompt, groupArray(word) AS ws FROM malign_logits.movement "
           "WHERE base='%s' AND aligned='%s' AND cls='fall' "
           "GROUP BY prompt FORMAT JSONEachRow" % (base, aligned))
    out = {}
    for line in q(sql).strip().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["prompt"] in prompts:
            out[r["prompt"]] = frozenset(r["ws"])
    return out


def jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else float("nan")


def load_pairs():
    P = [r for r in json.load(open(CATEG))["prompts"] if r.get("status") == "ACTIVE"]
    by = collections.defaultdict(list)
    for r in P:
        if r.get("contrast_type") == CONTRAST and str(r.get("source", "")).startswith(SOURCE_PREFIX):
            by[r.get("pair_id")].append(r)
    pairs, dropped = {}, collections.Counter()
    for pid, v in by.items():
        roles = {x.get("pair_role") for x in v}
        if len(v) != 2:
            dropped["incomplete (%d members)" % len(v)] += 1; continue
        if roles != {"MARKED", "UNMARKED"}:
            dropped["roles %s" % sorted(roles)] += 1; continue
        pairs[pid] = {x["pair_role"]: x for x in v}
    print("  pairs loaded            %d" % len(pairs))
    for k, n in dropped.most_common():
        print("     dropped %-28s %d" % (k, n))
    assert len(pairs) == N_PAIRS_EXPECTED, (
        "population is %d pairs, declared %d. The population is a claim: fix the "
        "declaration or the filter, never proceed on a number nobody chose."
        % (len(pairs), N_PAIRS_EXPECTED))
    return pairs


def load_arms():
    d = json.load(open(REGISTRY))
    rows = d if isinstance(d, list) else d.get("models", d.get("entries", []))
    ids = {r.get("model_id") or r.get("id") or r.get("hf") for r in rows if isinstance(r, dict)}
    ids = {i for i in ids if i}
    assert FULL_SFT in ids, "%s not in the registry; the fan has no reference arm" % FULL_SFT
    arms = {"full": FULL_SFT}
    for i in sorted(ids):
        if ABLATION_MARK in i and i.startswith(FULL_SFT.rsplit("-SFT", 1)[0]):
            arms[i.split(ABLATION_MARK)[1].replace("-data", "")] = i
    print("  arms from the registry  %d: %s" % (len(arms), ", ".join(sorted(arms))))
    assert len(arms) >= 3, "registry yielded %d arms -- check ABLATION_MARK" % len(arms)
    return arms


def main():
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser()
    #: ALL of them by default. `departed` and `arrived` are MASS; the js_*
    #: columns are divergence; `jaccard` is 3e's own statistic and is the only
    #: one that needs the per-word table, because a set of fallers is not in
    #: movement_cells. Running them together is the point: 3e's flat result and
    #: any mass result must be visible in one table or they get quoted apart.
    ap.add_argument("--metric", default="all",
                    choices=["all", "departed", "arrived", "js_total",
                             "js_fall", "js_rise", "jaccard"])
    ap.add_argument("--edge", default="sft_of", choices=["sft_of", "data_ablation_of"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    pairs = load_pairs()
    arms = load_arms()
    if a.limit:
        pairs = dict(list(pairs.items())[:a.limit])
        print("  --limit: %d pairs (SMOKE, not the declared population)" % len(pairs))
    print("  metric                  %s" % a.metric)
    print("  edge                    %s" % a.edge)

    metrics = CELL_METRICS + ["jaccard"] if a.metric == "all" else [a.metric]
    bad = [m for m in metrics if m in EMPTY_COLUMNS]
    assert not bad, (
        "%s is declared in movement_cells and populated in 0 of 568,977 rows. "
        "Asking for it returns zeros that read as a null result. Populate the "
        "column or do not ask for it." % ", ".join(bad))
    base = PRE if a.edge == "sft_of" else FULL_SFT
    need = {pairs[p][r]["prompt"] for p in pairs for r in ("MARKED", "UNMARKED")}

    #: One pass per metric. Every metric is computed on the SAME 684 pairs, so a
    #: reader comparing rows across metrics is comparing the same population --
    #: which is what 3e could not do, having one statistic on one partition.
    allrows, summary, coverage = [], [], {}
    for met in metrics:
        got = {}
        for name, ck in sorted(arms.items()):
            if a.edge == "data_ablation_of" and name == "full":
                continue
            if met == "jaccard":
                got[name] = load_fallers(base, ck, need)
            else:
                sql = ("SELECT prompt, %s AS v FROM malign_logits.movement_cells "
                       "WHERE base='%s' AND aligned='%s' FORMAT JSONEachRow"
                       % (met, base, ck))
                d = {}
                for line in q(sql).strip().split("\n"):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    d[r["prompt"]] = float(r["v"])
                got[name] = d
        coverage[met] = {k: len(v) for k, v in got.items()}
        if "full" not in got:
            print("  %s: no full arm" % met); continue

        #: COVERAGE ASSERTED PER METRIC. jaccard's population is smaller by
        #: construction -- a prompt with NO fallers has no row in `movement` --
        #: so it is reported rather than asserted equal, and the n travels.
        short = {n: len(need - set(d)) for n, d in got.items()}
        if met == "jaccard":
            for n, miss in sorted(short.items()):
                if miss:
                    print("  jaccard %-10s %d of %d prompts have no faller row (no fallers)"
                          % (n, miss, len(need)))
        else:
            assert not any(short.values()), (
                "%s: an arm misses declared prompts %s -- every number would be on a "
                "different set per arm" % (met, {k: v for k, v in short.items() if v}))

        for pid, d in pairs.items():
            mk, un = d["MARKED"]["prompt"], d["UNMARKED"]["prompt"]
            if not all(t in got["full"] for t in (mk, un)):
                continue
            fw = (jac(got["full"][mk], got["full"][mk]) if met == "jaccard" else
                  got["full"][mk] - got["full"][un])
            for name in got:
                if name == "full":
                    continue
                if not all(t in got[name] for t in (mk, un)):
                    continue
                if met == "jaccard":
                    #: 3e's statistic: overlap with full AT each member. The
                    #: within-pair contrast is (overlap at MARKED) - (at UNMARKED),
                    #: and there is no full-vs-full baseline to difference against,
                    #: so DiD is the within value itself.
                    v_mk = jac(got["full"][mk], got[name][mk])
                    v_un = jac(got["full"][un], got[name][un])
                    within = v_mk - v_un
                    did = within
                else:
                    v_mk, v_un = got[name][mk], got[name][un]
                    within = v_mk - v_un
                    did = within - (got["full"][mk] - got["full"][un])
                allrows.append(dict(metric=met, arm=name, pair_id=pid,
                                    domain=d["MARKED"].get("domain"),
                                    marked=v_mk, unmarked=v_un, within=within, did=did))

    R = pd.DataFrame(allrows)
    if not len(R):
        print("  no rows"); return 1

    from math import comb
    rng = np.random.default_rng(0)
    print("\n%-9s %-10s %7s %10s %10s %11s %9s %9s"
          % ("metric", "arm", "pairs", "MARKED", "UNMARKED", "DiD", "n_neg", "sign p"))
    for met in metrics:
        Rm = R[R.metric == met]
        if not len(Rm):
            continue
        for name, g in Rm.groupby("arm"):
            v = g["did"].to_numpy()
            neg = int((v < 0).sum()); pos = int((v > 0).sum()); n = pos + neg
            p = min((sum(comb(n, k) for k in range(min(pos, neg) + 1)) * 2 / 2 ** n), 1.0) if n else float("nan")
            boot = rng.choice(v, (2000, len(v)), replace=True).mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            star = " *" if (lo > 0 or hi < 0) else "  "
            print("%-9s %-10s %7d %10.5f %10.5f %+11.5f%s %8s %9.2g"
                  % (met, name, len(g), g["marked"].mean(), g["unmarked"].mean(),
                     v.mean(), star, "%d/%d" % (neg, pos), p))
            summary.append(dict(metric=met, arm=name, n_pairs=len(g),
                                marked=float(g["marked"].mean()),
                                unmarked=float(g["unmarked"].mean()),
                                did=float(v.mean()), ci_lo=float(lo), ci_hi=float(hi),
                                n_neg=neg, n_pos=pos, sign_p=float(p)))
        print()
    print("  * = 95%% bootstrap CI excludes 0 (2000 resamples). SIGN P is the")
    print("  robustness check: a mean that moves while the median does not means")
    print("  a few pairs carry it, and the two tests disagreeing is the finding.")
    print("  DiD<0 = the arm withdrew/diverged LESS at the transgressive member")
    print("  than full SFT did. For jaccard, DiD is the within-pair overlap gap.")

    R.to_csv(OUT, index=False)
    json.dump({"_about": {
        "question": ("Is any Tulu SFT ablation responsible for withdrawal of probability "
                     "MASS at transgressive sites? X 3e re-asked on mass and on pairs."),
        "metric": a.metric,
        "metric_note": ("3e's table is faller JACCARD, which is set overlap and cannot "
                        "see mass. This is movement_cells.%s. The two answer different "
                        "questions and 3e says so: 'magnitude normal, direction "
                        "different'." % a.metric),
        "edge": a.edge,
        "design": ("WITHIN-PAIR difference-in-differences. within(arm) = "
                   "arm(MARKED) - arm(UNMARKED); DiD = within(arm) - within(full). "
                   "3e compared two independent cells of DIFFERENT prompts."),
        "population": ("%d matched pairs: ACTIVE, contrast_type=%s, both roles present, "
                       "source starting %s. The 15 same-contrast pairs from "
                       "SETE/SETD/F13_CHINESE are excluded BY SOURCE."
                       % (N_PAIRS_EXPECTED, CONTRAST, SOURCE_PREFIX)),
        "source_table": "malign_logits.movement_cells (no model loading, no re-measurement)",
        "arms": "read from data/model_registry.json, never a literal",
        "no_reproduction_check": ("different population AND different metric from U's "
                                  "fan, so no assert against t_fans_jaccard.csv is "
                                  "available and none is faked."),
        "coverage": {k: len(v) for k, v in got.items()},
    }, "arms": summary}, open(OUT_JSON, "w"), indent=1)
    print("\n  wrote %s" % os.path.relpath(OUT, ROOT))
    print("  wrote %s" % os.path.relpath(OUT_JSON, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
