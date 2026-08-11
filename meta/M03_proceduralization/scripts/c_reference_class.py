"""Plan C: the reference class -- where the institutional effect sits.

    uv run python c_reference_class.py --population   # build + hash, no compute
    uv run python c_reference_class.py --run          # the three primaries
    uv run python c_reference_class.py --run --tall   # + the per-word dump

Plan: `meta/M03_proceduralization/plans/plan_c_reference_class.md`.

WHAT IT ANSWERS. Plan B's numbers have no denominator: `JS=0.073` on an
institutional prompt is uninterpretable until we know what alignment does to an
ordinary one. This runs three instruments -- JS, semantic fields, raw word
deltas -- across ~748 matched MARKED/UNMARKED pairs so the institutional effect
can be placed on a scale, on magnitude AND on content. The three primaries are
co-equal: a JS ladder answers HOW FAR and cannot answer TOWARD WHAT, and
toward-what is what Findings T makes falsifiable (plan §1a).

## THIS PRODUCER GOES THROUGH `Step` AND `Cell`, AND THE FIRST DRAFT DID NOT

The first draft read `twp_words` with its own SQL and reimplemented the
measurement layer: its own fetch, its own partition fold, its own residual, its
own JS. RH caught it. Every one of the following was already solved in
`malign_logits/cell.py` and `step.py`, and four of them the draft got wrong:

  SOURCE PRECEDENCE.  `twp_words` is ORDER BY (model, prompt, word, SOURCE), so
  a cell scored under two sources keeps BOTH rows, and `sum(p) GROUP BY word`
  adds them. **13,787 of 238,934 cells on this roster carry two sources**
  (`cloud_run_20260801 + f11_twp` 6,670; `twpfill0 + twpfill2` 2,579).
  `ch_read.SOURCE_PRECEDENCE` picks one and NAMES the models it cannot resolve.

  THE UNIT OF JS.  `Cell.js()` is in BITS (log2). Plan B's hand-rolled
  `js_with_residual` is in NATS (np.log). Same quantity, 1.4427x apart, and
  cell.py's own docstring is the warning: "a movement statistic that does not
  name its metric is not a number". **Plan C's JS is not comparable to plan B's
  printed values without that factor.**

  DIRECTION.  Plan B's producer names its variables `da, db = D[base], D[aligned]`
  and b_word_delta names them the other way round. The first draft of this file
  copied one naming and the other's arithmetic, and computed `base - aligned`:
  **the word deltas would have come out sign-flipped**, with the whole result
  reading backwards and nothing about it looking wrong. `c.pre` and `c.post`
  cannot be got backwards.

  RULE VERSION.  `Cell` RAISES on a v1 arm against a v3 arm, because that books
  an instrument change as alignment movement. The SQL path checked nothing.
  Measured here and clean -- all 92 models are `rule_version 3`, `dict_sha
  b16011275c42955c` -- so this is a guard that could have fired and did not,
  which is worth more than an unchecked assumption that happens to hold.

  Also, for free and also reimplemented in the draft: the (word, first_token)
  partition fold with its malformed-row refusal, TSV unescaping (`ch_read._unesc`,
  added there after the same `didn\\'t` defect), and `cell.prompt.domain`.

Speed was never a reason to avoid it: 0.3 ms/cell in steady state, the whole
population in under a minute after the per-model cache warms.

## WHAT REMAINS THIS FILE'S OWN

The population, which cannot come from `prompt_catalogue`: the JSON catalogue
and the database disagree on the fields we would select on -- source `OTHER` is
64 texts in the JSON and 4 in the DB, `QUINTUPLETS` 42 and 0, domain `other`
185 and 0 -- and RH has ruled the JSON authoritative. So membership is
enumerated from the JSON and hashed over TEXTS.

THE UNIT IS THE LINEAGE, n = 46. Every headline is a per-lineage median first
and a sign test over lineages second. Group sizes print before any grouped n.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from malign_logits.step import Step            # noqa: E402
from malign_logits.movement import CANONICAL   # noqa: E402
from b_twp_institutional import field_counts, pairs_and_models, _lineage  # noqa: E402

CAT = os.path.join(ROOT, "data", "prompt_categorisation.json")
POP = os.path.join(CAMP, "populations", "c_population.json")
R = os.path.join(CAMP, "results")
OUT_CELLS = os.path.join(R, "c_cells.csv.gz")
OUT_PAIRS = os.path.join(R, "c_pair_contrast.csv")
OUT_FIELDS = os.path.join(R, "c_fields_by_stratum.csv")
OUT_WORDS = os.path.join(R, "c_word_delta_by_word.csv")
OUT_TALL = os.path.join(R, "c_word_delta_cells.csv.gz")

THETA = 0.001

#: F21's own arm axis, in `subdomain` on its 24 `institutional_*` prompts --
#: NOT `pair_role`, which is null on those rows. An earlier analysis read the
#: absence of that column as the absence of a design and reported that F21 had
#: no arm contrast; it has one, and the finding was made on it.
F21_INDIV = ("worker", "tenant", "patient", "citizen")
F21_INST = ("mgmt", "landlord", "doctor", "agency", "officer", "party")

#: Named BEFORE the run, per plan §5 Primary 2. These five are where T's two
#: priors predict opposite things; everything else the field pass reports is
#: exploratory and is flagged so in the output.
PRENAMED = (("usas", "X2.4"), ("usas", "S1.1.2"), ("usas", "A1.7"),
            ("usas", "A1.3"), ("wordnet", "cognition"))

#: Plan B's institutional risers, entering as a HYPOTHESIS to test on this
#: population and never as a selector for it. That inversion is the C2 defect:
#: a list read off one arm's risers cannot fail on the population it came from,
#: and off it the same six words moved 5.5x the other way.
PLAN_B_WORDS = ("ensure", "prioritize", "document", "involve", "improve",
                "engage", "handle", "reassess", "gather", "maintain",
                "adjust", "carefully", "communicate", "conduct")


def sign_test(vals):
    v = [x for x in vals if x != 0.0]
    n, k = len(v), sum(1 for x in v if x > 0)
    if n == 0:
        return n, k, float("nan")
    tail = min(k, n - k)
    return n, k, min(1.0, 2 * sum(math.comb(n, i) for i in range(tail + 1)) / 2 ** n)


# ----------------------------------------------------------------- population

def coverage(models):
    """{text: n_models scored}. Model-only query; `uniqExact(model)` is immune
    to the multi-source duplication because it counts models, not rows."""
    from malign_logits.ch_read import _unesc
    import subprocess
    assert not any("'" in m for m in models), "apostrophe in a model id"
    q = ("SELECT prompt, uniqExact(model) FROM malign_logits.twp_words "
         "WHERE abs(theta - %s) < 1e-9 AND model IN (%s) GROUP BY prompt FORMAT TSV"
         % (THETA, ",".join("'%s'" % m for m in models)))
    r = subprocess.run(["/opt/homebrew/bin/clickhouse", "client",
                        "--max_query_size", "20000000", "-q", q], capture_output=True)
    if r.returncode:
        raise SystemExit(r.stderr.decode()[:500])
    cov = {}
    for line in r.stdout.decode().splitlines():
        p, n = line.split("\t")
        cov[_unesc(p)] = int(n)
    return cov


def labels(models):
    """Pair membership and arm membership over prompt TEXT, as INDEPENDENT maps.

    `pair` maps text -> LIST of memberships, because text->pair is ONE-TO-MANY:
    7 texts belong to more than one pair, and a one-to-one dict silently drops
    the earlier membership. That cost 4 pairs (744 where 748 was right) with
    nothing reporting it -- violence and contradiction each came up short and
    the total still looked like a plausible number.

    A text can carry both a pair label and an arm label. Exactly one does: the
    text that is both `e1_credit_M` and `institutional_labor_worker_2`. The
    SETE `e*` rows have `subdomain` null, so they are pair-labelled only.
    """
    cat = json.load(open(CAT))["prompts"]
    act = [r for r in cat if r.get("status") == "ACTIVE" and r.get("language") == "en"]
    cov = coverage(models)
    n_models = len(models)

    grouped = collections.defaultdict(list)
    for r in act:
        if r.get("pair_role") in ("MARKED", "UNMARKED"):
            grouped[r.get("pair_id")].append(r)
    pair, d_shape, d_cov = collections.defaultdict(list), [], []
    for pid, rs in grouped.items():
        if len(rs) != 2 or {x["pair_role"] for x in rs} != {"MARKED", "UNMARKED"}:
            d_shape.append((pid, len(rs)))
            continue
        if any(cov.get(x["prompt"], 0) != n_models for x in rs):
            d_cov.append((pid, rs[0].get("domain")))
            continue
        for x in rs:
            pair[x["prompt"]].append((x["pair_role"], pid, x.get("domain")))

    arm = {}
    for r in act:
        if r.get("finding") == "F21":
            sd = r.get("subdomain")
            if sd in F21_INDIV:
                arm[r["prompt"]] = ("f21_indiv", None)
            elif sd in F21_INST:
                arm[r["prompt"]] = ("f21_inst", None)
        if r.get("source") == "M03_SPEAKER_KERNEL":
            a, _, cond = r["group_role"].partition("_")
            arm[r["prompt"]] = ("m03_" + a, cond)
    return dict(pair), arm, d_shape, d_cov


def build_population(quiet=False):
    pairs, models = pairs_and_models()
    pair, arm, d_shape, d_cov = labels(models)
    texts = sorted(set(pair) | set(arm))
    sha = hashlib.sha256("\n".join(texts).encode()).hexdigest()

    pids = {}
    for memberships in pair.values():
        for role, pid, dom in memberships:
            pids[pid] = dom
    bydom = collections.Counter(pids.values())
    byarm = collections.Counter(v[0] for v in arm.values())

    if not quiet:
        print("matched pairs by domain (1 MARKED + 1 UNMARKED, both at full %d):"
              % len(models))
        for d, n in bydom.most_common():
            print("  %-16s %4d pairs" % (d, n))
        print("  %-16s %4d pairs   %d distinct texts in pairs"
              % ("TOTAL", len(pids), len(pair)))
        print("\ndropped, named not silent:")
        print("  not a 1+1 pair       %d pair_ids  member counts %s"
              % (len(d_shape), dict(collections.Counter(n for _, n in d_shape))))
        print("  a member below full  %d pair_ids  %s"
              % (len(d_cov), dict(collections.Counter(d for _, d in d_cov))))
        multi = sum(1 for v in pair.values() if len(v) > 1)
        print("  texts in >1 pair     %d  (counted once per pair, so a text count"
              % multi)
        print("                          and a pair count differ and both are right)")
        print("\narm-labelled strata:")
        for a, n in sorted(byarm.items()):
            print("  %-16s %4d texts" % (a, n))
        print("\ntexts with BOTH a pair and an arm label: %d"
              % len(set(pair) & set(arm)))
        print("distinct texts %d   x %d models = %d cells"
              % (len(texts), len(models), len(texts) * len(models)))
        print("sha16 %s" % sha[:16])

    doc = {
        "_what": "matched MARKED/UNMARKED pairs + F21's arm + the M03 kernel, "
                 "on the 46 lineage-representative pairs.",
        "_unit": "THE LINEAGE, n=46.",
        "_measured_through": "malign_logits.step.Step / cell.Cell -- source "
                             "precedence, the partition fold, the residual bin, "
                             "rule_version checking and TSV unescaping all come "
                             "from the library. JS IS IN BITS (log2), which is "
                             "NOT the unit plan B's hand-rolled js_with_residual "
                             "reports (nats). Factor 1.4427.",
        "_enumerated_from": "data/prompt_categorisation.json (RH 2026-08-11: the "
                            "JSON is authoritative where it disagrees with "
                            "prompt_catalogue, and it disagrees on source and domain).",
        "_cannot_measure": "AGENCY. Plan B §5; the addendum's 'do not narrate "
                           "submission' binds this output too.",
        "n_models": len(models), "n_texts": len(texts),
        "n_cells": len(texts) * len(models),
        "n_pairs": len(pids), "pairs_by_domain": dict(bydom),
        "arm_strata": dict(byarm),
        "dropped_not_a_pair": [{"pair_id": p, "members": n} for p, n in sorted(d_shape)],
        "dropped_coverage": [{"pair_id": p, "domain": d} for p, d in sorted(d_cov)],
        "population_sha256": sha, "population_sha256_16": sha[:16],
        "texts": texts,
    }
    os.makedirs(os.path.dirname(POP), exist_ok=True)
    json.dump(doc, open(POP, "w"), indent=1)
    if not quiet:
        print("wrote %s" % os.path.relpath(POP, ROOT))
    return pairs, models, pair, arm, texts


# -------------------------------------------------------------------- the run

def strata_of(text, pair, arm):
    """Every stratum this text belongs to. A list, because it can be several."""
    out = []
    for role, pid, dom in pair.get(text, ()):
        out.append(("%s_%s" % (dom, role.lower()), pid, dom, role))
    a = arm.get(text)
    if a:
        out.append((a[0], None, None, None))
    return out


def run(tall=False):
    pairs, models, pair, arm, texts = build_population(quiet=True)
    print("population %d texts, %d steps, %d models" % (len(texts), len(pairs), len(models)))

    os.makedirs(R, exist_ok=True)
    cells = gzip.open(OUT_CELLS, "wt", newline="")
    cw = csv.writer(cells)
    cw.writerow(["lineage", "step", "prompt", "js_bits", "l1", "n_risers",
                 "n_fallers", "inflation", "residual_pre", "residual_post",
                 "rule_version"])
    tallfh = gzip.open(OUT_TALL, "wt", newline="") if tall else None
    if tallfh:
        tw = csv.writer(tallfh)
        tw.writerow(["lineage", "stratum", "word", "delta"])

    fshare = collections.defaultdict(lambda: collections.defaultdict(list))
    fcov = collections.defaultdict(lambda: collections.defaultdict(list))
    wdelta = collections.defaultdict(lambda: collections.defaultdict(list))
    pairrows, absent = [], 0

    for i, p in enumerate(pairs, 1):
        base, aligned = p.split(">")
        lin = _lineage(base)
        step = Step(base, aligned)
        n = 0
        for text in texts:
            c = step.cell(text)
            if not c.is_present:
                absent += 1
                continue
            #: `c.pre` is the BASE arm and `c.post` the ALIGNED one, by
            #: construction. The first draft of this file hand-rolled the two
            #: dictionaries and subtracted them the wrong way round.
            mv = c.movement(CANONICAL)
            js = c.js()
            cw.writerow([lin, step.label, text, "%.8g" % js, "%.8g" % c.l1(),
                         len(mv.risers), len(mv.fallers),
                         "%.6f" % mv.inflation,
                         "%.6f" % c.pre.residual, "%.6f" % c.post.residual,
                         c.rule_version])
            n += 1

            delta = {w: c.post.probs.get(w, 0.0) - c.pre.probs.get(w, 0.0)
                     for w in set(c.pre.probs) | set(c.post.probs)}
            #: computed ONCE per cell; a text in two strata must not pay twice
            #: for the lexicon lookup, which dominates this loop.
            fc = field_counts(mv.risers)
            tot = collections.Counter()
            for src, fld, k in _flat(fc):
                tot[src] += k
            share = [(src, fld, k / tot[src]) for src, fld, k in _flat(fc) if tot[src]]
            cov = [(src, k) for src, k in _cov(fc) if k is not None]

            for stratum, pid, dom, role in strata_of(text, pair, arm):
                for w, v in delta.items():
                    wdelta[stratum][w].append((lin, v))
                    if tallfh:
                        tw.writerow([lin, stratum, w, "%+.8g" % v])
                for src, fld, v in share:
                    fshare[stratum][(src, fld)].append((lin, v))
                for src, k in cov:
                    fcov[stratum][src].append(k)
                if pid:
                    pairrows.append((lin, pid, dom, role, js))
        print("  [%2d/%2d] %-40s %s  %4d cells" %
              (i, len(pairs), base.split("/")[-1][:38], step.label, n), flush=True)

    cells.close()
    if tallfh:
        tallfh.close()
    print("\ncells absent (a text one arm has not scored): %d" % absent)
    write_pairs(pairrows)
    write_fields(fshare, fcov)
    write_words(wdelta)
    print("\nwrote %s" % os.path.relpath(OUT_CELLS, ROOT))


def _flat(f):
    for src, v in f.items():
        if src == "norms":
            for norm, w in v.items():
                for k, n in w.get("counts", {}).items():
                    yield "norms:" + norm, k, n
        else:
            for k, n in v.get("counts", {}).items():
                yield src, k, n


def _cov(f):
    for src, v in f.items():
        if src == "norms":
            for norm, w in v.items():
                yield "norms:" + norm, w.get("coverage")
        else:
            yield src, v.get("coverage")


# ------------------------------------------------------------------- reporting

def write_pairs(rows):
    """PRIMARY 1: d = JS(MARKED) - JS(UNMARKED), the scene held fixed."""
    idx = {}
    for lin, pid, dom, role, js in rows:
        idx.setdefault((lin, pid, dom), {})[role] = js
    out = [(lin, pid, dom, v["UNMARKED"], v["MARKED"], v["MARKED"] - v["UNMARKED"])
           for (lin, pid, dom), v in idx.items() if len(v) == 2]
    with open(OUT_PAIRS, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lineage", "pair_id", "domain", "js_unmarked", "js_marked", "d_js"])
        for r in sorted(out):
            w.writerow(list(r[:3]) + ["%.8g" % x for x in r[3:]])

    print("\n" + "=" * 78)
    print("PRIMARY 1 -- displacement ladder: JS(MARKED) - JS(UNMARKED), by domain")
    print("=" * 78)
    print("  JS IS IN BITS. Plan B's printed values are in nats; x1.4427 to compare.")
    print("  Never pooled across domains: a pooled number over 158 violence")
    print("  pairs and 1 profanity pair is a violence number.")
    bydom = collections.defaultdict(lambda: collections.defaultdict(list))
    pids = collections.defaultdict(set)
    for lin, pid, dom, u, m, d in out:
        bydom[dom][lin].append(d)
        pids[dom].add(pid)
    print("\n  %-16s %6s %6s %11s %11s  %7s %s" %
          ("domain", "pairs", "lins", "med UNMARKED", "median d", "lins>0", "p"))
    for dom in sorted(bydom, key=lambda d: -len(pids[d])):
        bylin = bydom[dom]
        lm = [st.median(v) for v in bylin.values()]
        n, k, p = sign_test(lm)
        umed = st.median([u for l, pi, dd, u, m, d in out if dd == dom])
        print("  %-16s %6d %6d %11.5f %+11.5f   %2d/%-2d  %.2g"
              % (dom, len(pids[dom]), len(bylin), umed, st.median(lm), k, n, p))
    big = max(pids, key=lambda d: len(pids[d]))
    print("\n  group sizes in %s: %s"
          % (big, dict(collections.Counter(len(v) for v in bydom[big].values()))))


def write_fields(fshare, fcov):
    """PRIMARY 2: is the rising vocabulary general or institutional?"""
    rows = []
    for stratum, d in fshare.items():
        for (src, fld), vals in d.items():
            bylin = collections.defaultdict(list)
            for lin, v in vals:
                bylin[lin].append(v)
            if len(bylin) < 40:
                continue
            lm = [st.median(v) for v in bylin.values()]
            rows.append({"stratum": stratum, "source": src, "field": fld,
                         "n_lineages": len(bylin), "median_share": st.median(lm),
                         "coverage": (st.median(fcov[stratum][src])
                                      if fcov[stratum].get(src) else ""),
                         "prenamed": (src, fld) in PRENAMED})
    if not rows:
        print("\n(no field rows reached 40 lineages)")
        return
    with open(OUT_FIELDS, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.8g" % v if isinstance(v, float) else v)
                        for k, v in r.items()})

    print("\n" + "=" * 78)
    print("PRIMARY 2 -- the PRE-NAMED cells, general vs institutional")
    print("=" * 78)
    print("  T §12/13 predicts these rise EVERYWHERE; T §11 predicts the four")
    print("  institutional strata reverse. Both cannot hold of one category.")
    idx = {(r["stratum"], r["source"], r["field"]): r for r in rows}
    strata = ["violence_unmarked", "violence_marked", "taboo_unmarked",
              "m03_indiv", "m03_inst", "f21_inst"]
    print("\n  %-20s %s" % ("cell", " ".join("%-12s" % s[:12] for s in strata)))
    for src, fld in PRENAMED:
        line = []
        for s in strata:
            r = idx.get((s, src, fld))
            line.append("%-12s" % ("%.5f" % r["median_share"] if r else "--"))
        print("  %-20s %s" % ("%s/%s" % (src, fld), " ".join(line)))
    print("\n  full table, all strata and fields, coverage on every row: %s"
          % os.path.relpath(OUT_FIELDS, ROOT))
    print("  READ COVERAGE BEFORE ANY FIELD: on plan B's population RID gave the")
    print("  largest differences at 40%% coverage, which is a composition over a")
    print("  small non-random subset rather than a finding.")


def write_words(wdelta):
    """PRIMARY 3: the same question with no lexicon at all."""
    rows = []
    for stratum, d in wdelta.items():
        for word, vals in d.items():
            bylin = collections.defaultdict(list)
            for lin, v in vals:
                bylin[lin].append(v)
            if len(bylin) < 40:
                continue
            lm = [st.median(v) for v in bylin.values()]
            n, k, p = sign_test(lm)
            rows.append({"stratum": stratum, "word": word, "n_lineages": len(bylin),
                         "median_delta": st.median(lm), "lineages_pos": k,
                         "lineages_tested": n, "p": p,
                         "plan_b_word": word in PLAN_B_WORDS})
    if not rows:
        print("\n(no word rows reached 40 lineages)")
        return
    with open(OUT_WORDS, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.8g" % v if isinstance(v, float) else v)
                        for k, v in r.items()})

    print("\n" + "=" * 78)
    print("PRIMARY 3 -- do plan B's institutional risers rise on ordinary prompts?")
    print("=" * 78)
    print("  delta = aligned - base. These 14 words are the HYPOTHESIS, not the")
    print("  selector: a list read off one population cannot fail on it.")
    idx = {(r["stratum"], r["word"]): r for r in rows}
    strata = ["violence_unmarked", "violence_marked", "taboo_unmarked",
              "m03_indiv", "m03_inst"]
    print("\n  %-13s %s" % ("word", " ".join("%-16s" % s[:16] for s in strata)))
    for word in PLAN_B_WORDS:
        line = []
        for s in strata:
            r = idx.get((s, word))
            line.append("%-16s" % ("%+.5f %2d/%-2d" %
                                   (r["median_delta"], r["lineages_pos"],
                                    r["lineages_tested"]) if r else "--"))
        print("  %-13s %s" % (word, " ".join(line)))
    print("\n  T §13 predicts YES, and predicts the rise is if anything LARGER")
    print("  where there was no transgression to remove.")
    print("  full table: %s" % os.path.relpath(OUT_WORDS, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--tall", action="store_true",
                    help="also dump the per-word table (large)")
    a = ap.parse_args()
    if a.population:
        return build_population()
    if a.run:
        return run(tall=a.tall)
    ap.print_help()


if __name__ == "__main__":
    main()
