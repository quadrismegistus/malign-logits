"""Plan B: the institutional effect on twp, at 46 lineages, with no annotator.

    uv run python b_twp_institutional.py --population
    uv run python b_twp_institutional.py --run            # movement.CANONICAL

Plan: `meta/M03_proceduralization/plans/plan_b_twp_institutional.md`.

THE MOVEMENT RULE IS `malign_logits.movement.CANONICAL` AND IS NOT A PARAMETER
OF THIS SCRIPT. The first draft invented `--riser-mass` / `--riser-topk` with no
default, on the reasoning that a threshold chosen after seeing the curves is
chosen on the outcome. That reasoning was right and the conclusion was wrong:
`movement.py` already exists BECAUSE "every seat derived fallers and risers on
its own constants", and inventing a fresh knob -- even an undefaulted one -- is
that failure with better manners. Cite the profile; do not re-type its numbers.

WHAT IT COMPUTES, per (lineage-representative pair, prompt):

    js          Jensen-Shannon between the base and aligned twp distributions,
                RESIDUAL KEPT AS A BIN (C1's settled default -- the unscored tail
                is a state, not something to normalise away)
    risers      CANONICAL: gained, AND beyond the renormalisation null --
                every word gains a little once a faller's mass is removed, and
                the null is what separates redistribution from bookkeeping
    fallers     a bare ratio rule. ASYMMETRY PRESERVED: fallers are NOT tested
                against the null and may never be called "beyond renormalisation"
    fields      semantic-field counts over the risers and over the fallers,
                from EVERY lexicon fields.available() reports, WITH COVERAGE

THE UNIT IS THE LINEAGE, n = 46. Not 92 models, not 290 prompts, not 26,680
cells. Falcon3 1B/3B/7B is one observation. The output carries `lineage` on
every row so no downstream aggregation has to rediscover that.

PROMPTS NEVER LEAVE THE DATABASE. The population is a subquery against
`prompt_catalogue`; the only strings this process puts INTO SQL are model ids,
and it asserts none of them contains an apostrophe. Three counts of this
population were wrong before that rule was adopted -- rows-not-texts once, then
twice more because `it's`, `can't` and `aren't` were double-escaped on the round
trip out of TSV and back into a query literal, and matched nothing. A lookup
under the wrong key returns a confident false negative, so the remedy is
structural rather than careful.

FIELDS COME FROM EXTERNAL LEXICONS, NEVER FROM A LIST DERIVED HERE. C2's word
list was read off this population's own risers, so it could not fail on it, and
off it the same six words moved 5.5x the other way. The risers are the INPUT to
the field counts and never the definition of them.

AND THIS INSTRUMENT CANNOT MEASURE AGENCY. F21's addendum binds the narration --
"Agency RISES in every family... do not narrate submission" -- and no output of
this script may be used to reopen that. See the plan §5.
"""
import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)

CH = "/opt/homebrew/bin/clickhouse"
DB = "malign_logits"
POP = os.path.join(CAMP, "populations", "b_population.json")
OUT = os.path.join(CAMP, "results", "b_pair_prompt.jsonl")
OUT_F = os.path.join(CAMP, "results", "b_fields.jsonl")

#: the two halves of the population, as PREDICATES rather than as text lists.
#: `finding='F21'` alone leaks 13 zh rows -- the same leak that hit the M05
#: battery build, caught on RH's "which institutional?".
SELECTORS = {
    "f21_institutional": "finding='F21' AND status='ACTIVE' AND language='en'",
    "m03_speaker_kernel": "source='M03_SPEAKER_KERNEL'",
}


#: CLICKHOUSE TSV ESCAPES ON OUTPUT, AND THE APOSTROPHE IS ONE OF THEM.
#: `can't` in the store comes back over `FORMAT TSV` as `can\'t` -- verified
#: with `od -c` against `prompt_catalogue`, where JSONEachRow returns the clean
#: text and both stores hold ZERO backslashes. The JS and movement numbers were
#: never wrong, because both arms travel this same pipe and their keys match
#: each other; what breaks is every join to anything OUTSIDE it. Three of F21's
#: 38 texts failed to join to `prompt_categorisation.json`, and `weren\'t`
#: appeared as a word key in the delta table.
#:
#: This is the SAME THREE WORDS as the old defect (`it's`, `can't`, `aren't`)
#: in the OPPOSITE DIRECTION. "Prompts never leave the database" governs text
#: going INTO SQL and is silent about text coming OUT, so the doctrine that was
#: supposed to have closed this covered one direction only.
TSV_UNESCAPE = {"\\\\": "\\", "\\'": "'", "\\t": "\t", "\\n": "\n",
                "\\r": "\r", "\\0": "\0", "\\b": "\b", "\\f": "\f"}


def tsv_unescape(s):
    """Reverse ClickHouse's TSV output escaping. Left-to-right, never a chain
    of str.replace: replacing `\\\\` first and `\\'` after would turn the two
    characters `\\` `\\'` into something neither escape produced."""
    if "\\" not in s:
        return s
    out, i = [], 0
    while i < len(s):
        two = s[i:i + 2]
        if two in TSV_UNESCAPE:
            out.append(TSV_UNESCAPE[two])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _ch(sql):
    r = subprocess.run([CH, "client", "--max_query_size", "20000000", "-q", sql],
                       capture_output=True)
    if r.returncode:
        raise SystemExit(r.stderr.decode()[:600])
    return r.stdout.decode("utf-8")


def pairs_and_models():
    f = os.path.join(ROOT, "data", "lineage_representative_pairs.txt")
    pairs = [l.strip() for l in open(f) if l.strip()]
    models = sorted({m for p in pairs for m in p.split(">")})
    #: THE ONLY STRINGS THIS PROCESS PUTS INTO SQL. Asserted, not hoped.
    bad = [m for m in models if "'" in m or "\\" in m]
    if bad:
        raise SystemExit("model id would need escaping, refusing: %s" % bad)
    return pairs, models


def _lineage(m):
    from malign_logits.lineage import lineage_of, UnmappedModel
    try:
        return lineage_of(m)
    except UnmappedModel:
        return None


def population_sql():
    return " UNION ALL ".join(
        "SELECT DISTINCT prompt, '%s' AS stratum FROM %s.prompt_catalogue WHERE %s"
        % (k, DB, v) for k, v in SELECTORS.items())


def build_population():
    pairs, models = pairs_and_models()
    mls = ",".join("'%s'" % m for m in models)
    rows = _ch("""
      SELECT c.stratum, count() AS n_texts, sum(full92) AS at_full, sum(part) AS partial
      FROM (
        SELECT p.stratum AS stratum,
               uniqExactIf(w.model, w.model IN (%s)) AS n,
               n = %d AS full92, n > 0 AND n < %d AS part
        FROM (%s) AS p
        LEFT JOIN %s.twp_words AS w ON w.prompt = p.prompt
        GROUP BY p.stratum, p.prompt
      ) AS c GROUP BY c.stratum ORDER BY c.stratum FORMAT TSV
    """ % (mls, len(models), len(models), population_sql(), DB))
    tally = {}
    for line in rows.strip().split("\n"):
        st, n, full, part = line.split("\t")
        tally[st] = dict(texts=int(n), at_full=int(full), partial=int(part))
        print("  %-22s %4s texts   at full %s-model coverage: %s   partial: %s"
              % (st, n, len(models), full, part))
    #: the hash is over the SELECTORS and the roster, not over the texts -- the
    #: texts are defined by the predicates and never materialise here.
    key = json.dumps({"selectors": SELECTORS, "models": models}, sort_keys=True)
    sha = hashlib.sha256(key.encode()).hexdigest()
    doc = {
        "_what": "F21 institutional + M03 speaker kernel, on the 46 lineage-"
                 "representative pairs. Unit: (lineage, prompt).",
        "_prompts_never_leave_the_database":
            "The population is these SELECTORS, applied as a subquery. No text is "
            "written into this file or into a query literal. Three earlier counts "
            "were wrong because texts round-tripped through TSV and back: `it's`, "
            "`can't`, `aren't` double-escaped and matched nothing.",
        "_unit": "THE LINEAGE, n=46. Not 92 models, not 290 prompts, not 26,680 cells.",
        "_cannot_measure": "AGENCY. See plan §5; the addendum's 'do not narrate "
                           "submission' is not dischargeable by this instrument.",
        "selectors": SELECTORS,
        "n_pairs": len(pairs), "n_models": len(models),
        "coverage": tally,
        "population_sha256": sha, "population_sha256_16": sha[:16],
    }
    os.makedirs(os.path.dirname(POP), exist_ok=True)
    json.dump(doc, open(POP, "w"), indent=1)
    print("\nsha16 %s   wrote %s" % (sha[:16], os.path.relpath(POP, ROOT)))
    return doc


def fetch(models):
    """{(model, prompt): {word: p}} and {(model, prompt): stratum}. ONE query."""
    mls = ",".join("'%s'" % m for m in models)
    out = _ch("""
      SELECT w.model, w.prompt, p.stratum, w.word, sum(w.p) AS pr
      FROM (%s) AS p
      INNER JOIN %s.twp_words AS w ON w.prompt = p.prompt
      WHERE w.model IN (%s) AND abs(w.theta - 0.001) < 1e-9
      GROUP BY w.model, w.prompt, p.stratum, w.word FORMAT TSV
    """ % (population_sql(), DB, mls))
    D, strat = collections.defaultdict(dict), {}
    for line in out.splitlines():
        m, pr, st, word, v = line.split("\t")
        pr, word = tsv_unescape(pr), tsv_unescape(word)
        D[(m, pr)][word] = float(v)
        strat[pr] = st
    return D, strat


def js_with_residual(a, b):
    """JS over the union of words PLUS a residual bin.

    **The residual is a BIN, not something to normalise away** (C1's settled
    default). The unscored tail is a state of the model -- mass it put on words
    below theta -- and dropping it renormalises two different tails to 1 and
    calls them comparable.
    """
    import numpy as np
    keys = sorted(set(a) | set(b))
    va = np.array([a.get(k, 0.0) for k in keys] + [max(0.0, 1.0 - sum(a.values()))])
    vb = np.array([b.get(k, 0.0) for k in keys] + [max(0.0, 1.0 - sum(b.values()))])
    va = np.clip(va / va.sum(), 1e-12, None)
    vb = np.clip(vb / vb.sum(), 1e-12, None)
    m = 0.5 * (va + vb)
    return float(0.5 * (va * np.log(va / m)).sum() + 0.5 * (vb * np.log(vb / m)).sum())


def movement_of(a, b, rule):
    """Risers and fallers from the CANONICAL rule. **Not a threshold of mine.**

    `malign_logits/movement.py` exists because "every seat derived fallers and
    risers on its own constants". The first draft of this producer invented
    `--riser-mass` / `--riser-topk` and was exactly that failure (RH caught it).

    THE NULL IS THE POINT. Without it a riser is any word that went up, and
    EVERY word goes up a little once a faller's mass is removed -- the
    renormalisation null is what separates redistribution from bookkeeping.

    THE RESIDUAL IS PASSED EXPLICITLY. twp is truncated at theta, so the scored
    words sum to 1 - residual and the null needs total mass. Omitting it makes
    `exact_null` False and `residual_share` 0.0, which the module's own
    docstring calls "a claim about the input, not a property of the data".

    ASYMMETRY, PRESERVED FROM THE ORIGINAL AND NOT TO BE NARRATED AWAY: risers
    are tested against the null, FALLERS ARE NOT. Nothing downstream may
    describe a faller as "beyond renormalisation".
    """
    from malign_logits.movement import movement
    return movement(a, b, rule=rule,
                    residual_pre=max(0.0, 1.0 - sum(a.values())),
                    residual_post=max(0.0, 1.0 - sum(b.values())))


def field_counts(words):
    """Every lexicon, both granularities, WITH COVERAGE.

    Coverage is not decoration: the General Inquirer has no entry for `raped`,
    `desecrated` or `stomped`, so on this corpus it silently drops the
    transgressive end. A count without its denominator compares how much of the
    text a lexicon happens to know.
    """
    from malign_logits import fields
    text = " ".join(sorted(words))
    out = {}
    for src in ("usas_fine", "meta", "usas", "gi", "wordnet", "rid", "byu"):
        try:
            out[src] = fields.count(text, source=src)
        except Exception as e:
            out[src] = {"_error": str(e)[:120]}
    try:
        out["norms"] = fields.norms(text)
    except Exception as e:
        out["norms"] = {"_error": str(e)[:120]}
    return out


def run(rule):
    pairs, models = pairs_and_models()
    print("fetching twp for %d models over the population ..." % len(models), flush=True)
    D, strat = fetch(models)
    print("cells fetched: %d" % len(D), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w") as fh, open(OUT_F, "w") as fg:
        for p in pairs:
            b, a = p.split(">")
            lin = _lineage(b)
            prompts = sorted({pr for (m, pr) in D if m == b} & {pr for (m, pr) in D if m == a})
            for pr in prompts:
                da, db = D[(b, pr)], D[(a, pr)]
                mv = movement_of(da, db, rule)
                ri, fa = mv.risers, mv.fallers
                fh.write(json.dumps({
                    "lineage": lin, "base": b, "aligned": a, "prompt": pr,
                    "stratum": strat.get(pr), "js": js_with_residual(da, db),
                    "n_words_base": len(da), "n_words_aligned": len(db),
                    "mass_base": sum(da.values()), "mass_aligned": sum(db.values()),
                    "n_risers": len(ri), "n_fallers": len(fa),
                    "risers": sorted(ri, key=lambda w: -mv.excess.get(w, 0.0))[:40],
                    "fallers": sorted(fa, key=lambda w: mv.delta.get(w, 0.0))[:40],
                    "rule": rule.name, "inflation": mv.inflation,
                    #: READ THESE BEFORE QUOTING AN EXCESS. A null over a
                    #: truncated support is approximate and says so; at a
                    #: residual share of 0.26 the bucket is larger than most
                    #: words in the cell.
                    "exact_null": mv.diagnostics.get("exact_null"),
                    "residual_share": mv.diagnostics.get("residual_share"),
                    "excess_top": {w: mv.excess.get(w) for w in
                                   sorted(ri, key=lambda x: -mv.excess.get(x, 0.0))[:10]},
                    }) + "\n")
                for side, ws in (("risers", ri), ("fallers", fa)):
                    if not ws:
                        continue
                    fg.write(json.dumps({
                        "lineage": lin, "base": b, "aligned": a, "prompt": pr,
                        "stratum": strat.get(pr), "side": side, "n_words": len(ws),
                        "fields": field_counts(ws)}) + "\n")
                n += 1
            print("  %-44s %3d prompts" % (b.split("/")[-1][:42], len(prompts)), flush=True)
    print("\nwrote %d (pair, prompt) rows -> %s" % (n, os.path.relpath(OUT, ROOT)))
    print("       fields               -> %s" % os.path.relpath(OUT_F, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", action="store_true")
    ap.add_argument("--run", action="store_true")
    #: THE RULE IS NAMED, NOT NUMBERED. Cite the profile; do not re-type its
    #: constants -- that is what movement.py exists to stop.
    ap.add_argument("--rule", choices=("canonical", "lens", "draw"),
                    default="canonical",
                    help="movement.py profile. CANONICAL tests risers against "
                         "the renormalisation null and is the default because "
                         "it is the one whose claims are about the riser SET.")
    a = ap.parse_args()
    if a.population:
        build_population()
    if a.run:
        from malign_logits import movement as M
        run({"canonical": M.CANONICAL, "lens": M.LENS, "draw": M.DRAW}[a.rule])
    if not (a.population or a.run):
        raise SystemExit("pass --population and/or --run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
