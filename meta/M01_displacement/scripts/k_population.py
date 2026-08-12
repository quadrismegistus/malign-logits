"""Plan K's population, both languages, straight off the movement table.

    uv run python meta/M01_displacement/scripts/k_population.py

THE QUESTION. At top-N per prompt per model, union across BOTH arms, over
lineage representatives: how many unique words, and how do they split into
fallers, risers and non-movers?

This is what F40 should have been. F40 pooled top-10s into 347 words and hand-
sorted them into seven bins whose TRANSGRESSIVE cell held 24 words -- 6.9% of the
list -- with the two categories a reader most wants (violence_explicit,
sexual_explicit) straddling zero. The list was the instrument and the list was
small. Here the population is derived rather than assembled, at every N, with
the movement class attached from the frozen rule instead of from a reading.

WHY N MATTERS AND IS NOT A TASTE. The mover set SATURATES: past some N every
additional word is a non-mover, so N stops buying coverage and starts buying
padding. That saturation point is a property of the data, and it is the
principled place to cut a population -- not a round number.

LINEAGE REPRESENTATIVES, one edge per lineage, chosen as the alphabetically
first aligned arm so the choice is reproducible and not "whichever the roster
listed first". Six Llama-3.1-8B families would otherwise vote six times.
"""
import collections
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
DB = os.environ.get("MALIGN_CH_DB", "malign_logits")
NS = (5, 10, 20, 50, 100, 200)


def q(sql):
    r = subprocess.run([CH, "client", "--query", sql + " FORMAT JSONEachRow"],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[:500])
    return [json.loads(l) for l in r.stdout.strip().split("\n") if l.strip()]


#: ORDINAL tier gate, applied INSIDE the roster. `x_zh_movement.py` records RH
#: finding the language-blind version of this in one question: `roster()` returns
#: every base>superego pair, and run on a Chinese prompt it pools Qwen and GLM
#: with SmolLM2, which has 77 CJK characters in its whole vocabulary -- their
#: "movement" at a Chinese prompt is tokenizer noise. Its lesson is explicit:
#: "the guard belongs in the roster rather than in the habit of whoever calls it."
#:
#: The first version of THIS file put it in neither, and the symptom was visible
#: in its own output: 53 lineages on zh prompts yielding MORE unique words than
#: English (101,851 against 51,359) off a sixth of the cells. A defect that
#: inflates a count reads as coverage.
CJK_OK = {"FLUENT", "MARGINAL", "PARTIAL"}


def _cjk_capable():
    import csv
    return {r["model"] for r in csv.DictReader(
        open(os.path.join(ROOT, "data/cjk_coverage.csv"))) if r["tier"] in CJK_OK}


def reps(lang):
    """One (base, aligned) edge per lineage, restricted to real training edges.

    `same_base_as` and the scale relations are EXCLUDED: a sibling pair is not
    an alignment step, and "movement" across it is a comparison of two trained
    models rather than the effect of training. Including them would inflate the
    word set with differences alignment never made.
    """
    rows = q("""
      SELECT m.lineage AS lineage, m.base AS base, m.aligned AS aligned,
             count() AS rows
      FROM %s.movement AS m
      INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                  WHERE status='ACTIVE' AND language='%s') AS p ON m.prompt = p.prompt
      WHERE m.rule='canonical'
        AND m.relation IN ('sft_of','dpo_of','kto_of','ppo_of','slic_of','rlvr_of')
      GROUP BY lineage, base, aligned
    """ % (DB, DB, lang))
    if lang == "zh":
        ok = _cjk_capable()
        rows = [r for r in rows if r["base"] in ok and r["aligned"] in ok]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["lineage"]].append((r["aligned"], r["base"]))
    return [(b, a) for lin, v in sorted(by.items()) for a, b in [sorted(v)[0]]]


def population(lang, edges):
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    pairs = " OR ".join("(m.base='%s' AND m.aligned='%s')" % (esc(b), esc(a))
                        for b, a in edges)
    out = {}
    for n in NS:
        rows = q("""
          SELECT
            countDistinct(word) AS uniq,
            countDistinct(if(ever_fall AND NOT ever_rise, word, NULL)) AS fall_only,
            countDistinct(if(ever_rise AND NOT ever_fall, word, NULL)) AS rise_only,
            countDistinct(if(ever_rise AND ever_fall, word, NULL))     AS both,
            countDistinct(if(NOT ever_rise AND NOT ever_fall, word, NULL)) AS still
          FROM (
            SELECT word,
                   max(cls = 'fall') AS ever_fall,
                   max(cls = 'rise') AS ever_rise
            FROM (
              SELECT m.word AS word, m.cls AS cls,
                     row_number() OVER (PARTITION BY m.base, m.aligned, m.prompt
                                        ORDER BY m.p_base DESC)    AS rb,
                     row_number() OVER (PARTITION BY m.base, m.aligned, m.prompt
                                        ORDER BY m.p_aligned DESC) AS ra
              FROM %s.movement AS m
              INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                          WHERE status='ACTIVE' AND language='%s') AS p
                ON m.prompt = p.prompt
              WHERE m.rule='canonical' AND (%s)
            )
            WHERE rb <= %d OR ra <= %d
            GROUP BY word
          )
        """ % (DB, DB, lang, pairs, n, n))[0]
        out[n] = rows
    return out


def main():
    for lang, label in (("en", "ENGLISH"), ("zh", "CHINESE")):
        edges = reps(lang)
        if not edges:
            print("\n%s: no edges\n" % label)
            continue
        ncells = q("""SELECT count() AS c FROM (SELECT DISTINCT m.base, m.aligned, m.prompt
          FROM %s.movement AS m
          INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                      WHERE status='ACTIVE' AND language='%s') AS p ON m.prompt=p.prompt
          WHERE m.rule='canonical' AND (%s))"""
                  % (DB, DB, lang,
                     " OR ".join("(m.base='%s' AND m.aligned='%s')"
                                 % (b.replace("'", "\\'"), a.replace("'", "\\'"))
                                 for b, a in edges)))[0]["c"]
        print("\n%s\n%s -- %d lineage representatives, %s cells\n%s"
              % ("=" * 78, label, len(edges), format(ncells, ","), "=" * 78))
        print("    N     unique   fall-only   rise-only    both    never-move")
        pop = population(lang, edges)
        prev = None
        for n in NS:
            r = pop[n]
            mv = r["fall_only"] + r["rise_only"] + r["both"]
            flag = ""
            if prev is not None and mv - prev < 10:
                flag = "  <- movers saturated"
            prev = mv
            print("  %4d  %8s   %9s   %9s  %6s   %7s (%2.0f%%)%s"
                  % (n, format(r["uniq"], ","), format(r["fall_only"], ","),
                     format(r["rise_only"], ","), format(r["both"], ","),
                     format(r["still"], ","),
                     100 * r["still"] / r["uniq"] if r["uniq"] else 0, flag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
