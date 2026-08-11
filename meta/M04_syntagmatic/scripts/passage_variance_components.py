#!/usr/bin/env python
"""What n_samples buys, measured rather than chosen. For the passage-corpus spec.

    uv run python passage_variance_components.py

[5450] asked the seat to attack `n_samples=10`, naming the crux as *"I have not
measured the within-site spread of mean surprisal"*. [5451] answered DO NOT
CHOOSE IT, MEASURE IT. It is measurable from two corpora already in ClickHouse,
at the same 256-token length the spec proposes:

    gen_scores  corpus='f11_l2'   20 samples/cell   226 tok   unforced
    gen_scores  corpus='y'        50 samples/cell   225 tok   25 forced words

**GROUP BY MUST INCLUDE forced_word.** Y carries 25 of them and only 34,000 of
its 231,200 rows are unforced; pooling them gave 341 "samples" in a 50-sample
corpus and an sd mixing conditions. The impossible sample count is what caught
it -- a spread computed over a pool that should not have existed.

THE STATISTIC is mean logprob per token per sample, i.e. A's `mean_lp`, so the
numbers below are directly comparable to A's primary of -0.0104/-0.0124.
"""
import json
import subprocess
import sys

CH = "/opt/homebrew/bin/clickhouse"


def q(sql):
    r = subprocess.run([CH, "client", "-q", sql + " FORMAT JSONEachRow"],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit("clickhouse: " + r.stderr[:400])
    return [json.loads(l) for l in r.stdout.splitlines() if l.strip()]


CELLS = """
WITH cells AS (
  SELECT corpus, model, prompt, forced_word, scorer,
         groupArray(arraySum(logprobs)/length(logprobs)) AS m
  FROM malign_logits.gen_scores
  WHERE corpus IN ('y','f11_l2') AND length(logprobs) > 0
  GROUP BY corpus, model, prompt, forced_word, scorer
  HAVING length(m) >= 10)
SELECT corpus, count() AS cells, median(length(m)) AS samples,
       median(arrayReduce('stddevSamp', m)) AS sd_w,
       quantile(0.1)(arrayReduce('stddevSamp', m)) AS p10,
       quantile(0.9)(arrayReduce('stddevSamp', m)) AS p90
FROM cells GROUP BY corpus"""

BETWEEN = """
WITH cells AS (
  SELECT corpus, model, scorer, prompt,
         avg(arraySum(logprobs)/length(logprobs)) AS site_mean
  FROM malign_logits.gen_scores
  WHERE corpus='f11_l2' AND length(logprobs) > 0
  GROUP BY corpus, model, scorer, prompt)
SELECT median(sd_b) AS sd_b, count() AS pairs FROM (
  SELECT stddevSamp(site_mean) AS sd_b FROM cells
  GROUP BY model, scorer HAVING count() >= 8)"""

SUB = """
SELECT * FROM (
WITH cells AS (
  SELECT groupArray(arraySum(logprobs)/length(logprobs)) AS m
  FROM malign_logits.gen_scores WHERE corpus='y' AND length(logprobs) > 0
  GROUP BY model, prompt, forced_word, scorer HAVING length(m) >= 50)
SELECT 5 AS N,  median(arrayReduce('stddevSamp', arraySlice(m,1,5)))  AS sd_at_N,
       median(arrayReduce('stddevSamp', arraySlice(m,1,5))/sqrt(5))   AS se_self FROM cells
UNION ALL SELECT 10, median(arrayReduce('stddevSamp', arraySlice(m,1,10))),
       median(arrayReduce('stddevSamp', arraySlice(m,1,10))/sqrt(10)) FROM cells
UNION ALL SELECT 20, median(arrayReduce('stddevSamp', arraySlice(m,1,20))),
       median(arrayReduce('stddevSamp', arraySlice(m,1,20))/sqrt(20)) FROM cells
UNION ALL SELECT 50, median(arrayReduce('stddevSamp', m)),
       median(arrayReduce('stddevSamp', m)/sqrt(50)) FROM cells
) ORDER BY N"""

S = 178          #: median sites per pair in the proposed population
A_EFFECT = 0.0104  #: A's primary, same units


def main():
    print("WITHIN-SITE SPREAD OF MEAN SURPRISAL, nats/token, two corpora\n")
    print("%-9s %8s %9s %9s %8s %8s" % ("corpus", "cells", "samples", "sd_w", "p10", "p90"))
    sdw = {}
    for r in q(CELLS):
        sdw[r["corpus"]] = r["sd_w"]
        print("%-9s %8d %9.0f %9.4f %8.4f %8.4f"
              % (r["corpus"], r["cells"], r["samples"], r["sd_w"], r["p10"], r["p90"]))
    print("\n  TWO INDEPENDENT CORPORA, DIFFERENT DOMAINS, %.0f%% APART."
          % (100 * abs(sdw["y"] - sdw["f11_l2"]) / sdw["f11_l2"]))

    b = q(BETWEEN)[0]
    w = sdw["f11_l2"]
    print("\nBETWEEN-SITE sd of the per-site mean: %.4f  (%d pairs)"
          % (b["sd_b"], b["pairs"]))
    print("  sd_w / sd_b = %.2f, so the VARIANCE ratio is %.1f -- the per-site"
          % (w / b["sd_b"], (w / b["sd_b"]) ** 2))
    print("  mean is the noisy layer, and N is what fixes it.")

    print("\nWHAT N BUYS AT THE PAIR LEVEL.  Var ~ (sd_b^2 + sd_w^2/N)/S,  S=%d" % S)
    floor = (b["sd_b"] ** 2 / S) ** 0.5
    print("  %4s %12s %14s" % ("N", "SE of pair", "above the floor"))
    for N in (5, 10, 20, 50):
        se = ((b["sd_b"] ** 2 + w * w / N) / S) ** 0.5
        print("  %4d %12.4f %13.0f%%" % (N, se, 100 * (se / floor - 1)))
    print("  %4s %12.4f %13s" % ("inf", floor, "--"))

    print("\nSUBSAMPLING Y'S 50-SAMPLE CELLS. The sd is BIASED LOW at small N,")
    print("so a run at N understates its OWN error bars:")
    rows = q(SUB)
    full = [r for r in rows if r["N"] == 50][0]["sd_at_N"]
    print("  %4s %10s %14s %14s" % ("N", "sd at N", "SE self-est", "SE true"))
    for r in rows:
        true = full / (r["N"] ** 0.5)
        print("  %4d %10.4f %14.4f %14.4f%s"
              % (r["N"], r["sd_at_N"], r["se_self"], true,
                 "   <- %.0f%% understated" % (100 * (true / r["se_self"] - 1))
                 if r["N"] < 50 else ""))

    print("\nSITE-GRAIN IS NOT AFFORDABLE AT ANY N. A per-site quantity has no S")
    print("to average over, so its SE is sd_w/sqrt(N) alone, against A's %.4f:" % A_EFFECT)
    for N in (10, 20, 50):
        se = w / (N ** 0.5)
        print("  N=%-3d  SE %.4f   = %4.0fx the effect" % (N, se, se / A_EFFECT))
    print("\nUNMEASURABLE HERE: A's Delta is a DOUBLE difference and its terms are")
    print("correlated within a site. Correlation cancels variance and neither")
    print("corpus carries paired forced arms, so the site-grain figures are an")
    print("UPPER bound. The pair-level table does not depend on it.")


if __name__ == "__main__":
    main()
