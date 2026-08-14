"""Check the claims the fluency finding RESTS on and nothing else tests.

    uv run python meta/M06_generation/scripts/m06_zh_fluency_audit.py

Every number in `findings/zh_fluency_and_ordering.md` is downstream of one
structural claim: **the judging was blind.** That claim lives in a docstring.
The asserts in the producers test counts and thresholds; none of them tests
whether a judge could have known which model wrote a passage, and a leak would
not make any output look wrong.

**That is dario's category from [6154]: a false reason for a correct design is
invisible to every check.** A figure whose numbers are right and whose stated
reason is false passes the asserts, passes the pixel audit, and misleads the
next reader. The same holds for a blind protocol that is not blind: the
verdicts would still be well-formed, the agreement would still be high, and
the finding would be worthless.

FIVE CHECKS, each naming what would have to be true for it to matter:

  1 FIELD LEAK      the batch files carry ONLY key/prompt/continuation.
                    A `model` field would end the study outright.
  2 BATCH CLUSTER   models must not concentrate in batches. If one agent saw
                    mostly one arm, its calibration becomes that arm's score.
  3 KEY ORDER       key index must not track model. Weak here and harmless --
                    the judge sees batch order, not key order, and could not
                    decode an alphabetical rank -- but it is checked rather
                    than argued away.
  4 RATING ORDER    position within a batch must not predict the verdict.
                    A fatiguing judge would impose a gradient on whatever
                    happened to be late in each file.
  5 ROUND DRIFT     the two rounds may rate differently; what matters is that
                    the offset is CONSTANT across models, or it does not
                    cancel in an aligned-minus-base contrast.

Check 5 is the one that could have bitten. Round 2 rates +0.087 more generously
than round 1 on the re-rates. That is harmless only because every model carries
exactly 6 round-1 and 14 round-2 first ratings -- balanced BY CONSTRUCTION,
since the draw is per model. Had the rounds been drawn any other way the offset
would have been a per-model bias wearing the shape of a result.
"""
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
OUT = os.path.join(OUTD, "zh_fluency_audit.json")
SCORE = {"fluent": 3, "flawed": 2, "broken": 1, "not_chinese": 0}
ALLOWED = {"key", "prompt", "continuation"}
ROUNDS = (("", "r1"), ("_r2", "r2"))


def main():
    from scipy import stats
    res, ok = {}, True

    def say(name, passed, detail):
        nonlocal ok
        ok = ok and passed
        print("  %-14s %-4s %s" % (name, "PASS" if passed else "FAIL", detail))
        res[name] = {"pass": bool(passed), "detail": detail}

    for sfx, label in ROUNDS:
        sp = os.path.join(OUTD, "zh_fluency_sample%s.json" % sfx)
        vp = os.path.join(OUTD, "zh_fluency_verdicts%s.json" % sfx)
        if not (os.path.exists(sp) and os.path.exists(vp)):
            continue
        truth = json.load(open(sp))["truth"]
        vd = json.load(open(vp))
        vd = vd["verdicts"] if isinstance(vd, dict) else vd
        sc = {r["key"]: SCORE[r["verdict"]] for r in vd}
        print("\nROUND %s  (%d sampled, %d judged)" % (label, len(truth), len(sc)))

        #: 1 FIELD LEAK
        extra, n = set(), 0
        pos_x, pos_y, bym = [], [], collections.defaultdict(set)
        for p in sorted(glob.glob(os.path.join(
                OUTD, "zh_fluency_batches%s" % sfx, "batch_*.json"))):
            items = json.load(open(p))
            for i, it in enumerate(items):
                n += 1
                extra |= set(it) - ALLOWED
                if it["key"] in truth:
                    bym[truth[it["key"]]["model"]].add(os.path.basename(p))
                if it["key"] in sc:
                    pos_x.append(i / max(1, len(items) - 1))
                    pos_y.append(sc[it["key"]])
        say("field-leak", not extra,
            "%d items, fields beyond the three: %s"
            % (n, sorted(extra) or "none"))

        #: 2 BATCH CLUSTER
        spread = [len(v) for v in bym.values()]
        nb = len({b for v in bym.values() for b in v})
        say("batch-cluster", min(spread) >= 3,
            "%d models over %d batches; min %d batches/model"
            % (len(bym), nb, min(spread)))

        #: 3 KEY ORDER
        ks = sorted(truth)
        mr = {m: i for i, m in enumerate(sorted({v["model"] for v in truth.values()}))}
        rho, pv = stats.spearmanr(list(range(len(ks))),
                                  [mr[truth[k]["model"]] for k in ks])
        say("key-order", abs(rho) < 0.2,
            "spearman(key index, model index) %+.4f p=%.3f" % (rho, pv))

        #: 4 RATING ORDER
        rho2, pv2 = stats.spearmanr(pos_x, pos_y)
        say("rating-order", pv2 > 0.05,
            "spearman(position in batch, score) %+.4f p=%.3f" % (rho2, pv2))

    #: 5 ROUND DRIFT -- needs both rounds
    t2 = json.load(open(os.path.join(OUTD, "zh_fluency_sample_r2.json")))["truth"]
    v1 = json.load(open(os.path.join(OUTD, "zh_fluency_verdicts.json")))["verdicts"]
    v2 = json.load(open(os.path.join(OUTD, "zh_fluency_verdicts_r2.json")))["verdicts"]
    s1 = {r["key"]: SCORE[r["verdict"]] for r in v1}
    s2 = {r["key"]: SCORE[r["verdict"]] for r in v2}
    d = [s2[k] - s1[m["first_key"]] for k, m in t2.items()
         if m.get("role") == "iaa" and k in s2 and m.get("first_key") in s1]
    t1 = json.load(open(os.path.join(OUTD, "zh_fluency_sample.json")))["truth"]
    per = collections.Counter()
    for v in t1.values():
        per[(v["model"], "r1")] += 1
    for v in t2.values():
        if v.get("role") == "new":
            per[(v["model"], "r2")] += 1
    models = sorted({m for m, _ in per})
    share = [per[(m, "r1")] / max(1, per[(m, "r1")] + per[(m, "r2")]) for m in models]
    print("\nBOTH ROUNDS")
    print("  %-14s %-4s round2-minus-round1 %+.3f on %d re-rates"
          % ("round-drift", "----", sum(d) / len(d), len(d)))
    say("drift-cancels", (max(share) - min(share)) < 1e-9,
        "round-1 share per model: min %.3f max %.3f spread %.2g -- a constant "
        "offset cancels in any within-model contrast"
        % (min(share), max(share), max(share) - min(share)))

    json.dump({"_about": "Structural checks on the blind fluency protocol. "
                         "Nothing else in this campaign tests them, and a "
                         "failure would leave every output well-formed.",
               "all_pass": ok, "checks": res},
              open(OUT, "w"), indent=1)
    print("\n%s -> %s" % ("ALL CHECKS PASS" if ok else "**A CHECK FAILED**",
                          os.path.relpath(OUT, ROOT)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
