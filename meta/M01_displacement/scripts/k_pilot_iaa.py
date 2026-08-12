"""Plan K's pilot sidecar: inter-annotator agreement, calibration, rarity check.

    uv run python meta/M01_displacement/scripts/k_pilot_iaa.py

Reads `results/k/pilot_en_475.json`, writes `results/k/iaa.json`.

**EVERY FIGURE IS RECOMPUTED FROM THE PILOT ROWS AT WRITE TIME.** Nothing is
transcribed, so the sidecar cannot drift from the artifact it describes. Re-run
it after any change to the pilot set and the band updates with it.

WHY A SIDECAR AND NOT A PARAGRAPH. Registrar's Findings G established the
pattern this follows: two-coder pilot with tie-break, single-coder bulk, and the
pilot's disagreement band riding in a sidecar so it travels with any drafted
rate. A band that lives only in a limits section is a band nobody applies.

TWO ENTRIES ATTACH AT THE POINT OF USE rather than in a limits section:

  concreteness    agrees EXACTLY only 39% of the time (within-1 80%) -- the
                  noisiest scale in absolute terms, though it is the one with
                  an external check the others lack (Brysbaert r 0.88).
  register_level  carries r 0.60 for the general population. Its 0.85 applies
                  ONLY to register-varying pairs and must not be quoted for
                  anything else.

AND ONE FIGURE HERE IS A LESSON, NOT A NUMBER. On the frequency-stratified 300
alone, vulgarity had sd 0.10 and a maximum of 2, and its inter-coder r read
0.28. That is a no-variance artefact, not disagreement: frequency stratification
draws almost no obscenity, so the sample lacked the construct it was being used
to measure. The USAS stratum took the same scale to 0.88. **A reliability figure
computed on a sample that lacks the construct is not a reliability figure**, and
it fails in the direction of looking like a broken instrument.
"""
import csv
import itertools
import json
import math
import os
import statistics as st
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

SCALES = ("vulgarity", "register_level", "transgressiveness", "charge",
          "valence", "bodily_harm", "concreteness")
PILOT = os.path.join(ROOT, "meta/M01_displacement/results/k/pilot_en_475.json")
OUT = os.path.join(ROOT, "meta/M01_displacement/results/k/iaa.json")
BYU = os.path.expanduser("~/Dropbox/Prof/Code/osp/worddb.byu.txt")


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else None


def main():
    from malign_logits.fields import _norms
    from malign_logits.tasks.rate_charge_v1 import ChargeTask7EN
    D = json.load(open(PILOT))

    agreement = {}
    for s in SCALES:
        a = [r["a_" + s] for r in D]
        b = [r["b_" + s] for r in D]
        agreement[s] = {
            "pearson_r": round(pearson(a, b), 4),
            "exact": round(sum(1 for x, y in zip(a, b) if x == y) / len(a), 4),
            "within_1": round(sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / len(a), 4),
            "mean_abs_diff": round(st.mean(abs(x - y) for x, y in zip(a, b)), 4),
            "coder_a_mean": round(st.mean(a), 3),
            "coder_a_sd": round(st.pstdev(a), 3),
            "n": len(a),
        }

    #: CALIBRATION IS WHAT LICENSES THE CHINESE HALF. Chinese has no Warriner and
    #: no Brysbaert; if the coder tracks human norms on English, the Chinese
    #: ratings inherit a human-normed warrant instead of resting on the coder's
    #: say-so. Charge-vs-arousal is reported and is NOT a failure at 0.54: charge
    #: is declared as intensity in either direction, arousal is its own
    #: construct, and the pair is recorded so nobody later pools them.
    N = _norms()
    N = N[0] if isinstance(N, tuple) else N
    calibration = {}
    for scale, key in (("valence", "valence"), ("charge", "arousal"),
                       ("concreteness", "concreteness")):
        p = [(r["a_" + scale], N[r["word"].strip().lower()][key]) for r in D
             if r["word"].strip().lower() in N and key in N[r["word"].strip().lower()]]
        if p:
            calibration[scale] = {"vs_human_norm": key, "n": len(p),
                                  "pearson_r": round(pearson([x for x, _ in p],
                                                             [y for _, y in p]), 4)}

    #: THE RARITY CHECK. X_metonymy's -0.33 nuisance floor means a coder reading
    #: unusual-ness as charge would manufacture the artefact the analysis then
    #: has to partial out. This is the number that says whether the instruction
    #: held.
    freq = {}
    with open(BYU, encoding="utf-8", errors="ignore") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            w = (row.get("word") or "").strip().lower()
            try:
                v = float(row.get("fpm_COCA") or 0)
                if v > 0:
                    freq[w] = v
            except (TypeError, ValueError):
                pass
    have = [r for r in D if r["word"].strip().lower() in freq]
    lf = [math.log10(freq[r["word"].strip().lower()]) for r in have]
    rarity = {s: round(pearson(lf, [r["a_" + s] for r in have]), 4) for s in SCALES}
    rarity["_n"] = len(have)

    collinearity = {"%s~%s" % (a, b): round(
        pearson([r["a_" + a] for r in D], [r["a_" + b] for r in D]), 4)
        for a, b in itertools.combinations(SCALES, 2)}

    out = {
        "_what": "Plan K inter-annotator agreement and calibration. THE BAND "
                 "TRAVELS WITH EVERY DRAFTED NUMBER. Two entries attach at the "
                 "point of use, not in a limits section: concreteness agrees "
                 "exactly only 39% of the time, and register_level carries 0.60 "
                 "for the general population (0.85 applies only where register "
                 "actually varies).",
        "_produced_by": "meta/M01_displacement/scripts/k_pilot_iaa.py",
        "instrument_sha256": ChargeTask7EN().instrument_sha256(),
        "coder_a": "deepseek/deepseek-v4-flash",
        "coder_b": "anthropic/claude-haiku-4-5",
        "temperature": 0.0,
        "n_words": len(D),
        "sample": {
            "frequency_stratified": 300,
            "usas_transgressive_stratum": 175,
            "usas_tags": ["S3.2", "G2.1", "G3", "G2.2", "B1", "E3", "A1.1.2"],
            "stratum_selected_by": "USAS primary tag, externally -- NEVER by the "
                                   "coder's own ratings",
            "note": "vulgarity read r=0.28 on the frequency-only 300 (sd 0.10, "
                    "max 2) and 0.88 with the stratum. A reliability figure "
                    "computed on a sample lacking the construct is not one.",
        },
        "agreement": agreement,
        "calibration_vs_human_norms": calibration,
        "rarity_check_vs_log10_coca_fpm": rarity,
        "collinearity": collinearity,
        "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip(),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote %s" % os.path.relpath(OUT, ROOT))
    for s in SCALES:
        a = agreement[s]
        print("   %-18s r %.2f  exact %3.0f%%  within-1 %3.0f%%"
              % (s, a["pearson_r"], 100 * a["exact"], 100 * a["within_1"]))
    worst = max(collinearity.items(), key=lambda kv: abs(kv[1]))
    print("   worst collinearity: %s %+.2f" % worst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
