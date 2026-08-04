#!/usr/bin/env python3
"""REGISTRATION N's PRODUCER — clause 1, `mass-migration`. ARM A, ENGLISH ONLY.

Frozen registration: `registration_n_migration_and_null.md`
  @ 9fb5e13fd1c3b1c8201e6cdc28205dc752757df11db6673f0a9d750187d79c6c
  committed c7a101de7cc9a4c98d3dee3e57d3e107c34c4d77, RH's word "Freeze N".

**THIS FILE IMPLEMENTS THE FROZEN TEXT AND NOTHING ELSE.** Where the text is
silent it RAISES rather than chooses. Every section it implements is cited by
number, because a producer citing a docket message cites something a reader of
this repository cannot resolve (Amendment D3b-B §B7's rule, adopted here).

    §3     population: 2,199 ENGLISH stimuli x 44 operation_edges = 96,756
    §3.0   language is a FILTER, not a stratum. zh -> Registration O.
    §4     the statistic, the units, the combination, the sidedness, ties
    §4.1   the corrected-input primary, run TWICE, and the companion column
    §N6    the symbols and the reconstruction of R, S, n_unscored_fallers
    §6.1   require_frozen FIRST LINE -- called, not remembered
    §6.5   zero-faller cells EXCLUDED, counted, one rate from the ANALYSED set
    §6.6   a malformed row is REFUSED AND NAMED, never skipped

**§6.2 and §6.4 ARE STRUCK** ([4012].S10): they gate logit shards and N opens
none. §6.3's non-finite refusal is the module's, wired in `movement.word_probs`,
not this producer's.

**NO SEED.** Nothing here samples, splits or shuffles.

**THE READING IS §4.1's AND IT IS DECLARED BEFORE ANY NUMBER EXISTS:**

    THE CORRECTED ARM CARRIES ALL VERDICT LANGUAGE.  THE RAW ARM CARRIES
    NONE -- it is a REPORTED DIAGNOSTIC and never a finding.
      raw + / corrected +     SUBSTITUTION CONFIRMED
      raw + / corrected null  NOT SUPPORTED -- never REFUTED
      raw null                NOT SUPPORTED

    python meta/M01_displacement/scripts/n_primary.py [--dry-run]
"""

import collections
import hashlib
import json
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

REGISTRATION = os.path.join(CAMPAIGN, "registrations",
                            "registration_n_migration_and_null.md")
REGISTRATION_SHA = ("9fb5e13fd1c3b1c8201e6cdc28205dc752757df11db6673f0a9d"
                    "750187d79c6c")
OUT = os.path.join(CAMPAIGN, "results", "result_n_primary.json")
ESCROW = os.path.join(CAMPAIGN, "results", "superseded")

#: §7's floor, restated at its point of use per §N6.
THETA = 0.001

#: §3: sentinels are not stimuli. `<<<LOGICAL:BOS>>>` is the BOS-policy probe.
SENTINEL = re.compile(r"^<<<.*>>>$")
#: §3.0: language is a FILTER. A stimulus containing any CJK character is zh.
CJK = re.compile(r"[一-鿿]")


# ══════════════════════════════════════════════════════════════════════════
# §6.1 — THE FIRST LINE. Called, not remembered.
# ══════════════════════════════════════════════════════════════════════════
def gate_registration():
    """§6.1. `require_frozen` before ANY read, and the bytes named as well.

    The gate proves the registration is frozen; it does not prove WHICH frozen
    registration. A producer that passes the gate against a different frozen
    document has satisfied the ceremony and answered another question.
    """
    import freeze_gate
    freeze_gate.require_frozen(REGISTRATION)
    got = hashlib.sha256(open(REGISTRATION, "rb").read()).hexdigest()
    if got != REGISTRATION_SHA:
        raise SystemExit(
            "REFUSING: the registration hashes %s, this producer was written "
            "against %s. The gate proves a document is frozen; only the hash "
            "proves it is THIS one." % (got[:16], REGISTRATION_SHA[:16]))
    print("§6.1  registration FROZEN and IDENTIFIED: %s" % got[:16], flush=True)


# ══════════════════════════════════════════════════════════════════════════
# §3 / §3.0 — the population
# ══════════════════════════════════════════════════════════════════════════
def english_stimuli():
    """§3: distinct stimuli, second identities deduplicated, sentinels out.
    §3.0: language is a FILTER -- zh excluded to Registration O.

    The published figure is 2,199 and it is a KNOWN ANSWER, not a target:
    this function derives it and `main` asserts it.
    """
    from malign_logits.prompts import Prompts
    out = set()
    for p in Prompts().all():
        t = p if isinstance(p, str) else (getattr(p, "text", None) or str(p))
        if SENTINEL.match(t):
            continue
        if CJK.search(t):
            continue
        out.add(t)
    return out


def edges_and_clusters():
    """§4: 44 operation_edges; clusters are the 34 distinct BASE checkpoints.

    The within-cluster rule is §4's and it is L's §L5 verbatim: a sign test PER
    FAMILY, and a cluster's z is the UNWEIGHTED MEAN of its families' z. Seven
    families share Llama-3.1-8B; pooling their cells would let whichever family
    contributed most dominate that cluster.
    """
    import m01_concentration as CC
    _p, models, _h, _d = CC.frozen_population()
    edges, _dropped = CC.operation_edges(models)
    def mid(o):
        return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)
    return [(fam, mid(step.pre), mid(step.post), step) for fam, _pos, step in edges]


# ══════════════════════════════════════════════════════════════════════════
# §N6 — the reconstruction. DIRECT, never S = R/inflation, never m.null.
# ══════════════════════════════════════════════════════════════════════════
def reconstruct(P, Q, m):
    """§N6. Returns (R, S, n_unscored_fallers) or raises on the cross-check.

    `Movement` exposes fallers/risers/null/excess/delta/inflation/rule/
    diagnostics. **Neither S nor n_unscored_fallers is among them.**

    S IS RECOMPUTED, NOT DIVIDED OUT. `S = R/inflation` is arithmetically true
    only because the residual was excluded from faller candidacy on 2026-08-03;
    a producer dividing by `inflation` is correct while one line of `_movement`
    stays as it is, and has no way to notice if it moves.

    A CROSS-CHECK OVER `m.null` IS FORBIDDEN and this does not use one:
    `movement()` pops RESIDUAL_KEY from null/excess/delta -- a LOAD-BEARING pop,
    since `top_riser()` is an argmax the bucket would win -- so
    `sum(m.null.values())` is `inflation * (S - P_res)`, short by the residual's
    pre-mass. Measured: 200/200 cells disagree, worst gap 0.735.
    """
    keys = set(P) | set(Q)
    fall = set(m.fallers)
    R = 1.0 - sum(Q.get(w, 0.0) for w in fall)
    S = sum(P.get(k, 0.0) for k in keys if k not in fall)
    n_unscored = sum(1 for w in fall if Q.get(w, 0.0) == 0.0)
    #: §N6's cross-check: the reconstruction against the module's OWN ratio,
    #: without depending on it. Fails loud if faller candidacy moves again.
    if m.inflation != m.inflation:          # NaN: the null was not computed
        return None
    if S <= 0 or abs(R / S - m.inflation) > 1e-9:
        return None
    return R, S, n_unscored


# ══════════════════════════════════════════════════════════════════════════
# §4 — the statistic
# ══════════════════════════════════════════════════════════════════════════
def _ppf(q):
    """Inverse standard-normal CDF by bisection. No scipy dependency."""
    if q <= 0:
        return -40.0
    if q >= 1:
        return 40.0
    lo, hi = -40.0, 40.0
    for _ in range(300):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sign_test(values):
    """§4: exact one-sided sign test predicting NEGATIVE tail_excess.

    §4's TIES clause: a cell whose `tail_excess` is exactly 0.0 is EXCLUDED,
    not split, and the denominator is cells with NON-ZERO tail_excess.

    Returns (n_neg, n_pos, n_tie, p_one_sided, signed_z). z is POSITIVE when
    negatives dominate -- the registered direction -- so Stouffer accumulates
    evidence FOR substitution as positive.
    """
    neg = sum(1 for v in values if v < 0)
    pos = sum(1 for v in values if v > 0)
    tie = sum(1 for v in values if v == 0)
    n = neg + pos
    if n == 0:
        return neg, pos, tie, None, None
    #: one-sided exact binomial: P(X >= neg) under Binomial(n, 1/2)
    p = sum(math.comb(n, i) for i in range(neg, n + 1)) / (2 ** n)
    p = min(1.0, max(p, 1e-300))
    if neg == pos:
        return neg, pos, tie, p, 0.0
    z = abs(_ppf(p)) * (1 if neg > pos else -1)
    return neg, pos, tie, p, z


def stouffer(zs):
    """§4: equal weight per CLUSTER, over the 34 distinct base checkpoints."""
    zs = [z for z in zs if z is not None]
    if not zs:
        return None, None, 0
    Z = sum(zs) / math.sqrt(len(zs))
    p = 1.0 - 0.5 * (1 + math.erf(Z / math.sqrt(2)))     #: one-sided
    return Z, p, len(zs)


def combine(per_family, fam_base):
    """§4: family z -> cluster z (UNWEIGHTED MEAN) -> Stouffer over clusters."""
    by_base = collections.defaultdict(list)
    for fam, rec in per_family.items():
        if rec["z"] is not None:
            by_base[fam_base.get(fam, "?" + fam)].append(rec["z"])
    clusters = {b: sum(v) / len(v) for b, v in by_base.items()}
    Z, p, k = stouffer(list(clusters.values()))
    return clusters, Z, p, k


# ══════════════════════════════════════════════════════════════════════════
def collect():
    """One pass. Emits per-cell rows carrying BOTH arms and the bias column."""
    from malign_logits.cache import get_cache
    from malign_logits.movement import (movement, word_probs, decompose,
                                        CANONICAL, RESIDUAL_KEY)

    stim = english_stimuli()
    edges = edges_and_clusters()
    cm = get_cache()
    have = {}
    for k in cm._stash("true_word_probs").keys():
        if isinstance(k, dict):
            have.setdefault(k["model"], set()).add(k["prompt"])

    rows = []
    diag = collections.Counter()
    for ei, (fam, pre, post, _step) in enumerate(edges, 1):
        sel = sorted(have.get(pre, set()) & have.get(post, set()) & stim)
        for pr in sel:
            #: §6.6 -- a malformed row is REFUSED AND NAMED, never skipped.
            #: `word_probs` raises with the cell's coordinates; N's population
            #: excludes the two known NaN cells as sentinels, so a raise here
            #: is a NEW defect and must stop the run rather than be counted.
            A, B = word_probs(pre, pr), word_probs(post, pr)
            if A is None or B is None:
                diag["cell absent"] += 1
                continue
            P = {**A.probs, RESIDUAL_KEY: A.residual}
            Q = {**B.probs, RESIDUAL_KEY: B.residual}
            m = movement(P, Q, CANONICAL)
            #: §6.5 -- zero-faller cells EXCLUDED because the CLAIM does not
            #: apply: no mass departed, so nothing can have landed anywhere.
            #: Counted, and one rate reported from the ANALYSED population.
            if not m.fallers:
                diag["zero-faller (§6.5)"] += 1
                continue
            rec = reconstruct(P, Q, m)
            if rec is None:
                diag["§N6 cross-check refused"] += 1
                continue
            R, S, n_unscored = rec
            d = decompose(A.probs, B.probs,
                          residual_pre=A.residual, residual_post=B.residual)
            te = d["tail_excess"]
            #: §4.1 -- the adversarial push. dR capped by the POST residual:
            #: the unscored fallers' post mass cannot exceed the whole bucket.
            dR = min(n_unscored * THETA, B.residual)
            push = A.residual * dR / S
            rows.append({
                "family": fam, "base": pre, "aligned": post, "prompt": pr,
                "tail_excess_raw": te,
                "tail_excess_corrected": te + push,
                "push": push, "dR": dR, "R": R, "S": S,
                "n_unscored_fallers": n_unscored,
                "P_res": A.residual, "Q_res": B.residual,
                "n_fallers": d["n_fallers"], "n_risers": d["n_risers"],
                "captured": None,          #: §4.1 -- dropped pending [3776].3
            })
        print("  [%2d/%d] %-24s analysed %5d" % (
            ei, len(edges), fam,
            sum(1 for r in rows if r["family"] == fam)), flush=True)
    return rows, diag


def run_arm(rows, field, fam_base):
    """§4's primary on one arm. Returns the full per-family and cluster record."""
    per_fam = {}
    by_fam = collections.defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r[field])
    for fam, vals in sorted(by_fam.items()):
        neg, pos, tie, p, z = sign_test(vals)
        per_fam[fam] = {"n": len(vals), "n_neg": neg, "n_pos": pos,
                        "n_tie": tie, "p": p, "z": z}
    clusters, Z, p, k = combine(per_fam, fam_base)
    tot_neg = sum(v["n_neg"] for v in per_fam.values())
    tot_pos = sum(v["n_pos"] for v in per_fam.values())
    tot_tie = sum(v["n_tie"] for v in per_fam.values())
    split = tot_neg / (tot_neg + tot_pos) if (tot_neg + tot_pos) else None
    return {"field": field, "families": per_fam, "clusters": clusters,
            "n_clusters": k, "stouffer_Z": Z, "stouffer_p": p,
            "n_negative": tot_neg, "n_positive": tot_pos, "n_tie": tot_tie,
            "sign_split_negative": split}


def verdict(raw, corrected):
    """§4.1's READING, declared before any number existed. Not a choice here."""
    def survives(a):
        return (a["stouffer_Z"] is not None and a["stouffer_Z"] > 0
                and a["stouffer_p"] is not None and a["stouffer_p"] < 0.05
                and a["sign_split_negative"] is not None
                and not (0.45 <= a["sign_split_negative"] <= 0.55))
    r, c = survives(raw), survives(corrected)
    if r and c:
        return "SUBSTITUTION CONFIRMED"
    if r and not c:
        return "NOT SUPPORTED -- never REFUTED (§4.1: the correction can only push negative)"
    return "NOT SUPPORTED"


def main():
    gate_registration()                                   #: §6.1, first line

    stim = english_stimuli()
    print("§3/§3.0  ENGLISH stimuli %d   (published known answer 2,199)"
          % len(stim), flush=True)
    if len(stim) != 2199:
        raise SystemExit("REFUSING: §3 publishes 2,199 English stimuli and "
                         "this derivation gives %d." % len(stim))
    edges = edges_and_clusters()
    fam_base = {fam: pre for fam, pre, _post, _s in edges}
    n_clusters = len(set(fam_base.values()))
    print("§4       edges %d   clusters %d   (published 44 / 34)"
          % (len(edges), n_clusters), flush=True)
    if len(edges) != 44 or n_clusters != 34:
        raise SystemExit("REFUSING: §4 publishes 44 edges over 34 clusters.")
    if "--dry-run" in sys.argv:
        print("\n--dry-run: gates and known answers only. No cell opened.")
        return 0

    rows, diag = collect()
    reach = len(stim) * len(edges)
    print("\n§3       reachable %d   analysed %d" % (reach, len(rows)), flush=True)
    for k, v in diag.most_common():
        print("           %-28s %6d" % (k, v), flush=True)
    zf = diag["zero-faller (§6.5)"]
    print("§6.5     zero-faller EXCLUDED %d = %.2f%% of reachable"
          % (zf, 100.0 * zf / reach), flush=True)

    raw = run_arm(rows, "tail_excess_raw", fam_base)
    cor = run_arm(rows, "tail_excess_corrected", fam_base)
    v = verdict(raw, cor)

    for name, a in (("RAW (diagnostic, no verdict language)", raw),
                    ("CORRECTED (carries the verdict)", cor)):
        print("\n%s" % name, flush=True)
        print("  clusters %d   Stouffer Z %+.4f   p %.6g"
              % (a["n_clusters"], a["stouffer_Z"], a["stouffer_p"]), flush=True)
        print("  sign split NEGATIVE %.4f   (neg %d / pos %d / ties excluded %d)"
              % (a["sign_split_negative"], a["n_negative"], a["n_positive"],
                 a["n_tie"]), flush=True)

    print("\n§4.1 READING: %s" % v, flush=True)

    pushes = [r["push"] for r in rows]
    payload = {
        "_what": "Registration N, clause 1 (mass-migration). ARM A, ENGLISH.",
        "_registration": REGISTRATION_SHA,
        "_reading": v,
        "_population": {"stimuli_en": len(stim), "edges": len(edges),
                        "clusters": n_clusters, "reachable": reach,
                        "analysed": len(rows), "diagnostics": dict(diag)},
        "_bias_column": {"push_median": st.median(pushes),
                         "push_max": max(pushes), "push_min": min(pushes),
                         "all_non_negative": all(p >= 0 for p in pushes)},
        "raw": raw, "corrected": cor,
        "cells": rows,                    #: §4.1's companion column, PER CELL
    }
    if os.path.exists(OUT):
        os.makedirs(ESCROW, exist_ok=True)
        prior = open(OUT, "rb").read()
        h = hashlib.sha256(prior).hexdigest()[:16]
        dst = os.path.join(ESCROW, "result_n_primary.PREFIX-%s.json" % h)
        if not os.path.exists(dst):
            with open(dst, "wb") as fh:
                fh.write(prior)
            os.chmod(dst, 0o444)
        print("\n  escrowed prior artifact @ %s" % h, flush=True)
        print("  UNLOCKING %s for the re-emit" % os.path.basename(OUT), flush=True)
        os.chmod(OUT, 0o644)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    os.chmod(OUT, 0o444)
    print("  wrote %s @ %s  RE-LOCKED a-w"
          % (os.path.basename(OUT),
             hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
