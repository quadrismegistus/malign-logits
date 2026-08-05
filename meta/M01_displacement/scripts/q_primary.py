"""q_primary.py — REGISTRATION Q's PRODUCER. Built to the parse checklist
published at [4492] and amended at [4494], BEFORE this file existed.

**THE CHECKLIST IS A SPEC, NOT A REVIEW.** It was published before a line of
this file was written, so a failing item here is a defect in this producer
and never a discovery about the checklist. Item numbers below are that
document's.

Registration Q is FROZEN at `sha256:c28ec53f2fe0276a` (commit 5c80c80a),
countersigned [4489]. This file reads that document's rules and no others.

WHAT Q ASKS, and the grid it fills:

                    (a) vs NEUTRAL TWIN      (b) vs CORPUS AT LARGE
    substitution    H1  TESTED               H2  TESTED
    magnitude       H5  TESTED               H4  TESTED
    norms           H6  ESTIMATED            H3  ESTIMATED

**THE FOUR TESTED ARMS RUN AT TWO-SIDED `p < 0.0125`** — alpha 0.05 split
four ways (4.1). **H3 AND H6 CONSUME NO ALPHA, APPEAR IN NO BRANCH, AND NO
VERDICT LANGUAGE ATTACHES TO THEM** (4.2); they emit a point estimate, an
interval and their registered MDE, and **this file deliberately emits NO
p-value for them** — a p invites a verdict, and the registration forbids one.

GATES, IN THE ORDER THEY FIRE (1.1-1.4, 2.1-2.4):

    1.1(a) `movement.py` BLOB-to-BLOB against commit e7864dab. **Never
           commit-to-commit** — §Q1's reason: a commit match passes
           identically whether the dirty edit sits in the pinned file or
           somewhere harmless.
    1.1(b) `m01_norms.py` + four norm sources, via `load_norms(verify=True)`.
    1.1(c) `sha256(data/prompt_categorisation.json)[:16]` RECOMPUTED and
           asserted against `population_d_684.json`'s DECLARED
           `categorisation_sha256_16`. **The pin existed; the assertion did
           not** ([4494]) — a declared hash nobody recomputes has never been
           checked.
    **NO TAG ASSERTION.** [4492].1.1 required one; it was STRUCK at [4494]
    because Q's producer reaches no provider library — all five modules it
    imports carry zero references — and §Q1 declares the scope itself:
    *"administers nothing to any model, calls no provider, renders no
    prompt."* **A gate whose premise is false passes for the wrong reason.**
    2.1-2.3 THREE known answers, fired **before any hypothesis quantity is
           read**, with TOLERANCE BY KIND: counts and A-yield rows are
           integers at EXACT equality; N's two floats at |obs-pub| <= 5e-5.
           **Any failure stops the run.**
    2.4    **G AND D2 ARE NOT RE-DERIVED.** Withdrawn at §Q6.1 on
           correctness; a producer re-deriving `d = 0.748` must guess a
           denominator and misses by 222x the tolerance while being
           entirely right about G.

WHAT THIS FILE CANNOT DO. It computes Q's registered quantities and applies
Q's registered reading rules. **It cannot tell you they were the right
quantities.** Four of the six arms have no measured effect size on their
scale, so their nulls are BOUNDS that may be reported and may not be called
small (5.2) — that limit is Q's, not this producer's, and no output here
softens it.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys

# ---------------------------------------------------------------- 1.2 defs
HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REG = os.path.join(CAMPAIGN, "registrations", "registration_q_bridge.md")
POP = os.path.join(CAMPAIGN, "results", "population_d_684.json")
N_ART = os.path.join(CAMPAIGN, "results", "result_n_primary.json")
CATALOGUE = os.path.join(ROOT, "data", "prompt_categorisation.json")
OUT = os.path.join(CAMPAIGN, "results", "result_q_primary.json")

REG_FROZEN_SHA16 = "c28ec53f2fe0276a"
MOVEMENT_REL = "malign_logits/movement.py"
MOVEMENT_PIN_COMMIT = "e7864dab"

SENTINEL = re.compile(r"^<<<.*>>>$")
CJK = re.compile(r"[一-鿿]")

#: §Q1.1's transgressive domains for the general-corpus partition.
TRANSGRESSIVE = {"violence", "sexual", "profanity", "substance", "death",
                 "taboo", "animal", "betrayal", "power", "property"}

#: §Q3 L327-332. The floor's VALUE is arbitrary; its EXISTENCE is not.
FLOOR = 10

#: 2.1 KNOWN ANSWERS -- Q's own population, all public.
KA_STIMULI, KA_EDGES, KA_CLUSTERS, KA_REACHABLE = 2199, 44, 34, 96756
#: N's two pooled floats, EACH WITH THE TOLERANCE ITS OWN PUBLISHED
#: PRECISION SUPPORTS -- half a unit in its last published place.
#:
#: **CHECKLIST 2.2 GIVES ONE TOLERANCE (5e-5) FOR BOTH AND IT IS RIGHT FOR
#: ONLY ONE OF THEM.** `-0.0738` is published to 4 dp, so 5e-5 is exactly
#: its half-place. `91.0%` is published to ONE decimal in percent, i.e.
#: 0.910 as a fraction, whose half-place is 5e-4 -- TEN TIMES LOOSER. At
#: 5e-5 the gate rejects 0.910178, a value that rounds to the published
#: figure exactly. **This is item 6.4 -- compare at the precision the
#: document publishes -- arriving in item 2.2.** Reported at [4495].
KA_TE_MEAN, KA_TE_MEAN_TOL = -0.0738, 5e-5       #: 4 dp -> half-place 5e-5
KA_TE_NEG_FRAC, KA_TE_NEG_TOL = 0.910, 5e-4      #: 3 dp -> half-place 5e-4

#: §Q3/§Q1.3 structural counts; integers, EXACT (2.2).
KA_STEMS, KA_PAIR_TEXTS, KA_DISTINCT_EDGES = 684, 1368, 43
KA_BOTHSIDES_ANALYSED, KA_ONESIDED_ANALYSED = 24606, 1253
KA_BOTHSIDES_A, KA_STEMS_H6 = 15152, 626

#: §Q4's registered MDEs, for the bound sentences (5.2/5.3).
MDE = {"H1": 0.00189, "H2": 0.00436, "H5": 0.00284, "H4": 0.00571,
       "H6": 0.01773, "H3": 0.03693}
ALPHA_TESTED = 0.0125                            #: 4.1, two-sided
L_BORROWED_SD = 0.0789                           #: 5.3, H4 prints this too

#: The sign-flip null. 2**684 is not enumerable, so this is Monte Carlo with
#: a DECLARED seed, and the p is never quoted finer than its own MC
#: resolution (6.1: no figure beyond what its inputs reproduce).
FLIP_DRAWS = 200000
FLIP_SEED = 20260805


def refuse(msg):
    raise SystemExit("REFUSING: %s" % msg)


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def wmean(vals, wts):
    s = sum(wts)
    return sum(v * w for v, w in zip(vals, wts)) / s if s > 0 else None


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def sign_flip_p(d, draws=FLIP_DRAWS, seed=FLIP_SEED):
    """TWO-SIDED sign-flip null over the ANALYSIS UNIT (4.3).

    **THE UNIT FLIPS AS A WHOLE.** Each `d[i]` is already a unit-level
    difference — a stem's mean over its ~36 edges (H1/H5), or a cluster's
    difference of means (H2/H4) — so flipping `d[i]` flips everything inside
    that unit together and **carries the within-unit dependence** the
    registration measures at ratio 1.040 (§Q4 L651).

    Returns (p, mc_se). The MC standard error is returned so the caller
    cannot quote p finer than the draw count supports.
    """
    obs = abs(mean(d))
    rng = random.Random(seed)
    k = len(d)
    hits = 0
    for _ in range(draws):
        s = 0.0
        for x in d:
            s += x if rng.getrandbits(1) else -x
        if abs(s / k) >= obs:
            hits += 1
    p = (hits + 1.0) / (draws + 1.0)             #: add-one, never reports 0
    return p, math.sqrt(max(p * (1 - p), 1e-12) / draws)


def t_interval(d, conf=0.95):
    """Point estimate and interval for an ESTIMATED arm (4.2).

    Normal quantile, not Student's t: at k = 626 and k = 32 the difference
    is immaterial and §Q4's own MDEs are built on the normal. **No p-value
    is computed here and none is returned** — H3 and H6 consume no alpha,
    and a p in the output invites the verdict the registration forbids.
    """
    from statistics import NormalDist
    m, s, k = mean(d), sd(d), len(d)
    half = NormalDist().inv_cdf(1 - (1 - conf) / 2) * s / math.sqrt(k)
    return m, (m - half, m + half), s, k


# ------------------------------------------------------------- 1.1 gates
def gate_registration_frozen():
    """The producer runs against the FROZEN bytes or not at all."""
    got = sha16(REG)
    if got != REG_FROZEN_SHA16:
        refuse("registration is %s, not the frozen %s — this producer reads "
               "one document" % (got, REG_FROZEN_SHA16))
    return got


def gate_movement_blob():
    """1.1(a) BLOB to BLOB. Never commit to commit -- §Q1's own reason."""
    wt = subprocess.run(["git", "hash-object", MOVEMENT_REL],
                        capture_output=True, text=True, cwd=ROOT).stdout.strip()
    pin = subprocess.run(["git", "rev-parse",
                          "%s:%s" % (MOVEMENT_PIN_COMMIT, MOVEMENT_REL)],
                         capture_output=True, text=True, cwd=ROOT).stdout.strip()
    if not wt or not pin:
        refuse("could not resolve movement.py blobs (wt=%r pin=%r)" % (wt, pin))
    if wt != pin:
        refuse("movement.py blob %s != pinned blob %s at %s — the faller rule "
               "may differ in 11%% of cells and every arm rests on it"
               % (wt[:16], pin[:16], MOVEMENT_PIN_COMMIT))
    return wt


def gate_catalogue_hash():
    """1.1(c) RECOMPUTE the catalogue hash and ASSERT it against the
    population artifact's DECLARED value. The pin existed; the assertion
    did not ([4494]). This is the only gate standing between a changed
    STIMULUS TEXT and every arm, because the counts would not move."""
    declared = json.load(open(POP)).get("categorisation_sha256_16")
    if not declared:
        refuse("population artifact declares no categorisation_sha256_16")
    actual = sha16(CATALOGUE)
    if declared != actual:
        refuse("catalogue hash %s != declared %s — a stimulus text may have "
               "moved while every count held" % (actual, declared))
    return actual


def gate_population_idset():
    """The 684 by the hash the file computes over its ID SET -- NOT a
    whole-file hash, which moves when metadata does (§Q1.1)."""
    pop = json.load(open(POP))
    ids = pop["ids"]
    rec = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]
    if rec != pop.get("id_set_sha256_16"):
        refuse("id-set hash %s != declared %s" % (rec, pop.get("id_set_sha256_16")))
    if len(ids) != KA_STEMS:
        refuse("population carries %d stems, not %d" % (len(ids), KA_STEMS))
    return ids, rec


# --------------------------------------------------------- 2.1 known answers
def known_answers(art):
    """2.1-2.3. Fired BEFORE any hypothesis quantity is read. TOLERANCE BY
    KIND (2.2): integers EXACT, N's two floats at 5e-5. ANY failure stops
    the run (2.3) -- these raise, they do not warn."""
    out = {}
    pop = art["_population"]
    for name, got, want in (("stimuli_en", pop["stimuli_en"], KA_STIMULI),
                            ("edges", pop["edges"], KA_EDGES),
                            ("clusters", pop["clusters"], KA_CLUSTERS),
                            ("reachable", pop["reachable"], KA_REACHABLE)):
        if got != want:
            refuse("known answer (population/%s): %r != %r EXACT" % (name, got, want))
        out[name] = got

    #: **THE PUBLISHED KNOWN ANSWER IS THE **RAW** ARM, AND §Q6 L879 DOES
    #: NOT SAY SO.** Q's TESTED arm is `tail_excess_corrected` (§Q2), so the
    #: natural reading of "N's pooled tail_excess mean -0.0738, 91.0%
    #: negative" is the corrected arm -- and on the corrected arm this gate
    #: FAILS, which is how it was found. The arm is IDENTIFIED, not guessed:
    #:
    #:     raw        mean -0.073796 -> -0.0738  OK   neg 0.9102 -> 91.0% OK
    #:     corrected  mean -0.072500             no   neg 0.9088 -> 90.9% no
    #:
    #: **TWO INDEPENDENT QUANTITIES BOTH MATCH RAW AND BOTH MISS CORRECTED**,
    #: so this is an identification with two confirmations rather than the
    #: unrunnable guess §Q6.1 withdrew G for. Reported at [4495]; both arms
    #: are recorded below so the record does not depend on this reading.
    te = [c["tail_excess_raw"] for c in art["cells"]]
    m = mean(te)
    negf = sum(1 for x in te if x < 0) / len(te)
    te_corr = [c["tail_excess_corrected"] for c in art["cells"]]
    if abs(m - KA_TE_MEAN) > KA_TE_MEAN_TOL:
        refuse("known answer (N pooled mean): %.6f vs %.4f, |d|=%.2e > %.0e"
               % (m, KA_TE_MEAN, abs(m - KA_TE_MEAN), KA_TE_MEAN_TOL))
    if abs(negf - KA_TE_NEG_FRAC) > KA_TE_NEG_TOL:
        refuse("known answer (N %% negative): %.6f vs %.3f, |d|=%.2e > %.0e"
               % (negf, KA_TE_NEG_FRAC, abs(negf - KA_TE_NEG_FRAC), KA_TE_NEG_TOL))
    out["N_pooled_tail_excess_mean_RAW"] = m
    out["N_frac_negative_RAW"] = negf
    #: recorded, NOT asserted -- no published figure exists for this arm.
    out["N_pooled_tail_excess_mean_CORRECTED"] = mean(te_corr)
    out["N_frac_negative_CORRECTED"] = sum(1 for x in te_corr if x < 0) / len(te_corr)
    out["_arm_note"] = ("published known answer reproduces on tail_excess_RAW; "
                        "Q's tested arm is tail_excess_CORRECTED (§Q2); "
                        "§Q6 L879 names no arm -- see [4495]")
    return out


def build_partition(pop_ids):
    """§Q1.1's precedence, verbatim: a pair member takes its PAIR ROLE and
    NEVER its domain; first-in-catalogue wins on multi-domain."""
    rows = json.load(open(CATALOGUE))["prompts"]
    pair_ids = set(pop_ids)
    pair_texts = set()
    for r in rows:
        if (r.get("pair_role") and r.get("contrast_type") == "transgressive_swap"
                and str(r.get("source", "")).startswith("M01_PAIRS")
                and r.get("pair_id") in pair_ids):
            pair_texts.add(r["prompt"])
    part = {}
    for r in rows:
        t = r["prompt"]
        if t in pair_texts or t in part:
            continue
        d = r.get("domain")
        if d in TRANSGRESSIVE:
            part[t] = "T"
        elif d == "neutral":
            part[t] = "N"
    return pair_texts, part


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="fire every gate and stop before the arms")
    ap.add_argument("--draws", type=int, default=FLIP_DRAWS)
    args = ap.parse_args(argv)

    # === 1.4: EVERY CALL-FREE GATE ABOVE ANY EARLY RETURN =================
    # P fired four times because --dry-run returned before a gate, leaving
    # that gate untested forever. Nothing below this block is skippable.
    print("=== GATES")
    reg_sha = gate_registration_frozen()
    print("  registration FROZEN        %s" % reg_sha)
    blob = gate_movement_blob()
    print("  1.1(a) movement.py blob    %s  == @%s" % (blob[:16], MOVEMENT_PIN_COMMIT))
    cat = gate_catalogue_hash()
    print("  1.1(c) catalogue sha256    %s  == declared" % cat)
    ids, idset = gate_population_idset()
    print("  population id-set          %s  n=%d" % (idset, len(ids)))

    from malign_logits.movement import CANONICAL
    import m01_concentration as CC
    import m01_norms as N
    import m01_registration_b as B
    from malign_logits.prompts import Prompts

    norms, _f, _r = N.load_norms(verify=True)          # 1.1(b)
    print("  1.1(b) norms verify=True   PASS (module-declared hashes)")
    tabs = {d: norms[("en", d, "primary")] for d in ("arousal", "valence")}

    art = json.load(open(N_ART))
    ka = known_answers(art)                            # 2.1-2.3
    print("=== 2.1 KNOWN ANSWERS  (integers EXACT; each float at ITS OWN half-place)")
    print("  population  %d stimuli, %d edges, %d clusters, %d cells  EXACT"
          % (ka["stimuli_en"], ka["edges"], ka["clusters"], ka["reachable"]))
    print("  N pooled    RAW mean %.6f (pub %.4f, tol 5e-5)  neg %.4f (pub %.3f)"
          % (ka["N_pooled_tail_excess_mean_RAW"], KA_TE_MEAN,
             ka["N_frac_negative_RAW"], KA_TE_NEG_FRAC))
    print("              tolerances: mean 5e-5 (4 dp), neg 5e-4 (3 dp) — "
          "EACH ITS OWN HALF-PLACE")
    print("              **the published figures are the RAW arm; Q TESTS the")
    print("              CORRECTED arm (mean %.6f, neg %.4f), and §Q6 names"
          % (ka["N_pooled_tail_excess_mean_CORRECTED"], ka["N_frac_negative_CORRECTED"]))
    print("              no arm at all — reported [4495], recorded not resolved**")
    print("  **G and D2 are NOT re-derived (2.4, §Q6.1 withdrawal).**")
    print("  **AND THE CORRECTION STEP CARRIES NO KNOWN ANSWER** — the two")
    print("  published floats guard the RAW arm; `push` is guarded by nothing")
    print("  and no published figure exists to guard it with ([4497].1).")

    if args.dry_run:
        print("\n--dry-run: every gate above FIRED. Stopping before the arms.")
        return 0

    # === 3. POPULATION AND KEYS ==========================================
    byid = {str(p.id): p for p in Prompts().all()}
    pair = {}
    for s in ids:
        m_, u_ = byid.get(s + "_M"), byid.get(s + "_U")
        if m_ is None or u_ is None:
            continue
        t_m, t_u = m_.text, u_.text
        if SENTINEL.match(t_m) or CJK.search(t_m) or SENTINEL.match(t_u) or CJK.search(t_u):
            continue
        pair[s] = (t_m, t_u)
    if 2 * len(pair) != KA_PAIR_TEXTS:
        refuse("pair map covers %d texts, not %d" % (2 * len(pair), KA_PAIR_TEXTS))

    _p, mods, _h, _d = CC.frozen_population()
    edges_raw, _drop = CC.operation_edges(mods)

    def mid(o):
        return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)

    #: 3.1/3.2 an edge is (base, aligned); FAMILY LABELS COLLAPSE.
    steps = {}
    for _fam, _pos, step in edges_raw:
        steps.setdefault((mid(step.pre), mid(step.post)), step)
    if len(steps) != KA_DISTINCT_EDGES:
        refuse("%d distinct transitions, not %d" % (len(steps), KA_DISTINCT_EDGES))
    print("=== 3. KEYS  %d distinct (base, aligned); family labels collapse"
          % len(steps))

    #: H1/H2's arm is READ from N's artifact (§Q2's provenance split): the
    #: correction `push` is defined by n_primary.py, and re-implementing a
    #: sibling registration's definition is the move §Q6 rejected for G.
    pair_texts, part = build_partition(ids)
    want_texts = pair_texts | set(part)
    te_art = {}
    for c in art["cells"]:
        if c["prompt"] not in want_texts:
            continue
        k = (c["prompt"], c["base"], c["aligned"])
        v = c["tail_excess_corrected"]
        if k in te_art:
            #: 3.2 a duplicate key carrying an UNEQUAL value is NOT one
            #: measurement -- refuse, per q_h1_sd_pass.
            if te_art[k] != v:
                refuse("collapsed edge carries UNEQUAL values at one key — "
                       "not one measurement")
            continue
        te_art[k] = v

    def measures(step, text):
        """-> (analysed?, departed, A_absvalence or None). MACHINERY
        quantities. `tail_excess` is NOT computed here -- it is read."""
        c = step.cell(text)
        if not c.is_present:
            return False, None, None
        try:
            dec = c.decompose(None)
        except Exception:
            return False, None, None
        if not dec:
            return False, None, None
        try:
            roles = N.cell_roles(c, CANONICAL)
        except Exception:
            roles = None
        if roles is None or not any(r == "faller" for _w, _wt, r in roles):
            return False, None, None
        ws_f, zs_f, ws_r, zs_r = [], [], [], []
        for w, wt, role in roles:
            key = N.norm_key(w, "en", fold=False)
            if N.is_function_word(key, "en"):
                continue
            zv = {}
            for dim in ("arousal", "valence"):
                val, _src = N.lookup(tabs[dim], key.casefold(), "en")
                zv[dim] = val
            if any(x is None for x in zv.values()):
                continue
            if role == "faller":
                ws_f.append(wt); zs_f.append(abs(zv["valence"]))
            else:
                ws_r.append(wt); zs_r.append(abs(zv["valence"]))
        a_val = None
        if len(ws_f) >= B.QUALIFYING_MIN and len(ws_r) >= B.QUALIFYING_MIN:
            mf, mr = wmean(zs_f, ws_f), wmean(zs_r, ws_r)
            if mf is not None and mr is not None:
                a_val = mf - mr
        return True, float(dec["departed"]), a_val

    # === 4. THE ARMS =====================================================
    per_stem = {m: collections.defaultdict(list) for m in ("H1", "H5", "H6")}
    clus = collections.defaultdict(lambda: {"H2": {"T": [], "N": []},
                                            "H4": {"T": [], "N": []},
                                            "H3": {"T": [], "N": []}})
    n_both_an = n_one_an = n_both_a = n_missing = 0

    for ei, ((b_, a_), step) in enumerate(sorted(steps.items()), 1):
        # -- pair arms: H1, H5, H6 (3.3 strict both-sides)
        for s, (t_m, t_u) in pair.items():
            an_m, dp_m, av_m = measures(step, t_m)
            an_u, dp_u, av_u = measures(step, t_u)
            if an_m and an_u:
                n_both_an += 1
                per_stem["H5"][s].append(dp_m - dp_u)
                k_m, k_u = (t_m, b_, a_), (t_u, b_, a_)
                if k_m in te_art and k_u in te_art:
                    per_stem["H1"][s].append(te_art[k_m] - te_art[k_u])
                else:
                    n_missing += 1
                if av_m is not None and av_u is not None:
                    n_both_a += 1
                    per_stem["H6"][s].append(av_m - av_u)
            elif an_m or an_u:
                n_one_an += 1
        # -- cluster arms: H2, H4, H3 on the NON-PAIR partition
        for t, side in part.items():
            an, dp, av = measures(step, t)
            if not an:
                continue
            key = (t, b_, a_)
            if key in te_art:
                clus[b_]["H2"][side].append(te_art[key])
            clus[b_]["H4"][side].append(dp)
            if av is not None:
                clus[b_]["H3"][side].append(av)
        print("  [%2d/%d] pair keys %6d   clusters seen %2d"
              % (ei, len(steps), n_both_an, len(clus)), flush=True)

    if n_missing:
        refuse("machinery/artifact key sets disagree on %d keys" % n_missing)
    for got, want, what in ((n_both_an, KA_BOTHSIDES_ANALYSED, "both-sides analysed"),
                            (n_one_an, KA_ONESIDED_ANALYSED, "one-sided analysed"),
                            (n_both_a, KA_BOTHSIDES_A, "both-sides A")):
        if got != want:
            refuse("%s keys = %d, not %d (EXACT)" % (what, got, want))
    print("=== structural counts reproduce §Q1.3 EXACTLY")

    d = {}
    for arm, want_k in (("H1", KA_STEMS), ("H5", KA_STEMS), ("H6", KA_STEMS_H6)):
        d[arm] = [mean(v) for v in per_stem[arm].values() if v]
        if len(d[arm]) != want_k:
            refuse("%s stems = %d, not %d" % (arm, len(d[arm]), want_k))

    #: 3.4 the floor. `pythia-2.8b` is REPORTED, never silently dropped.
    below = {}
    for arm in ("H2", "H4", "H3"):
        diffs, excluded = [], []
        for b_, slots in sorted(clus.items()):
            T, Nn = slots[arm]["T"], slots[arm]["N"]
            if len(T) >= FLOOR and len(Nn) >= FLOOR:
                diffs.append(mean(T) - mean(Nn))
            else:
                excluded.append((b_, len(T), len(Nn)))
        d[arm] = diffs
        below[arm] = excluded
    print("=== 3.4 FLOOR >= %d analysed cells BOTH sides" % FLOOR)
    for arm in ("H2", "H4", "H3"):
        print("  %s  k = %d      below floor, REPORTED not dropped: %s"
              % (arm, len(d[arm]),
                 ", ".join("%s (T=%d,N=%d)" % e for e in below[arm]) or "none"))

    # === 5. READING ======================================================
    res = {}
    print("\n=== 4.1 TESTED ARMS — two-sided sign-flip, p < %.4f" % ALPHA_TESTED)
    for arm in ("H1", "H2", "H5", "H4"):
        obs = mean(d[arm])
        p, mc = sign_flip_p(d[arm], draws=args.draws)
        realized = sd(d[arm])
        sig = p < ALPHA_TESTED
        #: 5.1 THREE branches, not two.
        if sig:
            branch = ("significant, AS PREDICTED (marked/transgressive more "
                      "negative)" if obs < 0 else
                      "**significant IN THE WRONG DIRECTION — REPORTED AS A "
                      "REVERSAL, never as an asymmetry**")
        else:
            branch = "NULL — quoted as a BOUND, never as an absence"
        res[arm] = {"mean": obs, "k": len(d[arm]), "p": p, "p_mc_se": mc,
                    "realized_sd": realized, "registered_mde": MDE[arm],
                    "significant": sig, "branch": branch,
                    #: 5.3 the bound at the REALIZED dispersion, always, with
                    #: the pre-registered figure beside it.
                    "bound_realized": 3.3393 * realized / math.sqrt(len(d[arm])),
                    "bound_registered": MDE[arm]}
        if arm == "H4":
            res[arm]["bound_borrowed_sd"] = L_BORROWED_SD   # 5.3
        print("  %s  k=%-4d mean %+.6f   p = %.5f (MC se %.5f)   %s"
              % (arm, len(d[arm]), obs, p, mc, "SIG" if sig else "null"))
        print("      %s" % branch)
        print("      bound: realized %.5f   registered %.5f%s"
              % (res[arm]["bound_realized"], MDE[arm],
                 "   L's borrowed SD %.4f" % L_BORROWED_SD if arm == "H4" else ""))
        if not sig:
            print("      **THIS BOUND MAY NOT BE CALLED SMALL** — no measured "
                  "effect size exists on this scale (5.2).")

    print("\n=== 4.2 ESTIMATED ARMS — no alpha, no test, NO VERDICT LANGUAGE")
    for arm in ("H6", "H3"):
        m, ci, s, k = t_interval(d[arm])
        res[arm] = {"mean": m, "ci95": list(ci), "realized_sd": s, "k": k,
                    "registered_mde": MDE[arm], "p": None,
                    "branch": "ESTIMATED — point estimate and interval only; "
                              "no alpha consumed, no verdict language, and "
                              "the word 'confirmed' may not attach"}
        print("  %s  k=%-4d mean %+.6f   95%% CI [%+.6f, %+.6f]   MDE %.5f"
              % (arm, k, m, ci[0], ci[1], MDE[arm]))
        print("      no p-value is emitted for this arm, by design")

    payload = {
        "_what": "Registration Q — primary result.",
        "_registration": {"file": os.path.basename(REG), "sha256_16": reg_sha,
                          "frozen": "2026-08-05 UTC, RH's word at [4487]"},
        "_pins": {"movement.py_blob": blob, "movement_pin_commit": MOVEMENT_PIN_COMMIT,
                  "catalogue_sha256_16": cat, "population_id_set_sha256_16": idset},
        "_known_answers": ka,
        "_null": {"kind": "two-sided sign-flip over the analysis unit",
                  "draws": args.draws, "seed": FLIP_SEED,
                  "note": "the unit flips as a whole, carrying within-unit "
                          "dependence; p is never quoted finer than its MC se"},
        "_counts": {"bothsides_analysed": n_both_an, "onesided_analysed": n_one_an,
                    "bothsides_A": n_both_a},
        "_floor": {"value": FLOOR,
                   "below_floor_reported_not_dropped":
                       {a: [{"cluster": b, "T": t, "N": n} for b, t, n in below[a]]
                        for a in below}},
        "arms": res,
        "_limits": [
            "Four of six arms have no measured effect size on their scale; "
            "their nulls are BOUNDS that may be reported and may NOT be "
            "called small.",
            "H2's and H4's transgressive arm is 13.0% of the transgressive "
            "corpus and is a RESIDUE, not a sample.",
            "H3 and H6 are ESTIMATED: no alpha, no test, no verdict language.",
        ],
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print("\nwrote %s" % OUT)
    return 0


# 1.3: __main__ AT THE END OF THE FILE, with every definition above it.
if __name__ == "__main__":
    sys.exit(main())
