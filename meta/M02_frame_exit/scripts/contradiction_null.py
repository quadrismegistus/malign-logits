"""Where does NEITHER POLE fall on F11's contradiction ratio? Compute the null.

    uv run python contradiction_null.py                 # twp, 46 lineages
    uv run python contradiction_null.py --logits        # full-vocab cross-check

F11's instrument (`findings/F11_contradiction.md`, `scripts/contradiction_compare.py`):

    ratio = JS(AB, mean(A,B)) / min( JS(AB,A), JS(AB,B) )

read with the rule "ratio < 1 = superposition (inclusive disjunction), ratio > 1
= resolution (exclusive disjunction)". **That rule assumes 1.0 is the boundary
between the two, and nobody ever computed what lands there.** A metric whose
scale has never been calibrated is read against the only number in it, and 1.0
is in the formula rather than in the data.

THREE THINGS PRODUCE AN INTERMEDIATE RATIO AND THEY ARE NOT THE SAME CLAIM:

    superposition   AB holds both poles       -- inclusive disjunction, D&G
    resolution      AB collapses to one pole  -- exclusive disjunction
    NEUTRALIZATION  AB holds NEITHER pole     -- frame exit

The first and third are opposite readings of the finding and F11's rule cannot
separate them, because it has a reference for superposition (the blend) and a
reference for resolution (each pole) and NO REFERENCE FOR NEITHER.

THE NULL, AND WHY IT IS THIS ONE. A neutralized AB is a real next-token
distribution that has nothing to do with these poles. So: for group g, take
ANOTHER live group's `both` distribution from the SAME MODEL and score it
against g's poles. It is a genuine model output with realistic peakedness and
the shared function-word/syntax structure a synthetic distribution cannot fake.
A Dirichlet draw was tried first and is not usable -- it has no shared tail, so
every distance inflates together.

    ITS SUSPECTED BIAS, TESTED RATHER THAN CARRIED. Groups differ in their
    trailing frame ("and she wanted to" vs "and chose to"), so the null prompt
    is distant from A and B in form as well as in content, and a uniformly
    distant third distribution drives the ratio toward 1 by construction. If
    the null were inflated that way, every signal here would be OVERSTATED.
    The battery DOES contain the stricter null -- groups sharing a word-level
    trailing frame with a disjoint pole contrast, 10 of 22 en and 4 of 21 zh --
    and on those the same-frame null is HIGHER, not lower (en +0.0125, zh
    +0.0062). The bias runs the other way and the signal is if anything
    understated.

SUBSTRATE. `twp_words` at theta=0.001 (word probs, ~200 words/cell) is the
tractable table; F11 used the FULL softmax. These are different estimators and
a null computed on one does not calibrate a number computed on the other --
so this file computes BOTH sides on the SAME substrate and never compares its
null to F11's published values. `--logits` re-runs on `logit_probs` (1e-6,
~3.6k tokens, 98-99% of mass) to show the truncation does not carry the result.

**`--logits` READS A SUPERSEDED TABLE. Flagged 2026-08-14 by registrar at
[6150]; recorded here because the warning was on the docket and not in the
file that needs it.** RH: *"logit_probs was abandoned, we decided to keep
logits in the f16 files."* The table was last written 2026-08-10 and holds 123
models frozen at that date.

    fetch_logits   ->  ClickHouse `logit_probs`   SUPERSEDED, frozen 08-10
    fetch_stash    ->  the `.f16` set via cache   LIVE

**Both paths are real and they point at DIFFERENT STORES**, which is how the
two got conflated: when a seat established at [6068] that this file "reads a
different store", that was true of `fetch_stash` and not of `fetch_logits`.
The `--logits` result cited in `findings/contradiction_ratio_has_no_null.md`
is NOT invalidated -- it measured what was in the table when it ran -- but a
re-run today reads a store that stopped moving on 08-10, and any model added
since is simply absent rather than erroring. Use `fetch_stash` for anything
new; keep `--logits` only to reproduce the published cross-check.

UNIT. The pair is the unit and the pairs are the 46 LINEAGE REPRESENTATIVES of
`data/lineage_representative_pairs.txt`, not 52 arms -- Falcon3 1B/3B/7B are one
lineage and three rows would be three counts of one observation.
"""
import argparse
import collections
import json
import os
import subprocess

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
CH = "/opt/homebrew/bin/clickhouse"
DB = "malign_logits"
#: the three historical directories, in the order `ch_read.prefetch` uses
PRECEDENCE = ("f11_twp_delta", "f11_twp", "cloud_run_20260801")


def _esc(s):
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _q(sql):
    r = subprocess.run([CH, "client", "--max_query_size", "20000000", "-q", sql],
                       capture_output=True)
    if r.returncode:
        raise SystemExit(r.stderr.decode()[:600])
    return r.stdout.decode("utf-8")


def groups(lang="en"):
    Q = json.load(open(os.path.join(ROOT, "data", "f11_quintuplets.json")))
    return [g for g in Q["quintuplets"]
            if g["status"] != "RETIRED" and g["language"] == lang]


def pairs():
    p = os.path.join(ROOT, "data", "lineage_representative_pairs.txt")
    return [l.strip() for l in open(p) if l.strip()]


def fetch_twp(models, prompts):
    #: **abs(theta - x) < 1e-9, NEVER `theta = 0.001`.** theta is Float32 and
    #: the literal is Float64; the equality matches ZERO rows and returns an
    #: empty result rather than an error, which reads as "not scored".
    order = " ".join("WHEN source='%s' THEN %d" % (s, i)
                     for i, s in enumerate(PRECEDENCE))
    out = _q("SELECT model, prompt, word, argMin(p, CASE %s ELSE 99 END) AS p "
             "FROM %s.twp_words WHERE abs(theta - 0.001) < 1e-9 "
             "AND model IN (%s) AND prompt IN (%s) "
             "GROUP BY model, prompt, word FORMAT TSV"
             % (order, DB,
                ",".join("'%s'" % _esc(m) for m in models),
                ",".join("'%s'" % _esc(p) for p in prompts)))
    #: GROUP BY (prompt, word) with argMin is safe HERE and was checked rather
    #: than assumed: twp rows partition over (word, FIRST TOKEN) and must be
    #: summed, but the summation already happened before ingest -- 0 of
    #: 47,140,883 word cells carry more than one t1.
    D = collections.defaultdict(dict)
    for line in out.splitlines():
        m, p, w, v = line.split("\t")
        D[(m, p)][w] = float(v)
    return D


def fetch_logits(models, prompts):
    out = _q("SELECT model, prompt, token_id, exp(logprob) FROM %s.logit_probs "
             "WHERE model IN (%s) AND prompt IN (%s) FORMAT TSV"
             % (DB, ",".join("'%s'" % _esc(m) for m in models),
                ",".join("'%s'" % _esc(p) for p in prompts)))
    D = collections.defaultdict(dict)
    for line in out.splitlines():
        m, p, t, v = line.split("\t")
        D[(m, p)][t] = float(v)
    return D


def fetch_stash(models, prompts):
    """FULL-VOCABULARY logits from the .f16/.f32 stash -- F11's own estimator.

    The CH tables are both thresholded (twp at theta=0.001 over words, logit_probs
    at 1e-6 over tokens); F11 used the complete softmax. This is the only
    substrate that matches it. Olmo-3-1025-7B: 100,278 tokens, of which 4,202
    clear 1e-6 and carry 99.70% of the mass.

    **DTYPE IS DECLARED, NOT DEFAULTED.** The store is mixed and cache.get_logits
    REFUSES a read that names no dtype -- "a dtype difference is a logit
    difference" -- which is the guard working. Precedence float32 then float16,
    matching the ingester's stated preference. Returns the per-model dtype tally
    alongside the data so a caller can check dtype is CONSTANT WITHIN a model:
    every contrast here is AB against its own poles, so a dtype that varied
    across prompts of one model would put the difference inside the contrast.
    """
    import numpy as _np
    from malign_logits.cache import get_cache
    cm = get_cache()
    D, dts = {}, collections.defaultdict(collections.Counter)
    for m in models:
        for pr in prompts:
            v = None
            for dt in ("float32", "float16"):
                try:
                    v = cm.get_logits(m, pr, dtype=dt)
                except Exception:
                    v = None
                if v is not None:
                    dts[m][dt] += 1
                    break
            if v is None:
                continue
            v = _np.asarray(v, dtype=_np.float64)
            e = _np.exp(v - v.max())
            pv = e / e.sum()
            #: kept as a dense dict on nonzero support so _vecs can align it
            #: with the other substrates without special-casing.
            nz = _np.flatnonzero(pv > 0)
            D[(m, pr)] = {str(int(t)): float(pv[t]) for t in nz}
    return D, dts


def _is_cjk(s):
    return any("一" <= c <= "鿿" for c in s)


def _units(s):
    """Tokens for suffix comparison: words for spaced scripts, CHARACTERS for CJK.

    **CHINESE IS NOT WHITESPACE TOKENIZED and `.split()` silently returns the
    whole prompt as one token.** The first version used `.split()` for
    everything, so all 21 zh groups got the empty frame, every group counted as
    every other group's same-frame partner (19-20 each), and the same-frame null
    came out EQUAL TO the loose null to four decimals. That reads as a clean
    replication and is an identity.

    Word-level is still right for English: the poles share adjective endings
    ("loved"/"hated"), so a character suffix returns 'ed him deeply and wanted
    to' and folds content into the frame.
    """
    return list(s) if _is_cjk(s) else s.split()


def frame_of(g):
    """A group's trailing frame: the common suffix of its two poles."""
    A, B = _units(g["pole_a"]), _units(g["pole_b"])
    n = 0
    while n < min(len(A), len(B)) and A[-1 - n] == B[-1 - n]:
        n += 1
    if not n:
        return ""
    tail = A[len(A) - n:]
    return "".join(tail) if _is_cjk(g["pole_a"]) else " ".join(tail)


def content_of(g):
    """The words a group's two poles DISAGREE on: its contrast, and nothing else.

    **Not "everything outside the trailing frame".** That was the first version
    and it returned 0 eligible partners out of 22, because the poles also share
    a head ("He was ...") whose function words -- he, was, a, and -- then sat in
    every group's content set and made every pair overlap. The symmetric
    difference isolates the contrasting terms: {beautiful, disgusting} against
    {man, woman}.
    """
    A = {str(w).strip(".,，。").lower() for w in _units(g["pole_a"])}
    B = {str(w).strip(".,，。").lower() for w in _units(g["pole_b"])}
    return (A ^ B) - {""}


def same_frame_partners(G):
    """group -> [groups sharing its frame AND sharing no pole content].

    THE CONTENT TEST IS NOT OPTIONAL. `f11_beauty` (beautiful/disgusting) and
    `f11_beauty_ugly` (beautiful/ugly) share a frame and a POLE, so scoring one
    against the other's poles is not a neutralization null at all -- it is a
    near-replicate, and it would drive the null DOWN toward the observed value
    and manufacture the conclusion that the frame mismatch was inflating it.
    """
    out = {}
    for g in G:
        fg, cg = frame_of(g), content_of(g)
        #: **AN EMPTY FRAME IS NOT A SHARED FRAME.** If the extractor cannot
        #: find a suffix, every group's frame is "" and every group becomes
        #: every other's partner -- which is what happened to all 21 zh groups
        #: and produced a same-frame null identical to the loose one. Refusing
        #: on an empty frame turns that from a false replication into a
        #: visible zero.
        out[g["group"]] = [] if not fg else [
            h for h in G
            if h["group"] != g["group"]
            and frame_of(h) == fg
            and not (content_of(h) & cg)]
    return out


def _vecs(ds):
    """Align dicts on their union and renormalise to sum 1.

    Both stores are THRESHOLDED, so each dict is a truncated distribution and
    an absent key means "below threshold", not zero. Renormalising makes the
    three comparable; the clamp at 1e-10 matches contradiction_compare.py.
    """
    keys = sorted(set().union(*[d.keys() for d in ds]))
    out = []
    for d in ds:
        v = np.array([d.get(k, 0.0) for k in keys], dtype=float)
        s = v.sum()
        out.append(v / s if s > 0 else v)
    return out


def _js(p, q):
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def _ratio(ab, A, B):
    den = min(_js(ab, A), _js(ab, B))
    return _js(ab, 0.5 * (A + B)) / den if den > 0 else np.nan


def score(D, models, G):
    """One row per (model, group): observed ratio and two nulls.

    `null` is ANY other live group's BOTH. `null_sf` restricts to a same-frame,
    content-disjoint partner and is NaN where no such partner exists -- the
    stricter null, and the one that says whether frame mismatch inflated the
    loose one.
    """
    SF = same_frame_partners(G)
    rows = []
    for mid in models:
        for g in G:
            k = [(mid, g[x]) for x in ("pole_a", "pole_b", "both")]
            if not all(x in D for x in k):
                continue
            A, B, AB = _vecs([D[k[0]], D[k[1]], D[k[2]]])
            obs = _ratio(AB, A, B)
            #: the resolution reference: NOT a pole itself, which divides by
            #: JS(A,A) = 0. 0.9A + 0.1B is a strongly resolved distribution.
            res = _ratio(0.9 * A + 0.1 * B, A, B)
            sf_names = {h["group"] for h in SF[g["group"]]}
            nulls, nulls_sf = [], []
            for g2 in G:
                if g2["group"] == g["group"] or (mid, g2["both"]) not in D:
                    continue
                A2, B2, N2 = _vecs([D[k[0]], D[k[1]], D[(mid, g2["both"])]])
                r = _ratio(N2, A2, B2)
                if not np.isfinite(r):
                    continue
                nulls.append(r)
                if g2["group"] in sf_names:
                    nulls_sf.append(r)
            if np.isfinite(obs) and nulls:
                rows.append((mid, g["group"], obs, float(np.median(nulls)),
                             float(np.median(nulls_sf)) if nulls_sf else np.nan, res))
    return pd.DataFrame(rows, columns=["model", "group", "obs", "null",
                                       "null_sf", "res"])


def main():
    from scipy import stats
    ap = argparse.ArgumentParser()
    ap.add_argument("--stash", action="store_true",
                    help="also score on FULL-VOCABULARY logits from the .f16/.f32 "
                         "stash -- F11's own estimator, and the only unthresholded one")
    ap.add_argument("--logits", action="store_true",
                    help="cross-check on logit_probs for the models that have "
                         "them; 2.8M rows for 12 models, so not the default")
    ap.add_argument("--lang", default="en")
    a = ap.parse_args()

    G = groups(a.lang)
    PR = pairs()
    models = sorted({m for p in PR for m in p.split(">")})
    prompts = sorted({g[k] for g in G for k in ("pole_a", "pole_b", "both")})
    print("live %s groups %d   lineage-representative pairs %d   models %d   prompts %d"
          % (a.lang, len(G), len(PR), len(models), len(prompts)))

    D = fetch_twp(models, prompts)
    R = score(D, models, G)
    R["gap"] = R["null"] - R["obs"]
    #: **THE LANGUAGE IS PART OF THE FILENAME.** Without it `--lang zh` silently
    #: overwrote the en results under the en name, and the two populations are
    #: not comparable cell for cell.
    out = os.path.join(CAMP, "results", "contradiction_null_%s.csv" % a.lang)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    R.to_csv(out, index=False)

    print("\n%s\nCALIBRATION -- where each reference lands on F11's ratio scale\n%s"
          % ("=" * 78, "=" * 78))
    print("   perfect superposition (A+B)/2        0.000   by construction")
    print("   OBSERVED contradiction               %6.3f" % R.obs.median())
    print("   NEUTRALIZATION null (other BOTH)     %6.3f   <- F11 reads 1.0 as the boundary"
          % R["null"].median())
    print("   RESOLUTION 0.9A + 0.1B               %6.3f" % R.res.median())

    #: THE CAVEAT, TESTED RATHER THAN CARRIED. The loose null draws from any
    #: group, so it differs from the target in trailing FRAME as well as
    #: content, and a uniformly distant third distribution drifts toward 1 by
    #: construction -- which would mean the loose null is inflated and every
    #: signal computed against it is overstated. The same-frame null holds the
    #: frame fixed and varies only content.
    S = R.dropna(subset=["null_sf"])
    sf_groups = sorted(S.group.unique())
    print("\n   SAME-FRAME, CONTENT-DISJOINT NULL   (%d of %d groups have a partner)"
          % (len(sf_groups), R.group.nunique()))
    if len(S):
        print("      loose null      %6.3f" % S["null"].median())
        print("      same-frame null %6.3f   delta %+0.4f"
              % (S.null_sf.median(), S.null_sf.median() - S["null"].median()))
        print("      observed        %6.3f" % S.obs.median())
        print("      signal vs loose %+0.4f   vs same-frame %+0.4f"
              % (S["null"].median() - S.obs.median(),
                 S.null_sf.median() - S.obs.median()))
        print("      groups: %s" % ", ".join(sf_groups))

    g = R.groupby("model").gap.median()
    rows = [(b, g.get(b, np.nan), g.get(al, np.nan))
            for b, al in (p.split(">") for p in PR)
            if b in g.index and al in g.index]
    P = pd.DataFrame(rows, columns=["base", "gap_base", "gap_aligned"])
    P["delta"] = P.gap_aligned - P.gap_base
    P.to_csv(os.path.join(CAMP, "results", "contradiction_null_by_pair_%s.csv" % a.lang), index=False)

    print("\nSUPERPOSITION SIGNAL = null - observed, per lineage (n = %d)" % len(P))
    print("   base     %+0.4f   > 0 in %d of %d"
          % (P.gap_base.median(), int((P.gap_base > 0).sum()), len(P)))
    print("   aligned  %+0.4f   > 0 in %d of %d"
          % (P.gap_aligned.median(), int((P.gap_aligned > 0).sum()), len(P)))
    w = stats.wilcoxon(P.gap_base, P.gap_aligned)
    print("   delta    %+0.4f   reduced in %d of %d   Wilcoxon p=%.3g"
          % (P.delta.median(), int((P.delta < 0).sum()), len(P), w.pvalue))
    print("   AGAINST F11's 'universal': %d of %d lineages move the other way"
          % (int((P.delta >= 0).sum()), len(P)))

    if a.logits:
        have = set(_q("SELECT DISTINCT model FROM %s.logit_probs WHERE model IN (%s) "
                      "FORMAT TSV" % (DB, ",".join("'%s'" % _esc(m) for m in models))).split())
        sub = sorted(have)
        print("\n%s\nCROSS-CHECK: DOES THE TRUNCATION CARRY THE RESULT?  (%d models)\n%s"
              % ("=" * 78, len(sub), "=" * 78))
        L = score(fetch_logits(sub, prompts), sub, G)
        T = score(D, sub, G)
        subs = [("logit_probs 1e-6", L), ("twp words 0.001", T)]
        if a.stash:
            SD, dts = fetch_stash(sub, prompts)
            mixed = sorted(m for m, c in dts.items() if len(c) > 1)
            print("   stash dtype: %d models float32-only, %d float16-only, "
                  "%d MIXED WITHIN MODEL"
                  % (sum(1 for c in dts.values() if set(c) == {"float32"}),
                     sum(1 for c in dts.values() if set(c) == {"float16"}),
                     len(mixed)))
            for m in mixed:
                print("      MIXED %s %s -- a dtype difference inside the contrast"
                      % (m, dict(dts[m])))
            subs.insert(0, ("full logits (stash)", score(SD, sub, G)))
        for lab, X in subs:
            print("   %-20s obs %.3f  null %.3f  resolution %5.2f  signal %+0.3f"
                  % (lab, X.obs.median(), X["null"].median(), X.res.median(),
                     X["null"].median() - X.obs.median()))
    print("\nwrote %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
