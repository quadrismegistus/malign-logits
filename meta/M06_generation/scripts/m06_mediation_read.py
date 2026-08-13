"""Stage 2: do M01's movers carry M06's passage effect? The three readings.

    uv run python meta/M06_generation/scripts/m06_mediation_read.py
    -> results/mediation_readings.json + mediation_pairs.parquet

Runs plan_mediation.md (d79b6c0f). Stage 1 (`m06_mediation.py`) produced, per
(pair, generating role, word): occurrences, and summed surprisal under BOTH the
base and the aligned scorer. That gives the full 2x2 -- each arm's text scored
by each arm's model -- which is what makes the decomposition exact rather than
approximate.

THE DECOMPOSITION. Writing E_r[s_m] for "mean surprisal of role r's text under
model m", the self-surprisal difference splits two ways:

    order 1   Delta = (E_a[s_a] - E_b[s_a])  +  (E_b[s_a] - E_b[s_b])
                       COMPOSITION                LEVEL on base text
    order 2   Delta = (E_a[s_b] - E_b[s_b])  +  (E_a[s_a] - E_a[s_b])
                       COMPOSITION                LEVEL on aligned text

Both are exact and they do not generally agree, so BOTH are reported and the
symmetric mean is the headline. A single order would be an undeclared choice,
and this campaign has already had a top-of-site-versus-summed choice flip a
refutation into an agreement.

The LEVEL term is the one with no frequency confound in it: it compares two
models on IDENTICAL TEXT, so every word is its own control.
"""
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")
SEED = 20260813
N_BOOT = 2000


def ch_rows(q):
    pr = subprocess.Popen([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                          stdout=subprocess.PIPE, text=True, bufsize=1 << 22)
    for line in pr.stdout:
        try:
            yield json.loads(line)
        except Exception:
            continue
    pr.wait()


def main():
    import numpy as np
    import pandas as pd
    from scipy import stats

    rng = np.random.default_rng(SEED)
    W = pd.read_parquet(os.path.join(OUTD, "mediation_words.parquet"))
    gate = {g["pair"]: g for g in json.load(
        open(os.path.join(OUTD, "mediation_gate.json")))}
    pairs = sorted(W["pair"].unique())
    print("stage 1 kept %d pairs, %d word rows" % (len(pairs), len(W)))

    #: a pair can carry two labels; prefer the named training relation over the
    #: registry's base_to_superego fallback, and never let dict order decide.
    relset = collections.defaultdict(set)
    for r in ch_rows("SELECT DISTINCT base, aligned, relation "
                     "FROM malign_logits.movement"):
        relset[r["base"] + ">" + r["aligned"]].add(r["relation"])
    rel = {}
    for p, rs in relset.items():
        named = sorted(rs - {"base_to_superego"})
        rel[p] = named[0] if named else sorted(rs)[0]

    out, per_word_all = [], []
    for pair in pairs:
        base, aligned = pair.split(">", 1)
        d = W[W["pair"] == pair]
        b = d[d["role"] == "base"].set_index("word")
        a = d[d["role"] == "aligned"].set_index("word")
        if not len(b) or not len(a):
            continue

        words = sorted(set(b.index) | set(a.index))
        occ_b = b["occurrences"].reindex(words).fillna(0.0).to_numpy()
        occ_a = a["occurrences"].reindex(words).fillna(0.0).to_numpy()
        # mean surprisal of word w, by (text role | scoring model)
        with np.errstate(invalid="ignore", divide="ignore"):
            sbb = (b["sum_s_base"] / b["occurrences"]).reindex(words).to_numpy()
            sab = (b["sum_s_aligned"] / b["occurrences"]).reindex(words).to_numpy()
            sba = (a["sum_s_base"] / a["occurrences"]).reindex(words).to_numpy()
            saa = (a["sum_s_aligned"] / a["occurrences"]).reindex(words).to_numpy()
        f_b = occ_b / occ_b.sum()
        f_a = occ_a / occ_a.sum()

        def dot(f, s):
            m = (f > 0) & np.isfinite(s)
            return float((f[m] * s[m]).sum())

        E_bb, E_ba = dot(f_b, sbb), dot(f_b, sab)   # base text, each scorer
        E_ab, E_aa = dot(f_a, sba), dot(f_a, saa)   # aligned text, each scorer
        delta = E_aa - E_bb
        comp1, lev1 = E_aa - E_ba, E_ba - E_bb
        comp2, lev2 = E_ab - E_bb, E_aa - E_ab
        # GATE R: each order must reconstruct Delta exactly.
        resid = max(abs(comp1 + lev1 - delta), abs(comp2 + lev2 - delta))

        # LEVEL per word on identical text: >0 = aligned finds it MORE
        # surprising. No frequency confound; each word is its own control.
        # Computed on BOTH text populations -- same contrast, different corpus
        # of occasions -- because reporting only one would be a silent choice.
        lvl = sab - sbb          # on base-generated text
        lvl_a = saa - sba        # on aligned-generated text
        comp = f_a - f_b

        out.append(dict(
            pair=pair, relation=rel.get(pair, "?"),
            retention=gate.get(pair, {}).get("retention"),
            n_words=len(words), delta=delta,
            composition=0.5 * (comp1 + comp2), level=0.5 * (lev1 + lev2),
            comp1=comp1, lev1=lev1, comp2=comp2, lev2=lev2, residual=resid))

        per_word_all.append(pd.DataFrame(dict(
            pair=pair, word=words, occ_b=occ_b, occ_a=occ_a,
            f_b=f_b, f_a=f_a, level=lvl, level_a=lvl_a, comp=comp)))

    P = pd.DataFrame(out)
    PW = pd.concat(per_word_all, ignore_index=True)
    P.to_parquet(os.path.join(OUTD, "mediation_pairs.parquet"), index=False)

    print("\nGATE R (decomposition reconciles)   max residual %.2e"
          % P["residual"].max())
    print("\nDECOMPOSITION over %d pairs, nats per word" % len(P))
    print("  Delta (aligned self - base self)   %+.4f" % P["delta"].mean())
    print("  composition (symmetric)            %+.4f" % P["composition"].mean())
    print("  level       (symmetric)            %+.4f" % P["level"].mean())
    print("  order 1: comp %+.4f  level %+.4f" % (P["comp1"].mean(), P["lev1"].mean()))
    print("  order 2: comp %+.4f  level %+.4f" % (P["comp2"].mean(), P["lev2"].mean()))
    for r, g in P.groupby("relation"):
        print("    %-20s n=%2d  Delta %+.4f  comp %+.4f  level %+.4f"
              % (r, len(g), g["delta"].mean(), g["composition"].mean(),
                 g["level"].mean()))

    # ---- the M01 join -------------------------------------------------
    print("\nloading movement (canonical rule) for %d pairs" % len(P))
    mv = collections.defaultdict(lambda: [0, 0, 0.0])
    plist = "','".join(p.replace("'", "\\'") for p in P["pair"])
    #: 13 of 206 pairs carry TWO relation labels (a pair can be both e.g.
    #: dpo_of and base_to_superego), which duplicates its rows. Counting
    #: DISTINCT PROMPTS rather than rows is immune to that; countIf would
    #: silently double the evidence for exactly those 13.
    q = ("SELECT concat(base,'>',aligned) AS pair, word, "
         "uniqExactIf(prompt, cls='fall') AS nf, "
         "uniqExactIf(prompt, cls='rise') AS nr, "
         "avg(delta) AS md FROM malign_logits.movement "
         "WHERE rule='canonical' AND concat(base,'>',aligned) IN ('%s') "
         "GROUP BY pair, word" % plist)
    for r in ch_rows(q):
        mv[(r["pair"], r["word"])] = [r["nf"], r["nr"], r["md"]]
    print("  movement (pair, word) entries: %d" % len(mv))

    key = list(zip(PW["pair"], PW["word"]))
    nf = np.array([mv.get(k, (0, 0, 0.0))[0] for k in key], dtype=float)
    nr = np.array([mv.get(k, (0, 0, 0.0))[1] for k in key], dtype=float)
    PW["is_faller"] = (nf > 0) & (nf > nr)
    PW["is_riser"] = (nr > 0) & (nr > nf)
    PW["in_m01"] = np.array([k in mv for k in key])

    # GATE V: movers must actually occur in passage text before any share claim
    occ = PW["occ_b"] + PW["occ_a"]
    tot = occ.sum()
    print("\nGATE V (do M01's words occur in passages?)")
    print("  word types in passages          %d" % len(PW))
    print("  types M01 measured              %d (%.1f%%)"
          % (PW["in_m01"].sum(), 100 * PW["in_m01"].mean()))
    print("  token share of M01-measured     %.1f%%"
          % (100 * occ[PW["in_m01"]].sum() / tot))
    print("  faller types %d, token share %.2f%%"
          % (PW["is_faller"].sum(), 100 * occ[PW["is_faller"]].sum() / tot))
    print("  riser  types %d, token share %.2f%%"
          % (PW["is_riser"].sum(), 100 * occ[PW["is_riser"]].sum() / tot))

    res = {"decomposition": {k: float(P[k].mean()) for k in
                             ("delta", "composition", "level", "comp1", "lev1",
                              "comp2", "lev2")},
           "n_pairs": int(len(P)), "gate_r_residual": float(P["residual"].max()),
           "mover_token_share": {
               "faller": float(occ[PW["is_faller"]].sum() / tot),
               "riser": float(occ[PW["is_riser"]].sum() / tot),
               "in_m01": float(occ[PW["in_m01"]].sum() / tot)}}

    # ---- M1: LEVEL, per pair, weighted by occurrence -------------------
    print("\nM1  LEVEL on identical text (>0 = aligned finds it MORE surprising)")
    print("    declared: fallers > 0, risers < 0, both halves required")
    rows = []
    for pair, g in PW.groupby("pair"):
        w = g["occ_b"].to_numpy()
        lv = g["level"].to_numpy()
        ok = np.isfinite(lv) & (w > 0)
        fa, ri = g["is_faller"].to_numpy(), g["is_riser"].to_numpy()
        inm = g["in_m01"].to_numpy()
        def wm(mask, vals=lv):
            m = mask & ok
            return float(np.average(vals[m], weights=w[m])) if m.sum() else np.nan
        # `still` = M01 MEASURED it and it did not move; `unmeasured` = M01
        # never saw it. Pooling those two would hide the thing GATE V asks.
        rows.append((pair, wm(fa), wm(ri), wm(inm & ~fa & ~ri), wm(~inm),
                     wm(fa, g["level_a"].to_numpy()),
                     wm(ri, g["level_a"].to_numpy())))
    M1 = pd.DataFrame(rows, columns=["pair", "faller", "riser", "still",
                                     "unmeasured", "faller_alt", "riser_alt"])
    M1 = M1.dropna(subset=["faller", "riser"])
    for c in ("faller", "riser", "still", "unmeasured"):
        v = M1[c].dropna()
        t = stats.wilcoxon(v) if len(v) > 5 else None
        print("    %-11s mean %+.4f   n=%d  %s"
              % (c, v.mean(), len(v),
                 ("wilcoxon p %.3g" % t.pvalue) if t else ""))
    da = (M1["faller_alt"] - M1["riser_alt"]).dropna()
    print("    [on aligned-generated text: faller %+.4f riser %+.4f diff %+.4f]"
          % (M1["faller_alt"].mean(), M1["riser_alt"].mean(), da.mean()))
    dif = M1["faller"] - M1["riser"]
    tt = stats.wilcoxon(dif) if len(M1) > 5 else None
    print("    faller - riser  %+.4f  n=%d  %s  [%d/%d pairs positive]"
          % (dif.mean(), len(M1), ("wilcoxon p %.3g" % tt.pvalue) if tt else "",
             int((dif > 0).sum()), len(dif)))
    res["M1"] = {"faller": float(M1["faller"].mean()),
                 "riser": float(M1["riser"].mean()),
                 "still": float(M1["still"].mean()),
                 "unmeasured": float(M1["unmeasured"].mean()),
                 "faller_alt_text": float(M1["faller_alt"].mean()),
                 "riser_alt_text": float(M1["riser_alt"].mean()),
                 "faller_minus_riser": float(dif.mean()),
                 "n_pairs": int(len(M1)),
                 "p": float(tt.pvalue) if tt else None,
                 "pairs_positive": int((dif > 0).sum())}

    # ---- M2: COMPOSITION ----------------------------------------------
    print("\nM2  COMPOSITION (f_aligned - f_base, per 10k tokens)")
    print("    declared: fallers < 0, risers > 0")
    rows = []
    for pair, g in PW.groupby("pair"):
        c = np.nan_to_num(g["comp"].to_numpy()) * 1e4
        rows.append((pair, float(c[g["is_faller"].to_numpy()].sum()),
                     float(c[g["is_riser"].to_numpy()].sum())))
    M2 = pd.DataFrame(rows, columns=["pair", "faller", "riser"])
    for c in ("faller", "riser"):
        t = stats.wilcoxon(M2[c]) if len(M2) > 5 else None
        print("    %-8s summed %+.2f  n=%d  %s  [%d/%d pairs negative]"
              % (c, M2[c].mean(), len(M2),
                 ("wilcoxon p %.3g" % t.pvalue) if t else "",
                 int((M2[c] < 0).sum()), len(M2)))
    res["M2"] = {"faller": float(M2["faller"].mean()),
                 "riser": float(M2["riser"].mean()),
                 "n_pairs": int(len(M2))}

    # ---- M3: SHARE against a frequency-matched non-mover set -----------
    print("\nM3  SHARE of the level term carried by movers")
    print("    declared NULL: enrichment <= 1 against frequency-matched non-movers")
    contrib = PW["f_b"].to_numpy() * np.nan_to_num(PW["level"].to_numpy())
    total = float(np.nansum(contrib))
    mover = PW["is_faller"].to_numpy() | PW["is_riser"].to_numpy()
    mv_share = float(np.nansum(contrib[mover]) / total) if total else np.nan
    mv_tok = float(PW["f_b"][mover].sum() / PW["f_b"].sum())
    #: frequency-matched: resample non-movers to the movers' log-frequency
    #: distribution, in deciles of base frequency
    lf = np.log10(PW["f_b"].to_numpy() + 1e-12)
    edges = np.quantile(lf[mover], np.linspace(0, 1, 11))
    boots = []
    nonmv_idx = np.where(~mover)[0]
    lf_non = lf[nonmv_idx]
    for _ in range(N_BOOT):
        pick = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            need = int(((lf[mover] >= lo) & (lf[mover] < hi)).sum())
            pool = nonmv_idx[(lf_non >= lo) & (lf_non < hi)]
            if need and len(pool):
                pick.append(rng.choice(pool, size=need, replace=True))
        if pick:
            boots.append(float(np.nansum(contrib[np.concatenate(pick)]) / total))
    boots = np.array(boots)
    enrich = mv_share / np.median(boots) if len(boots) and np.median(boots) else np.nan
    lo, hi = (np.quantile(boots, [0.025, 0.975]) if len(boots) else (np.nan, np.nan))
    print("    mover token share of text        %.2f%%" % (100 * mv_tok))
    print("    mover share of the level term    %.2f%%" % (100 * mv_share))
    print("    freq-matched non-movers          %.2f%%  [95%% %.2f-%.2f]"
          % (100 * np.median(boots), 100 * lo, 100 * hi))
    print("    ENRICHMENT                       %.2fx" % enrich)
    res["M3"] = {"mover_token_share": mv_tok, "mover_level_share": mv_share,
                 "matched_median": float(np.median(boots)) if len(boots) else None,
                 "matched_ci": [float(lo), float(hi)],
                 "enrichment": float(enrich)}

    json.dump(res, open(os.path.join(OUTD, "mediation_readings.json"), "w"),
              indent=1)
    PW.to_parquet(os.path.join(OUTD, "mediation_words_joined.parquet"), index=False)
    print("\nwrote mediation_readings.json, mediation_pairs.parquet, "
          "mediation_words_joined.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
