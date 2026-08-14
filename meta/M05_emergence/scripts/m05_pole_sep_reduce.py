"""The pole_sep per-checkpoint reduction, declared at plan_pole_sep_reduction.md.

    uv run python meta/M05_emergence/scripts/m05_pole_sep_reduce.py
    -> results/m05_pole_sep_reduced.{csv,json}

The plan was committed ALONE at 570afad4, before this file existed. That
ordering is the reason these numbers are worth more than the six they replace:
the six in `findings/pole_sep_is_not_about_poles.md` cannot be reproduced from
the committed artifact by any stated rule (@dario [5954], six-target check at
[5958]), and a reduction chosen because it reproduced them would be a recipe
fitted to a target ([5935]).

**THE SUPERSEDED VALUES ARE NOT AVAILABLE TO THE REDUCTION.** They are printed
at the end, after every number is computed, purely so a reader sees the size of
the change. No branch reads them. If they had been in scope, choosing among the
plan's alternatives would have been contaminated by knowing which one hit.

THE RULE, in the plan's words:

    1. role is NOT a dimension: pole_sep is bit-identical across role
       (max diff exactly 0.0), but only 50,193 of the cells carry all three
       while 15,675 carry one, so pooling roles is a median weighted by which
       controls happened to run. Deduplicate to one row per
       (checkpoint, group, layer).
    2. two stages: median over the 33 layers within (checkpoint, group), then
       median over groups. The GROUP is the unit; the layer is summarised.
    3. the group set is the COMMON set across the ladder, count published.
    4. median, not mean, throughout (M05's ranks-not-levels riders).

The null column reads `m05_pole_sep_crossgroup_null.csv`, which holds BOTH arms
distinguished only by `group_x == group_y` -- the `kind` field says REAL on the
9,009 real rows and is EMPTY on the 90,090 null ones. Two traps in that file,
both hit here before they were noticed: OLMo's `step` column is -1 on every
OLMo row (the step lives in the model string), and an unsorted `.unique()[:4]`
shows only Pythia though OLMo is 53,361 of the rows.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTD = os.path.join(ROOT, "meta/M05_emergence/results")

#: printed at the end only; NEVER read by the reduction. See the docstring.
SUPERSEDED = {
    "allenai/Olmo-3-1025-7B@stage1-step0": (0.795, 1.396),
    "allenai/Olmo-3-1025-7B@stage1-step1000": (0.227, 0.526),
    "allenai/Olmo-3-1025-7B@stage1-step16000": (0.475, 0.805),
    "EleutherAI/pythia-6.9b@step0": (0.347, 0.748),
    "EleutherAI/pythia-6.9b@step128": (0.115, 0.270),
    "EleutherAI/pythia-6.9b@step16000": (0.384, 0.701),
}


def ck_order(c):
    """Sort key for a checkpoint. LEXICOGRAPHIC SORTING PUTS step1000 BEFORE
    step128 -- the first run of this producer printed the Pythia ladder in that
    order, which is harmless for an unordered rho and wrong for every curve
    drawn from the CSV. Stage first, then the step as an integer."""
    import re
    s = str(c)
    stage = re.search(r"stage(\d+)", s)
    step = re.search(r"step(\d+)", s)
    return (int(stage.group(1)) if stage else 0, int(step.group(1)) if step else -1, s)


def reduce_two_stage(df, ckcol, val, groups):
    """Rule 2 + 3 + 4: median over layers within group, then median over the
    COMMON group set. Returns {checkpoint: value}."""
    d = df[df["__g"].isin(groups)]
    per_group = d.groupby([ckcol, "__g"])[val].median()
    return per_group.groupby(level=0).median().to_dict()


def main():
    import pandas as pd

    real = pd.read_csv(os.path.join(OUTD, "m05_pole_sep.csv"))
    nullf = pd.read_csv(os.path.join(OUTD, "m05_pole_sep_crossgroup_null.csv"))

    #: RULE 1. Assert the identity the dedup rests on, rather than trusting the
    #: finding's prose for it -- if role ever stops being bit-identical, this
    #: reduction is wrong and must not run.
    p = real.pivot_table(index=["checkpoint", "group", "layer"], columns="role",
                         values="pole_sep").dropna()
    worst = max(float((p["both"] - p[c]).abs().max()) for c in p.columns if c != "both")
    if worst != 0.0:
        raise SystemExit("role is NOT bit-identical (max |diff| = %r); rule 1 of "
                         "plan_pole_sep_reduction.md does not hold and this "
                         "reduction is void." % worst)
    n_before = len(real)
    real_olmo = real.drop_duplicates(["checkpoint", "group", "layer"]).copy()
    print("RULE 1  role dedup: %s -> %s rows (identity asserted, max|diff| %.1f)"
          % (format(n_before, ","), format(len(real_olmo), ","), worst))

    #: THE REAL COLUMN COMES FROM THE NULL FILE, NOT FROM m05_pole_sep.csv.
    #: The first draft of this producer took real from m05_pole_sep.csv and
    #: silently emitted NO pythia|real ladder, because that file is OLMo-only
    #: (four OLMo variants, zero Pythia) -- the finding's Pythia real column
    #: lives in the null file's `kind == "REAL"` rows, which cover BOTH
    #: lineages. Taking both arms from one file also puts them at one stored
    #: precision, which the cross-check below then has to allow for.
    nullf["__ck"] = nullf["model"].astype(str)
    nullf["__g"] = nullf["group_x"].astype(str) + "|" + nullf["group_y"].astype(str)
    cross = nullf[nullf.group_x != nullf.group_y].copy()
    realf = nullf[nullf.group_x == nullf.group_y].copy()
    print("RULE 0  null file: %s rows | %s cross-group (null) | %s same-group (real)"
          % (format(len(nullf), ","), format(len(cross), ","), format(len(realf), ",")))
    print("        `kind` says REAL on the real rows and is EMPTY on the null "
          "rows, so the arms are separated by the join condition, not the label")

    #: CROSS-CHECK, since OLMo real exists in BOTH files and so is checkable.
    #: TOLERANCE IS NOT ARBITRARY: the null file stores ~6 significant figures,
    #: so the half-ulp grows with magnitude -- measured, the error is 0 below
    #: 0.01, 5e-7 in [0.01,0.1), 0 in [0.1,1) and 5e-6 in [1,10), i.e. max
    #: RELATIVE 2.02e-5. An absolute 1e-9 threshold called this a disagreement
    #: on the first run; comparing at unequal stored precision is its own defect.
    x = realf[realf.model.astype(str).str.contains("Olmo")].copy()
    x["group"] = x["group_x"]
    x["checkpoint"] = x["model"]
    ck = x[["checkpoint", "group", "layer", "sep"]].merge(
        real_olmo[["checkpoint", "group", "layer", "pole_sep"]],
        on=["checkpoint", "group", "layer"], how="inner")
    rel = ((ck["sep"] - ck["pole_sep"]).abs()
           / ck["pole_sep"].abs().clip(lower=1e-12)).max()
    if len(ck) and rel > 1e-4:
        raise SystemExit("the two OLMo real sources disagree beyond stored "
                         "precision (max relative %.2e over %d cells)" % (rel, len(ck)))
    print("CROSS    OLMo real in both sources: %s cells agree, max relative %.2e"
          % (format(len(ck), ","), rel))

    out = {"_about":
           "pole_sep reduced to ONE NUMBER PER CHECKPOINT under the rule declared "
           "at plans/plan_pole_sep_reduction.md, committed at 570afad4 BEFORE this "
           "producer existed. These SUPERSEDE the six values in "
           "findings/pole_sep_is_not_about_poles.md, which have no stated rule and "
           "do not reproduce (32-recipe and six-target checks, [5954]/[5958]). The "
           "superseded values are reproduced in `superseded` for comparison only "
           "and were not available to the reduction. The finding's CLAIM is "
           "co-movement of the real and null columns, not their levels -- the "
           "finding's own text says THE LEVEL GAP LICENSES NOTHING.",
           "plan": "plans/plan_pole_sep_reduction.md",
           "rule": {"1_role": "deduplicated; bit-identical, asserted at run",
                    "2_stages": "median over layers within group, then over groups",
                    "3_groups": "common set across the ladder, published below",
                    "4_statistic": "median throughout"},
           "ladders": {}, "superseded": {}}
    rows = []

    #: ARM LABELS ARE `REAL` / `CROSSGROUP`, NOT `real` / `null`.
    #: The first version labelled the second arm `null`, which is in pandas'
    #: DEFAULT missing-value set -- so `pd.read_csv` turned the label of the
    #: null arm into NaN for every reader, and `df[df.column == "null"]`
    #: returned ZERO rows against 13 present on disk. @dario found it within
    #: minutes of opening the file ([5966]). Renaming beats telling readers to
    #: pass `keep_default_na=False`, which is a fix each one has to remember.
    #: Note what it reproduced: the SOURCE file labels its real arm and leaves
    #: the null arm empty, and this file fixed that and then had the reader put
    #: it back. Same two-arm-one-label defect, opposite end of the pipe.
    for name, sub, ckcol, val, src in (
            ("REAL", realf, "__ck", "sep",
             "m05_pole_sep_crossgroup_null.csv (group_x == group_y)"),
            ("CROSSGROUP", cross, "__ck", "sep",
             "m05_pole_sep_crossgroup_null.csv (group_x != group_y)")):
        #: RULE 3, per ladder: a group must be present at EVERY checkpoint of
        #: that ladder, else a curve moves when composition moves.
        sub = sub.copy()
        sub["__lad"] = ["olmo" if "Olmo" in str(c) else "pythia" for c in sub[ckcol]]
        for lad in ("olmo", "pythia"):
            s = sub[sub["__lad"] == lad]
            if not len(s):
                continue
            cks = sorted(s[ckcol].unique())
            per_ck = {c: set(s[s[ckcol] == c]["__g"]) for c in cks}
            common = set.intersection(*per_ck.values()) if per_ck else set()
            allg = set.union(*per_ck.values()) if per_ck else set()
            vals = reduce_two_stage(s, ckcol, val, common)
            key = "%s|%s" % (lad, name)
            out["ladders"][key] = {
                "n_checkpoints": len(cks), "n_groups_common": len(common),
                "n_groups_any": len(allg),
                "groups_excluded": sorted(allg - common), "source": src,
                "values": {str(k): float(vals[k]) for k in sorted(vals, key=ck_order)}}
            print("  %-12s %2d checkpoints | %d common groups of %d | %d excluded"
                  % (key, len(cks), len(common), len(allg), len(allg - common)))
            for c in sorted(vals, key=ck_order):
                v = vals[c]
                rows.append({"ladder": lad, "column": name, "checkpoint": str(c),
                             "value": float(v), "n_groups": len(common)})

    df = pd.DataFrame(rows)

    #: THE GENERAL GUARD, worth more than the rename above. A categorical whose
    #: VALUE collides with a reserved missing-value token is unreadable by
    #: default, and the collision is invisible in the file -- it happens in the
    #: reader. Refuse to write any object column holding one, rather than fix
    #: the one instance that was found. Round-trip asserted on top, since the
    #: guard is a claim about pandas and the round trip is a measurement.
    from pandas._libs.parsers import STR_NA_VALUES
    for col in df.columns:
        if df[col].dtype != object:
            continue
        bad = sorted(set(df[col].astype(str)) & set(STR_NA_VALUES))
        if bad:
            raise SystemExit(
                "column %r holds value(s) %r that pandas reads as NaN by "
                "default; rename them (see the arm-label comment above)."
                % (col, bad))
    path = os.path.join(OUTD, "m05_pole_sep_reduced.csv")
    df.to_csv(path, index=False)
    back = pd.read_csv(path)
    if len(back) != len(df) or back.isna().any().any():
        raise SystemExit("round trip lost data: %d rows out, %d back, %d NaN"
                         % (len(df), len(back), int(back.isna().sum().sum())))
    print("GUARD    no emitted label collides with pandas NA tokens; "
          "round trip %d rows, 0 NaN" % len(back))

    #: EVERYTHING ABOVE IS COMPUTED. Only now are the superseded values read.
    print("\nAGAINST THE SUPERSEDED VALUES (comparison only; not used above)")
    print("  %-42s %-14s %-14s" % ("checkpoint", "real  was->now", "null  was->now"))
    look = {(r["checkpoint"], r["column"]): r["value"] for _, r in df.iterrows()}
    for ck, (was_r, was_n) in SUPERSEDED.items():
        now_r = look.get((ck, "REAL"))
        now_n = look.get((ck, "CROSSGROUP"))
        out["superseded"][ck] = {"real_was": was_r, "real_now": now_r,
                                 "null_was": was_n, "null_now": now_n}
        f = lambda w, n: "%.3f->%s" % (w, "%.4f" % n if n is not None else "ABSENT")
        print("  %-42s %-14s %-14s" % (ck.split("@")[-1][:40], f(was_r, now_r),
                                       f(was_n, now_n)))

    #: THE PREDICTION THE PLAN RECORDED: co-movement, not levels.
    for lad in ("olmo", "pythia"):
        r = out["ladders"].get("%s|REAL" % lad, {}).get("values", {})
        n = out["ladders"].get("%s|CROSSGROUP" % lad, {}).get("values", {})
        both = sorted(set(r) & set(n), key=ck_order)
        if len(both) > 2:
            from scipy import stats
            rho, pv = stats.spearmanr([r[c] for c in both], [n[c] for c in both])
            out["ladders"]["%s|comovement_spearman" % lad] = {
                "rho": float(rho), "p": float(pv), "n": len(both),
                "fence": "n is 6-7 checkpoints; this cannot establish "
                         "co-movement, only fail to contradict it"}
            print("  CO-MOVEMENT %-7s Spearman rho %+.3f  p %.3f  n=%d"
                  % (lad, rho, pv, len(both)))

    p = os.path.join(OUTD, "m05_pole_sep_reduced.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n-> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
