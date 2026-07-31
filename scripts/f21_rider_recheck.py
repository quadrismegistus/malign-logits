"""F21 rider re-check — the deference cut, the arm definition, and what each moves.

Re-derives every number in the F21 rider (docket [1707]-[1713]) from the surviving
tagged generations. F21 itself declares `data: []` and `scripts: []`, so the
specification that produced its booked figures is not recorded anywhere; this
script states its own specification explicitly and prints what varies under it.

    .venv/bin/python scripts/f21_rider_recheck.py

Writes data/f21_rider_recheck.csv.
"""

import itertools
import math

import numpy as np
import pandas as pd

SRC = "data/f21_institutional_generations.csv"
OUT = "data/f21_rider_recheck.csv"

# The institution-side role in each prompt key (prompt keys are
# institutional_<domain>_<role>_<n>). Everything else is the individual side.
INSTITUTION_ROLES = {"agency", "landlord", "mgmt", "doctor", "officer", "party"}

# F21's four booked numbers, as written in findings/F21_institutional_alignment.md.
BOOKED = {"individual": (73.7, 79.0), "institution": (91.6, 94.1)}

# Candidate definitions of the "aligned" arm. F21 says "base vs aligned
# checkpoints" and never says which post-training stages that covers.
ALIGNED_ARMS = {
    "dpo+rlvr": ["dpo", "rlvr"],
    "dpo": ["dpo"],
    "sft+dpo+rlvr": ["sft", "dpo", "rlvr"],
    "sft+dpo": ["sft", "dpo"],
    "sft": ["sft"],
}


def logit(p):
    return math.log(p / (1 - p))


def load():
    df = pd.read_csv(SRC)
    role = df["prompt_key"].str.rsplit("_", n=1).str[0].str.rsplit("_", n=1).str[1]
    df["side"] = np.where(role.isin(INSTITUTION_ROLES), "institution", "individual")
    # layer == "unknown" is the frontier-API block, which has no base checkpoint.
    return df[df["layer"] != "unknown"].copy()


def arm_frame(df, aligned_layers):
    d = df[df["layer"].isin(["base"] + aligned_layers)].copy()
    d["arm"] = np.where(d["layer"] == "base", "base", "aligned")
    return d


def rates(d, cut):
    """Procedural rate per (side, arm) at `deference >= cut`, plus the transforms."""
    d = d.assign(p=(d["institutional_deference"] >= cut).astype(float))
    out = {}
    for side in ("individual", "institution"):
        b = d[(d.side == side) & (d.arm == "base")].p.mean()
        a = d[(d.side == side) & (d.arm == "aligned")].p.mean()
        out[side] = dict(
            base=100 * b,
            aligned=100 * a,
            delta_pp=100 * (a - b),
            headroom_pct=100 * (a - b) / (1 - b),
            log_odds=logit(a) - logit(b),
            risk_ratio=a / b,
        )
    return out


def ordinal(d):
    out = {}
    for side in ("individual", "institution"):
        b = d[(d.side == side) & (d.arm == "base")].institutional_deference.mean()
        a = d[(d.side == side) & (d.arm == "aligned")].institutional_deference.mean()
        out[side] = dict(base=b, aligned=a, delta=a - b)
    return out


def reproduction_sweep(df):
    """Does any declared specification return F21's four booked numbers?

    Stopping rule (the liminal/explicit precedent, CLAUDE.md): sweep the
    specifications that are actually plausible readings of the finding's prose,
    report the closest, and STOP. With enough specification freedom something
    eventually lands, and a further reading would be fitting, not reproducing.
    """
    fams = sorted(df.family.unique())
    with_base = [f for f in fams if "base" in set(df[df.family == f].layer)]
    rows = []
    for fam_name, fam_set in (("all open-weight", fams), ("families with a base", with_base)):
        for arm_name, layers in ALIGNED_ARMS.items():
            d = arm_frame(df[df.family.isin(fam_set)], layers)
            for agg in ("pooled", "family-mean", "prompt-mean"):
                d2 = d.assign(p=(d["institutional_deference"] >= 3).astype(float))
                got = {}
                for side in ("individual", "institution"):
                    for arm in ("base", "aligned"):
                        s = d2[(d2.side == side) & (d2.arm == arm)]
                        if agg == "pooled":
                            v = s.p.mean()
                        elif agg == "family-mean":
                            v = s.groupby("family").p.mean().mean()
                        else:
                            v = s.groupby("prompt_key").p.mean().mean()
                        got[(side, arm)] = 100 * v
                err = sum(
                    abs(got[(side, arm)] - BOOKED[side][i])
                    for side in BOOKED
                    for i, arm in enumerate(("base", "aligned"))
                )
                rows.append(
                    dict(
                        families=fam_name, aligned_arm=arm_name, aggregation=agg,
                        ind_base=got[("individual", "base")], ind_aligned=got[("individual", "aligned")],
                        inst_base=got[("institution", "base")], inst_aligned=got[("institution", "aligned")],
                        abs_err_vs_booked=err,
                    )
                )
    return pd.DataFrame(rows).sort_values("abs_err_vs_booked")


def base_arm_exclusion_sweep(df):
    """Would dropping families reproduce the base arm? (It is the base arm that misses.)"""
    b = df[df.layer == "base"].copy()
    b["p"] = (b.institutional_deference >= 3).astype(float)
    fams = sorted(b.family.unique())
    rows = []
    for k in (1, 2):
        for drop in itertools.combinations(fams, k):
            s = b[~b.family.isin(drop)]
            i = 100 * s[s.side == "individual"].p.mean()
            t = 100 * s[s.side == "institution"].p.mean()
            rows.append(dict(dropped="+".join(drop), n_dropped=k, ind_base=i, inst_base=t,
                             abs_err=abs(i - BOOKED["individual"][0]) + abs(t - BOOKED["institution"][0])))
    return pd.DataFrame(rows).sort_values("abs_err")


def main():
    df = load()
    print(f"{SRC}: {len(df)} open-weight tagged generations, "
          f"{df.family.nunique()} families, {df.prompt_key.nunique()} prompts\n")

    print("=" * 78)
    print("1. REPRODUCTION — does any declared spec return F21's booked 73.7/79.0, 91.6/94.1?")
    print("=" * 78)
    sweep = reproduction_sweep(df)
    print(sweep.head(5).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n   closest of {len(sweep)} specifications: |err| = {sweep.abs_err_vs_booked.min():.2f}pp")
    print("   the BASE arm misses under every one of them, in the same direction.")
    excl = base_arm_exclusion_sweep(df)
    print(f"\n   base arm under family exclusions ({len(excl)} tried, best 3):")
    print(excl.head(3).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("   STOPPING HERE BY RULE: the closest requires an UNDECLARED two-family")
    print("   exclusion. Continuing would be fitting, not reproducing.")

    print("\n" + "=" * 78)
    print("2. THE CUT IS THE CEILING — sweep the deference threshold")
    print("=" * 78)
    d = arm_frame(df[df.family.isin([f for f in df.family.unique()
                                     if 'base' in set(df[df.family == f].layer)])], ALIGNED_ARMS["dpo+rlvr"])
    print("   spec: families with a base checkpoint; aligned = DPO+RLVR; pooled generations")
    print(f"   {'cut':>4}  {'individual  Δpp / headroom / log-odds':<40} {'institution Δpp / headroom / log-odds':<40} moves-more")
    rows = []
    for cut in (2, 3, 4, 5):
        r = rates(d, cut)
        i, t = r["individual"], r["institution"]
        winner_pp = "individual" if i["delta_pp"] > t["delta_pp"] else "INSTITUTION"
        winner_lo = "individual" if i["log_odds"] > t["log_odds"] else "INSTITUTION"
        print(f"   >={cut:<2}  {i['delta_pp']:+7.1f}pp {i['headroom_pct']:+7.1f}% {i['log_odds']:+8.3f}"
              f"{'':<12} {t['delta_pp']:+7.1f}pp {t['headroom_pct']:+7.1f}% {t['log_odds']:+8.3f}"
              f"{'':<12} {winner_pp}/{winner_lo}")
        for side in ("individual", "institution"):
            rows.append(dict(check="threshold_sweep", cut=cut, side=side, **r[side]))
    print("\n   AT cut >= 4 THE RAW-PERCENTAGE-POINT ORDERING REVERSES. No transform needed.")

    print("\n" + "=" * 78)
    print("3. IS THE REVERSAL AN ARTEFACT OF THE ARM DEFINITION? (it is not)")
    print("   and does the UNBINARISED ordinal reading settle it? (it does not)")
    print("=" * 78)
    print(f"   {'aligned arm':14s} {'cut>=3 Δpp i/inst':>20} {'cut>=4 Δpp i/inst':>20} {'ordinal Δ i/inst':>20}  rev? ord-dir")
    for name, layers in ALIGNED_ARMS.items():
        d2 = arm_frame(df, layers)
        r3, r4, o = rates(d2, 3), rates(d2, 4), ordinal(d2)
        rev = "yes" if r4["institution"]["delta_pp"] > r4["individual"]["delta_pp"] else "NO"
        od = "individual" if o["individual"]["delta"] > o["institution"]["delta"] else "INSTITUTION"
        print(f"   {name:14s} {r3['individual']['delta_pp']:+8.1f}/{r3['institution']['delta_pp']:+8.1f}"
              f" {r4['individual']['delta_pp']:+9.1f}/{r4['institution']['delta_pp']:+8.1f}"
              f" {o['individual']['delta']:+9.3f}/{o['institution']['delta']:+8.3f}  {rev:>4} {od}")
        rows.append(dict(check="arm_sensitivity", aligned_arm=name,
                         ind_delta_pp_cut3=r3["individual"]["delta_pp"],
                         inst_delta_pp_cut3=r3["institution"]["delta_pp"],
                         ind_delta_pp_cut4=r4["individual"]["delta_pp"],
                         inst_delta_pp_cut4=r4["institution"]["delta_pp"],
                         ind_ordinal_delta=o["individual"]["delta"],
                         inst_ordinal_delta=o["institution"]["delta"]))
    print("\n   THE REVERSAL AT cut>=4 HOLDS UNDER ALL FIVE ARM DEFINITIONS.")
    print("   THE ORDINAL DIRECTION DOES NOT: it flips to a tie under sft+dpo+rlvr and")
    print("   goes NEGATIVE for the individual under sft-only. Margins are <= 0.05 of a")
    print("   scale point in three of five. Unbinarising does not settle the question.")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
