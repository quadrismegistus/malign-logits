#!/usr/bin/env python
"""f11_hidden_backfill.py — which checkpoints ran BEFORE output_hidden_states=True?

    scripts/f11_hidden_backfill.py            report
    scripts/f11_hidden_backfill.py --write    emit the backfill specs

**THE SPLIT IS A RECORDED FIELD, NOT A DISCOVERED ABSENCE** (registrar [5142].a).
The flag landed mid-fleet, so the roster is split: checkpoints completed before it
carry word probabilities and logits but no residuals. That split has to be
derivable rather than remembered.

**AND "MISSING" IS NOT ONE THING.** Three cases, and only the first is a backfill
target:

    NO SIDECAR / EMPTY   ran before the flag. Recoverable by re-running.
    EMPTY, NO ROWS       the model produced nothing at all -- refused on
                         round-trip (Pharia) or failed to load (Baichuan2).
                         Re-running gets nothing; it is not missing data, it is
                         a model that has none.
    PRESENT              rows == logit-bearing lines, nothing to do.

Counting the second as a backfill target would buy downloads for checkpoints that
cannot produce a residual, which is the shape of paying for a re-run that changes
nothing.
"""
import argparse, collections, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data", "f11_twp")


def state():
    rows = {}
    for jp in sorted(glob.glob(os.path.join(D, "*.jsonl"))):
        base = os.path.basename(jp)[:-6]
        mid = base.replace("__", "/")
        n_rows = n_logit = 0
        for ln in open(jp, errors="ignore"):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            n_rows += 1
            if r.get("logit_row") is not None:
                n_logit += 1
        hp = os.path.join(D, base + ".hidden.f32")
        hsz = os.path.getsize(hp) if os.path.exists(hp) else 0
        if n_logit == 0:
            kind = "no-usable-rows"
        elif hsz > 0:
            kind = "has-hidden"
        else:
            kind = "NEEDS-BACKFILL"
        rows[mid] = {"rows": n_rows, "logit_rows": n_logit,
                     "hidden_bytes": hsz, "kind": kind}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    st = state()
    by = collections.defaultdict(list)
    for mid, v in st.items():
        by[v["kind"]].append(mid)

    print("SYNCED SO FAR: %d checkpoints" % len(st))
    for k in ("has-hidden", "NEEDS-BACKFILL", "no-usable-rows"):
        print("  %-16s %d" % (k, len(by.get(k, []))))
    print("\nNEEDS-BACKFILL — ran before the flag, residuals recoverable:")
    for m in sorted(by.get("NEEDS-BACKFILL", [])):
        print("   %-52s %d rows" % (m, st[m]["logit_rows"]))
    print("\nno-usable-rows — produced nothing; a re-run gets nothing:")
    for m in sorted(by.get("no-usable-rows", [])):
        print("   %-52s %d rows, 0 logit-bearing" % (m, st[m]["rows"]))

    if a.write:
        env = json.load(open(os.path.join(ROOT, "data", "f11_env_plan.json")))
        full = json.load(open(os.path.join(ROOT, "data", "f11_twp_spec.json")))
        bym = {e["model"]: e for e in full["spec"]}
        want = set(by.get("NEEDS-BACKFILL", []))
        BOX = {"default": "box1_dense", "torch26": "box1_dense",
               "ssm": "box2_ssm", "twogpu": "box3_70b"}
        groups = collections.defaultdict(list)
        for e, v in env["environments"].items():
            for m in v["models"]:
                if m in want:
                    groups[BOX[e]].append(m)
        for box, ms in sorted(groups.items()):
            spec = [bym[m] for m in sorted(ms) if m in bym]
            p = os.path.join(ROOT, "data", "f11_twp_backfill.%s.json" % box)
            json.dump({"_meta": {
                "about": "HIDDEN-STATE BACKFILL. These checkpoints completed "
                         "before output_hidden_states=True landed mid-fleet. "
                         "Word probabilities and logits are already stored and "
                         "regenerate bit-identically (batch-1, no RNG); the "
                         "residuals are what this pass is for.",
                "producer": "scripts/f11_hidden_backfill.py",
                "box": box, "models": len(spec),
                "run_into_a_SEPARATE_out_dir": "resume is by completed-prompt "
                    "readback, so pointing this at the original directory would "
                    "skip every model in it and produce nothing.",
            }, "spec": spec}, open(p, "w"), ensure_ascii=False, indent=1)
            print("\nwrote %s  (%d models)" % (os.path.relpath(p, ROOT), len(spec)))


if __name__ == "__main__":
    main()
