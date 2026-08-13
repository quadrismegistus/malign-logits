"""Do the ambient interiority shift and the second-order ascent covary by pair?

    uv run python meta/M02_frame_exit/scripts/ambient_vs_ascent_covariance.py
    -> results/ambient_vs_ascent.json

THE QUESTION. M02 holds two arm effects on one corpus: an AMBIENT semantic-field
shift toward interiority (39 of 79 fields, roster-wide, NOT contradiction-
specific -- `field_signature_not_contradiction_specific.md`) and a TRIGGERED
second-order ascent at the contradiction (3.37x readers / 2.18x regex, controls
at 1.00 / 0.93 -- `second_order_naming.md`). If they are one operation at two
grains, pairs with a larger ambient shift should show a larger ascent. If the
ascent is a separate mechanism, they need not covary.

**THE READING IS ASYMMETRIC AND IS DECLARED BEFORE ANY NUMBER EXISTS.** A
positive correlation is AMBIGUOUS: lineages differ in overall alignment dose
(how hard the aligned checkpoint was pushed), and dose would drive both
quantities without them sharing a mechanism. A null is the DISCRIMINATING
outcome -- two effects installed separately -- bounded by the MDE at n=26,
which is reported because n=26 cannot see a small correlation and a null must
not be read past its power.

**THE TWO VARIABLES ARE MEASURED ON DISJOINT PASSAGES**, or the test is
trivially coupled through shared text:

    AMBIENT   aligned-minus-base interiority composite, CONTROL_A and CONTROL_B
              continuations ONLY (consonant pairs; no contradiction in sight)
    ASCENT    aligned-minus-base rate of ANY second-order marker, exit-free,
              BOTH role ONLY (from `z_second_order_cells.csv`, the committed
              cell table of the regex instrument)

THE COMPOSITE IS THE PUBLISHED SIGNATURE, NOT A CHOICE MADE HERE. Fields enter
if `l2_fields_meta.json` records them as surviving BH on the general effect
(`bh_both`), signed by the sign of the published roster median `d_both`. The
composite is the mean of signed per-pair rate differences over those fields.
Nothing in this script selects fields by looking at the covariance.

RATE DIFFERENCES, NOT RATIOS: marker rates run 0-10% per cell and several base
cells are zero, so a ratio is undefined or explosive exactly where the data are
thinnest. `meta` lexicon only -- one lexicon, 13 fields, no cross-source
summing, per that module's own rule.
"""
import collections
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

RES = os.path.join(ROOT, "meta/M02_frame_exit/results")
CH = "clickhouse"
SEED = 20260813


def fetch(models):
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    q = ("SELECT model, prompt, text FROM malign_logits.gen_sequences "
         "WHERE corpus='f11_l2' AND model IN (%s) FORMAT JSONEachRow"
         % ",".join("'%s'" % esc(m) for m in models))
    pr = subprocess.Popen([CH, "client", "-q", q], stdout=subprocess.PIPE,
                          text=True, bufsize=1 << 20)
    for line in pr.stdout:
        try:
            r = json.loads(line)
        except Exception:
            continue
        yield r["model"], r["prompt"].strip(), r["text"] or ""
    pr.wait()


def main():
    from scipy.stats import spearmanr
    from malign_logits import fields as FL

    rec = json.load(open(os.path.join(ROOT, "data/f11_l2_receipt.json")))
    pairs = [(c["base"], c["aligned"]) for c in rec["complete"]]
    print("%d complete pairs from the receipt" % len(pairs))

    #: the published signature: membership = bh_both, sign = sign(d_both)
    meta = json.load(open(os.path.join(RES, "l2_fields_meta.json")))["results"]
    sig = {r["field"]: (1.0 if r["d_both"] > 0 else -1.0)
           for r in meta if r.get("bh_both")}
    print("signature fields (meta, bh_both): %s"
          % ", ".join("%s%s" % ("+" if s > 0 else "-", f) for f, s in sorted(sig.items())))

    #: prompt -> roles, English groups only, CONTROLS only
    pop = json.load(open(os.path.join(ROOT, "data/f11_l2_population.json")))
    ctrl_prompts = set()
    for p in pop["prompts"]:
        if p.get("lang") != "en":
            continue
        if any(c["role"] in ("control_a", "control_b") for c in p["claims"]):
            ctrl_prompts.add(p["text"].strip())
    print("%d English control prompts" % len(ctrl_prompts))

    models = sorted({m for p in pairs for m in p})
    counts = {m: collections.Counter() for m in models}
    denom = collections.Counter()
    n_pass = 0
    for m, prompt, text in fetch(models):
        if prompt not in ctrl_prompts or not text:
            continue
        c = FL.count(text, source="meta")
        if not c["n_counted"]:
            continue
        n_pass += 1
        denom[m] += c["n_counted"]
        for f, k in c["counts"].items():
            counts[m][f] += k
    print("%s control continuations counted" % format(n_pass, ","))

    def rate(m, f):
        return counts[m].get(f, 0) / max(denom[m], 1)

    #: FL.count(source='meta') keys are BARE field names ('cognition_mental');
    #: the 'meta:' prefix belongs to both_vs_controls_fields.py's OUTPUT
    #: convention, not to count(). The first run of this script prefixed the
    #: lookups, every rate came back 0, and the composite was a CONSTANT --
    #: caught only because scipy warns on constant input. A lookup under the
    #: wrong key returns a confident zero, not an error, so the guard below
    #: refuses to proceed if the signature fields are absent from the counts.
    seen = set().union(*[set(counts[m]) for m in models])
    missing = [f for f in sig if f not in seen]
    if missing:
        raise SystemExit("REFUSING: signature fields absent from count() keys: %s\n"
                         "seen keys sample: %s" % (missing, sorted(seen)[:8]))
    ambient = {}
    for b, a in pairs:
        if denom[b] == 0 or denom[a] == 0:
            continue
        ambient[(b, a)] = float(np.mean(
            [s * (rate(a, f) - rate(b, f)) for f, s in sig.items()]))

    #: ASCENT from the committed cell table
    hdr = None
    cells = collections.defaultdict(lambda: [0, 0])   # model -> [hits, n]
    for ln in open(os.path.join(RES, "z_second_order_cells.csv"), encoding="utf-8"):
        p = ln.rstrip("\n").split(",")
        if hdr is None:
            hdr = {c: i for i, c in enumerate(p)}
            continue
        if p[hdr["role"]] != "BOTH":
            continue
        m = p[hdr["model"]]
        cells[m][0] += int(p[hdr["ANY_SO|exitfree"]])
        cells[m][1] += int(p[hdr["n_exitfree"]])
    ascent = {}
    for b, a in pairs:
        if b in cells and a in cells and cells[b][1] and cells[a][1]:
            ascent[(b, a)] = (cells[a][0] / cells[a][1]
                              - cells[b][0] / cells[b][1])

    keys = sorted(set(ambient) & set(ascent))
    x = np.array([ambient[k] for k in keys])
    y = np.array([ascent[k] for k in keys])
    n = len(keys)
    rho = float(spearmanr(x, y).statistic)
    #: permutation p, and the MDE at this n so a null cannot be over-read
    rng = np.random.default_rng(SEED)
    perm = np.array([spearmanr(x, rng.permutation(y)).statistic
                     for _ in range(10000)])
    pv = float((np.abs(perm) >= abs(rho)).mean())
    mde = float(np.quantile(np.abs(perm), 0.95))
    print("\n  pairs with both quantities  %d" % n)
    print("  AMBIENT composite (controls only)   median %+.5f  [%+.5f, %+.5f]"
          % (float(np.median(x)), float(x.min()), float(x.max())))
    print("  ASCENT  (BOTH role, exit-free)      median %+.5f  [%+.5f, %+.5f]"
          % (float(np.median(y)), float(y.min()), float(y.max())))
    print("\n  Spearman(ambient, ascent) = %+.3f   permutation p = %.3f" % (rho, pv))
    print("  MDE at n=%d: |rho| >= %.3f detectable at alpha 0.05" % (n, mde))
    print("  declared reading: positive = ambiguous (dose); null = dissociation, "
          "bounded by the MDE")

    out = {"n_pairs": n, "rho": rho, "p_perm": pv, "mde_abs_rho_05": mde,
           "signature": {f: s for f, s in sig.items()},
           "ambient": {"%s>%s" % k: v for k, v in ambient.items()},
           "ascent": {"%s>%s" % k: v for k, v in ascent.items()}}
    p = os.path.join(RES, "ambient_vs_ascent.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
