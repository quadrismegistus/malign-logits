"""L3 PILOT: where does the BOTH representation sit between its poles, per layer?

Two checkpoints, three prompts, six forward passes. OLMo-2-0425-1B and its DPO
because it is 1B, cached, loads on MPS, and its family carries the LARGEST ratio
shift in F11's table (+0.22; OLMo-tiny +0.28). If a layerwise signal exists
anywhere it should be here, and its absence here is informative rather than
inconclusive.

THE MEASURE. F11's construct at representation level. With h_A, h_B, h_AB the
final-position residual at a given layer for POLE_A / POLE_B / BOTH:

    t = (h_AB - h_B) . (h_A - h_B) / |h_A - h_B|^2

    t ~ 0.5   BOTH sits between the poles          SUPERPOSITION
    t ~ 1     BOTH sits at A                       RESOLVED to A
    t ~ 0     BOTH sits at B                       RESOLVED to B

`resid` is the component of h_AB off that axis, as a fraction of the pole gap —
large resid means BOTH is not on the A-B line at all, which is what FRAME EXIT
would look like geometrically: not between the poles, and not at either.

WHAT F05'S RERUN PREDICTS. The displacement operation was found
"final-layer/unembedding-uniform in 13/17 families". If that generalises, t
should track together in base and aligned across every layer and separate only
at the end -- alignment as a readout change, the representation preserved.
The alternative is separation at depth. Either is a result; they say opposite
things about what alignment is.

NOT A FINDING. Two checkpoints, one triplet, no controls. This decides whether
L3 is worth a flag on the running fleet, nothing else.

## WHAT IT RETURNED, 8 Aug 2026 (docket [5141].3, registered exploratory [5142])

    layer     BASE t     DPO t     shift
      1       -0.053     0.157    +0.210
      6        0.443     0.229    -0.214
      7        0.416     0.180    -0.236   <- largest
      8        0.424     0.197    -0.226
     12        0.462     0.463    +0.001
     16        0.444     0.482    +0.038   <- final

    mean |shift| interior : 0.1032
    |shift| at final layer: 0.0383

The interior shift is 2.7x the final-layer shift. The base holds t ~ 0.42-0.46 --
the midpoint, superposition -- stably from layer 3 up. The DPO model tracks it to
layer 5, drops to 0.18 at layer 7 (hauled toward the hate pole), and climbs back
to 0.48 by the top. **The arms diverge at mid-depth and RECONVERGE before the
output**, which is consistent with F05's unembedding-uniform rerun and invisible
to it.

Registrar's sentence for it, if the shape survives a real L3: *alignment's
pole-pull is an INTERIOR event that the output layer repairs* -- which would say
the campaign's output-grain instruments are not wrong but BLIND, since every one
of them reads at or after the unembedding.

## THE DEFECT IN THIS MEASURE, WHICH A REAL L3 MUST FIX

`resid` runs near 1.0 in BOTH arms: the BOTH point sits about as far off the
pole axis as the poles sit from each other. So `t` is a projection onto a line
the point is not really on, and reading it as "between the poles" overstates
what a scalar on that axis can carry. The residual is the geometric signature of
FRAME EXIT -- not at either pole and not between them -- and it belongs in the
readout as a measure, not divided out as a nuisance.

## STATUS

Pilot only. Superseded by the fleet L3 if the residual capture lands (RH ordered
the flag at [5141]; registrar ruled it additive and non-material at [5142], with
a per-record `residuals_present` stamp because the roster splits at the ~40
checkpoints completed before the flag).
"""
import os
import sys

import numpy as np
import torch

os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

PAIR = [("base", "allenai/OLMo-2-0425-1B"), ("DPO", "allenai/OLMo-2-0425-1B-DPO")]
CELLS = {
    "POLE_A": "She loved him deeply and wanted to",
    "POLE_B": "She hated him deeply and wanted to",
    "BOTH":   "She loved him and hated him and wanted to",
}
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def resid_by_layer(model, tok, prompt):
    """Final-position hidden state at every layer. (n_layers+1, d_model)."""
    ids = tok.encode(prompt, return_tensors="pt").to(DEV)
    #: ROUND-TRIP, the same guard the fleet uses. A silently mangled prompt
    #: would produce a perfectly plausible geometry for the wrong sentence.
    back = tok.decode(ids[0], skip_special_tokens=True)
    if back.strip() != prompt.strip():
        raise SystemExit("round-trip failed: %r -> %r" % (prompt, back))
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states]).numpy()


def main():
    res = {}
    for arm, mid in PAIR:
        print("loading %s ..." % mid, flush=True)
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(DEV).eval()
        res[arm] = {k: resid_by_layer(mdl, tok, p) for k, p in CELLS.items()}
        del mdl
        if DEV == "mps":
            torch.mps.empty_cache()

    nL = res["base"]["BOTH"].shape[0]
    print("\nlayers (incl. embedding): %d   d_model: %d\n" % (nL, res["base"]["BOTH"].shape[1]))
    print("  t = where BOTH sits on the POLE_B -> POLE_A axis   (0.5 = between, 1 = at A, 0 = at B)")
    print("  resid = distance off that axis, as a fraction of the pole gap\n")
    print("  %5s | %-22s | %-22s | %s" % ("layer", "BASE  t     resid", "DPO   t     resid", "t shift"))
    print("  " + "-" * 76)
    rows = []
    for L in range(nL):
        line = []
        for arm in ("base", "DPO"):
            hA, hB, hAB = (res[arm][k][L] for k in ("POLE_A", "POLE_B", "BOTH"))
            ax = hA - hB
            n2 = float(ax @ ax)
            #: LAYER 0 IS DEGENERATE BY CONSTRUCTION, not by defect. All three
            #: prompts END IN THE SAME TOKEN ("to"), so before any attention the
            #: final-position embedding is identical across cells and the pole
            #: axis has zero length. That it is EXACTLY zero is a check on the
            #: measure: anything else at layer 0 would mean the cells differ
            #: somewhere they cannot.
            if n2 == 0.0:
                line.append((float("nan"), float("nan")))
                continue
            t = float((hAB - hB) @ ax) / n2
            off = (hAB - hB) - t * ax
            line.append((t, float(np.linalg.norm(off)) / float(np.linalg.norm(ax))))
        rows.append((L, line[0][0], line[0][1], line[1][0], line[1][1]))
        mark = ""
        if L == nL - 1:
            mark = "  <- final"
        print("  %5d | %8.3f  %9.2f | %8.3f  %9.2f | %+7.3f%s"
              % (L, line[0][0], line[0][1], line[1][0], line[1][1],
                 line[1][0] - line[0][0], mark))

    a = np.array(rows)[1:]   #: drop the degenerate embedding row
    print("\n  mean |t shift| over interior layers: %.4f" % np.abs(a[:-1, 3] - a[:-1, 1]).mean())
    print("  |t shift| at the FINAL layer      : %.4f" % abs(a[-1, 3] - a[-1, 1]))
    print("  mean resid, base / DPO            : %.2f / %.2f" % (a[:, 2].mean(), a[:, 4].mean()))
    print("\n  READ: if the final-layer shift dwarfs the mean interior shift, the")
    print("  representation is preserved and alignment moves the READOUT (F05's")
    print("  rerun generalising). If interior layers shift comparably, it does not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
