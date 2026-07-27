#!/usr/bin/env python3
"""Post-hoc characterisation of the failed gate: undetected, or excluded?

NOT a new test of the convention hypothesis, and no chain-pair `D` is computed
here -- the standing rule that a failed gate forbids the chain-pair sign test is
untouched. This asks a question about the GATE'S OWN THREE NUMBERS, which are
already public in the gate output.

The distinction matters because the ruled cell's power was its disclosed weak
point: 0.804 as registered, 0.759 shown, so roughly a fifth to a quarter of
failures would be expected even on a live instrument. That is an argument about
the DESIGN. But the design's alternative hypothesis is a specific number -- the
anchor, each corpus's median chain-pair MDE, log(1.19) = 0.1740 for hh -- and
the observed markers can be tested directly against it.

If the observed effects are merely small and noisy, the failure is ambiguous and
the power caveat governs. If they are significantly BELOW the anchor, then the
alternative the design was powered against is excluded on the evidence, and the
power caveat is answered rather than merely disclosed.

The pooled estimate assumes the three markers share a common effect. They are
one per construct and the constructs were chosen to be independent, so this is
an assumption and is labelled as one; each marker is therefore also reported
individually, and the conclusion does not depend on pooling.
"""
import json, math

ANCHOR = math.log(1.19)          # hh median chain-pair MDE -- the powered-against effect
MARKERS = [("require->prefer", 0.0178, 0.0370),
           ("entirely->mostly", 0.0332, 0.0544),
           ("angry->concerned", 0.0234, 0.0541)]


def norm_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def main():
    print(f"anchor (the alternative the gate was powered against) = {ANCHOR:.4f}\n")
    print(f"{'marker':20s}{'D':>9s}{'SE':>8s}{'z vs 0':>9s}{'z vs anchor':>13s}"
          f"{'p(<=obs|anchor)':>17s}")
    rows = []
    for n, d, s in MARKERS:
        za = (d - ANCHOR) / s
        p = norm_cdf(za)
        print(f"{n:20s}{d:>9.4f}{s:>8.4f}{d / s:>9.2f}{za:>13.2f}{p:>17.5f}")
        rows.append(dict(marker=n, D=d, se=s, z_vs_zero=d / s, z_vs_anchor=za, p=p))

    w = [1 / s ** 2 for _, _, s in MARKERS]
    dbar = sum(wi * d for wi, (_, d, _) in zip(w, MARKERS)) / sum(w)
    sebar = math.sqrt(1 / sum(w))
    za = (dbar - ANCHOR) / sebar
    hi = dbar + 1.96 * sebar
    print(f"\npooled (inverse variance; assumes a common effect across constructs)")
    print(f"   D = {dbar:+.4f} +/- {sebar:.4f}")
    print(f"   vs zero    z = {dbar / sebar:+.2f}   -- indistinguishable from no effect")
    print(f"   vs anchor  z = {za:+.2f}   p = {norm_cdf(za):.3g}")
    print(f"   95% upper bound on the effect = {hi:.4f} = {hi / ANCHOR:.2f}x the anchor")
    print("\nThe gate did not merely fail to detect an effect at the anchor scale.")
    print("It EXCLUDES one. Every marker is individually below the anchor at")
    print("p < 0.005, so the conclusion does not rest on the pooling assumption.")

    json.dump(dict(anchor=ANCHOR, markers=rows, pooled=dict(
        D=dbar, se=sebar, z_vs_anchor=za, p=norm_cdf(za), upper95=hi,
        upper95_over_anchor=hi / ANCHOR)),
        open("data/preference_corpus_exclusion.json", "w"), indent=1)
    print("\n-> data/preference_corpus_exclusion.json")


if __name__ == "__main__":
    main()
