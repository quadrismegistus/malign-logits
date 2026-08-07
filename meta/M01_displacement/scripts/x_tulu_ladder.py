"""WHICH EDGE does the erogenous-zone movement? SFT, DPO, RLVR, and the ablations.

    uv run --with lemminflect python x_tulu_ladder.py

Section 3d shows that at `suck his ___` alignment moves off the erogenous zones
and onto the extremities, over 44 base>aligned pairs. **That contrast collapses
the whole post-training pipeline into one arrow.** RH's question: which stage
actually does it, and does it survive the Tulu SFT data ablations.

Findings U answered the first question for the M01 population -- SFT does the
cutting and the later rungs mostly stop -- and found that removing the safety
split leaves the operation at 90% strength. **Neither result was ever run on
these prompts**, and X's scene-local movement is not the same measurement as
M01's, so it does not inherit them.

THE LINEAGE. One pretraining run, every rung a real checkpoint, all nine with
true_word_probs at this prompt already. Nothing here needs a GPU.

    meta-llama/Llama-3.1-8B
      |-- sft_of --> Tulu-3-8B-SFT --- dpo_of --> Tulu-3-8B-DPO --- rlvr_of --> Tulu-3.1-8B
      |                |
      |                +-- data_ablation_of --> SFT-no-safety-data
      |                +-- data_ablation_of --> SFT-no-math-data
      |                +-- data_ablation_of --> SFT-no-persona-data
      |                +-- data_ablation_of --> SFT-no-wildchat-data
      |
      +-- dpo_of --> Llama-3.1-8B-Instruct        (the OTHER recipe, same base)

**EDGES, NOT `position`.** malign found that `position` and `stage` disagree in
the registry: three of the four SFT ablation arms carry `position=superego`
though they are SFT checkpoints, and eight rows carry an empty position. A rule
keyed on `position` is right here only by accident. The edges are what the
lineage actually asserts, so the ladder is built from them and each comparison
names its own parent.

THE PER-EDGE STATISTIC, because 3d's counting rule needs more than one pair.
For each word present in BOTH arms of an edge, CANONICAL says faller, riser or
neither. Per class that gives a net rate, (risers - fallers) / present, and

    EFFECT = net rate of DIGITS_LIMBS  -  net rate of GENITALS + BREAST

Positive means the edge moves probability off the zones and onto the
extremities, which is the 3d direction. A rate rather than a count because the
classes are different sizes and coverage differs by arm.

**The mean-delta version is printed beside it and they are not the same test.**
The rate ignores magnitude; the delta ignores how many words moved. A finding
that only appears in one of them is a finding about that summary.
"""
import collections
import inspect
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

TWP = dict(dict_sha="b16011275c42955c", mode="raw", rule_version=3, theta=0.001)
TAG = "sexual_explicit_1"
SPEC = os.path.join(CAMP, "results", "x_bodypart_classes.json")
L = "meta-llama/Llama-3.1-8B"
T = "allenai/Llama-3.1-Tulu-3-8B-"

#: (parent, child, relation, label). Parent named explicitly at every edge so a
#: reader never has to infer which comparison a row is.
LADDER = [
    (L, T + "SFT", "sft_of", "base -> SFT"),
    (T + "SFT", T + "DPO", "dpo_of", "SFT -> DPO"),
    (T + "DPO", "allenai/Llama-3.1-Tulu-3.1-8B", "rlvr_of", "DPO -> RLVR"),
]
CUMULATIVE = [
    (L, T + "SFT", "", "base -> SFT"),
    (L, T + "DPO", "", "base -> DPO"),
    (L, "allenai/Llama-3.1-Tulu-3.1-8B", "", "base -> RLVR"),
    (L, "meta-llama/Llama-3.1-8B-Instruct", "", "base -> Instruct (other recipe)"),
]
ABLATIONS = [
    (L, T + "SFT", "", "full SFT"),
    (L, T + "SFT-no-safety-data", "data_ablation_of", "SFT minus safety"),
    (L, T + "SFT-no-math-data", "data_ablation_of", "SFT minus math"),
    (L, T + "SFT-no-persona-data", "data_ablation_of", "SFT minus persona"),
    (L, T + "SFT-no-wildchat-data", "data_ablation_of", "SFT minus wildchat"),
]


def edge_effect(st, prompt, parent, child, inv, zones, per):
    from malign_logits.movement import movement, CANONICAL, RESIDUAL_KEY
    from m05_sites import prepare

    def rows(m):
        k = dict(TWP); k["model"] = m; k["prompt"] = prompt
        try:
            v = st[k]
        except Exception:
            return None
        return v.get("rows") if isinstance(v, dict) else None

    rp, rc = rows(parent), rows(child)
    if not rp or not rc:
        return None
    op, pp = prepare(rp)
    oc, pc = prepare(rc)
    mv = movement({w: pp[w] for w in op}, {w: pc[w] for w in oc}, CANONICAL)
    F = {w for w in mv.fallers if w != RESIDUAL_KEY}
    R = {w for w in mv.risers if w != RESIDUAL_KEY}
    out = {}
    for name, classes in (("zone", zones), ("digit", per)):
        #: PRESENT IN THE PARENT is the denominator. A word absent from the
        #: parent cannot fall, so scoring it would make a narrower child look
        #: like a milder edge.
        ws = [w for w in op if inv.get(w) in classes]
        if not ws:
            return None
        rise = sum(1 for w in ws if w in R)
        fall = sum(1 for w in ws if w in F)
        out[name] = dict(n=len(ws), rise=rise, fall=fall,
                         rate=(rise - fall) / float(len(ws)),
                         delta=sum(mv.delta.get(w, 0.0) for w in ws) / float(len(ws)))
    out["effect"] = out["digit"]["rate"] - out["zone"]["rate"]
    out["effect_delta"] = out["digit"]["delta"] - out["zone"]["delta"]
    return out


def main():
    from malign_logits.cache import get_cache
    from malign_logits import experiments as E

    spec = json.load(open(SPEC))
    inv = {w: c for c, ws in spec["classes"].items() for w in ws}
    zones, per = spec["_zones"], spec["_periphery"]
    st = get_cache()._stash("true_word_probs")
    src = inspect.getsource(E)
    P = {k: v for k, v in re.findall(
        r'"((?:sexual|violence)_(?:liminal|explicit)_\d+)":\s*"([^"]+)"', src) if v.isascii()}
    prompt = P[TAG]
    print("%s  %r\n" % (TAG, prompt))

    def block(title, edges, note=""):
        print(title)
        if note:
            print("   %s" % note)
        print("   %-34s %-14s %18s %18s %9s %9s"
              % ("comparison", "relation", "zones rise/fall/n", "digits rise/fall/n",
                 "EFFECT", "by delta"))
        for parent, child, rel, lab in edges:
            r = edge_effect(st, prompt, parent, child, inv, zones, per)
            if r is None:
                print("   %-34s %-14s  no data at this prompt" % (lab, rel))
                continue
            z, d = r["zone"], r["digit"]
            print("   %-34s %-14s %6d/%d/%-8d %6d/%d/%-8d %+9.3f %+9.4f"
                  % (lab, rel, z["rise"], z["fall"], z["n"],
                     d["rise"], d["fall"], d["n"], r["effect"], r["effect_delta"]))
        print()

    block("EACH EDGE ON ITS OWN. Which stage does the work?", LADDER,
          "parent is the PREVIOUS RUNG, so these do not accumulate")
    block("CUMULATIVE FROM THE BASE. Where does the effect stand after each stage?", CUMULATIVE,
          "parent is Llama-3.1-8B throughout; the last row is the same base, a different recipe")
    block("THE SFT DATA ABLATIONS. Does removing a data split remove the effect?", ABLATIONS,
          "parent is Llama-3.1-8B throughout, so every row is base -> some SFT")

    #: THE STATISTIC ON CLASS PAIRS WITH NO HYPOTHESIS ATTACHED. If it returns
    #: the same magnitude for arbitrary classes it is measuring redistribution
    #: rather than the zone contrast. Argument order is (reference, target) and
    #: the sign is target minus reference -- stated because getting it backwards
    #: in a throwaway check made the last row look like a refutation.
    print("ARTEFACT CHECK: same statistic, other class pairs.  sign = target minus reference")
    CP = [(zones, per, "zones -> digits          THE CLAIM"),
          (["MOUTH_FACE"], ["OTHER"], "mouth/face -> other      no hypothesis"),
          (zones, ["MOUTH_FACE"], "zones -> mouth/face      small if mouth is a zone"),
          (["OTHER"], per, "other -> digits          digits should still gain")]
    rows = LADDER + [r for r in ABLATIONS if r[2]]
    print("   %-40s%s" % ("", "".join("%-11s" % l.split()[-1][:10] for _, _, _, l in rows)))
    for a, b, lab in CP:
        vals = []
        for parent, child, rel, l in rows:
            r = edge_effect(st, prompt, parent, child, inv, a, b)
            vals.append("%+.3f" % r["effect"] if r else "   -   ")
        print("   %-40s%s" % (lab, "".join("%-11s" % v for v in vals)))

    print()
    print("READ WITH CARE, AND THE LIMIT IS THE DENOMINATOR, NOT THE LINEAGE.")
    print("Classes present in the parent: zones %d, digits %d, mouth/face %d, other %d."
          % (len([w for w in _parent_words(st, prompt) if inv.get(w) in zones]),
             len([w for w in _parent_words(st, prompt) if inv.get(w) in per]),
             len([w for w in _parent_words(st, prompt) if inv.get(w) == "MOUTH_FACE"]),
             len([w for w in _parent_words(st, prompt) if inv.get(w) == "OTHER"])))
    print("So the digit rate moves in steps of 1/6 and the zone rate in steps of 1/13:")
    print("**the gaps between edges above are one or two words and are not resolvable.**")
    print("What IS resolvable is presence, and the effect is present at every edge and")
    print("every ablation. To rank the edges you need more PROMPTS, not more models.")


def _parent_words(st, prompt):
    from m05_sites import prepare
    k = dict(TWP); k["model"] = L; k["prompt"] = prompt
    v = st[k]
    op, _ = prepare(v.get("rows"))
    return op


if __name__ == "__main__":
    main()
