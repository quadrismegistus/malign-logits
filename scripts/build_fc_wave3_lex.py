#!/usr/bin/env python
"""build_fc_wave3_lex.py — the LEXICAL ARM manifest. RECOVERED, NOT REWRITTEN.

    scripts/build_fc_wave3_lex.py --show          print the selection
    scripts/build_fc_wave3_lex.py --out-suffix X  write data/fc_wave3_lex_{t}X.json
    scripts/build_fc_wave3_lex.py --verify        reproduce the shipped manifest

WHY THIS FILE EXISTS NOW AND NOT THEN. The manifests that ran wave 3
(`data/fc_wave3_lex_{mps,vast}.json`, 7 Aug 00:37, 29 pairs, 17,066 units) were
produced by an INLINE HEREDOC. No file was deleted -- one was never written.
The rule survived only in a chat transcript, and the manifest's own `producer`
field names `scripts/build_fc_manifest.py`, which contains no part of it.

**AN ARTIFACT THAT NAMES THE WRONG PRODUCER IS WORSE THAN ONE THAT NAMES NONE.**
Asked to check the rule, I read that field, found no lexical arm in it, and
reported that the producer did not exist -- then reimplemented the rule from
the manifest's one-line `note` and got 96.8% of fallers and 81.6% of risers.
The gap was entirely the function-word list: the note says "OPEN-CLASS" and
there are two open-class lists in this repo. With the right one the recovered
rule reproduces the shipped manifest at 766/766 EXACT on the four pairs
checked -- so the 18% was my substitution, not the producer's uncertainty.

    fc_analyse.FUNCTION_WORDS   what wave 3 ran. Fixed before the split that
                                motivated the arm, and not tuned to it ([4754]).
    f13_draw_relation_items.FUNC  a DIFFERENT list for a different question.
                                Swapping them moves ~18% of riser picks.

THE RULE, exactly as it ran:

    F = fallers, residual out, w.lower() not in FUNCTION_WORDS
    R = risers,  residual out, w.lower() not in FUNCTION_WORDS
    a site needs BOTH; cells with neither are counted (`nolex`) not dropped
    faller = min(F, key=delta)          the biggest DROP
    riser  = max(R, key=excess)         BY EXCESS -- ranking risers by delta
                                        re-introduces what the null removes

**NOTE `min` ON delta AND `max` ON excess.** They look inconsistent and are
not: delta is negative for a faller, so `min` is the largest fall.

SOURCE MANIFEST is `data/fc_wave2_top1_{mps,vast}.json` -- this arm RE-PICKS
the words at sites that manifest already chose; it does not re-select sites.
So its site count is inherited and its prompt population is
`beam_sample_105.csv` (210 prompts), by way of pass 2.

WHAT THE ARM IS FOR ([4752]/[4754]): the marked-vs-unmarked withdrawal effect
was +0.00773 (detected) on open-class top fallers and -0.00061 (null) on
closed-class, so the pooled +0.00294 was a lexical effect diluted by a
function-word half carrying nothing. This is an ADDITIONAL arm -- the
magnitude-ranked mover stays as the neutral reference.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "meta", "M01_displacement", "scripts"))

TWP = dict(dict_sha="b16011275c42955c", mode="raw", rule_version=3, theta=0.001)
NOTE = ("LEXICAL ARM: largest OPEN-CLASS faller and riser per site, chosen "
        "regardless of overall rank. A DECLARED SELECTION ON WORD CLASS and "
        "an ADDITIONAL arm -- the magnitude-ranked mover stays as the neutral "
        "reference. Motivated by a measured 2.6x: the marked-vs-unmarked "
        "withdrawal effect was +0.00773 (detected) on open-class top fallers "
        "and -0.00061 (null) on closed-class, so the pooled +0.00294 was a "
        "lexical effect diluted by a function-word half that carries nothing.")


def dist(st, mid, prompt):
    from m05_sites import prepare
    k = dict(TWP); k["model"] = mid; k["prompt"] = prompt
    try:
        v = st[k]
    except Exception:
        return None
    r = v.get("rows") if isinstance(v, dict) else None
    if not r:
        return None
    o, pr = prepare(r)
    return {w: pr[w] for w in o}


def pick(st, base, aligned, prompt):
    """(faller, riser) or None. THE RULE -- one implementation, imported by callers."""
    from fc_analyse import FUNCTION_WORDS as FW
    from malign_logits.movement import movement, CANONICAL, RESIDUAL_KEY
    P, Q = dist(st, base, prompt), dist(st, aligned, prompt)
    if not P or not Q:
        return None
    mv = movement(P, Q, CANONICAL)
    F = [w for w in mv.fallers if w != RESIDUAL_KEY and w.lower() not in FW]
    R = [w for w in mv.risers if w != RESIDUAL_KEY and w.lower() not in FW]
    if not F or not R:
        return None
    key = mv.excess if mv.rule.null_test else mv.delta
    return (min(F, key=lambda w: mv.delta.get(w, 0.0)),
            max(R, key=lambda w: key.get(w, 0.0)))


def build(src_path, st):
    src = json.load(open(src_path))
    out, stats = [], collections.Counter()
    for p in src["pairs"]:
        sites = []
        for s in p["sites"]:
            got = pick(st, p["base"], p["aligned"], s["prompt"])
            if got is None:
                stats["nolex_or_no_twp"] += 1
                continue
            f, r = got
            stats["site"] += 1
            sites.append({"prompt": s["prompt"], "stem": s.get("stem"),
                          "member": s.get("member"),
                          "fallers": [f], "risers": [r]})
        if sites:
            q = dict(p); q["sites"] = sites
            q["n_forced_per_checkpoint"] = 2 * len(sites)
            out.append(q)
    return src, out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="re-pick every site in the SHIPPED manifests and "
                         "require an exact match")
    ap.add_argument("--out-suffix", default=None,
                    help="write data/fc_wave3_lex_{target}SUFFIX.json; "
                         "REFUSES to overwrite")
    a = ap.parse_args()

    from malign_logits.cache import get_cache
    st = get_cache()._stash("true_word_probs")

    if a.verify:
        #: **THE KNOWN-ANSWER COLUMN, AND IT IS THE POINT OF THE FILE.** A
        #: recovered rule that does not reproduce its own artifact is a new
        #: rule wearing the old one's name.
        tot = collections.Counter()
        for tgt in ("mps", "vast"):
            f = os.path.join(ROOT, "data", "fc_wave3_lex_%s.json" % tgt)
            if not os.path.exists(f):
                print("%-6s SHIPPED MANIFEST ABSENT: %s" % (tgt, f))
                continue
            d = json.load(open(f))
            ok = bad = 0
            for p in d["pairs"]:
                for s in p["sites"]:
                    got = pick(st, p["base"], p["aligned"], s["prompt"])
                    if got and [got[0]] == s["fallers"] and [got[1]] == s["risers"]:
                        ok += 1
                    else:
                        bad += 1
            tot["ok"] += ok; tot["bad"] += bad
            print("%-6s %2d pairs   exact %5d   mismatch %4d" % (tgt, len(d["pairs"]), ok, bad))
        n = tot["ok"] + tot["bad"]
        print("\nTOTAL exact %d / %d = %.4f" % (tot["ok"], n, tot["ok"] / n if n else 0))
        sys.exit(0 if tot["bad"] == 0 else 1)

    for tgt in ("mps", "vast"):
        src_path = os.path.join(ROOT, "data", "fc_wave2_top1_%s.json" % tgt)
        if not os.path.exists(src_path):
            print("%-6s SOURCE ABSENT: %s" % (tgt, os.path.relpath(src_path, ROOT)))
            continue
        src, pairs, stats = build(src_path, st)
        tot = sum(p["n_forced_per_checkpoint"] for p in pairs)
        print("%-6s %2d pairs | %5d sites | forced per checkpoint %6d | x2 = %6d"
              % (tgt, len(pairs), stats["site"], tot, 2 * tot))
        if a.out_suffix:
            out = os.path.join(ROOT, "data",
                               "fc_wave3_lex_%s%s.json" % (tgt, a.out_suffix))
            if os.path.exists(out):
                sys.exit("REFUSING to overwrite %s" % out)
            json.dump(dict(src, pairs=pairs, target="wave3-lexical-%s" % tgt,
                           arms=["force_faller", "force_riser"], note=NOTE,
                           producer="scripts/build_fc_wave3_lex.py"),
                      open(out, "w"), indent=1)
            print("       wrote %s" % os.path.relpath(out, ROOT))
    if not a.out_suffix:
        print("\n(--out-suffix X to write; --verify to check against what shipped)")


if __name__ == "__main__":
    main()
