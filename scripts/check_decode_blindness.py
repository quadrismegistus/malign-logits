#!/usr/bin/env python
"""Can each arm's text instrument SEE the words it counts? Run when the population changes.

    scripts/check_decode_blindness.py
    scripts/check_decode_blindness.py --words assistant user system think
    scripts/check_decode_blindness.py --trust-remote-code     # test the 10

THE DEFECT THIS EXISTS TO CATCH. `assistant` is a SPECIAL TOKEN in the Falcon
Mamba vocabularies, and every one of the eleven `decode` call sites in this
repo passes `skip_special_tokens=True` -- including `core.py:311`, the
generation path that wrote the text E-ASSIST-AMBIENT regexes over. So for those
models the counter's own search term is deleted before it can be matched.

    falcon-mamba-7b-instruct    'assistant' -> ''            ALIGNED
    Falcon3-Mamba-7B-Instruct   'assistant' -> ''            ALIGNED
    falcon-mamba-7b             'assistant' -> 'assistant'   base
    Falcon3-Mamba-7B-Base       'assistant' -> 'assistant'   base

**The aligned arms lose it and the base arms keep it**, and the quantity is
aligned-minus-base, so the contrast was biased against its own effect. One
sentence of E-ASSIST-AMBIENT was withdrawn on this ([5325]-[5328]); its 17/18
sign test at p 7.2e-05 does not rest on those pairs and stands.

THREE RULES, EACH PAID FOR
--------------------------

**1. TEST THE BEHAVIOUR, NOT THE DECLARATION.** A scan over
`all_special_tokens` reported 1 affected arm of 54. The behaviour reports more:
`Falcon3-Mamba-7B-Instruct` drops `assistant` and is NOT in that list. The
declaration was wrong in the reassuring direction. Only
`decode(encode(w), skip_special_tokens=True) == w` is the fact.

**2. UNTESTED IS NOT CLEAN.** Ten arms fail to load without
`trust_remote_code`; they are reported in their own bucket and never counted as
passing. A load failure scoring as a pass is how a scan reports coverage it
never had. The honest headline is "N affected, M clean, K unknown", never
"N of N+M+K".

**3. SYMMETRIC BLINDNESS IS MORE DANGEROUS THAN ASYMMETRIC**, which inverts the
usual intuition. The Mamba case biased a contrast and was findable because the
two halves disagreed. **Pharia loses all four words on BOTH arms**, so it
returns a clean symmetric zero -- and E-ASSIST-AMBIENT excludes ties from its
sign test, so that pair would have been discarded as agreement-at-zero rather
than flagged as unmeasurable. *A defect that produces a tie is invisible to any
test that discards ties.* Pharia is not in the generations stash today, which
is the only reason it is latent rather than live.

WHY IT IS STANDING RATHER THAN ONE-OFF. The exposure is a fact about the
CURRENT population, not about the roster. Pharia's arms went into the twp
census on 2026-08-10; a generation run would move it from LATENT to LIVE
without anyone touching the finding. Run this when the population changes.
"""
import argparse
import json
import os
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

#: the words the text instruments actually search for. Keep in step with
#: `meta/M02_frame_exit/scripts/eassist_ambient.py`'s LOOSE and STRICT patterns.
DEFAULT_WORDS = ("assistant", "user", "system", "think")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="+", default=list(DEFAULT_WORDS))
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="also test the arms that need custom code")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer

    pairs = json.load(open(os.path.join(ROOT, "data", "base_aligned_pairs.json")))

    #: WHICH ARMS ARE IN THE POPULATION. An arm that is blind but absent is
    #: LATENT; the same arm once generations exist for it is LIVE. That
    #: distinction is the whole reason this is re-runnable.
    from malign_logits.cache import CacheManager
    present = set()
    try:
        for k in CacheManager()._stash("generations"):
            m = k.get("model")
            if m:
                present.add(m)
    except Exception as e:
        print("WARNING: could not read the generations stash (%s). Every "
              "finding below is reported as UNKNOWN-population." % type(e).__name__)
        present = None

    lost, untested, clean = [], [], 0
    for p in pairs:
        for role in ("base", "aligned"):
            m = p[role]
            try:
                t = AutoTokenizer.from_pretrained(
                    m, trust_remote_code=a.trust_remote_code)
            except Exception as e:
                untested.append((m, role, type(e).__name__))
                continue
            bad = []
            for w in a.words:
                try:
                    if t.decode(t.encode(w, add_special_tokens=False),
                                skip_special_tokens=True) != w:
                        bad.append(w)
                except Exception:
                    pass
            if bad:
                live = None if present is None else (m in present)
                lost.append((m, role, bad, live))
            else:
                clean += 1

    n = sum(1 for p in pairs for _ in ("base", "aligned"))
    print("words tested   %s" % ", ".join(a.words))
    print("arms           %d" % n)
    print()
    print("  AFFECTED %d   CLEAN %d   UNTESTED %d" % (len(lost), clean, len(untested)))
    print("  (never report these as one denominator: UNTESTED is not CLEAN)")
    if lost:
        print("\nARMS THAT CANNOT SEE A WORD THEY ARE COUNTED FOR:")
        for m, role, bad, live in sorted(lost):
            tag = "LIVE" if live else ("LATENT" if live is False else "UNKNOWN")
            print("   %-6s %-44s %-8s loses %s" % (tag, m[:44], role, bad))
        #: a pair blind on BOTH arms returns a symmetric zero, which a
        #: tie-discarding test drops silently. Called out separately.
        by_model = {m: bad for m, _r, bad, _l in lost}
        for p in pairs:
            if p["base"] in by_model and p["aligned"] in by_model:
                print("   ** BOTH ARMS BLIND: %s -- returns a symmetric zero, which a "
                      "tie-discarding test discards as agreement **"
                      % p["base"].split("/")[-1])
    if untested:
        print("\nUNTESTED (load failed -- NOT clean, NOT counted as passing):")
        for m, role, e in untested:
            print("   %-46s %-8s %s" % (m[:46], role, e))
        print("   re-run with --trust-remote-code to test these")

    live_n = sum(1 for _m, _r, _b, l in lost if l)
    print()
    print("VERDICT: %s" % ("no LIVE exposure" if not live_n else
                           "*** %d LIVE arm(s): a text instrument cannot see a "
                           "word it counts ***" % live_n))
    return 1 if live_n else 0


if __name__ == "__main__":
    sys.exit(main())
