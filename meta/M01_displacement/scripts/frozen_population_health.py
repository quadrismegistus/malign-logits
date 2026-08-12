#!/usr/bin/env python
"""Is M01's shared population rule still satisfiable? As of 2026-08-12: NO.

    uv run python frozen_population_health.py

`m01_concentration.frozen_population()` derives the model population by a RULE
rather than a stored list, deliberately:

    prompts = every ACTIVE prompt
    models  = every model whose true_word_probs cover ALL of them

Its docstring defends this: "A population frozen as a COUNT goes stale ... So
the rule is the artifact and the digest is the check on it."

**THE RULE IS MONOTONE IN THE WRONG DIRECTION.** Adding an ACTIVE prompt can
only SHRINK the qualifying model set, never grow it, and nothing in the campaign
gates prompt registration on coverage. Measured here on 2026-08-12:

    ACTIVE prompts                        2,699
    models with any true_word_probs         401
    models covering ALL active prompts        0   <- the population is EMPTY
    best coverage                          97.4%
    median coverage                        15.2%
    models missing only 1-5 prompts           0

Not a near miss. The best model is ~70 prompts short.

**CONSEQUENCE: `frozen_population()` raises POPULATION DRIFT and refuses, so
every producer gating on it cannot re-run.** That is ~55 scripts including
registrations B, C3, E, L, M, N, O, P, Q, R and S. Registration E's discharge
(producer-debt class 1A) is blocked by exactly this and is how it was found.

**THE GUARD IS WORKING, NOT BROKEN.** It refuses rather than computing on a
population that is not the frozen one, which is correct and is why nobody has
silently published a drifted number. What has failed is the rule's stability,
and the failure is invisible until someone tries to re-run.

**ONE HONEST CAVEAT.** The model side is derived from the LOCAL cache, so
coverage depends on what this machine holds; the PROMPT side is repo-level
(`Prompts.all(status="ACTIVE")`) and its digest has drifted from the frozen one
for everybody. Run this on another machine before concluding the model count is
universal -- the prompt drift is not in doubt.

## THE DRIFT IS ENTIRELY ON THE PROMPT SIDE, AND THE FROZEN SET IS RECOVERABLE

Added after the first version of this file called the rule "unsatisfiable" flatly.
RH asked whether the population at freeze time could be reconstructed. It can, and
the digests prove it:

    frozen prompts   RECOVERED from `git show efe27885:data/prompt_categorisation.json`,
                     status=ACTIVE -> 959 prompts, digest fd3f14796ba9481b,
                     EXACT MATCH to PROMPTS_SHA.

    frozen models    NOT recovered as the exact 93. Against the 959 frozen
                     prompts, 144 models qualify TODAY -- the cache GREW, so the
                     frozen 93 is a SUBSET of today's 144 and the digest differs
                     for that reason, not because anything was lost.

**So the rule is satisfiable against the FROZEN prompt set and unsatisfiable only
against the LIVE one.** The producers fail because they re-derive prompts from
`Prompts.all(status="ACTIVE")`, which has gone 959 -> 2,699. Pinning the prompt
side to the recovered frozen set unblocks them -- on 144 models rather than 93,
which CHANGES THE POPULATION and is therefore a campaign decision, not a repair.

Routes tried and closed for isolating the original 93: the local cache keys carry
no timestamp (`dict_sha, mode, model, prompt, rule_version, theta`), and
ClickHouse's `twp_words.ingested` is a load date -- every row postdates the
freeze, so a date filter returns zero at any cutoff.

## WHAT THE PIN UNBLOCKED, AND WHAT IT DID NOT (2026-08-12)

The pin (approved [5560]; opt-in via `pin_prompts=True` or `M01_PIN_PROMPTS=1`)
works: `frozen_population()` returns 959 x 144 with no drift, and the default
path is unchanged and still refuses.

**Registration E then failed on something else, and it is worth more than the
pin.** Under the pin E reaches the data and dies on a malformed twp row --
`Qwen/Qwen3-8B-Base`, prompt `<<<LOGICAL:BOS>>>`, row 76, p is NaN.

Three facts, each checked:

    the BOS prompt IS in the frozen 959 (the only synthetic prompt in it)
    that model WAS in the 31-July roster -- not an expansion artefact
    the NaN guard was added 2026-08-03 (e7864dab); E RAN 2026-08-01 (4993d539)

**E predates the guard that now refuses its data.** NOT established: whether the
NaN was present on 2026-08-01 -- the row could have been re-ingested since. The
honest statement is that E ran before this class of row could be detected, and
the data now contains one the instrument rejects. Whether E consumed it is
unknown and may not be knowable.

That is the third distinct thing discharging E's producer debt has surfaced, and
none of them was the debt itself: a wrong count in a comment beside a digest, a
population rule unpinned on the prompt side, and a published number whose
substrate now fails a guard written after it.
"""
import collections
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")


def main():
    from malign_logits.cache import get_cache
    from malign_logits.prompts import Prompts
    prompts = sorted({p.text for p in Prompts.all(status="ACTIVE")})
    per = collections.defaultdict(set)
    for d in get_cache().iter_keys("true_word_probs"):
        per[d.get("model")].add(d.get("prompt"))
    need = set(prompts)
    full = [m for m, got in per.items() if need <= got]
    h = lambda s: hashlib.sha256("\n".join(s).encode()).hexdigest()   # noqa: E731
    print("ACTIVE prompts                     %6d" % len(prompts))
    print("  digest                           %s" % h(prompts)[:16])
    print("models with any true_word_probs    %6d" % len(per))
    print("models covering ALL active prompts %6d   <- the population" % len(full))
    print("  digest                           %s" % h(sorted(full))[:16])
    if not per:
        print("\nNO CACHE VISIBLE -- this machine cannot answer the model half.")
        return 1
    cov = sorted((len(need & got) / len(need) for got in per.values()), reverse=True)
    import statistics as st
    print("\nbest coverage   %5.1f%%" % (100 * cov[0]))
    print("median coverage %5.1f%%" % (100 * st.median(cov)))
    short = [len(need - got) for got in per.values()]
    print("models missing only 1-5 prompts    %6d" % sum(1 for x in short if 1 <= x <= 5))
    print("smallest shortfall                 %6d prompts" % min(short))
    print("\nVERDICT: %s" % ("SATISFIABLE" if full else
          "UNSATISFIABLE -- every producer gating on frozen_population() refuses"))
    return 0 if full else 2


if __name__ == "__main__":
    sys.exit(main())
