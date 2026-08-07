"""Ask-list item 2 at corpus scale: WHAT is the base estranged BY, by stratum and prompt.

    uv run --with transformers --with lemminflect python x_estrangement_strata.py

The first run of this used the slot probe -- **6 pairs, 1 prompt, 7,200 beams**.
The corpus is **36 pairs, 232 prompts, 22,788 records**, and the estrangement
result the drafting seat means was measured on the UNDISTURBED arm of that
corpus, not on the probe. RH caught it.

    arm            undisturbed 14,380 | force_riser 4,222 | force_faller 4,186
    max_tokens     10 throughout

**UNDISTURBED ONLY.** The forced arms are a different manipulation and pooling
the three would be pooling across a manipulated axis.

**STRATUM FIRST, THEN PROMPT WITHIN IT**, per RH — and the stratum is
`pair_role`, MARKED against UNMARKED, 105 prompts each.

    THE KEY IS MARKEDNESS, NOT DOMAIN, and an earlier version got this wrong.
    Both members of a `transgressive_swap` pair carry the SAME domain: "he
    kicked the dog" and "he patted the dog" are both `animal`. Slicing by
    domain therefore cuts ACROSS the manipulation instead of along it, and it
    produced the conclusion that the corpus held no neutral prompts. It holds
    105, balanced by construction against 105 marked, within domain.

    excess(token) = logp_aligned - logp_base       positive = base more surprised

USAS ONLY: 83.4% of continuation types, 95.5% of tokens. VerbNet 23.9% and
FrameNet 34.1% of token mass, and the quantity is token-weighted, so a result on
those would describe whichever third of the continuation is verbal -- the defect
that killed the slot-openness measure.

UNIT: the PAIR within a cell. A category's excess is averaged inside a pair
before being tested across pairs; pooling word-observations would let one
verbose pair carry a category.

SCOPE: ten tokens, and BEAM. malign showed beam degenerates asymmetrically by
role at 100 tokens; at 10 the loop rate is 3% but beam-versus-sampling agreement
is unchecked and a clipped sampled run is commissioned. **If that comes back
dirty this wants rerunning, not patching.**
"""
import collections
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

#: THE SPLIT IS pair_role, NOT domain. Both members of a transgressive_swap
#: pair share a domain -- "he kicked the dog" and "he patted the dog" are both
#: `animal` -- so the transgressive/neutral contrast runs INSIDE each domain,
#: along the markedness axis. An earlier version keyed on domain, imported the
#: partition from x_wildchat_split.py (built for the 2,182-prompt battery where
#: a literal `neutral` domain exists), and concluded there were no neutral
#: prompts in the corpus. There are 105, exactly balanced against 105 marked.
MARKED, UNMARKED = "MARKED", "UNMARKED"
MIN_OBS_IN_PAIR = 5
MIN_PAIRS = 6

#: INTEGRITY EXCLUSION, malign's [4962] verdict on the rule declared at [4961].
#: `llama-7b > beaver-7b-v1.0` has 130 records where the two arms scored
#: different token counts -- the 32000-vs-32001 pad-token case -- so it is not
#: commensurable across arms under any reading.
#:
#: The first run of this script logged the resulting truncations as a 0.01%
#: rate and moved on. **The rate was the wrong summary: 100% of them came from
#: this one pair of 36.** A defect concentrated entirely in one pair is a fact
#: about that pair, not noise, and the rate is exactly the statistic that hides
#: it. The independent corroboration ran the other way too -- malign found the
#: pair from its score structure without knowing about these skips.
EXCLUDE_PAIRS = {"huggyllama/llama-7b>PKU-Alignment/beaver-7b-v1.0"}


def main():
    import numpy as np
    from scipy import stats
    from malign_logits.cache import get_cache
    import s_lexicon_crosstab as X

    D = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]
    role = {}
    for r in D:
        if r.get("status") == "ACTIVE" and r.get("prompt"):
            role.setdefault(r["prompt"], r.get("pair_role"))

    #: DESIGN FILTER, and it is not optional. beam_fc now holds four designs
    #: (legacy pass1+waves1-2, wave3-lexical, explicit-battery-v1,
    #: slot-probe-sexexp1) and wave 3 landed WHILE an earlier version of this
    #: was running, so it pooled them. malign's [4932] says it plainly: do not
    #: filter by `arm` alone, that returns every forced record from every
    #: design ever run. The estrangement result is the wave-1 population, which
    #: carries design=None and reads as `legacy-pass1`.
    WANT = {None, "legacy-pass1"}
    st = get_cache()._stash("beam_fc")
    keys, bydesign = [], collections.Counter()
    for k in st.keys():
        if k.get("arm") != "undisturbed":
            continue
        try:
            d = st[k].get("design")
        except Exception:
            continue
        bydesign[str(d)] += 1
        if d in WANT:
            keys.append(k)
    print("undisturbed by design: %s" % dict(bydesign))
    print("kept design in {None, legacy-pass1}: %d records" % len(keys))

    #: TOKEN DECODING GOES THROUGH THE CACHE (cache.decode_tokens), added to
    #: CacheManager for this: the map (model, token_id) -> string is a pure
    #: function and was being recomputed on every script over 7M observations.
    #: Cold cost is one tokenizer load per model, warm cost is a dict lookup.
    #: Keyed on the MODEL, not the pair -- the H4 checkpoints double-encode the
    #: leading space, so two arms of one pair can decode the same id
    #: differently.
    cache = get_cache()
    DROPPED, SKIPPED, EXCLUDED = {}, {}, {}

    #: cell -> category -> pair -> [excess]
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    perprompt = collections.defaultdict(lambda: collections.defaultdict(list))
    seen_words = set()
    staged = []
    done = 0
    for k in keys:
        prompt = k.get("prompt")
        cell = role.get(prompt)
        if cell not in (MARKED, UNMARKED):
            continue
        try:
            v = st[k]
        except Exception:
            continue
        sb, sa = v.get("scored_by_base"), v.get("scored_by_aligned")
        if not sb or not sa:
            continue
        pair = k.get("pair")
        if pair in EXCLUDE_PAIRS:
            EXCLUDED[pair] = EXCLUDED.get(pair, 0) + 1
            continue
        for i, bm in enumerate(v["beams"]):
            #: 6 beams in 119,994 have a TRUNCATED scored_by_base (shape
            #: (10,-1,10), all beaver-7b-v1.0, 0.01%). Counted, not silently
            #: skipped: a guard that drops rows without saying so is how a
            #: non-random subset leaves an analysis unnoticed.
            if i >= len(sb) or i >= len(sa) or len(sb[i]) != len(bm["tokens"]) or len(sa[i]) != len(bm["tokens"]):
                SKIPPED[pair] = SKIPPED.get(pair, 0) + 1
                continue
            base_model = pair.split(">")[0]
            try:
                pieces = cache.decode_tokens(base_model, bm["tokens"])
            except Exception as e:
                DROPPED[pair] = type(e).__name__ + ": " + str(e)[:80]
                break
            cur, cur_ex = "", 0.0
            for j, s in enumerate(pieces):
                if (s.startswith((" ", "Ġ", "\n", "\t"))) and cur:
                    staged.append((cell, prompt, pair, cur, cur_ex))
                    cur, cur_ex = "", 0.0
                cur += s.replace("Ġ", " ")
                cur_ex += sa[i][j] - sb[i][j]
            if cur:
                staged.append((cell, prompt, pair, cur, cur_ex))
        done += 1
        if done % 2000 == 0:
            print("   %d records, %d word-observations" % (done, len(staged)), flush=True)
    print("%d records used, %d word-observations" % (done, len(staged)))
    if DROPPED:
        print("DROP LEDGER -- pairs whose tokenizer would not load:")
        for k, v in DROPPED.items():
            print("   %-46s %s" % (k.split("/")[-1][:46], v))
    else:
        print("DROP LEDGER: no pair dropped for tokenizer reasons")
    #: the exclusion ledger prints unconditionally, including when it is empty,
    #: so "no pair was excluded" and "the exclusion never ran" are different
    #: lines rather than the same silence.
    print("INTEGRITY EXCLUSION [4962]: %d records dropped -- %s"
          % (sum(EXCLUDED.values()), dict(EXCLUDED) or "NONE (check EXCLUDE_PAIRS spelling)"))
    if SKIPPED:
        print("BEAMS SKIPPED for truncated cross-scores: %d total" % sum(SKIPPED.values()))
        for kk, vv in sorted(SKIPPED.items(), key=lambda x: -x[1])[:5]:
            print("   %-46s %d" % (kk.split("/")[-1][:46], vv))
    print()

    words = sorted({re.sub(r"[^A-Za-z']", "", w).lower() for _, _, _, w, _ in staged} - {""})
    usas = X.usas_labels(words)[0]
    T = {}
    for line in open(os.path.join(CAMP, "lexicons", "usas_tagset.tsv"), encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            T[p[0].strip()] = p[1].strip()

    ncov = 0
    for cell, prompt, pair, w, ex in staged:
        c = re.sub(r"[^A-Za-z']", "", w).lower()
        cat = usas.get(c)
        if not cat:
            continue
        ncov += 1
        acc[cell][cat][pair].append(ex)
        perprompt[(cell, prompt)][cat].append(ex)
    print("USAS covers %.1f%% of word-observations\n" % (100 * ncov / max(len(staged), 1)))

    rows = []
    for cell in (MARKED, UNMARKED):
        res = []
        for cat, bypair in acc[cell].items():
            means = [float(np.mean(v)) for v in bypair.values() if len(v) >= MIN_OBS_IN_PAIR]
            if len(means) >= MIN_PAIRS:
                res.append((cat, float(np.mean(means)), len(means),
                            sum(len(v) for v in bypair.values()),
                            float(stats.ttest_1samp(means, 0)[1])))
        res.sort(key=lambda r: -r[1])
        rows += [(cell,) + r for r in res]
        bonf = 0.05 / max(len(res), 1)
        print("=" * 78)
        print("%s  —  %d categories, Bonferroni %.5f" % (cell, len(res), bonf))
        print("=" * 78)
        print("   %-8s %9s %6s %8s %10s  %s" % ("USAS", "excess", "pairs", "n_obs", "p", "gloss"))
        for cat, m, npair, nobs, pv in res[:10]:
            print("   %-8s %+9.4f %6d %8d %10.5f%s %s"
                  % (cat, m, npair, nobs, pv, " *" if pv < bonf else "  ", T.get(cat, "")[:34]))
        print("   " + "." * 62)
        for cat, m, npair, nobs, pv in res[-4:]:
            print("   %-8s %+9.4f %6d %8d %10.5f%s %s"
                  % (cat, m, npair, nobs, pv, " *" if pv < bonf else "  ", T.get(cat, "")[:34]))
        print()

    out = os.path.join(CAMP, "results", "x_estrangement_strata.csv")
    with open(out, "w", newline="") as f:
        c = csv.writer(f)
        c.writerow(["cell", "usas", "mean_excess", "n_pairs", "n_obs", "p"])
        c.writerows(rows)
    pp = os.path.join(CAMP, "results", "x_estrangement_perprompt.csv")
    with open(pp, "w", newline="") as f:
        c = csv.writer(f)
        c.writerow(["cell", "prompt", "usas", "mean_excess", "n_obs"])
        for (cell, prompt), cats in perprompt.items():
            for cat, v in cats.items():
                if len(v) >= 20:
                    c.writerow([cell, prompt, cat, float(np.mean(v)), len(v)])
    print("wrote %s\nwrote %s" % (os.path.relpath(out, ROOT), os.path.relpath(pp, ROOT)))


if __name__ == "__main__":
    main()
