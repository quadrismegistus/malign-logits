"""Read the forced-continuation beam stash and print sites as text.

    uv run python scripts/fc_beams.py                       one site, all arms
    uv run python scripts/fc_beams.py --grep wrist          prompts matching
    uv run python scripts/fc_beams.py --pair Falcon3-7B -n 5
    uv run python scripts/fc_beams.py --list-pairs

Written 2026-08-07 after RH asked to see example beams. The analysis in
`fc_analyse.py` reduces these to nats/token; this prints the continuations
themselves, because the one channel nobody has measured -- what the model DOES
after the forced word, as against what it COSTS -- is legible by eye and is not
in any statistic yet. See findings/DISPLACEMENT_EVIDENCE.md, OPEN.

READ PATH, the same one `fc_analyse.load()` uses:

    st = get_cache()._stash("beam_fc")
    key    {type: fc_v1, pair, prompt, role, arm, word, n_beams, max_tokens, ...}
    record {beams, forced_token_ids, n_forced_tokens, prompt_len,
            scored_by_base, scored_by_aligned}
    beam   {text, tokens, full_ids, log_prob}      100 per record

`role` is which model generated; `arm` is undisturbed / force_faller /
force_riser; `word` is the pinned token and is empty on the undisturbed arm.
**The forced design is a full 2x2** -- {base, aligned} x {demoted, promoted},
1,627 sites in each cell -- because `dd` is the difference-in-differences over
those four. Print all four or the picture is half a measure.
**A forced beam's position i sits one token later in the sentence than an
unforced beam's position i**, because the pinned word consumes a slot -- so any
index-to-index comparison of the two is comparing different sentence positions.
`fc_analyse.mean_lp()` offsets for it; this script only prints, so it does not.
"""

import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

TOP = 5


def index(st, pair=None, grep=None):
    """(pair, prompt) -> {(role, arm, word): key}, filtered."""
    by = collections.defaultdict(dict)
    for k in st.keys():
        if not isinstance(k, dict) or k.get("type") != "fc_v1":
            continue
        if pair and pair not in k["pair"]:
            continue
        if grep and grep.lower() not in k["prompt"].lower():
            continue
        by[(k["pair"], k["prompt"])][(k["role"], k["arm"], k["word"] or "")] = k
    return by


def show(st, pair, prompt, d, top=TOP):
    def beams(key):
        rec = st[key]
        return sorted(rec["beams"], key=lambda b: -b["log_prob"])[:top]

    print("=" * 78)
    print("PAIR   ", pair)
    print("PROMPT ", repr(prompt))
    print("=" * 78)
    for role in ("base", "aligned"):
        k = d.get((role, "undisturbed", ""))
        if not k:
            continue
        print("\n--- %s, undisturbed" % role.upper())
        for b in beams(k):
            print("    %+8.2f  %s" % (b["log_prob"], repr(b["text"])[:66]))
    #: **BOTH ROLES, BOTH WORDS -- the design is a 2x2 and printing one row of
    #: it misrepresents the measure.** `dd` is the difference-in-differences
    #: over exactly these four cells: what forcing the demoted word costs the
    #: aligned model, MINUS what it costs the base, against the same contrast
    #: for the promoted word. An earlier version of this script filtered the
    #: forced arms to role == "aligned" and hid the base half; the stash is
    #: symmetric, 1,627 sites in each of the four cells.
    for arm, label in (("force_faller", "forced to the DEMOTED word"),
                       ("force_riser", "forced to the PROMOTED word")):
        for role in ("base", "aligned"):
            for (r, a, w), k in sorted(d.items()):
                if a != arm or r != role:
                    continue
                print("\n--- %-7s %s: %r" % (role.upper(), label, w))
                for b in beams(k):
                    print("    %+8.2f  %s" % (b["log_prob"], repr(b["text"])[:66]))
    dem = sorted({w for (r, a, w) in d if a == "force_faller"})
    pro = sorted({w for (r, a, w) in d if a == "force_riser"})
    print("\n    demoted at this site :", dem)
    print("    promoted at this site:", pro)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", help="substring of the pair id")
    ap.add_argument("--grep", help="substring of the prompt")
    ap.add_argument("-n", type=int, default=1, help="how many sites to print")
    ap.add_argument("--top", type=int, default=TOP, help="beams per arm")
    ap.add_argument("--list-pairs", action="store_true")
    ap.add_argument("--all-arms", action="store_true", default=True,
                    help="only sites carrying base+aligned undisturbed AND both forced arms")
    a = ap.parse_args()

    from malign_logits.cache import get_cache
    st = get_cache()._stash("beam_fc")

    if a.list_pairs:
        ps = collections.Counter(k["pair"] for k in st.keys()
                                 if isinstance(k, dict) and k.get("type") == "fc_v1")
        for p, c in sorted(ps.items()):
            print("%6d  %s" % (c, p))
        print("\n%d pairs, %d records" % (len(ps), sum(ps.values())))
        return

    by = index(st, a.pair, a.grep)
    ok = []
    for (pair, prompt), d in by.items():
        has_forced = any(x[1] == "force_faller" for x in d) and any(x[1] == "force_riser" for x in d)
        if not a.all_arms or (has_forced and ("base", "undisturbed", "") in d
                              and ("aligned", "undisturbed", "") in d):
            ok.append((pair, prompt, d))
    #: sorted so repeated invocations print the same sites -- the stash key
    #: order is not stable and an unsorted pick would quietly change what a
    #: cited example refers to.
    ok.sort(key=lambda x: (x[0], x[1]))
    print("matching sites: %d (of %d before the all-arms filter)\n" % (len(ok), len(by)))
    for pair, prompt, d in ok[:a.n]:
        show(st, pair, prompt, d, a.top)


if __name__ == "__main__":
    main()
