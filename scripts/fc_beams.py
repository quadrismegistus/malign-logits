"""Read the forced-continuation beam stash and print sites as text.

    uv run python scripts/fc_beams.py                       one site, all arms
    uv run python scripts/fc_beams.py --grep wrist          prompts matching
    uv run python scripts/fc_beams.py --pair Falcon3-7B -n 5
    uv run python scripts/fc_beams.py --domain violence --transgressive
    uv run python scripts/fc_beams.py --twin --domain taboo      both members
    uv run python scripts/fc_beams.py --list-pairs | --list

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

DOMAIN AND MARKEDNESS are not in the stash key; they come from the frozen
sample via `fc_analyse.site_meta()`, joined on the prompt string. That function
is imported rather than reimplemented so its uniqueness assertion runs -- two
stems sharing a prompt would merge silently. Coverage checked 2026-08-07: all
210 distinct stash prompts are in the sample, none uncovered.

    domain    animal betrayal power property sexual taboo violence   (30 each)
    member    MARKED (105) / UNMARKED (105) -- MARKED IS THE TRANSGRESSIVE ARM.
              `--transgressive` and `--neutral` are accepted as synonyms so the
              flag does not require translating the vault's vocabulary.
    stratum   R_COMPARABLE (120) / R_INVISIBLE (90)
"""

import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

TOP = 5


def index(st, pair=None, grep=None, meta=None, want=None):
    """(pair, prompt) -> {(role, arm, word): key}, filtered.

    `want` is {field: value} over the sample metadata; matched case-insensitively
    so `--domain Violence` and `--member marked` both work.
    """
    by = collections.defaultdict(dict)
    for k in st.keys():
        if not isinstance(k, dict) or k.get("type") != "fc_v1":
            continue
        if pair and pair not in k["pair"]:
            continue
        if grep and grep.lower() not in k["prompt"].lower():
            continue
        if want:
            m = meta.get(k["prompt"])
            #: A prompt absent from the sample is NOT silently kept. Coverage is
            #: complete today; if that changes, a filtered run must drop the row
            #: rather than pass it through unmatched.
            if not m or any(str(m.get(f, "")).lower() != v.lower()
                            for f, v in want.items()):
                continue
        by[(k["pair"], k["prompt"])][(k["role"], k["arm"], k["word"] or "")] = k
    return by


def show(st, pair, prompt, d, top=TOP, meta=None):
    def beams(key):
        rec = st[key]
        return sorted(rec["beams"], key=lambda b: -b["log_prob"])[:top]

    print("=" * 78)
    print("PAIR   ", pair)
    print("PROMPT ", repr(prompt))
    m = (meta or {}).get(prompt)
    if m:
        print("META    domain=%s  subdomain=%s  member=%s  stratum=%s  stem=%s"
              % (m.get("domain"), m.get("subdomain"), m.get("member"),
                 m.get("stratum"), m.get("stem")))
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
    ap.add_argument("--domain", help="animal betrayal power property sexual taboo violence")
    ap.add_argument("--member", help="MARKED/transgressive or UNMARKED/neutral")
    ap.add_argument("--transgressive", action="store_true", help="= --member MARKED")
    ap.add_argument("--neutral", action="store_true", help="= --member UNMARKED")
    ap.add_argument("--twin", action="store_true",
                    help="print both members of each stem together, transgressive first")
    ap.add_argument("--stratum", help="R_COMPARABLE or R_INVISIBLE")
    ap.add_argument("--stem", help="exact stem id")
    ap.add_argument("--list-pairs", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="summarise the filterable metadata values")
    ap.add_argument("--all-arms", action="store_true", default=True,
                    help="only sites carrying base+aligned undisturbed AND both forced arms")
    a = ap.parse_args()

    from malign_logits.cache import get_cache
    from fc_analyse import site_meta
    st = get_cache()._stash("beam_fc")
    meta = site_meta()

    if a.list:
        for f in ("domain", "subdomain", "member", "stratum"):
            c = collections.Counter(r.get(f, "") for r in meta.values())
            print("%-10s %s" % (f, "  ".join("%s=%d" % kv for kv in sorted(c.items()))))
        return

    if a.list_pairs:
        ps = collections.Counter(k["pair"] for k in st.keys()
                                 if isinstance(k, dict) and k.get("type") == "fc_v1")
        for p, c in sorted(ps.items()):
            print("%6d  %s" % (c, p))
        print("\n%d pairs, %d records" % (len(ps), sum(ps.values())))
        return

    #: MARKED is the transgressive arm; accept the plain words for it, and
    #: refuse a contradiction rather than silently letting one flag win.
    SYN = {"transgressive": "MARKED", "marked": "MARKED",
           "neutral": "UNMARKED", "unmarked": "UNMARKED"}
    if a.transgressive and a.neutral:
        sys.exit("--transgressive and --neutral are mutually exclusive")
    if a.transgressive:
        a.member = "MARKED"
    elif a.neutral:
        a.member = "UNMARKED"
    if a.member:
        if a.member.lower() not in SYN:
            sys.exit("--member: expected MARKED/transgressive or UNMARKED/neutral")
        a.member = SYN[a.member.lower()]
    #: `--twin` needs BOTH members, so a member filter would empty it.
    if a.twin and a.member:
        sys.exit("--twin prints both members; drop --member/--transgressive/--neutral")
    want = {f: getattr(a, f) for f in ("domain", "member", "stratum", "stem")
            if getattr(a, f)}
    by = index(st, a.pair, a.grep, meta, want)
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

    if a.twin:
        #: group by (pair, stem) and print the transgressive member first, so
        #: the two prompts differing by one word sit adjacent. Stems where only
        #: one member survived the all-arms filter are reported, not dropped
        #: silently -- a twin comparison missing its twin is not one.
        byst = collections.defaultdict(list)
        for pair, prompt, d in ok:
            m = meta.get(prompt) or {}
            byst[(pair, m.get("stem"))].append((m.get("member"), prompt, d))
        full = {k: v for k, v in byst.items() if len(v) == 2}
        half = len(byst) - len(full)
        print("stems with BOTH members present: %d   (%d have only one, skipped)\n"
              % (len(full), half))
        for (pair, stem), members in sorted(full.items())[:a.n]:
            for _, prompt, d in sorted(members, key=lambda x: x[0] != "MARKED"):
                show(st, pair, prompt, d, a.top, meta)
        return

    for pair, prompt, d in ok[:a.n]:
        show(st, pair, prompt, d, a.top, meta)


if __name__ == "__main__":
    main()
