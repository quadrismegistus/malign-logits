"""M05 SITE FINDER -- the rule as an EXECUTABLE SPECIFICATION.

WHY THIS EXISTS. Two seats implemented one written rule and differed by 11%
(1,223 vs 1,359). [2801]: "the underspecification is the finding." A rule
stated in prose has as many implementations as readers; this file IS the rule,
so a disagreement becomes a diff against one artifact instead of two readings
of one paragraph.

EVERY SEMANTIC CHOICE THE PROSE LEFT OPEN IS PINNED HERE AND CASED:

  PAIRING     base := model_to_base[aligned].  [2801], verbatim.
              NOT the lineage's first base: `lineage` pools Falcon3-1B/3B/
              7B/10B and Qwen2.5-0.5B with Qwen2.5-7B, so that formed
              CROSS-SIZE pairs (Qwen2.5-7B-Base vs Qwen2.5-0.5B-Instruct).

  PROMPTS     INTERSECTION of the two grids. A prompt in one arm only is not
              a comparison; it is counted as `prompt_one_arm`.

  TOP WORD    argmax by probability. A TIE AT RANK 1 HAS NO TOP WORD, so the
              site is REFUSED and counted (`tie_at_top`) -- never resolved by
              file order, which is what makes two runs of one rule disagree.

  CLAUSE 4    REPORTED BOTH WAYS, because it is a construct question RH owns
              and not an implementation detail ([2801]):
                LEX_STRICT   the substitute must be OUTSIDE the category --
                             displacement means leaving it. `kill->beat` NO.
                LEX_OVERRIDE any change from a transgressive base word --
                             the base's choice was suppressed, wherever it
                             landed. `kill->beat` YES.

  FREE        the ADOPTED M05 rule ([2797]), lexicon-free:
                w_b != w_a  AND  rank_base(w_a) <= AVAIL_MAX
              No category, so no clause 4 and no tie between readings.

    ./m05_sites.py --selftest      known-answer, runs first
    ./m05_sites.py                 the counts
"""
import sys, os, json, csv, collections

ROOT = os.path.expanduser("~/github/malign-logits")
LAC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAC); sys.path.insert(0, ROOT)

GRID_DIR = os.path.join(ROOT, "data/twp_grid_v3")
VOCAB = os.path.join(ROOT, "data/f40_vocab/vocab_tagged_v2.csv")
AVAIL_MAX = 19            # rank index, so "top 20"
NOT_PRESENT = 10 ** 6


def load_vocab(path=VOCAB):
    return {r["word"] for r in csv.DictReader(open(path))
            if r["primary"] == "TRANSGRESSIVE"}


def prepare(rows):
    """Raw grid `rows` -> (ordered words, summed probs). AGGREGATE, THEN RANK.

    RULING [2815]. `rows` is one entry PER TOKEN PATH, not per word: 17.9% of
    prompt-rows carry the same surface form more than once. The availability
    clause asks whether a word was AVAILABLE to the base model, and a word
    reachable by two paths is available at the SUM of them. Ranking by a single
    path's probability answers a question nobody asked.

    Net effect on the corpus is 7 sites. MEMBERSHIP effect is 205 -- they
    nearly cancel, so a net figure understates the churn thirtyfold, and a
    corpus is a SET.

    The bias is also directional: duplicates concentrate where one word has
    many token paths, which in this corpus is Chinese. Row-ranking
    systematically under-ranks the words the resolver splits most.

    AND IT MAKES THE TIE GUARD NON-VACUOUS BY CONSTRUCTION. Before this, the
    word list could hold one word twice, so `probs[order[1]] == probs[order[0]]`
    compared a value to ITSELF and fired unconditionally -- 463 phantom ties.
    After aggregation the keys are unique, so a tie is a real tie.
    """
    by = collections.defaultdict(float)
    for r in rows:
        by[r["word"]] += r["p"]
    order = [w for w, _ in sorted(by.items(), key=lambda kv: -kv[1])]
    return order, dict(by)


def top_word(order, probs):
    """argmax by probability, or None on a TIE AT RANK 1.

    A tie has no top word. Resolving it by file order is the single most
    likely source of two implementations disagreeing, because neither author
    would think to mention it. (On this corpus there are ZERO exact ties, so
    the rule's exposure is nil -- but arbitrary is arbitrary.)

    REQUIRES an aggregated order: with duplicates present this compares a
    value to itself. `prepare()` guarantees uniqueness; the case below pins it.
    """
    if not order:
        return None
    if len(order) != len(set(order)):
        raise ValueError("top_word requires aggregated rows: duplicate words "
                         "make the tie guard compare a value to itself")
    best = order[0]
    if len(order) > 1 and probs[order[1]] == probs[best]:
        return None
    return best


def classify(base_row, algn_row, transgressive, avail_max=AVAIL_MAX):
    """One prompt -> the set of labels it earns. Pure; the cases drive it."""
    ob, pb = base_row
    oa, pa = algn_row
    wb, wa = top_word(ob, pb), top_word(oa, pa)
    if wb is None or wa is None:
        return {"tie_at_top"}
    if wb == wa:
        return set()
    out = {"top_changed"}
    try:
        avail = ob.index(wa)
    except ValueError:
        avail = NOT_PRESENT
    if avail <= avail_max:
        out.add("FREE")
    else:
        out.add("substitute_novel")
    if wb in transgressive:
        out.add("LEX_OVERRIDE")
        if wa not in transgressive:
            out.add("LEX_STRICT")
    return out


def pairs_from_map(grid, m2b, arm_of):
    """EXACT pairs, [2801]. Returns (pairs, unpairable)."""
    pairs, bad = [], []
    for mid in sorted(grid):
        if arm_of(mid) != "aligned":
            continue
        b = m2b.get(mid)
        if b is None:
            bad.append((mid, "no model_to_base entry"))
        elif b not in grid:
            bad.append((mid, f"base {b} absent from grid"))
        else:
            pairs.append((b, mid))
    return pairs, bad


def count(grid, pairs, transgressive, avail_max=AVAIL_MAX):
    led = collections.Counter()
    per = []
    for b, a in pairs:
        sb, sa = set(grid[b]), set(grid[a])
        led["prompt_one_arm"] += len(sb ^ sa)
        row = collections.Counter()
        for p in sb & sa:
            for lab in classify(grid[b][p], grid[a][p], transgressive,
                                avail_max):
                row[lab] += 1
            row["shared"] += 1
        per.append((b, a, row))
        led.update(row)
    return per, led


# ─────────────────────────────────────────────────────────────────────────────

def selftest(verbose=False):
    ok, bad = [], []
    def case(n, fn, why):
        try: r = bool(fn())
        except Exception as e:
            r = False; n = f"{n} [raised {type(e).__name__}: {e}]"
        (ok if r else bad).append(n)
        if verbose and r: print(f"  [ok]   {n}\n         {why}")

    TR = {"kill", "beat", "stab"}
    def row(ws):                       # descending, distinct probs
        return ([w for w, _ in ws], {w: p for w, p in ws})

    #: [2815] -- AGGREGATE, THEN RANK. Three claims, each of which failed in
    #: the pre-fix code: the probability is the SUM, the ranking uses the sum,
    #: and the top word can CHANGE because of it.
    def _aggregate_then_rank():
        rows = [{"word": "a", "p": 0.30}, {"word": "b", "p": 0.25},
                {"word": "a", "p": 0.20}, {"word": "c", "p": 0.10}]
        order, probs = prepare(rows)
        return (abs(probs["a"] - 0.50) < 1e-12          # SUMMED, not last
                and order == ["a", "b", "c"]
                and len(order) == len(set(order)))
    case("row probabilities are SUMMED per word before ranking",
         _aggregate_then_rank,
         "[2815]: `rows` is one entry per TOKEN PATH; a word reachable two "
         "ways is available at the sum. Net 7 sites, MEMBERSHIP 205")

    def _aggregation_can_move_the_top():
        #: the real shape found in the corpus: a split word loses rank 1 to an
        #: unsplit rival on row probability and wins it back on the sum.
        rows = [{"word": "rival", "p": 0.28},
                {"word": "split", "p": 0.20}, {"word": "split", "p": 0.15}]
        order, probs = prepare(rows)
        row_top = max(rows, key=lambda r: r["p"])["word"]
        return row_top == "rival" and order[0] == "split"
    case("aggregation can CHANGE which word is top",
         _aggregation_can_move_the_top,
         "71 prompt-rows in the corpus; concentrated in Chinese, where one "
         "word has more token paths -- so row-ranking is DIRECTIONALLY biased")

    def _guard_refuses_unaggregated():
        try:
            top_word(["a", "a", "b"], {"a": 0.5, "b": 0.1})
        except ValueError:
            return True
        return False
    case("top_word REFUSES unaggregated input rather than silently lying",
         _guard_refuses_unaggregated,
         "with duplicates the tie guard compares probs[w] to probs[w] and "
         "fires always -- 463 phantom refusals. It now raises instead")

    case("a tie at rank 1 is REFUSED, not resolved by order",
         lambda: (top_word(*row([("a", 0.4), ("b", 0.4)])) is None
                  and classify(row([("a", .4), ("b", .4)]),
                               row([("c", .9), ("d", .1)]), TR)
                  == {"tie_at_top"}),
         "[2801]: file-order resolution is invisible and is the most likely "
         "source of two implementations of one rule disagreeing")

    case("an unchanged top word earns NO label",
         lambda: classify(row([("x", .9), ("y", .1)]),
                          row([("x", .8), ("y", .2)]), TR) == set(),
         "no substitution, so not a site under any reading")

    case("kill->scream is STRICT and OVERRIDE and FREE",
         lambda: classify(row([("kill", .6), ("scream", .3)]),
                          row([("scream", .7), ("kill", .1)]), TR)
         == {"top_changed", "FREE", "LEX_OVERRIDE", "LEX_STRICT"},
         "the canonical displacement: substitute was rank 2 in the base")

    case("kill->beat is OVERRIDE but NOT STRICT",
         lambda: classify(row([("kill", .6), ("beat", .3)]),
                          row([("beat", .7), ("kill", .1)]), TR)
         == {"top_changed", "FREE", "LEX_OVERRIDE"},
         "[2801]: the construct question, both readings reported, neither "
         "silently chosen -- the substitute is still inside the category")

    case("a NOVEL substitute is not FREE",
         lambda: classify(row([("kill", .9), ("run", .1)]),
                          row([("hydroponics", .8), ("kill", .1)]), TR)
         == {"top_changed", "substitute_novel", "LEX_OVERRIDE", "LEX_STRICT"},
         "a word the base never ranked is a topic change, not a metonymic "
         "slide; it must not enter the design's corpus")

    def _avail_boundary():
        #: `kill` occupies index 0, so w_i sits at index i+1. The boundary is
        #: the INDEX (<= AVAIL_MAX = 19), i.e. rank 20 counting from 1. My
        #: first version of this case used w19/w20 and failed -- it was
        #: written from the rank and applied to the index, off by one in the
        #: case rather than in the code.
        ob = [("kill", .5)] + [(f"w{i}", .4 - i * 0.001) for i in range(30)]
        assert ob[19][0] == "w18" and ob[20][0] == "w19"
        inside = classify(row(ob), row([("w18", .9), ("kill", .05)]), TR)
        outside = classify(row(ob), row([("w19", .9), ("kill", .05)]), TR)
        return "FREE" in inside and "FREE" not in outside
    case("the availability boundary is rank 20 inclusive",
         _avail_boundary,
         "an off-by-one here moves thousands of sites and no prose "
         "statement of the rule pins it")

    def _pairing():
        grid = {"B7": {}, "B1": {}, "A7": {}, "A1": {}, "orphan": {}}
        m2b = {"A7": "B7", "A1": "B1", "orphan": "B9"}
        arm = lambda x: "base" if x.startswith("B") else "aligned"
        ps, bad = pairs_from_map(grid, m2b, arm)
        return (sorted(ps) == [("B1", "A1"), ("B7", "A7")]
                and len(bad) == 1 and bad[0][0] == "orphan")
    case("pairing is model_to_base, and unpairable is COUNTED",
         _pairing,
         "[2801]: each aligned checkpoint's OWN base. The pooled-lineage "
         "form produced cross-size pairs and more pairs than checkpoints")

    def _one_arm_counted():
        grid = {"B": {"p1": row([("a", .9), ("b", .1)]),
                      "p2": row([("a", .9), ("b", .1)])},
                "A": {"p1": row([("b", .9), ("a", .1)])}}
        per, led = count(grid, [("B", "A")], TR)
        return led["prompt_one_arm"] == 1 and led["shared"] == 1
    case("prompts present in ONE arm are excluded and counted",
         _one_arm_counted,
         "intersect-vs-union is another silent divergence between two "
         "readings of the same sentence")

    for n in bad: print(f"  [FAIL] {n}")
    print(f"m05_sites self-test: {len(ok)} of {len(ok)+len(bad)}")
    return not bad


def main():
    import m04_producer as P
    m = json.load(open(os.path.join(ROOT, "data/lineage_map_models.json")))
    m2b, m2s = m["model_to_base"], m["model_to_stage"]
    n2s = {}
    for _m, _s in m2s.items():
        n2s.setdefault(P.norm(_m), _s)
    def arm_of(cp):
        s = m2s.get(cp) or n2s.get(P.norm(cp))
        return None if s is None else ("base" if s == "base" else "aligned")

    grid = collections.defaultdict(dict)
    for f in sorted(os.listdir(GRID_DIR)):
        if not f.endswith(".jsonl"):
            continue
        mid = f[:-6].replace("__", "/")
        with open(os.path.join(GRID_DIR, f)) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                rs = d.get("rows") or []
                if not rs:
                    continue
                #: RULING 2815 -- aggregate, then rank. Never the raw rows.
                grid[mid][d["prompt"]] = prepare(rs)

    TR = load_vocab()
    pairs, bad = pairs_from_map(grid, m2b, arm_of)
    per, led = count(grid, pairs, TR)

    print(f"grid models {len(grid)}   pairs {len(pairs)}   unpairable {len(bad)}")
    for mid, why in bad:
        print(f"    {mid:46s} {why}")
    print(f"\n{'label':22s} {'sites':>10s}   meaning")
    print("-" * 78)
    for k, txt in (("shared", "prompts compared, both arms present"),
                   ("tie_at_top", "REFUSED: no top word (tie at rank 1)"),
                   ("prompt_one_arm", "excluded: present in one arm only"),
                   ("top_changed", "the top word changed"),
                   ("FREE", "ADOPTED M05 rule: substitute was in base top 20"),
                   ("substitute_novel", "substitute absent from base: topic change"),
                   ("LEX_OVERRIDE", "base top word transgressive, any change"),
                   ("LEX_STRICT", "...and the substitute is OUTSIDE the category")):
        print(f"{k:22s} {led[k]:10,d}   {txt}")
    print("-" * 78)
    print(f"\nTHE DESIGN'S CORPUS (FREE)          {led['FREE']:,}")
    print(f"the stratum, both readings          "
          f"STRICT {led['LEX_STRICT']:,} / OVERRIDE {led['LEX_OVERRIDE']:,}")
    print(f"  the gap between them IS `kill->beat`: "
          f"{led['LEX_OVERRIDE'] - led['LEX_STRICT']:,} sites where the "
          f"substitute is\n  also transgressive. RH's construct call, not "
          f"a seat's.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest("-v" in sys.argv) else 1)
    main()
