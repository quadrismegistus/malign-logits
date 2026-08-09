"""Draw the L1 coding pilot: 150 (surface, group) units per language, stratified.

    uv run python l1_pilot_sample.py

Per [5162] (unit is (surface, pair), not surface), [5170] (no shape filter),
[5172] (five classes; keep the blank-template surfaces visible). Draws from the
candidate set re-derived under `>= theta` only at 498c88b4.

WHY 150 UNITS AND NOT 150 SURFACES. A surface's class is pair-relative -- `kill`
is POLE2 for love/hate and IN-FRAME for loyal/rebellious -- so a resolved share
estimated per surface does not transfer to a full pass whose unit is the pair.
That was the error that repriced this arm from 3,284 to 15,968 to 55,055.

STRATIFIED BY GROUP so no pair dominates and 2.2's per-pair kappa has somewhere
to land. Languages are separate strata and are NEVER pooled: en and zh differ by
a factor of two in candidate coverage (0.823 vs 0.465), so a pooled resolved
share would describe neither.

BLANK-TEMPLATE SURFACES ARE TRACKED, NOT FORCED. [5172].3 asks that the
stratification not bury whether OLMo-3's blank-template behaviour is a family
signature. Forcing them in would bias the resolved share, so the draw is
untouched and their count is REPORTED -- if the sample contains none, that is
itself the answer to how common they are, and the report says so.

THE SAMPLE IS THE UNIT OF BLINDING. A unit is (surface, group). It carries no
model, no arm, no role, no probability -- there is nothing in the drawn record
for a coder to be unblinded by.
"""
import collections
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

THETA = 0.001
N_PER_LANG = 150
SEED = 4946          #: the campaign's seed, so the draw is reproducible


def main():
    from malign_logits.cache import CacheManager

    V = json.load(open(os.path.join(ROOT, "data", "f11_l1_candidate_vocab.json")))
    vocab = set(V["surfaces"])
    Q = json.load(open(os.path.join(ROOT, "data", "f11_quintuplets.json")))
    active = [q for q in Q["quintuplets"] if q["status"] == "ACTIVE"]
    print("candidate surfaces %d   ACTIVE groups %d (en %d, zh %d)"
          % (len(vocab), len(active),
             sum(1 for q in active if q["language"] == "en"),
             sum(1 for q in active if q["language"] == "zh")))

    cm = CacheManager()
    mods = set()
    for d in sorted(glob.glob(os.path.join(ROOT, "data", "f11_twp*"))):
        if os.path.isdir(d):
            for p in glob.glob(d + "/*.jsonl"):
                for line in open(p):
                    mods.add(json.loads(line)["model"])
                    break
    print("models contributing: %d\n" % len(mods))

    #: (surface, group) units that occur at or above theta, over ALL SIX PROMPT
    #: ROLES and surviving k >= K_MIN models.
    #:
    #: SIX ROLES, NOT THREE. An earlier draw used pole_a/pole_b/both because that
    #: is what the L3 GEOMETRY needs -- the pole axis requires exactly those --
    #: and carrying it here would have priced a resolved share on a population
    #: the full pass does not code. The redo analyses the controls and
    #: both_matched; the controls are why the delta ran at all. [5178].
    #:
    #: k >= 2 is RH's amendment ([5176].1): a unit enters only if at least two
    #: models put that surface above theta in that group. It halves the volume
    #: for 0.1% of en mass and 2.2% of zh, and it keeps the blank-template runs
    #: (17 models) while dropping the glued fragments (1 model) WITHOUT knowing
    #: anything about underscores -- evidential where the abolished filter was
    #: orthographic. Safe only because the L1 measure is mass-based.
    K_MIN = 2
    seen = collections.defaultdict(lambda: collections.defaultdict(set))
    for mid in sorted(mods):
        for q in active:
            for role in ("pole_a", "pole_b", "both",
                         "control_a", "control_b", "both_matched"):
                txt = q.get(role)
                if not txt:
                    continue
                v = cm.get_true_word_probs(mid, txt)
                if not v or not v.get("rows"):
                    continue
                for r in v["rows"]:
                    if r["p"] >= THETA and r["word"] in vocab:
                        seen[q["group"]][r["word"]].add(mid)
    units = collections.defaultdict(set)
    dropped = 0
    for g, d in seen.items():
        for w, ms in d.items():
            if len(ms) >= K_MIN:
                units[g].add(w)
            else:
                dropped += 1
    print("k >= %d: kept %d units, dropped %d singletons (%.1f%% kept)"
          % (K_MIN, sum(len(v) for v in units.values()), dropped,
             100 * sum(len(v) for v in units.values())
             / max(sum(len(v) for v in seen.values()), 1)))

    lang = {q["group"]: q["language"] for q in active}
    poles = {q["group"]: (q["pole_a"], q["pole_b"]) for q in active}
    rng = random.Random(SEED)
    out = {}
    for L in ("en", "zh"):
        gs = sorted(g for g in units if lang.get(g) == L)
        pool = [(g, s) for g in gs for s in sorted(units[g])]
        print("%s: %d groups, %d (surface, group) units in the frame" % (L, len(gs), len(pool)))
        #: proportional-ish stratification: at least one per group, remainder
        #: allocated by group size, so a small pair is represented and a large
        #: one is not allowed to swamp the per-pair kappa.
        #: LARGEST REMAINDER, because plain truncation loses one unit per group
        #: and silently returned 140 of a declared 150 -- an n nobody chose.
        per = {g: 1 for g in gs}
        left = N_PER_LANG - len(gs)
        if left > 0:
            sizes = {g: len(units[g]) for g in gs}
            tot = sum(sizes.values()) or 1
            exact = {g: left * sizes[g] / tot for g in gs}
            for g in gs:
                per[g] += int(exact[g])
            short = N_PER_LANG - sum(per.values())
            for g in sorted(gs, key=lambda x: -(exact[x] - int(exact[x])))[:short]:
                per[g] += 1
        #: and a group cannot give more units than it has
        for g in gs:
            per[g] = min(per[g], len(units[g]))
        drawn = []
        for g in gs:
            cand = sorted(units[g])
            rng.shuffle(cand)
            drawn += [(g, s) for s in cand[:per[g]]]
        rng.shuffle(drawn)
        drawn = drawn[:N_PER_LANG]
        blank = [(g, s) for g, s in drawn if s and set(s) <= set("_")]
        print("   drawn %d units over %d groups; blank-template surfaces present: %d"
              % (len(drawn), len({g for g, _ in drawn}), len(blank)))
        if not blank:
            allb = [(g, s) for g in gs for s in units[g] if s and set(s) <= set("_")]
            print("   NONE DRAWN, and the frame holds %d -- reported, not forced "
                  "([5172].3): forcing them in would bias the resolved share." % len(allb))
        out[L] = [{"group": g, "surface": s,
                   "pole_first": poles[g][0], "pole_second": poles[g][1]}
                  for g, s in drawn]

    p = os.path.join(ROOT, "data", "f11_l1_pilot_sample.json")
    json.dump({"_about": "L1 coding pilot draw: 150 (surface, group) units per "
                         "language, stratified by group. Units carry NO model, "
                         "arm, role or probability -- blinding is structural.",
               "_producer": "meta/M02_frame_exit/scripts/l1_pilot_sample.py",
               "_rulings": ["[5162] unit is (surface,pair)", "[5170] no shape filter",
                            "[5172] five classes"],
               "seed": SEED, "theta": THETA,
               "n": {k: len(v) for k, v in out.items()}, "units": out},
              open(p, "w"), ensure_ascii=False, indent=1)
    print("\nwrote %s" % os.path.relpath(p, ROOT))
    for L in out:
        print("\n  %s sample, first 8:" % L)
        for u in out[L][:8]:
            print("     %-18s %-10s  %r / %r"
                  % (u["group"], u["surface"], u["pole_first"][:34], u["pole_second"][:34]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
