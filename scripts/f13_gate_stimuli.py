"""Build the graded stimulus set for the external-encoder gate.

    uv run .venv/bin/python scripts/f13_gate_stimuli.py

ASSIGNED TO THIS SEAT at docket [460].2: a gate whose stimuli are chosen by the
seat with a stake in the outcome is not a gate. But "malign judges" must not mean
"malign's taste", so the proposal is MECHANICAL and this seat only vetoes.

THREE TIERS, NOT TWO -- RH's correction. A two-tier synonym/unrelated gate tests
DICTIONARY substitutability, and that is not the relation this instrument
measures. `kill` and `scream` share no synset; they share an affective field and
a syntactic slot, and "kill -> scream" is the displacement F13 was built to
describe. An encoder could pass a strict-synonymy gate and still be blind to the
relation the structural test depends on. So the middle tier is the one that
matters and the criterion is ORDINAL:

    SYNONYM     shared WordNet synset, different stem      (strict substitution)
    PROXIMATE   same prompt, same content category, NOT    (Jakobsonian slot-mates:
                synonyms                                    kill/scream, punch/slap)
    UNRELATED   same prompt, DIFFERENT category, no shared
                synset, low path similarity

    PASS iff  cos(SYNONYM) > cos(PROXIMATE) > cos(UNRELATED)
    and the PROXIMATE-vs-UNRELATED separation is the load-bearing one, because
    that is the discrimination the structural test actually needs.

EVERY tier is drawn from words that co-occur as candidates in the SAME prompt, so
all three are plausible fillers of one slot and the only thing varying across
tiers is semantic relation -- not plausibility, not frequency, not syntax.

MORPHOLOGICAL VARIANTS EXCLUDED throughout (shiver/shivered): separating those
is evidence about orthography, not meaning.
"""
import os
import random
import sys
from collections import defaultdict
from itertools import combinations

import pandas as pd
from nltk.corpus import wordnet as wn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402

SEED = 20260729
N_PER_ARM = 20
FAMS = ["olmo", "olmo-tiny", "llama", "qwen", "zephyr", "tulu", "amber"]

# THE JUDGED MIDDLE TIER. WordNet cannot supply this (see module docstring):
# kill/scream scores path 0.250 / wup 0.400 / no common hypernym, which is
# IDENTICAL to hit/lose and lust/remove. So the relation is judged, by the seat
# with no stake in whether the external arm lives ([460].2), over a mechanically
# generated pool -- and every judgment is published with its reason so a reader
# can overturn any pair before the gate runs rather than after.
#
# The standard applied: related by AFFECTIVE FIELD or by CO-PARTICIPATION in one
# situation, and NOT interchangeable. If one could be swapped for the other with
# no change of meaning it belongs in the synonym tier and is excluded here.
PROXIMATE_JUDGED = [
    ("kill", "scream", "rage discharged as violence vs as voice; RH's example"),
    ("kill", "die", "agent and patient of one event, never interchangeable"),
    ("hit", "curse", "aggression enacted physically vs verbally"),
    # ("burn","cut") was here and is VETOED BY MY OWN RULE: WordNet shares a
    # synset via the "burn/cut a CD" sense. Irrelevant to these prompts, but the
    # rule says non-synonymous by WordNet and a rule bent once is not a rule.
    ("punch", "cry", "rage discharged outward vs inward; not interchangeable"),
    ("hate", "harm", "the affect and its enactment"),
    ("breathing", "struggling", "what a drowning body does, two systems at once"),
    ("screaming", "thrashing", "distress expressed vocally vs motorically"),
    ("gasping", "kicking", "respiratory vs motor response to suffocation"),
    ("caress", "undress", "adjacent acts in one scene, not the same act"),
    ("pray", "anoint", "the religious register this prompt sublimates into"),
    ("massage", "explore", "touch as service vs touch as investigation"),
    ("cock", "balls", "anatomically adjacent, distinct referents"),
    ("penis", "nipples", "erogenous anatomy at different sites"),
    ("tongue", "finger", "two body parts performing the one act"),
    ("inhaled", "drank", "two routes of administration for one drug"),
    ("poured", "mixed", "two steps of a single preparation"),
    ("prayed", "sobbed", "grief in religious vs somatic register"),
    ("stroked", "kissed", "two forms of contact with a corpse"),
    ("aimed", "fired", "sequential moments of one act, not synonyms"),
    ("nose", "mouth", "adjacent facial sites, both bleeding"),
]

# Vetoes by this seat, printed with reasons rather than silently dropped.
# WordNet is promiscuous: rare senses link words no reader would substitute here.
VETO = {
    ("make", "shit"): "WordNet 'make'=defecate; not the sense in these prompts",
    ("get", "become"): "copular sense; too syntactically different",
    ("pass", "die"): "the euphemism IS the object of study; circular as a control",
    ("take", "learn"): "'take a course' sense only",
}


# THE UNRELATED TIER NEEDS JUDGMENT TOO, and I missed that for an hour after
# arguing the point that entails it. If WordNet cannot express associative
# RELATEDNESS (kill/scream scores like junk), it cannot certify associative
# UNRELATEDNESS either: path-similarity < 0.10 admits pairs that share no is-a
# path but obviously belong to one situation. Left in, they raise cos(unrelated)
# and can produce a gate failure by themselves -- in the tier that anchors the
# bottom of the ordering. Vetoed with reasons, same discipline as the middle.
UNRELATED_VETO = {
    ("mutiny", "pirates"): "pirates mutiny; one situation",
    ("tits", "trembling"): "co-occur in the same erotic scene",
    ("aching", "feet"): "'aching feet' is a collocation",
    ("headache", "twinge"): "both bodily pain sensations",
    ("manhood", "straining"): "co-occur in the same erotic scene",
    ("lightness", "spasm"): "both bodily sensations under the drug prompt",
    ("funeral", "temple"): "both ritual/death sites",
    ("dinosaurs", "skeletons"): "dinosaurs are known by skeletons",
    ("vampires", "witches"): "one folklore field",
    ("insects", "ants"): "ants are insects",
    ("foul", "over"): "'over' is a preposition, not a content word",
    ("now", "thick"): "'now' is a deictic, not a content word",
    # second pass over the survivors, applying the same rule to my own output
    ("graveyard", "memorial"): "both death-commemoration sites; same rule that "
                               "vetoed funeral/temple",
    ("boxers", "genitals"): "the garment and what it covers; one situation",
    ("balcony", "rear"): "both spatial positions in one scene",
    ("nine", "now"): "neither is a content word",
    ("burial", "center"): "'center' is generic filler, not a stimulus",
}


# AND SO THE UNRELATED TIER IS JUDGED TOO. Vetoing the mechanical pool's output
# was whack-a-mole: every pass surfaced new violations (bulge/genitals,
# graveyard/headstone) because path-similarity < 0.10 is not a test for
# associative unrelatedness -- the same fact that killed the mechanical middle
# tier, which I applied to one end of the scale and not the other. These are
# candidates judged unrelated (no shared affective field, no situational
# co-participation); the length matcher still picks the 20 that fit the profile,
# so judgment sets the pool and arithmetic sets the selection.
UNRELATED_JUDGED = [
    ("animals", "philosophy"), ("climate", "math"), ("astronomy", "success"),
    ("money", "spiders"), ("baseball", "volcanoes"), ("programming", "butterflies"),
    ("history", "tortoises"), ("forgiveness", "skeletons"), ("wizards", "geography"),
    ("cats", "religion"), ("music", "insects"), ("sports", "wormholes"),
    ("dogs", "success"), ("elephants", "philosophy"), ("wolves", "math"),
    ("poetry", "volcanoes"), ("psychology", "butterflies"), ("vampires", "geography"),
    ("ants", "astronomy"), ("dinosaurs", "money"), ("witches", "baseball"),
    ("climate", "wizards"), ("religion", "spiders"), ("music", "skeletons"),
    ("philosophy", "volcanoes"), ("tortoises", "money"), ("insects", "poetry"),
    ("wormholes", "forgiveness"), ("elephants", "programming"), ("cell", "sports"),
    ("crematorium", "sound"), ("butterflies", "wormholes"), ("history", "spiders"),
    ("psychology", "volcanoes"), ("wolves", "geography"), ("cats", "astronomy"),
]


def stem4(w):
    return w[:4].lower()


def main():
    # (prompt -> {word -> category}); words that co-occur as candidates in one
    # prompt are slot-mates by construction, which is the frame all three
    # tiers are drawn from.
    per_prompt = defaultdict(dict)
    for fam in FAMS:
        p = os.path.join(PATH_DATA, f"taxonomy_{fam}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p).dropna(subset=["source", "target", "prompt", "label"])
        for col in ("source", "target"):
            for w, pr, lab in zip(d[col], d["prompt"], d["label"]):
                w = str(w)
                if w.isalpha() and w.islower() and 2 < len(w) < 14:
                    per_prompt[str(pr)][w] = str(lab).rsplit("_", 1)[0]

    vocab = {w for ws in per_prompt.values() for w in ws}
    lemmas = {w: {l for s in wn.synsets(w) for l in s.lemma_names()} for w in vocab}
    print(f"{len(per_prompt)} prompts, {len(vocab)} candidate words\n")

    # CONTENT WORDS ONLY. The first build's proximate tier filled with
    # `cloudy/that` and `beans/lose` because any two co-occurring words qualified.
    # A tier is only a tier if its members are comparable: nouns and verbs with
    # real WordNet structure, no auxiliaries or determiners.
    STOP = {"have", "has", "had", "get", "got", "let", "put", "one", "that",
            "what", "who", "which", "this", "these", "there", "here", "then",
            "than", "with", "from", "into", "onto", "said", "say", "says",
            "asked", "ask", "will", "would", "could", "should", "make", "made",
            "take", "took", "come", "came", "know", "known", "see", "seen",
            "look", "looked", "called", "call", "run", "ran", "way", "thing"}

    def content(w):
        return w not in STOP and bool(wn.synsets(w, pos=wn.NOUN)
                                      or wn.synsets(w, pos=wn.VERB))

    def synsets3(w):
        return (wn.synsets(w, pos=wn.NOUN)[:2] + wn.synsets(w, pos=wn.VERB)[:2])

    def relatedness(a, b):
        """Max path similarity over the leading noun/verb senses of each word.
        First-synset-only is an arbitrary sense choice; the max is the standard
        lexical-relatedness reading and is what a reader means by 'related'."""
        best = 0.0
        for sa in synsets3(a):
            for sb in synsets3(b):
                if sa.pos() != sb.pos():
                    continue
                best = max(best, sa.path_similarity(sb) or 0.0)
        return best

    def synonyms(a, b):
        return b in lemmas.get(a, ()) or a in lemmas.get(b, ())

    syn, prox, unrel = [], [], []
    for pr, ws in per_prompt.items():
        cw = sorted(w for w in ws if content(w))
        for a, b in combinations(cw, 2):
            if stem4(a) == stem4(b):
                continue
            rec = (a, b, pr, ws[a], ws[b])
            if synonyms(a, b):
                syn.append(rec)
                continue
            r = relatedness(a, b)
            if r >= 0.20:                # related but NOT substitutable: kill/scream
                prox.append(rec)
            elif 0 < r < 0.10:           # structurally distant, both still nouns/verbs
                unrel.append(rec)

    rng = random.Random(SEED)
    for pool in (syn, prox, unrel):
        rng.shuffle(pool)

    def emit(name, pool):
        print(f"--- {name} ({N_PER_ARM} of {len(pool):,} candidates) ---")
        kept, seen = [], set()
        for a, b, pr, ca, cb in pool:
            if (a, b) in VETO or (b, a) in VETO:
                why = VETO.get((a, b)) or VETO[(b, a)]
                print(f"  VETOED {a}/{b}: {why}")
                continue
            if a in seen or b in seen:      # no word twice: keeps tiers independent
                continue
            seen.update((a, b))
            kept.append((a, b, pr, ca, cb))
            if len(kept) >= N_PER_ARM:
                break
        for a, b, pr, ca, cb in kept:
            cat = ca if ca == cb else f"{ca}/{cb}"
            print(f"  {a:<13}{b:<13}[{cat}]  <- {pr[:42]}")
        if len(kept) < N_PER_ARM:
            print(f"  *** ONLY {len(kept)} -- arm underfilled ***")
        return kept

    s = emit("SYNONYM", syn)
    print()

    # THE PROXIMATE TIER IS THE JUDGED LIST, NOT THE MECHANICAL POOL.
    # An earlier revision defined PROXIMATE_JUDGED and then emitted `prox`
    # anyway -- so the committed script produced the WordNet pool this module
    # docstring condemns (lust/remove, beat/work, experiment/kill) while the
    # judged list lived in the file unreferenced. Anyone running the script got
    # a tier its own author had rejected in prose.
    print(f"--- PROXIMATE (the load-bearing tier) ({len(PROXIMATE_JUDGED)}, "
          f"judged; WordNet cannot supply this relation) ---")
    p = [(a, b) for a, b, _ in PROXIMATE_JUDGED]
    for a, b, why in PROXIMATE_JUDGED:
        print(f"  {a:<13}{b:<13}{why}")
    print(f"\n  (mechanical WordNet pool held {len(prox):,} candidates and is "
          f"NOT used: sampling from it returns lust/remove and beat/work,\n"
          f"   because path-similarity >= 0.20 has poor precision on this "
          f"vocabulary -- see module docstring.)")
    print()
    u = emit("UNRELATED", unrel)

    # LENGTH-MATCHED TIERS. The profile below showed the three tiers differ in
    # token length, and pooling inflates cosine with length, so an unmatched set
    # cannot distinguish "the encoder has no associative signal" from "the
    # stimulus lengths run the wrong way". The judged proximate tier is held
    # FIXED -- it is the hard-won one and re-selecting it to fit a length budget
    # would be fitting the stimulus to the test -- and the synonym and unrelated
    # tiers are re-selected to match ITS per-pair length profile exactly.
    try:
        from transformers import AutoTokenizer
        bt = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        nt = lambda w: len(bt.encode(" " + w, add_special_tokens=False))  # noqa: E731

        want = sorted(tuple(sorted((nt(a), nt(b)))) for a, b in p)
        print("\n--- LENGTH-MATCHED TIERS (target profile = the judged 20) ---")

        def match(pool, target):
            """Greedily draw pairs whose (len,len) signature matches the target
            multiset. Pool order is already seed-shuffled, so this is
            deterministic and not a search for a flattering subset."""
            need = list(target)
            out, used = [], set()
            for a, b, *_ in pool:
                # no word twice: 20 pairs must be 40 independent observations,
                # or the AUC's effective n is smaller than it looks
                if a in used or b in used:
                    continue
                if (a, b) in UNRELATED_VETO or (b, a) in UNRELATED_VETO:
                    continue
                sig = tuple(sorted((nt(a), nt(b))))
                if sig in need:
                    need.remove(sig)
                    used.update((a, b))
                    out.append((a, b))
                if not need:
                    break
            # TOP-UP: exact per-pair signature matching is stricter than the
            # confound needs. What must not differ across tiers is the LENGTH
            # DISTRIBUTION, so unfilled slots take the remaining pair that moves
            # the running mean closest to target rather than leaving the tier short.
            if need:
                tgt = sum(sum(x) for x in target) / (2 * len(target))
                rest = [(a, b) for a, b, *_ in pool
                        if a not in used and b not in used
                        and (a, b) not in UNRELATED_VETO
                        and (b, a) not in UNRELATED_VETO]
                while need and rest:
                    cur = [nt(w) for pr in out for w in pr]
                    best = min(rest, key=lambda ab: abs(
                        (sum(cur) + nt(ab[0]) + nt(ab[1])) / (len(cur) + 2) - tgt))
                    out.append(best)
                    used.update(best)
                    rest.remove(best)
                    need.pop()
            return out, need

        for name, pool in (("SYNONYM", syn),
                           ("UNRELATED", [(a, b) for a, b in UNRELATED_JUDGED])):
            got, missing = match(pool, want)
            ts = [nt(w) for pr in got for w in pr]
            print(f"  {name:<11}matched {len(got)}/20  "
                  f"mean {sum(ts)/len(ts):.2f} tok/word"
                  + (f"  UNMATCHED signatures: {missing}" if missing else ""))
            for a, b in got:
                print(f"      {a:<14}{b:<14}({nt(a)},{nt(b)})")
        pts = [nt(w) for pr in p for w in pr]
        print(f"  {'PROXIMATE':<11}held fixed 20/20  mean {sum(pts)/len(pts):.2f} "
              f"tok/word (the target)")
    except Exception as e:
        print(f"\n(length matching unavailable: {e})")

    # TOKEN-LENGTH PROFILE, declared because subword pooling inflates cosine.
    # The encoder pools over a word's subwords, so longer words are pulled
    # toward a shared centroid and score MORE similar for reasons that are not
    # semantic (measured in the model's own space: +0.07 to +0.14, docket [488]).
    # If the tiers differ in token length the ordinal test is confounded -- so
    # the profile is printed with the direction of its bias stated.
    try:
        from transformers import AutoTokenizer
        bt = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        print("\n--- TOKEN-LENGTH PROFILE (bge-m3) ---")
        for name, pairs in (("SYNONYM", s), ("PROXIMATE", p), ("UNRELATED", u)):
            ts = [len(bt.encode(" " + w, add_special_tokens=False))
                  for pr in pairs for w in pr[:2]]
            if ts:
                print(f"  {name:<12}mean {sum(ts)/len(ts):.2f} tokens/word")
        print("  Longer words pool more and so score MORE similar. Measured on "
              "this set the\n  order is SYNONYM < PROXIMATE < UNRELATED, so the "
              "artifact pushes AGAINST the\n  required ordering -- it makes the "
              "gate harder to pass, never easier.")
    except Exception as e:                      # tokenizer optional, never fatal
        print(f"\n(token-length profile unavailable: {e})")
    print(f"\nseed {SEED} | emitted {len(s)} synonym, {len(p)} proximate, "
          f"{len(u)} unrelated")
    print("PASS iff cos(SYN) > cos(PROX) > cos(UNREL), with PROX-vs-UNREL the "
          "criterion the structural test depends on.")


if __name__ == "__main__":
    main()
