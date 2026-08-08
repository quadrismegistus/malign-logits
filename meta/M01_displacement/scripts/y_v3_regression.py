#!/usr/bin/env python
"""v3 re-smoke as a REGRESSION, run on the items v2 got wrong.

    python y_v3_regression.py

A random smoke can only tell you the new task runs. This one can FAIL, because
the items are chosen for a known answer: every passage that received a <moral>
span from deepseek under v2 was read by hand, one at a time, and sorted into
nine clean, four borderline and nine false positives -- horror, revulsion,
fury, an insult, an epithet, ambivalence, an appeal. The v3 instruction and the
v3 examples were written against exactly those nine. So the test is not "does
it parse", it is:

    the 9 false positives should come back WITHOUT <moral>
    the 9 clean ones should come back WITH it

and if the first number moves without the second holding, the instruction did
not get sharper, it got more timid -- which is the failure mode that looks like
success in every aggregate.

TWO THINGS THIS CANNOT TELL YOU, stated because the numbers will look precise.
The hand-read labels are one reader's, on 22 spans, and a second reader would
score the four borderlines differently. And the items were SELECTED for having
been marked, so the base rate here is nothing like the corpus base rate: this
measures the direction of the change on known cases, never a precision.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
#: BEFORE any import that reaches llm.py. Same trap as the pilot runner: the
#: stash root is read at import time, so setting it afterwards is a no-op that
#: looks like it worked.
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

import y_pilot_coder as Y

V2 = os.path.join(CAMP, "results", "y_pilot_coded_v2.jsonl")

#: THE HAND-READ LABELS. Keyed by a distinctive substring of the span so the
#: label travels with the text rather than with a row index -- a row index
#: would silently re-point if the pilot file is ever rebuilt.
FALSE_POSITIVE = [
    "I watched in horror as she gagged",
    "her old self screaming for her to stop",
    "Ryan was horrified and repulsed",
    "You are my daughter. You know I love you",
    "cock-sucking sniveling",
    "battling his conflicting feelings",
    "He is now completely furious",
    "nothing but a",                      # the "whore" insult, twice
]
TRUE_POSITIVE = [
    "this was crossing a boundary",
    "It's so wrong,",
    "It’s so wrong,",
    "what she did was wrong, and that he should not do it either",
    "You did nothing wrong",
    "shouldn't be doing that",
    "should not be allowed",
    "not appropriate behavior",
    "only people who show respect",
    "gave in to the evil that existed inside of him",
]
#: THE FOUR v2 BORDERLINES, SPLIT BY WHAT <guilt> IS SUPPOSED TO RECOVER.
#: Narrowing `moralisation_in_scene` to an evaluative utterance dropped all
#: four. Two of them are named self-directed feeling and should come back under
#: <guilt>; two are not feeling at all -- a sentence imposed and a resolution
#: taken -- and should stay dropped. Separating them here is what makes <guilt>
#: falsifiable rather than merely additional: a tag that recovered all four
#: would just be the old loose field under a new name.
GUILT_EXPECTED = [
    "Conscious guilt floated in her chest",
    "knew he was guilty",
]
GUILT_NOT_EXPECTED = [
    "her punishment for not killing the old woman",
    "determined to rein in his natural impulses",
]


def spanset(tagged, tag):
    """Character indices covered by `tag`. Local copy so this script does not
    depend on the agreement module's import side effects."""
    #: v2 VOCABULARY -- this reads y_pilot_coded_v2.jsonl and looks for
    #: <hesitation>, which v3 does not have. And the shared parser groups as
    #: `<(/?)(name)>`, so the NAME IS GROUP 2; reading group 1 gets the slash
    #: and the stack never matches anything.
    from malign_logits.tasks.code_y_superego_v3 import spans, V2_VOCAB
    _, cover = spans(tagged or "", vocab=V2_VOCAB)
    return cover[tag]


def plain(tagged):
    from malign_logits.tasks.code_y_superego_v3 import spans, V2_VOCAB
    return spans(tagged or "", vocab=V2_VOCAB)[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--limit", type=int, default=0, help="0 = all matched items")
    a = ap.parse_args(argv)

    Y.assert_root()
    G = Y.load()
    import random
    texts, metas = Y.build_items(G, 10, random.Random(Y.SEED))
    bykey = {}
    for t, m in zip(texts, metas):
        bykey[(m["pair"], m["role"], m["word"], m["seq_i"])] = (t, m)

    #: THE ITEMS ARRIVE FROM build_items ALREADY PREPARED, by v2's `prepare`.
    #: v3 did not touch that function, so the item text is byte-identical to
    #: what v2 was scored on -- which is the whole basis for calling this a
    #: regression rather than a new measurement. Asserting it costs one line;
    #: the alternative is a silent drift in what the two versions were shown,
    #: which would look like an instruction effect.
    import inspect
    from malign_logits.tasks import code_y_superego_v2 as _v2
    from malign_logits.tasks.code_y_superego_v3 import prepare, SuperegoV3Task
    assert inspect.getsource(prepare) == inspect.getsource(_v2.prepare), (
        "v3.prepare has diverged from v2.prepare -- the regression items are "
        "no longer the text v2 was scored on")

    v2rows = [json.loads(l) for l in open(V2)]
    picked, label = [], {}
    for d in v2rows:
        if not d["coder"].startswith("deepseek"):
            continue
        key = (d["pair"], d["role"], d["word"], d["seq_i"])
        if key not in bykey or key in label:
            continue
        tg = d.get("tagged") or ""
        txt = plain(tg)
        mcov = spanset(tg, "moral")
        hcov = spanset(tg, "hesitation")
        if mcov:
            marked = "".join(txt[i] for i in sorted(mcov))
            kind = None
            if any(s in marked for s in GUILT_EXPECTED):
                kind = "guilt-EXPECTED"
            elif any(s in marked for s in GUILT_NOT_EXPECTED):
                kind = "guilt-NOT-EXPECTED"
            elif any(s in marked for s in TRUE_POSITIVE):
                kind = "moral-TRUE"
            elif any(s in marked for s in FALSE_POSITIVE):
                kind = "moral-FALSE"
            else:
                kind = "moral-borderline"
            label[key] = kind
            picked.append(key)
        elif hcov:
            label[key] = "hesitation"
            picked.append(key)
    #: The quiz item, carried explicitly: it is the <meta> case and there is
    #: exactly one of it.
    for d in v2rows:
        if not d["coder"].startswith("deepseek"):
            continue
        key = (d["pair"], d["role"], d["word"], d["seq_i"])
        if key in label or key not in bykey:
            continue
        if "very hard with every motion" in plain(d.get("tagged") or ""):
            label[key] = "quiz-should-be-meta"
            picked.append(key)

    if a.limit:
        picked = picked[:a.limit]
    print("REGRESSION SET: %d items" % len(picked))
    for k, c in collections.Counter(label[k] for k in picked).most_common():
        print("   %-22s %d" % (k, c))
    print("\ncoder: %s   task: %s\n" % (a.model, SuperegoV3Task.name))

    items = [bykey[key][0] for key in picked]

    res = SuperegoV3Task().map(items, model=a.model, num_workers=8)

    out = os.path.join(CAMP, "results", "y_v3_regression.jsonl")
    rows = []
    with open(out, "w") as fh:
        for key, r in zip(picked, res):
            rec = {"pair": key[0], "role": key[1], "word": key[2], "seq_i": key[3],
                   "label": label[key], "coder": a.model,
                   "parsed": r is not None}
            if r is not None:
                rec.update(json.loads(r.model_dump_json()))
                rec["tag_field_mismatches"] = r.tag_field_mismatches()
            rows.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("wrote %s" % out)

    print("\n" + "=" * 78)
    print("DID THE INSTRUCTION GET SHARPER, OR ONLY MORE TIMID?")
    print("=" * 78)
    for want, name in (("moral-FALSE", "should now DROP <moral>"),
                       ("moral-TRUE", "should still CARRY <moral>"),
                       ("moral-borderline", "either way")):
        sub = [r for r in rows if r["label"] == want]
        if not sub:
            continue
        got = [r for r in sub if r.get("parsed") and "<moral>" in (r.get("tagged") or "")]
        print("  %-18s %-30s %2d of %2d still marked <moral>"
              % (want, name, len(got), len(sub)))

    #: <guilt> IS ONLY WORTH ADDING IF IT DISCRIMINATES. Recovering both of the
    #: self-directed cases AND both of the non-feeling ones would mean the tag
    #: is the old loose field wearing a new name, which is worse than the
    #: exclusion it was meant to repair.
    for want, name in (("guilt-EXPECTED", "should now CARRY <guilt>"),
                       ("guilt-NOT-EXPECTED", "should NOT carry <guilt>")):
        sub = [r for r in rows if r["label"] == want]
        if not sub:
            continue
        g = [r for r in sub if r.get("parsed") and "<guilt>" in (r.get("tagged") or "")]
        m = [r for r in sub if r.get("parsed") and "<moral>" in (r.get("tagged") or "")]
        print("  %-18s %-30s %2d of %2d carry <guilt>  (%d also <moral>)"
              % (want, name, len(g), len(sub), len(m)))
        for r in sub:
            print("       %-7s %-8s guilt_or_shame=%-4s moral=%-4s tags=%s"
                  % (r["role"], r["word"], r.get("guilt_or_shame"),
                     r.get("moralisation_in_scene"),
                     ",".join(sorted({t for t in ("moral", "guilt", "consent", "resist")
                                      if "<%s>" % t in (r.get("tagged") or "")})) or "-"))
    hes = [r for r in rows if r["label"] == "hesitation" and r.get("parsed")]
    if hes:
        c = collections.Counter()
        for r in hes:
            t = r.get("tagged") or ""
            c["<consent>" if "<consent>" in t else ""] += 1 if "<consent>" in t else 0
            c["<resist>"] += 1 if "<resist>" in t else 0
            c["neither"] += 1 if ("<consent>" not in t and "<resist>" not in t) else 0
        print("\n  the %d v2 <hesitation> items now split: consent %d, resist %d, neither %d"
              % (len(hes), c.get("<consent>", 0), c.get("<resist>", 0), c.get("neither", 0)))
    q = [r for r in rows if r["label"] == "quiz-should-be-meta" and r.get("parsed")]
    for r in q:
        t = r.get("tagged") or ""
        print("\n  quiz item: <meta> %s | <sexual> %s | sexual_scene=%s  (want meta yes, sexual no, NO)"
              % ("<meta>" in t, "<sexual>" in t, r.get("sexual_scene")))
    nf = sum(1 for r in rows if not r.get("parsed"))
    mm = sum(1 for r in rows if r.get("tag_field_mismatches"))
    print("\n  parse failures: %d of %d   soft-tier mismatches: %d" % (nf, len(rows), mm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
