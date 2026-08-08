#!/usr/bin/env python
"""M02: code the clean 819 frame. Pilot gate first, census second.

    python m02_code_pilot.py --pilot            # 120 stratified, 2 families
    python m02_code_pilot.py --pilot --gate     # gate stats from what exists
    python m02_code_pilot.py --census           # the remaining 819

LABEL, STATED FIRST AND CARRIED EVERYWHERE. **This is EXPLORATORY.** The 819
frame is 7 triplets x 13 checkpoints x 3 roles at THREE samples per cell. It has
no powered cell anywhere ([5044].4), its BOTH cells are 1-3 words longer than
their poles with no conjunction control in the corpus ([5034].1), and its
ambient baseline would have to be imported from another battery ([5059]). It
cannot carry a primary and nothing here will be written as one.

What it IS good for, and why it runs now rather than after the redo:

  1. GATING the coder against real continuation text at the right length,
     which is free and has to happen before the new corpus lands anyway.
  2. A FIRST LOOK at the M02 question at coded grain, which nobody has taken.
     Per RH's standing directive ([5060]): look first, label honestly; the
     expensive failure is the question nobody asked.

The frame is verified clean -- 0 of 156 flagged for chat-wrapping or
space-stripping at [5044].3, and malign confirmed the raw path from the
producer side. So the text is genuine raw continuation.

BLINDING. `prepare()` receives pole terms, prompt and continuation only. The
runner shuffles arms within a (group, role) before writing the work list, and
the coder never sees checkpoint, arm, or role. Role is inferable from the prompt
and that is unavoidable; arm is not, and that is what matters.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

from malign_logits.cache import CacheManager                      # noqa: E402
#: VERSION IS A FLAG, NOT AN EDIT. v1 stays runnable so its gate result
#: remains reproducible from the same script that produced it; v2 re-gates on
#: the SAME slice and seed, because a new draw would confound the field
#: redefinition with the sample.
from malign_logits.tasks import code_m02_contradiction_v1 as V1     # noqa: E402
from malign_logits.tasks import code_m02_contradiction_v2 as V2     # noqa: E402
from malign_logits.tasks.code_m02_contradiction_v1 import prepare   # noqa: E402
GATE = V1.GATE

MANIFEST = os.path.join(CAMP, "results", "exit_contradiction_manifest.csv")
CATS = os.path.join(ROOT, "malign-logits", "data", "prompt_categorisation.json")
if not os.path.exists(CATS):
    CATS = os.path.join(CAMP, "..", "..", "data", "prompt_categorisation.json")
OUT = os.path.join(CAMP, "results", "m02_coded.jsonl")
FRAME = ["f11_love", "f11_captive", "f11_create", "f11_desire",
         "f11_loyal", "f11_sensation", "f11_trust"]
SEED = 20260808
#: TWO FAMILIES, DIFFERENT VENDORS. Same-vendor agreement measures a
#: shared prior, not reliability -- two nulls that match can be one error
#: twice. deepseek is the coder the fields were written against; openai is
#: the one meeting them cold, which is the harder and more informative side.
FAMILIES = ["deepseek/deepseek-v4-flash", "openai/gpt-5.4-mini"]


def catalogue():
    """group -> {role: prompt text}, and group -> (pole_a term, pole_b term)."""
    rows = json.load(open(CATS))
    rows = rows if isinstance(rows, list) else rows.get("prompts", [])
    text, poles = collections.defaultdict(dict), {}
    for r in rows:
        if not isinstance(r, dict) or r.get("group_id") not in FRAME:
            continue
        role = r.get("group_role")
        if role in ("POLE_A", "POLE_B", "BOTH"):
            text[r["group_id"]][role] = r.get("prompt") or ""
        pc = r.get("pair_contrast")
        if pc and "/" in pc:
            poles[r["group_id"]] = tuple(pc.split("/", 1))
    return text, poles


def worklist():
    """One row per passage in the clean frame, arms shuffled, arm never in it."""
    import csv
    man = list(csv.DictReader(open(MANIFEST)))
    byg = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in man:
        if r["group"] in FRAME:
            byg[r["group"]][r["checkpoint"]][r["role"]] = r
    text, poles = catalogue()
    st = CacheManager()._stash("generations")

    #: index the stash once by (model, prompt); reading 256k keys per lookup
    #: is the difference between seconds and an hour
    idx = collections.defaultdict(list)
    for k in st.keys():
        if k.get("temp") == 1.0:
            idx[(k.get("model"), k.get("prompt"))].append(k)

    rows = []
    for g in FRAME:
        pa, pb = poles.get(g, ("", ""))
        for ck, roles in byg[g].items():
            if len(roles) != 3:
                continue          #: complete checkpoints only
            #: THE DEEPSEEK CELLS ARE QUARANTINED ([5042]) -- chat-wrapped
            #: reasoning transcripts, not continuations. Excluded by name so
            #: the exclusion is visible rather than a count that silently
            #: differs from the manifest.
            if "DeepSeek-R1-Distill" in ck:
                continue
            for role in ("POLE_A", "POLE_B", "BOTH"):
                p = text[g].get(role)
                if not p:
                    continue
                for i, k in enumerate(sorted(idx.get((ck, p), []),
                                             key=lambda k: k.get("idx") or 0)):
                    try:
                        t = st.get(k)
                    except Exception:
                        continue
                    if not isinstance(t, str) or len(t.strip()) < 20:
                        continue
                    rows.append(dict(
                        mid="%s|%s|%s|%d" % (g, ck, role, i),
                        group=g, checkpoint=ck, role=role, seq=i,
                        pole_a=pa, pole_b=pb, prompt=p, continuation=t,
                        text_sha16=hashlib.sha256(t.encode()).hexdigest()[:16]))
    rng = random.Random(SEED)
    rng.shuffle(rows)             #: arms interleaved before anything reads them
    return rows


def pilot_slice(rows, n):
    """Stratified over (group, role), deterministic."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["group"], r["role"])].append(r)
    per = max(1, n // max(1, len(by)))
    rng = random.Random(SEED + 1)
    out = []
    for k in sorted(by):
        v = sorted(by[k], key=lambda r: r["mid"])
        rng.shuffle(v)
        out.extend(v[:per])
    return out


def load_done():
    if not os.path.exists(OUT):
        return {}
    done = {}
    for line in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        done[(r["mid"], r["coder"], r.get("ver", "v1"))] = r
    return done


def run(items, model, workers, ver):
    task = (V2.ContradictionV2Task if ver == "v2" else V1.ContradictionV1Task)()
    COMPOSITES = V2.COMPOSITES if ver == "v2" else V1.COMPOSITES
    errors = {}
    CHUNK = 400
    n_ok = 0
    for s in range(0, len(items), CHUNK):
        part = items[s:s + CHUNK]
        pit = [prepare(r["pole_a"], r["pole_b"], r["prompt"], r["continuation"])
               for r in part]
        res = task.map(pit, model=model, num_workers=workers, errors=errors)
        if len(res) != len(pit):
            raise RuntimeError("map returned %d for %d items; zip would truncate"
                               % (len(res), len(pit)))
        with open(OUT, "a", encoding="utf-8") as fh:
            for r, out in zip(part, res):
                row = dict(r)
                row.pop("continuation", None)   #: the text lives in the stash
                row["coder"] = model
                row["ver"] = ver
                row["parsed"] = out is not None
                if out is not None:
                    d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
                    row.update(d)
                    for name, fn in COMPOSITES.items():
                        row[name] = fn(row)
                    n_ok += 1
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("   %s: %d/%d written" % (model, min(s + CHUNK, len(items)), len(items)))
    return n_ok


def gate(ver="v2"):
    """Agreement between families, per field, with the rare-event clause."""
    done = load_done()
    by = collections.defaultdict(dict)
    for (mid, coder, v), r in done.items():
        if r.get("parsed") and v == ver:
            by[mid][coder] = r
    both = {m: v for m, v in by.items() if len(v) >= 2}
    print("\nGATE  pilot n=%d coded by both families (of %d coded at all)"
          % (len(both), len(by)))
    if not both:
        return
    FIELDS = (["scene_share_GUARD", "frame_exit", "refusal", "pole_a_alive",
               "pole_b_alive", "tension_remarked", "degenerate"] if ver == "v2"
              else ["in_scene", "frame_exit", "refusal", "pole_a_alive",
                    "pole_b_alive", "tension_remarked", "degenerate"])
    #: scene_share is ORDINAL. Its agreement is reported as the BINARY THE GUARD
    #: ACTUALLY USES (MOST/ALL vs NONE/SOME), because that is the dichotomy the
    #: collider guard thresholds on and therefore the one that has to clear.
    #: Exact-match on all four levels is printed separately below.
    for v in both.values():
        for r in v.values():
            if "scene_share" in r:
                r["scene_share_GUARD"] = ("YES" if r["scene_share"]
                                          in V2.IN_SCENE_LEVELS else "NO")
    print("  %-18s %7s %8s %8s %9s  %s"
          % ("field", "rate", "agree", "kappa", "pos-cell", "verdict"))
    print("  " + "-" * 76)
    for f in FIELDS:
        a = [v[FAMILIES[0]].get(f) for v in both.values() if FAMILIES[0] in v and FAMILIES[1] in v]
        b = [v[FAMILIES[1]].get(f) for v in both.values() if FAMILIES[0] in v and FAMILIES[1] in v]
        n = len(a)
        if not n:
            continue
        yy = sum(1 for x, y in zip(a, b) if x == "YES" and y == "YES")
        nn = sum(1 for x, y in zip(a, b) if x == "NO" and y == "NO")
        po = (yy + nn) / n
        rate = (sum(1 for x in a if x == "YES") + sum(1 for y in b if y == "YES")) / (2 * n)
        pa = sum(1 for x in a if x == "YES") / n
        pb = sum(1 for y in b if y == "YES") / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        kap = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        #: RARE-EVENT CLAUSE. Under 5%, kappa is unstable to a single flip and
        #: POSITIVE-CELL agreement is reported instead. A field firing twice
        #: gets no reliability estimate and is SAID to have none.
        rare = rate < GATE["rare_event_below"]
        pos = yy / max(1, sum(1 for x, y in zip(a, b) if x == "YES" or y == "YES"))
        if rare:
            v = "RARE: no kappa; pos-cell %.2f on %d positives" % (
                pos, sum(1 for x, y in zip(a, b) if x == "YES" or y == "YES"))
        else:
            v = "PASS" if kap >= GATE["kappa_floor"] else "FAIL (floor %.2f)" % GATE["kappa_floor"]
        print("  %-18s %6.1f%% %7.3f %8s %8.2f  %s"
              % (f, 100 * rate, po, ("%.3f" % kap) if not rare else "-", pos, v))
    #: the cross-check the redundancy exists for
    inc = sum(1 for v in both.values() for r in v.values()
              if r.get("resolves") == "NEITHER"
              and (r.get("pole_a_alive") == "YES" or r.get("pole_b_alive") == "YES"))
    tot = sum(len(v) for v in both.values())
    print("\n  NEITHER-with-a-live-pole: %d of %d codings (%.1f%%) -- legal by design,"
          "\n  reported because the partition and the pole fields are deliberately"
          "\n  redundant and their disagreement is the quality signal." % (inc, tot, 100 * inc / max(1, tot)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--ver", choices=("v1", "v2"), default="v2")
    ap.add_argument("--n", type=int, default=GATE["pilot_n"])
    a = ap.parse_args(argv)

    if a.gate and not (a.pilot or a.census):
        gate(a.ver)
        return 0

    rows = worklist()
    print("clean frame: %d passages, %d groups, %d checkpoints"
          % (len(rows), len({r["group"] for r in rows}),
             len({r["checkpoint"] for r in rows})))
    done = load_done()
    if a.pilot:
        sl = pilot_slice(rows, a.n)
        print("pilot slice: %d passages over %d strata, %d coder families"
              % (len(sl), len({(r["group"], r["role"]) for r in sl}), len(FAMILIES)))
        for m in FAMILIES:
            todo = [r for r in sl if (r["mid"], m, a.ver) not in done]
            print("  %s: %d to code (%d already done)" % (m, len(todo), len(sl) - len(todo)))
            if todo:
                run(todo, m, a.workers, a.ver)
        gate(a.ver)
    elif a.census:
        m = FAMILIES[0]
        todo = [r for r in rows if (r["mid"], m, a.ver) not in done]
        print("census: %d to code with %s (%d done)" % (len(todo), m, len(rows) - len(todo)))
        if todo:
            run(todo, m, a.workers, a.ver)
    return 0


if __name__ == "__main__":
    sys.exit(main())
