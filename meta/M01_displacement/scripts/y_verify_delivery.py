#!/usr/bin/env python
"""Independent verification of the Y generation delivery, from the consumer's seat.

    python y_verify_delivery.py

malign verified this before releasing the boxes ([4986]) and their checks were
the right ones. This is not a re-audit of their work: it is the CONSUMER
reading the corpus it is about to spend an annotation budget on, which is a
different job and catches a different class of problem -- a file that exists
and parses can still not contain what the analysis needs.

**Counted against the SPEC, not against what arrived.** The spec is
`registration_y_slots.json`: 34 cells per model (29 forced + 5 undisturbed)
over 5 prompts. A census that reports what is present cannot see what is
absent, which is why the cell list comes from the spec and the delivered data
is checked into it rather than enumerated out of it.
"""
import collections
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
DATA = os.path.join(ROOT, "data", "raw")
SPEC = os.path.join(CAMP, "registrations", "registration_y_slots.json")


def main():
    spec = json.load(open(SPEC))
    want = set()
    for p in spec["prompts"]:
        for c in p["cells"]:
            want.add((p["prompt_id"], c["word"]))
    print("SPEC: %d prompts, %d cells per model" % (len(spec["prompts"]), len(want)))
    print("      max_tokens=%s n_samples=%s design=%s"
          % (spec["run"]["max_tokens"], spec["run"]["n_samples"], spec["run"]["design"]))

    files = sorted(glob.glob(os.path.join(DATA, "y_y-*", "*.jsonl")))
    print("\nfiles: %d across %d directories" % (
        len(files), len({os.path.dirname(f) for f in files})))

    by_model = collections.defaultdict(set)     # model -> {(prompt_id, word)}
    nseq = collections.Counter()
    tok_len, bad, designs, arms = [], collections.Counter(), collections.Counter(), collections.Counter()
    short = []                                  # the <=2 token population
    #: FAILED.jsonl records sit in the same directories and are NOT data: one
    #: row per failed attempt carrying the exception and traceback. That is
    #: malign's attrition ledger written down beside the corpus rather than
    #: summarised in a post, which is the right place for it -- but a reader
    #: that globs *.jsonl will treat them as malformed generations and report
    #: an integrity problem that does not exist.
    failed = collections.Counter()
    for f in files:
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                bad["unparseable line"] += 1
                continue
            if "error" in r and "sequences" not in r:
                failed[r.get("error", "?")] += 1
                continue
            m = r.get("model")
            pid = r.get("prompt_id") or r.get("prompt")
            by_model[m].add((pid, r.get("word")))
            designs[r.get("design")] += 1
            arms[r.get("arm")] += 1
            seqs = r.get("sequences") or []
            nseq[m] += len(seqs)
            for s in seqs:
                tk = s.get("tokens")
                if tk is None:
                    bad["sequence with no tokens"] += 1
                    continue
                if s.get("full_ids") is None:
                    bad["sequence with no full_ids"] += 1
                tok_len.append(len(tk))
                if len(tk) <= 2:
                    short.append((m, r.get("word"), pid, (s.get("text") or "")[:40], len(tk)))

    print("\nMODELS: %d" % len(by_model))
    print("designs: %s" % dict(designs))
    print("arms   : %s" % dict(arms))
    print("sequences: %d  |  token length median %d  min %d  max %d"
          % (sum(nseq.values()), statistics.median(tok_len), min(tok_len), max(tok_len)))
    print("integrity: %s" % (dict(bad) or "no missing tokens or full_ids"))
    print("attrition ledger (FAILED.jsonl, not data): %d rows -- %s"
          % (sum(failed.values()), dict(failed.most_common(6))))

    #: THE CHECK THAT MATTERS: cells the SPEC declares and the data lacks.
    print("\nCELL COVERAGE against the spec's %d" % len(want))
    incomplete = []
    for m, got in sorted(by_model.items(), key=lambda kv: str(kv[0])):
        missing = want - got
        extra = got - want
        if missing or extra:
            incomplete.append((m, len(missing), len(extra)))
    if incomplete:
        for m, nm, nx in incomplete[:12]:
            print("   %-52s missing %2d  unexpected %2d" % (m[:52], nm, nx))
        print("   %d of %d models incomplete" % (len(incomplete), len(by_model)))
    else:
        print("   every model carries every declared cell")

    print("\nSEQUENCES PER MODEL: %s" % (
        "all %d" % nseq.most_common(1)[0][1]
        if len(set(nseq.values())) == 1
        else "UNEVEN -- %s" % dict(collections.Counter(nseq.values()))))

    #: THE <=2 TOKEN POPULATION, handed over by malign at [4986] rather than
    #: filtered. Not truncation: a complete observation of a model closing the
    #: sentence and stopping. It splits a denominator -- it counts fully toward
    #: a refusal rate and contributes nothing to a per-token hazard -- so it is
    #: characterised here and never silently pooled.
    print("\n" + "=" * 78)
    print("THE <=2 TOKEN POPULATION: %d of %d (%.2f%%)"
          % (len(short), len(tok_len), 100 * len(short) / len(tok_len)))
    if short:
        arm_of = {}
        for m in by_model:
            arm_of[m] = None
        byw = collections.Counter(w for m, w, p, t, n in short)
        print("  by forced word: %s" % dict(byw.most_common(8)))
        print("  by length     : %s" % dict(collections.Counter(n for *_, n in short)))
        print("  distinct texts: %d" % len({t.strip() for *_, t, n in short}))
        for t, c in collections.Counter(t.strip() for *_, t, n in short).most_common(6):
            print("     %-30r %d" % (t, c))
        topm = collections.Counter(m for m, *_ in short).most_common(6)
        print("  top models    :")
        for m, c in topm:
            print("     %-52s %d" % (m[:52], c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
