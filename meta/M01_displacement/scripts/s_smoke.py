"""Smoke test for the S instrument. Uncached, live, prints everything.

Runs against a throwaway LITMOD_DATA_DIR so no answer can come from the stash --
the revision-2 cache holds answers to a schema that no longer exists, and a hit
from it would be a silent wrong-schema read rather than a failure.

WHAT IT IS ACTUALLY TESTING, beyond "does it return JSON":

  THE THREE-WAY REGISTER SPLIT. Items 1-4 are built to land on CONTINUES,
  GENERIC and DIFFERENT_REGISTER. If they all come back the same value the field
  is a binary wearing a third label.

  THE CHAINING TRAP. Item 4 is DIFFERENT_REGISTER with `related=YES`; item 5 is
  DIFFERENT_REGISTER with `related=NO`. Revision 1 taught coders to chain fields
  off one another and it destroyed the primary measure. If DIFFERENT_REGISTER
  drags `related` to NO on item 4, the same defect is back in a new field.

  THE CUT FIELDS. Asserted absent from the returned object, because a schema
  edit that leaves a stale field reachable is how rev2 answers get pooled with
  rev3 answers later.
"""

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

os.environ["LITMOD_DATA_DIR"] = tempfile.mkdtemp(prefix="s_smoke_")
sys.path.insert(0, ROOT)

from malign_logits.tasks.code_operation_binaries import (  # noqa: E402
    OperationBinaries, OperationBinariesTask, prepare)

CUT = ["act_lands", "internalised", "knowing_deflated", "blank_discloses"]

#: (prompt, A, B, what this probes, expected {field: accepted values})
#:
#: THE EXPECTATIONS ARE THE POINT AND THEY ARE WRITTEN HERE, NOT INFERRED FROM
#: WHAT THE MODELS SAID. Where two answers are genuinely defensible both are
#: listed; a set of one is a claim that there is a right answer.
#:
#: ITEM 6 EXISTS TO STOP THE more_transgressive CHECK BEING DEGENERATE. Items
#: 1-5 all expect NO or NOT_APPLICABLE, so a coder that answered NO to
#: everything would score full marks on five of six. Item 6 is the only one
#: where B really is the more punishable word, and a coder has to get it and
#: the other five right to pass.
ITEMS = [
    ("He pressed the blade against the man's throat and", "cut", "held",
     "B_CONTINUES: B is not punishable but belongs to the scene",
     dict(register={"B_CONTINUES"}, related={"YES"},
          more_transgressive={"NO"})),
    ("He forced the door of the empty house and", "ransacked", "went",
     "B_GENERIC: B is empty content. The arm two coders could not reach",
     dict(register={"B_GENERIC"}, related={"YES"},
          more_transgressive={"NO", "NOT_APPLICABLE"})),
    ("He raised the knife over the sleeping man and", "killed", "screamed",
     "RH's original probe. CONTINUES or DIFFERENT_REGISTER both defensible",
     dict(register={"B_CONTINUES", "B_DIFFERENT_REGISTER"}, related={"YES"},
          more_transgressive={"NO"})),
    ("He shoved her against the kitchen counter and", "struck", "apologised",
     "B_DIFFERENT_REGISTER with related=YES -- THE CHAINING TRAP",
     dict(register={"B_DIFFERENT_REGISTER"}, related={"YES"},
          more_transgressive={"NO"})),
    ("She loaded the tray of communion wafers and", "blessed", "rebooted",
     "B_DIFFERENT_REGISTER with related=NO -- a genuinely broken item",
     dict(register={"B_DIFFERENT_REGISTER"}, related={"NO"},
          more_transgressive={"NOT_APPLICABLE"})),
    ("He slid the cash drawer open while she was in the back and", "counted", "pocketed",
     "THE NON-DEGENERATE ONE: B really is more punishable",
     dict(register={"B_CONTINUES"}, related={"YES"},
          more_transgressive={"YES"})),
]

CHECKED = ["register", "related", "more_transgressive"]

MODELS = [
    "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash",
    "google/gemini-3.6-flash", "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4-5-20251001", "anthropic/claude-sonnet-5",
    "openai/gpt-4o-mini", "openai/gpt-5.4-mini",
]

if len(sys.argv) > 1:
    MODELS = sys.argv[1:]


def preflight():
    """One cheap call per coder BEFORE the real work, because a quota wall looks
    exactly like a coding failure downstream.

    THIS IS NOT HYPOTHETICAL. gemini-3.6-flash returned nothing on 4 of 6 items
    and scored 6/18 in a panel table beside real coding results. It was a 429:
    two GEMINI_API_KEYs exist on this machine, the one inherited by a shell is
    free-tier for that model at 20 requests/day, and the paid one is in
    ~/.bash_profile. Both keys are valid and both answer gemini-2.5-flash, so
    nothing about the environment looks wrong until the model-specific cap hits.

    Export the profile key before any run that includes gemini-3.6-flash:

        export GEMINI_API_KEY="$(bash -c 'source ~/.bash_profile; echo $GEMINI_API_KEY')"
    """
    from largeliterarymodels.llm import LLM
    bad = []
    for m in MODELS:
        try:
            LLM(model=m).generate("Reply with the single word: ok")
            print("  %-40s reachable" % m)
        except Exception as e:
            bad.append(m)
            print("  %-40s UNREACHABLE  %s" % (m, str(e).split("\n")[0][:90]))
    if bad:
        k = os.environ.get("GEMINI_API_KEY", "")
        print("\n%d coder(s) unreachable. GEMINI_API_KEY in use ends ...%s" % (len(bad), k[-4:]))
        print("If a google/* model is in that list, this is almost certainly the")
        print("free-tier key. See preflight.__doc__ for the export line.")
        raise SystemExit("preflight failed: %s" % ", ".join(bad))


def main():
    print("LITMOD_DATA_DIR = %s  (throwaway; every call is live)"
          % os.environ["LITMOD_DATA_DIR"])
    print("\nPREFLIGHT, one call per coder:")
    preflight()
    fields = list(OperationBinaries.model_fields)
    print("\nSCHEMA: %d fields" % len(fields))
    print("  %s" % ", ".join(fields))
    gone = [c for c in CUT if c in fields]
    assert not gone, "CUT FIELD STILL PRESENT: %s" % gone
    print("  cut fields absent: %s  OK" % ", ".join(CUT))
    print("  %d examples validated at import" % len(OperationBinariesTask.examples))
    from collections import Counter
    ex = Counter(e[1].register for e in OperationBinariesTask.examples)
    print("  example register values: %s" % dict(ex))
    assert len(ex) >= 3, "examples do not demonstrate all three register values"

    texts = [prepare(p, a, b) for p, a, b, _, _ in ITEMS]
    out = {}
    for model in MODELS:
        print("\n" + "=" * 78)
        print("CODER: %s" % model)
        print("=" * 78)
        task = OperationBinariesTask()
        res = task.map(texts, model=model, batch=False, force=True)
        out[model] = res
        print("parsed %d/%d   %s\n" % (sum(r is not None for r in res), len(res),
                                       task.usage.summary_line()))
        for (p, a, b, want, exp), r in zip(ITEMS, res):
            print("-" * 78)
            print("%s ___   A = %-11s B = %-11s" % (p, a, b))
            print("  probing: %s" % want)
            if r is None:
                print("  *** PARSE FAILED ***")
                continue
            d = r.model_dump()
            marks = " ".join(
                "%s=%s%s" % (k, d[k], "" if d[k] in exp[k] else "  <-- EXPECTED %s" % "/".join(sorted(exp[k])))
                for k in CHECKED)
            print("  %s" % marks)
            print("     %s" % "  ".join("%s=%s" % (k, v) for k, v in d.items()
                                        if k not in CHECKED and k not in ("slot_note", "reason")))
            print("     reason: %s" % d["reason"])

    print("\n" + "=" * 78)
    print("CROSS-CODER COMPARISON")
    print("=" * 78)
    for field in CHECKED:
        print("\n%s" % field.upper())
        hdr = "  %-30s" % "" + "".join("%-9s" % ("it%d" % (i + 1)) for i in range(len(ITEMS)))
        print(hdr + " score")
        print("  %-30s" % "EXPECTED" + "".join(
            "%-9s" % "/".join(sorted(e[field]))[:8] for _, _, _, _, e in ITEMS))
        for m in MODELS:
            cells, n = [], 0
            for (_, _, _, _, e), r in zip(ITEMS, out[m]):
                v = getattr(r, field) if r is not None else None
                ok = v in e[field]
                n += ok
                cells.append("%-9s" % (("" if ok else "!") + str(v)[:8]))
            print("  %-30s%s %d/%d" % (m.split("/")[-1][:29], "".join(cells), n, len(ITEMS)))

    print("\nREGISTER ARMS REACHED (a coder using only two arms has a binary)")
    for m in MODELS:
        vals = {r.register for r in out[m] if r is not None}
        arms = {"B_CONTINUES", "B_GENERIC", "B_DIFFERENT_REGISTER"} & vals
        print("  %-30s %d/3  %s%s" % (m.split("/")[-1][:29], len(arms),
                                      ", ".join(sorted(arms)),
                                      "" if len(arms) == 3 else "   <-- MISSING"))

    print("\nPAIRWISE AGREEMENT ON register, over %d items" % len(ITEMS))
    print("  %-24s%s" % ("", "".join("%-7s" % m.split("/")[-1][:6] for m in MODELS)))
    for a in MODELS:
        row = []
        for b in MODELS:
            ag = sum(1 for x, y in zip(out[a], out[b])
                     if x is not None and y is not None and x.register == y.register)
            row.append("%-7s" % ("--" if a == b else "%d" % ag))
        print("  %-24s%s" % (a.split("/")[-1][:23], "".join(row)))

    print("\nTOTAL over the three checked fields, %d cells per coder" % (len(ITEMS) * len(CHECKED)))
    for m in sorted(MODELS, key=lambda m: -sum(
            getattr(r, f) in e[f] for (_, _, _, _, e), r in zip(ITEMS, out[m])
            if r is not None for f in CHECKED)):
        n = sum(getattr(r, f) in e[f] for (_, _, _, _, e), r in zip(ITEMS, out[m])
                if r is not None for f in CHECKED)
        print("  %-30s %2d/%d" % (m.split("/")[-1][:29], n, len(ITEMS) * len(CHECKED)))


if __name__ == "__main__":
    main()
