#!/usr/bin/env python
"""Smoke `m02_opposition_v1` against passages whose right answer is known.

    uv run python z_opposition_smoke.py

NOT A PILOT AND NOT A GATE. Eleven passages chosen BECAUSE I read them and
formed a view, which is the opposite of a representative draw and makes every
rate here meaningless. What it can do is show whether the instrument behaves as
designed on cases where the design has an opinion -- and in particular whether
it says NONE where `tension_named` said YES.

The expectation column was written before the run. Where the coder disagrees
with it, the coder is not automatically wrong: two of these (`faith/doubt`,
`could be both`) are cases where I expect the DERIVATION to score OTHER/NONE on
a passage I judged a true naming, because `derive_match` is deliberately crude
and its recall cost on paraphrase is declared in the task docstring.
"""
import json
import os
import sys

os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from malign_logits.tasks.code_m02_opposition_v1 import (  # noqa: E402
    OppositionV1Task, code, derive_match, looped)

CAMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: (model, group, why it is here, what I expect)
WANT = [
    ("beaver-7b-v1.0", "f11_loyal", "true naming, lexical", "MATCH"),
    ("Qwen3-8B-Base", "f11_sensation", "true naming, lexical", "MATCH"),
    ("llama-7b", "f11_loyal", "true naming, PARAPHRASE", "OTHER, and that is the rule's cost"),
    ("SmolLM3-3B-Base", "f11_beauty_ugly", "true naming, no terms quotable", "NONE or PARTIAL"),
    ("llama-7b", "f11_love_he", "'he was all confused' -- generic affect", "NONE"),
    ("Amber", "f11_guilt", "'couldnt help feeling really guilty' -- one pole", "NONE"),
    ("AmberSafe", "f11_trust", "names a DIFFERENT tension", "OTHER"),
    ("zephyr-7b-beta", "f11_love", "loops the prompt four times", "degenerate / looped"),
    ("Qwen2.5-7B-Instruct", "f11_species", "'Write the above sentence...'", "exit_span present"),
    ("pythia-2.8b", "f11_parent", "DRIFT into EU parliament prose", "exit_span EMPTY"),
    ("Falcon3-7B-Base", "f11_love_he", "in-fiction, lurid", "exit_span EMPTY"),
]


def load():
    seen = {}
    for f in ("l2_treatment_paired500", "l2_treatment_paired100_v2",
              "l2_treatment_n100"):
        p = os.path.join(CAMP, "results", f + ".jsonl")
        for line in open(p):
            r = json.loads(line)
            seen.setdefault((r["model"], r["group"], r.get("sample_idx"),
                             (r.get("prompt") or "")[:40]), r)
    out = []
    for m, g, why, exp in WANT:
        hit = [r for r in seen.values()
               if r["model"].split("/")[-1] == m and r["group"] == g]
        if not hit:
            print("MISSING %s / %s" % (m, g))
            continue
        out.append((sorted(hit, key=lambda r: r.get("sample_idx") or 0)[0], why, exp))
    return out


def main():
    task = OppositionV1Task()
    rows = load()
    print("smoking %d passages\n" % len(rows))
    for r, why, exp in rows:
        print("=" * 78)
        print("%s  [%s]  %s" % (r["model"].split("/")[-1], r.get("arm"), r["group"]))
        print("  why here : %s" % why)
        print("  expect   : %s" % exp)
        print("  prompt   : %s" % r.get("prompt"))
        print("  text     : %s" % (r.get("text") or "")[:300].replace("\n", " "))
        try:
            out = code(task, r.get("prompt"), r.get("text"))
        except Exception as e:
            print("  !! %s: %s" % (type(e).__name__, str(e)[:300]))
            continue
        d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
        verdict = derive_match(d, r.get("pole_a"), r.get("pole_b"))
        print("  ---")
        print("  poles    : %s / %s" % (r.get("pole_a"), r.get("pole_b")))
        print("  span     : %r" % (d.get("opposition_span") or "")[:110])
        print("  terms    : %r  vs  %r" % (d.get("term_a"), d.get("term_b")))
        print("  DERIVED  : %s" % verdict)
        print("  exit_span: %r" % (d.get("exit_span") or "")[:90])
        print("  degen    : %s   looped(mech): %s   coerced: %s"
              % (d.get("degenerate"), looped(r.get("text")), d.get("coerced")))
        print("  old task : tension_named=%s  frame_exit=%s"
              % (r.get("tension_named"), r.get("frame_exit")))


if __name__ == "__main__":
    main()
