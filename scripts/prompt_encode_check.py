"""Precondition 7: does EVERY grid prompt survive encoding on EVERY arm it runs on?

    uv run .venv/bin/python scripts/prompt_encode_check.py

Preconditions 5 and 6 probed each tokenizer with a handful of FIXED strings.
This checks the actual grid: every (model, prompt) pair the spec will run.

The gap it closes is real. A probe set answers "is this tokenizer broken"; it
cannot answer "does THIS prompt survive THIS tokenizer". A prompt may carry a
character no probe used -- a curly apostrophe, a fullwidth comma, an emoji, a
rare CJK glyph -- and lose it silently, and the failure would surface as a
plausible number rather than an error.

WHAT COUNTS AS SURVIVING: decode(encode(p)) equals p after whitespace
normalisation. Anything else is a corrupted prompt and the pair is EXCLUDED, not
repaired -- with the exclusion recorded per [849].1, arm-wise.

Runs through twp_cloud.load_tokenizer, so it measures the loader the runner
uses, not AutoTokenizer's default. Tokenizers only; no weights.
"""
import csv, json, os, sys, importlib.util as ilu
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA

_sp = ilu.spec_from_file_location("tc", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "twp_cloud.py"))
_tc = ilu.module_from_spec(_sp); _sp.loader.exec_module(_tc)

SPEC = os.path.join(PATH_DATA, "grid_spec.json")
OUT = os.path.join(PATH_DATA, "prompt_encode_check.csv")


def main():
    spec = json.load(open(SPEC))
    total = sum(len(e["prompts"]) for e in spec)
    print(f"{len(spec)} models, {total:,} (model, prompt) pairs\n")
    fails, per_model, checked = [], Counter(), 0
    for i, entry in enumerate(spec, 1):
        mid = entry["model"]
        try:
            tok, loader = _tc.load_tokenizer(mid)
        except Exception as e:
            print(f"  [{i}/{len(spec)}] {mid[:44]:<46} LOAD FAILED {type(e).__name__}")
            fails.append(dict(model=mid, prompt="", reason=f"load:{type(e).__name__}"))
            continue
        bad = 0
        for p in entry["prompts"]:
            checked += 1
            try:
                ids = tok.encode(p, add_special_tokens=False)
                back = tok.decode(ids)
            except Exception as e:
                fails.append(dict(model=mid, prompt=p[:80], reason=type(e).__name__))
                bad += 1; continue
            if not ids:
                fails.append(dict(model=mid, prompt=p[:80], reason="empty_ids"))
                bad += 1; continue
            if " ".join(back.split()) != " ".join(p.split()):
                fails.append(dict(model=mid, prompt=p[:80], reason="roundtrip"))
                bad += 1
        per_model[mid] = bad
        flag = f"  *** {bad} FAIL" if bad else ""
        if bad or i % 20 == 0:
            print(f"  [{i}/{len(spec)}] {mid[:44]:<46}{loader:<10}{flag}", flush=True)

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "prompt", "reason"])
        w.writeheader(); w.writerows(fails)
    print(f"\nchecked {checked:,} pairs.  FAILURES {len(fails)}")
    if fails:
        print("by reason:", dict(Counter(f["reason"] for f in fails)))
        print("by model:")
        for m, n in sorted(per_model.items(), key=lambda kv: -kv[1]):
            if n: print(f"   {m:<48}{n:>6}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
