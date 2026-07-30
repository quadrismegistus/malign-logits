"""Roster precondition 6: does the PROMPT SURVIVE ENCODING, per model?

    uv run .venv/bin/python scripts/tokenizer_roundtrip_sweep.py

Preconditions 1-5 asked what a tokenizer CONTAINS (vocab size, CJK tokens, CJK
chars, added tokens, index divergence) and one asked what it does with CJK
input. None asked the general question: does encode->decode return the text we
sent, in ANY script?

deepseek-llm-7b forced it. Its English:

    'She was so angry she wanted to'  ->  'Shewassoangryshewantedto'

Every space silently deleted. Seven ids, no exception, no UNK -- transformers v5
(#45488) installs a SentencePiece Metaspace pre-tokenizer over the ByteLevel one
the repo declares; whitespace fails to remap, and `unk_token: null` means
nothing raises. The CJK sweep caught this model for the wrong reason: it drops
Chinese completely, so it looked like a CJK problem. It is a WHITESPACE problem
that also destroys Chinese.

WHAT IS COMPARED. Exact round-trip first; then a whitespace-normalised compare,
because a tokenizer may legitimately alter leading/trailing space. A model that
fails the second has corrupted the prompt in a way no downstream number can
survive.

PROBES span the scripts and shapes the battery actually uses -- plain English,
contractions, digits with punctuation, CJK, and mixed -- because a defect can be
script-specific (deepseek: whitespace AND CJK) or shape-specific.
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import MODEL_FAMILIES, PATH_DATA
from transformers import AutoTokenizer

PROBES = {
    "english":   "She was so angry she wanted to",
    "contract":  "He didn't know what she'd say",
    "numeric":   "The salary is $100,000 and rising",
    "cjk":       "她非常生气，想要",
    "mixed":     "The word 她 means she",
}


def main():
    seen, rows = set(), []
    for fam, F in sorted(MODEL_FAMILIES.items()):
        for pos in ("base", "ego", "superego", "reinforced_superego"):
            mid = getattr(F, pos, None)
            if not mid or mid in seen:
                continue
            seen.add(mid)
            try:
                tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            except Exception as e:
                rows.append(dict(model=mid, family=fam, tokenizer=type(e).__name__,
                                 **{k: "LOADFAIL" for k in PROBES}, verdict="UNMEASURED"))
                continue
            r = dict(model=mid, family=fam, tokenizer=type(tok).__name__)
            bad = []
            for name, probe in PROBES.items():
                ids = tok.encode(probe, add_special_tokens=False)
                back = tok.decode(ids)
                if back.strip() == probe.strip():
                    r[name] = "exact"
                elif " ".join(back.split()) == " ".join(probe.split()):
                    r[name] = "ws-only"
                else:
                    r[name] = "CORRUPT"
                    bad.append(name)
            r["verdict"] = "CORRUPT:" + ",".join(bad) if bad else "clean"
            rows.append(r)
            if bad:
                print(f"  *** {mid[:48]:<50}{type(tok).__name__[:20]:<22}{r['verdict']}",
                      flush=True)

    out = os.path.join(PATH_DATA, "tokenizer_roundtrip.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["verdict"].startswith("CORRUPT")]
    print(f"\n{len(rows)} models.  CLEAN {len(rows)-len(bad)}   CORRUPT {len(bad)}")
    for r in bad:
        print(f"   {r['model']:<46}{r['tokenizer'][:18]:<20}{r['verdict']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
