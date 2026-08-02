"""The check that matters: do a family's arms tokenize a scored WORD identically?

    uv run .venv/bin/python scripts/check_word_tokenizer_equivalence.py

WHY THE PROMPT-LEVEL CHECK WAS NOT ENOUGH, and this is the point of the file.
check_tokenizer_equivalence.py tokenizes every PROMPT in every arm and found 42 of 46
families clean. Zephyr was among the clean ones -- and zephyr is the family already known
to double-encode a leading space. Measured:

    'He picked up the knife and'   base [1, 650, 7715, ...]   aligned [1, 650, 7715, ...]   SAME
    ' kill'                        base [1, 4015]             aligned [1, 28705, 4015]      DIFFERS
    ' scream'                      base [1, 8933]             aligned [1, 28705, 8933]      DIFFERS

The prompt agrees; the WORD does not. The defect lives at the word boundary, and the
prompt-level check cannot see it because a prompt's internal spaces are interior to the
sequence while a scored word's leading space is at position 0 of its own encoding.

AND THE WORD IS THE MEASURED QUANTITY. Every displacement number in this project is a
difference of WORD probabilities between two arms. If the arms encode ' kill' as two token
sequences of different length, then p(' kill') is being computed over different objects and
the delta is partly a tokenization delta. `kill -> scream` is the paper's exhibit pair and
BOTH words differ for zephyr.

So: a clean prompt-level result is not a clean bill of health, it is a clean bill for the
prompt. This runs the same comparison over the scored vocabulary, with the leading space
that the scorer actually supplies.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "word_tokenizer_equivalence.json")
ARMS = ("base", "ego", "superego", "reinforced_superego")
_c = {}


def tok(mid):
    if mid in _c:
        return _c[mid]
    from transformers import AutoTokenizer
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            t = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    except Exception:
        t = None
    _c[mid] = t
    return t


def words():
    """The ACTUAL scored vocabulary, harvested from true_word_probs payloads.

    A CORRECTION TO THIS SCRIPT'S FIRST VERSION, and the reason matters for a check that
    is now a roster precondition. The first version scraped string lists out of the
    taxonomy module and found 49 words. taxonomy.py holds PROMPTS, not candidate words --
    the scored vocabulary lives in the cache, one row per (word, first_token) in each
    true_word_probs payload. So the sweep that reported "zephyr is the only affected
    family" ran on 49 words, not on the vocabulary, and I described it as a full sweep.

    The CONCLUSION survived the correction because the defect is structural rather than
    lexical -- zephyr prepends a bare space token before EVERY space-prefixed token, so it
    shows on any word -- but "no other family is affected" is a much stronger claim on
    1,805 words than on 49, and it was the claim being made.

    Built by scripts/check_word_tokenizer_equivalence.py itself on first run and cached to
    data/scored_vocabulary.json so the precondition does not need the cache to re-run.
    """
    path = os.path.join(ROOT, "data", "scored_vocabulary.json")
    if os.path.exists(path):
        return json.load(open(path))
    from malign_logits.cache import get_cache
    cm = get_cache()
    out, n = set(), 0
    for _k, pay in cm.iter_items("true_word_probs"):
        n += 1
        for r in (pay.get("rows") or []):
            w = (r.get("word") or "").strip()
            if w and w.isascii() and w.isalpha():
                out.add(w)
        if n >= 60:
            break
    out = sorted(out)
    json.dump(out, open(path, "w"))
    return out


def main(limit):
    import malign_logits.taxonomy as T
    W = words()
    if limit:
        W = W[:limit]
    print(f"{len(W)} distinct scored words x arms of {len(T.MODEL_FAMILIES)} families")
    print("comparing BOTH bare 'word' and ' word' with the leading space the scorer supplies\n")

    report = {}
    for name, f in T.MODEL_FAMILIES.items():
        loaded = [(a, tok(m)) for a in ARMS for m in [getattr(f, a, None)] if m]
        loaded = [(a, t) for a, t in loaded if t is not None]
        if len(loaded) < 2:
            continue
        bare = lead = 0
        ex = []
        for w in W:
            for form, s in (("bare", w), ("leading_space", " " + w)):
                ids = {}
                for a, t in loaded:
                    try:
                        ids[a] = tuple(t(s, add_special_tokens=False).input_ids)
                    except Exception:
                        ids[a] = ("ERR",)
                if len({v for v in ids.values()}) > 1:
                    if form == "bare":
                        bare += 1
                    else:
                        lead += 1
                    if len(ex) < 4:
                        ex.append({"word": w, "form": form,
                                   "ids": {a: list(v) for a, v in ids.items()}})
        if bare or lead:
            report[name] = {"n_words": len(W), "bare_differs": bare,
                            "leading_space_differs": lead,
                            "rate_leading": round(lead / len(W), 4), "examples": ex}
            print(f"  {name:<22}bare {bare:>5}   ' word' {lead:>5}   "
                  f"({100*lead/len(W):.1f}% of the vocabulary)")
        else:
            print(f"  {name:<22}clean")

    print(f"\n{'='*76}")
    print(f"families whose arms disagree on a SCORED WORD: {len(report)}")
    for k, v in sorted(report.items(), key=lambda x: -x[1]["rate_leading"]):
        print(f"\n  {k}: {v['leading_space_differs']}/{v['n_words']} with a leading space "
              f"({100*v['rate_leading']:.1f}%), {v['bare_differs']} bare")
        for e in v["examples"][:2]:
            print(f"      {e['form']:<14}{e['word']!r}")
            for a, ids in e["ids"].items():
                print(f"         {a:<22}{ids}")
    json.dump({"_n_words": len(W), "_families_disagreeing": len(report),
               "families": report}, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args().limit)
