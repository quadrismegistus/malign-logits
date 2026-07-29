"""Audit RH's defect: F13's word probabilities are FIRST-TOKEN probabilities.

    uv run .venv/bin/python scripts/f13_token_collision_audit.py

`core.py:score_words_from_logits` takes token_ids[0] for every candidate word, so
any two words sharing a first token carry the SAME probability at every layer in
both arms. lacan measured this on the vocabularies ([449]). This measures what it
does to the ROWS -- the denominator F13's numbers were actually computed over --
because a defect affecting 36% of a vocabulary and a defect affecting 36% of the
pairs are different sizes, and only the second is the finding's exposure.

Tokenizers only. No model weights, no GPU: the format battery owns MPS.
"""
import os
import sys
from collections import defaultdict

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import MODEL_FAMILIES, PATH_DATA  # noqa: E402

FAMS = ["olmo", "olmo-tiny", "llama", "qwen", "zephyr", "tulu", "amber"]


def first_token(tok, word):
    """Reproduce score_words_from_logits' token choice EXACTLY, including the
    leading-whitespace skip -- an approximation of the approximation would
    measure a different defect than the one in the data."""
    ids = tok.encode(" " + word, add_special_tokens=False)
    if not ids:
        return None, 0
    tid = ids[0]
    if len(ids) > 1 and not tok.decode([tid]).strip():
        tid = ids[1]
    return tid, len(ids)


def main():
    print(f"{'family':<11}{'words':>7}{'multi-tok':>11}{'in collision':>14}"
          f"{'rows':>9}{'rows hit':>10}{'% rows':>8}")
    total = []
    for fam in FAMS:
        path = os.path.join(PATH_DATA, f"taxonomy_{fam}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path).dropna(subset=["source", "target"])
        words = sorted(set(d.source) | set(d.target))
        base_id = MODEL_FAMILIES[fam].base
        try:
            tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        except Exception as e:
            print(f"{fam:<11}  tokenizer unavailable: {str(e)[:40]}")
            continue

        by_tid, nmulti = defaultdict(list), 0
        for w in words:
            tid, n = first_token(tok, w)
            if tid is None:
                continue
            by_tid[tid].append(w)
            nmulti += (n > 1)
        colliding = {w for ws in by_tid.values() if len(ws) > 1 for w in ws}

        # THE ROW-LEVEL EXPOSURE: a pair is hit if EITHER member collides, because
        # either side's probability may belong to a different word.
        hit = d.source.isin(colliding) | d.target.isin(colliding)
        print(f"{fam:<11}{len(words):>7}{nmulti/len(words):>10.1%}"
              f"{len(colliding)/len(words):>13.1%}{len(d):>9,}{hit.sum():>10,}"
              f"{hit.mean():>7.1%}")
        total.append((fam, hit.mean(), sorted(
            [ws for ws in by_tid.values() if len(ws) > 1], key=len, reverse=True)))

    print("\nLargest collision groups per family (the words that share a number):")
    for fam, _, groups in total:
        for g in groups[:3]:
            print(f"  {fam:<11}{g}")


if __name__ == "__main__":
    main()
