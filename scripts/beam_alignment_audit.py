"""THE ALIGNMENT AUDIT. Commission [2405].

    .venv/bin/python scripts/beam_alignment_audit.py --sample 200

WHY. The beam stash's per-position probabilities are indexed by a
RE-TOKENIZATION, not by the generation. `beam.py`'s `_batched_tf` does

    prompt_ids = base_tok.encode(ptext)
    full       = base_tok.encode(ptext + " " + s.text)

so it DECODES the story to text and RE-ENCODES it rather than reusing the stored
ids. `tokens` is always 10 (the cap); the probability vectors are 4, 9, 10, 11
or 12. **In a third of cells the two indices disagree, and nothing stored aligns
them.**

Every positional claim — the depth profile, the slope over positions 1..9, the
construct's "damage at the words immediately following" — asserts that position
*j* is the *j*-th generated token. Under a re-segmentation it is the *j*-th
token of a different segmentation.

**LENGTH AGREEMENT IS NOT ALIGNMENT** ([2405].1). A re-encoding can preserve the
count while moving a boundary, so the equal-length cells are not certified by
being equal-length. That would be the truncation detector's error at the
position level: a coincidence dressed as a criterion. **This audit compares
TOKEN IDS.**

THE CLASSIFICATION, per [2405]:

    EXACT        re-encoded suffix ids == stored generation ids
    OFFSET-K     stored ids appear intact at a constant shift K; positions are
                 relabelable, then certified
    RESEGMENTED  the story re-segments internally; no relabeling recovers
                 position identity; UNALIGNABLE for positional readouts

**No probability value is opened. This reads `tokens`, `text`, the prompt, and
the tokenizer — nothing else.** The re-tokenization is deterministic given
those, which is the whole reason the audit is possible without spending
anything.
"""

import argparse
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def classify(stored, reenc):
    """EXACT / OFFSET-K / RESEGMENTED, on token IDS.

    OFFSET-K means the stored sequence survives INTACT as a contiguous run
    inside the re-encoding: a boundary merge added or consumed tokens at the
    start, and every generated token still exists as itself, K positions over.
    Anything else re-segments the story internally and no relabeling helps.
    """
    if not stored or not reenc:
        return "EMPTY", None
    if list(reenc) == list(stored):
        return "EXACT", 0
    n = len(stored)
    for k in range(0, len(reenc) - n + 1):
        if list(reenc[k:k + n]) == list(stored):
            return "OFFSET", k
    return "RESEGMENTED", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200,
                    help="beam_cross_v1 keys to audit (stated as a sample)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from malign_logits.cache import get_cache
    cm = get_cache()

    toks, counts = {}, collections.Counter()
    offsets = collections.Counter()
    per_model = collections.defaultdict(collections.Counter)
    n_keys = 0

    for k in cm.iter_beam_keys():
        if not isinstance(k, dict) or k.get("type") != "beam_cross_v1":
            continue
        if n_keys >= args.sample:
            break
        mid, ptext = k.get("model"), k.get("prompt")
        if not mid or ptext is None:
            continue
        if mid not in toks:
            try:
                toks[mid] = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            except Exception as e:
                counts["TOKENIZER-UNAVAILABLE"] += 1
                toks[mid] = None
        tok = toks[mid]
        if tok is None:
            continue
        n_keys += 1
        plen = len(tok.encode(ptext))
        for s in cm.get_beams(k) or []:
            stored = s.get("tokens") or []
            text = s.get("text") or ""
            #: the producer's EXACT concatenation, quoted at [2404].2
            full = tok.encode(ptext + " " + text)
            cls, off = classify(stored, full[plen:])
            counts[cls] += 1
            per_model[mid][cls] += 1
            if cls == "OFFSET":
                offsets[off] += 1

    tot = sum(counts.values())
    print(f"ALIGNMENT AUDIT — {n_keys} keys, {tot:,} beams, token IDS compared\n")
    for c, n in counts.most_common():
        print(f"  {c:<22}{n:>8,}  {n / tot * 100:5.1f}%")
    if offsets:
        print(f"\n  OFFSET distribution: {dict(sorted(offsets.items()))}")
    print(f"\n  POSITIONALLY USABLE (EXACT + OFFSET): "
          f"{(counts['EXACT'] + counts['OFFSET']) / tot * 100:.1f}%")
    print(f"  UNALIGNABLE (RESEGMENTED):            "
          f"{counts['RESEGMENTED'] / tot * 100:.1f}%")

    worst = sorted(per_model.items(),
                   key=lambda kv: -kv[1]["RESEGMENTED"] / max(sum(kv[1].values()), 1))
    print("\n  worst models by RESEGMENTED share:")
    for mid, c in worst[:5]:
        t = sum(c.values())
        print(f"    {mid:<44}{c['RESEGMENTED'] / t * 100:5.1f}%  (n={t})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
