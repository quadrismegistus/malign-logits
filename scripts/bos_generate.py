"""Generate unconditional (BOS-only) text from all layers of a model family.

Usage:
    python scripts/bos_generate.py --family smol --n 10
    python scripts/bos_generate.py --family olmo --n 100 --max-tokens 200
"""

import argparse

from malign_logits import MODEL_FAMILIES
from malign_logits.psyche import Psyche


def main():
    parser = argparse.ArgumentParser(description="BOS-only generation")
    parser.add_argument("--family", default="smol", choices=list(MODEL_FAMILIES))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    psyche = Psyche.from_family(args.family, load=True)
    tok = psyche.tokenizer
    prompt = tok.bos_token or tok.eos_token or ""

    print(f"family={args.family}  bos={prompt!r}  n={args.n}  max_tokens={args.max_tokens}", flush=True)

    psyche.generate(
        prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        n=args.n,
    )

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
