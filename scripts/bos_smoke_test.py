"""Smoke test: generate from BOS token only (no prompt).

Measures pure source entropy — what each model layer "wants to say"
when given no steering from a prompt.
"""

from malign_logits.psyche import Psyche

FAMILY = "smol"

def main():
    psyche = Psyche.from_family(FAMILY, load=True)
    tok = psyche.tokenizer

    bos = tok.bos_token
    print(f"Family: {FAMILY}")
    print(f"bos_token={bos!r}  bos_token_id={tok.bos_token_id}")

    prompt = bos if bos else ""

    results = psyche.generate(
        prompt,
        max_new_tokens=100,
        temperature=1.0,
        n=5,
    )

    print(f"\n{'='*60}")
    print(f"Got {len(results)} results back")
    for i, r in enumerate(results):
        layers = [k for k in r if k != "prompt"]
        print(f"  [{i}] layers: {layers}")


if __name__ == "__main__":
    main()
