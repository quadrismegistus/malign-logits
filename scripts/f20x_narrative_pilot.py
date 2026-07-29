"""Pilot: does a NARRATIVE opener behave differently from the Q:/A: rung, and how
does that change with length?

    uv run .venv/bin/python scripts/f20x_narrative_pilot.py

WHY. Every measurement of the F20x drift effect -- beam search, the published
sampled finding, the third-person battery, the whole 2x2 -- has been on ONE prompt
format, the `Q:`/`A:` rung. That is a single-format dependency nobody has tested,
and the rung is also where nearly every artefact of the campaign came from: 63.5%
self-written Q/A looping, multiple-choice capture, a self-written `Q:` tail that
made fact opportunity arm-imbalanced, and a gate firing on a quarter to a half of
completions.

`Q:`/`A:` is also closer to instruction format, which advantages aligned models by
construction. "Let me tell you about X." is a plain continuation that base models
handle natively.

RH's question, and this pilot answers only the descriptive half of it: what do these
look like at 60, 100, 200 and 500 tokens? Specifically --

  - does the narrative opener avoid the Q/A loop, or fall into it later?
  - do base and aligned diverge in LENGTH the way they do on the rung
    (base bails at 67.7% against aligned 48.9%, base answers 19.8 words to 22.7)?
  - does the model RESTATE its subject -- the missing requirement for fact drift,
    which needs one thing described twice and never gets it on the rung?

ONE MODEL, ONE PAIR, DESCRIPTIVE ONLY. This decides whether a battery is worth
registering. It is not a measurement and no rate from it should be quoted.
"""
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAIR = [("base", "allenai/OLMo-2-0425-1B"),
        ("aligned", "allenai/OLMo-2-0425-1B-Instruct")]

PROMPTS = [
    ("1P-narr", "Let me tell you about myself."),
    ("3P-narr", "Let me tell you about her."),
    ("O-narr", "Let me tell you about the quern."),
    ("N-narr", "Let me tell you about a glorp."),
    ("1P-rung", "Q: Who are you?\nA:"),          # the incumbent, for contrast
]
LENGTHS = [60, 100, 200, 500]
N_DRAWS = 4
TEMP = 1.0
SEED0 = 20260729
OUT = "data/f20x_narrative_pilot.parquet"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = []
    for arm, mid in PAIR:
        print(f"loading {mid}", flush=True)
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float16, device_map="mps", trust_remote_code=True)
        model.eval()
        cell = 0
        for pid, prompt in PROMPTS:
            for n_tok in LENGTHS:
                ids = tok(prompt, return_tensors="pt").to(model.device)
                for k in range(N_DRAWS):
                    torch.manual_seed(SEED0 + cell)
                    cell += 1
                    with torch.no_grad():
                        o = model.generate(**ids, do_sample=True, temperature=TEMP,
                                           top_p=1.0, max_new_tokens=n_tok,
                                           pad_token_id=tok.eos_token_id)
                    text = tok.decode(o[0][ids.input_ids.shape[1]:],
                                      skip_special_tokens=True)
                    rows.append(dict(arm=arm, model_id=mid, pid=pid, prompt=prompt,
                                     n_tok=n_tok, draw=k, text=text))
                print(f"  {arm:7s} {pid:8s} {n_tok:4d} tok  x{N_DRAWS}", flush=True)
        del model
        import gc
        gc.collect()
        torch.mps.empty_cache()

    d = pd.DataFrame(rows)
    d.to_parquet(OUT, compression="zstd", index=False)
    print(f"\nwrote {OUT}: {len(d)} completions")


if __name__ == "__main__":
    main()
