"""Is the DEICTIC cell realizable across all four format levels, or does it drop?

    uv run .venv/bin/python scripts/f20x_format_pilot.py

WHY. The four-level ordering (rung / prose-question / narrative / document) requires
the IDENTICAL stimulus in every cell -- within-`N-bare` stimulus range is 0.089
against a level of +0.085, so an unpaired format contrast cannot separate
"attenuated" from "persisted". Registrar's draft lists `O-deictic` as an open item
for this seat: does `that` survive the non-question frames?

THE WORRY IS NOT THAT IT READS AWKWARDLY. It is that `that` may not be the SAME
manipulation across formats:

    rung / prose-question   "What is that?"        SITUATIONAL deixis -- points at
                                                   an object in a scene that does
                                                   not exist
    narrative / document    "Let me tell you       DISCOURSE anaphora -- refers back
                            about that."           to prior text that does not exist

Both lack an antecedent, which is what RH added the condition to test. But they lack
DIFFERENT antecedents, and a stimulus that changes kind across the format axis is
exactly what the identical-stimulus constraint forbids. If the model resolves the two
differently -- inventing a scene in one and inventing prior discourse in the other --
the deictic cell is measuring format twice rather than once.

DESCRIPTIVE, ONE MODEL PAIR, NO RATES QUOTED. Same standing as the narrative pilot:
this decides whether a cell enters the battery, and nothing else. A control stimulus
(`a glorp`, which has no antecedent problem in any frame) runs alongside so that
anything seen in the deictic cell can be checked against a stimulus where the frame
is known to be fine.
"""
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402

OUT = os.path.join(PATH_DATA, "f20x_format_pilot.parquet")
PAIR = [("base", "allenai/OLMo-2-0425-1B"),
        ("aligned", "allenai/OLMo-2-0425-1B-Instruct")]

# Four format levels x two stimuli. The deictic is the cell under test; the nonce
# word is the control -- it has no antecedent problem in any frame, so a difference
# that appears for `that` and not for `glorp` is about deixis and not about format.
CELLS = [
    ("deictic", "rung",     "Q: What is that?\nA:"),
    ("deictic", "prose_q",  "Someone asks you: What is that? You reply:"),
    ("deictic", "narrative", "Let me tell you about that."),
    ("deictic", "document",  "That is"),
    ("nonce",   "rung",     "Q: What is a glorp?\nA:"),
    ("nonce",   "prose_q",  "Someone asks you: What is a glorp? You reply:"),
    ("nonce",   "narrative", "Let me tell you about a glorp."),
    ("nonce",   "document",  "A glorp is"),
]
N_DRAWS = 6
MAX_TOK = 200
TEMP = 1.0
SEED0 = 20260729


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = []
    for arm, mid in PAIR:
        print(f"\n=== {arm}  {mid} ===", flush=True)
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.float16, device_map="mps")
        model.eval()
        for ci, (stim, fmt, prompt) in enumerate(CELLS):
            torch.manual_seed(SEED0 + ci)
            enc = tok(prompt, return_tensors="pt").to("mps")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=TEMP,
                                     max_new_tokens=MAX_TOK,
                                     num_return_sequences=N_DRAWS,
                                     pad_token_id=tok.eos_token_id)
            for d in range(N_DRAWS):
                text = tok.decode(gen[d][enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
                rows.append({"arm": arm, "model_id": mid, "stimulus": stim,
                             "format": fmt, "prompt": prompt, "draw": d,
                             "text": text, "n_words": len(text.split())})
            print(f"  {stim:8s} {fmt:10s} done", flush=True)
        del model
        torch.mps.empty_cache()

    d = pd.DataFrame(rows)
    d.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  {len(d)} rows\n")

    # Descriptive only. The question is whether `that` acquires a referent at all,
    # and of what kind -- reported as text for a human to read, not as a rate.
    print(d.groupby(["stimulus", "format", "arm"]).n_words.mean().unstack()
          .round(1).to_string())
    for stim in ("deictic", "nonce"):
        for fmt in ("rung", "prose_q", "narrative", "document"):
            print(f"\n{'='*74}\n{stim} / {fmt}\n{'='*74}")
            for arm in ("base", "aligned"):
                s = d[(d.stimulus == stim) & (d["format"] == fmt) & (d.arm == arm)]
                if len(s):
                    print(f"  [{arm}] {s.iloc[0].text.strip()[:330]}")


if __name__ == "__main__":
    main()
