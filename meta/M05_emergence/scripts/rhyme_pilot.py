"""PILOT: do the OLMo ladder rungs rhyme, given the first 5 lines of a poem?

RH's word, 2026-08-13. plan_rhyme.md's generative arm at pilot grade:
primers are the first 5 real lines (line_real, line_num 1-5) of poems from
the paper's own completion roster (genai_rhyme_completions.csv.gz), so the
HUMAN continuation (line_real 6+) scores as the built-in baseline.

Pure continuation via HF generate, NO chat template on any rung -- the base
rung cannot follow instructions, and template effects are part of what the
stuckness question is about; a chat-formatted variant is a later arm, not
this one. Scoring: prosodic (pinned at the paper's commit) on the
CONTINUATION lines only; perfect rime = distance 0; the paper's
rhyming-lines-per-10 quantity beside it.

PILOT NUMBERS ARE NEVER QUOTED AS RESULTS.

Output: meta/M05_emergence/data/rhyme_pilot.parquet (one row per
continuation: model, id_human, sample_idx, n_lines, n_rhyming, n_perfect,
per10) + the same for the human baselines (model='human/line_real').
"""

import json
import os
import sys

import pandas as pd

REPO = os.path.expanduser("~/github/malign-logits")
CSV = os.path.expanduser(
    "~/github/generative-formalism1/data/data_as_in_paper/genai_rhyme_completions.csv.gz")
OUT = os.path.join(REPO, "meta/M05_emergence/data/rhyme_pilot.parquet")

LADDER = [
    "allenai/Olmo-3-1025-7B",           # base
    "allenai/Olmo-3-7B-Instruct-SFT",   # SFT
    "allenai/Olmo-3-7B-Instruct-DPO",   # DPO
    "allenai/Olmo-3-7B-Instruct",       # RLVR (final)
]
N_PRIMERS = 12
N_SAMPLES = 2
MAX_NEW = 160
TEMP = 1.0


def pick_primers():
    df = pd.read_csv(CSV)
    df5 = df[df.first_n_lines == 5]
    rows = []
    seen = set()
    # spread across collections; need >= 10 human continuation lines
    for id_human, g in df5.groupby("id_human"):
        if id_human in seen:
            continue
        g0 = g[g.id == g.id.iloc[0]].sort_values("line_num")
        primer = g0[g0.line_num <= 5]["line_real"].tolist()
        human_cont = g0[(g0.line_num > 5)]["line_real"].dropna().tolist()
        if len(primer) == 5 and len(human_cont) >= 8 and all(isinstance(x, str) for x in primer):
            rows.append({"id_human": id_human,
                         "primer": "\n".join(primer),
                         "human_cont": "\n".join(human_cont[:12])})
            seen.add(id_human)
        if len(rows) >= N_PRIMERS:
            break
    return rows


def rhyme_score(txt):
    """EXACT mirror of generative_formalism get_rhyme_for_txt semantics:
    max_dist = RHYME_MAX_DIST = 1 (constants.py:351); rhyming/perfect counted
    over the SET OF LINES participating (union of both members per pair),
    perfect = score 0 (the paper's exact-rime criterion); denominator =
    prosodic's text.num_lines."""
    import prosodic
    lines = [l for l in txt.splitlines() if l.strip()]
    if len(lines) < 4:
        return None
    try:
        t = prosodic.Text("\n".join(lines))
        rhyme_d = t.get_rhyming_lines(max_dist=1)
        all_rhyming, all_perfect = set(), set()
        for l1, (score, l2) in rhyme_d.items():
            all_rhyming.update({l1, l2})
            if not score:
                all_perfect.update({l1, l2})
        n_lines = t.num_lines
    except Exception:
        return None
    if not n_lines:
        return None
    return {"n_lines": n_lines,
            "n_rhyming": len(all_rhyming), "n_perfect": len(all_perfect),
            "rhyming_per10": 10.0 * len(all_rhyming) / n_lines,
            "perfect_per10": 10.0 * len(all_perfect) / n_lines}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    primers = pick_primers()
    print(f"{len(primers)} primers selected", flush=True)
    rows = []

    # human baseline first (free)
    for p in primers:
        sc = rhyme_score(p["human_cont"])
        if sc:
            rows.append({"model": "human/line_real", "id_human": p["id_human"],
                         "sample_idx": 0, **sc})
    print("human baseline scored", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    for model_id in LADDER:
        print(f"== {model_id}", flush=True)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(dev).eval()
        for p in primers:
            enc = tok(p["primer"] + "\n", return_tensors="pt").to(dev)
            for si in range(N_SAMPLES):
                with torch.no_grad():
                    out = model.generate(
                        **enc, max_new_tokens=MAX_NEW, do_sample=True,
                        temperature=TEMP, top_p=1.0,
                        pad_token_id=tok.eos_token_id)
                cont = tok.decode(out[0][enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
                sc = rhyme_score(cont)
                row = {"model": model_id, "id_human": p["id_human"],
                       "sample_idx": si}
                if sc:
                    row.update(sc)
                row["text"] = cont[:2000]
                rows.append(row)
            print(f"   {p['id_human']} done", flush=True)
        del model
        if dev == "mps":
            torch.mps.empty_cache()
        pd.DataFrame(rows).to_parquet(OUT)  # checkpoint after each model
        print(f"   checkpointed {len(rows)} rows", flush=True)

    pd.DataFrame(rows).to_parquet(OUT)
    print(f"wrote {OUT}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
