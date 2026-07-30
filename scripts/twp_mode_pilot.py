"""Comparison-2 pilot: does prompt FORMAT move the word distribution?

    uv run .venv/bin/python scripts/twp_mode_pilot.py --n-prompts 40

Decides whether the 41,418-cell comparison-2 run is worth spending. Pre-declared
per [828].2(b) / [830]: the pilot's number, not a plausibility argument, settles
it.

THREE MODES, and THINK IS DELIBERATELY ABSENT (RH). Think-mode needs generation
until the end-of-think token and word probabilities from the FIRST REAL TOKEN
after it — custom logic this pilot does not have. Measuring think without it
would score the model's first *reasoning* token as if it were its answer.

    raw        the prompt as-is
    chat       apply_chat_template([{user: prompt}], add_generation_prompt=True)
    continue   apply_chat_template([{user: "Continue this text: " + prompt}])

CONTINUE IS A DIFFERENT STIMULUS AND IS REPORTED SEPARATELY. It injects an
instruction the other two do not carry, so a raw-vs-continue difference confounds
format with content. It is here because the project's existing CONTINUE data
uses this exact wording, so the pilot must price what exists, not an idealised
variant. It never pools with raw/chat.

CONDITION (d) IS A HARD GATE, NOT A DIAGNOSTIC ([830].3). If chat's resolved
mass sits materially below raw's, the boundary rule is reworked BEFORE anything
is measured. The reason is the project's blindest failure mode: a chat template
with add_generation_prompt=True ends mid-scaffold, so the next token may open an
assistant turn rather than a word — the mask calls that a boundary, terminates
into an empty surface, and the mass drains to `drop`. CONSERVATION STILL HOLDS.
Every existing defence fires on impossible numbers; this one produces plausible
ones. The gate is the only eye on it, so it runs FIRST and aborts.

THE DECISION RULE, declared before the run:

    format effect (raw vs chat) comparable to or above the alignment effect
        -> the 41,418 is worth spending
    at the noise band (~0.05, where family-level effects sit)
        -> the pilot's null IS the result and the full run is not spent

Scope that travels with any result: comparison 2 reaches only arms that SHIP a
chat template. The 18 unreachable arms are the RLHF-era checkpoints (archangel
x4, beaver, bloom, pythia, amber), so nothing found here carries back to
kill->scream, the ego->superego edge, or the register result — all of which live
on amber.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "tc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "twp_cloud.py"))
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

from malign_logits import PATH_DATA  # noqa: E402

# 5 arms; Qwen and MiniCPM have TEMPLATED BASES, which is what prices
# comparison 3 (is the format effect the size of the alignment effect) --
# lacan's [830].2 addition. The rest are template-on-aligned-only.
ARMS = [
    ("qwen",     "Qwen/Qwen2.5-0.5B",              "Qwen/Qwen2.5-0.5B-Instruct"),
    ("minicpm",  "openbmb/MiniCPM5-1B-Base",       "openbmb/MiniCPM5-1B"),
    ("olmo2",    "allenai/OLMo-2-0425-1B",         "allenai/OLMo-2-0425-1B-Instruct"),
    ("smol",     "HuggingFaceTB/SmolLM2-360M",     "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("falcon3",  "tiiuae/Falcon3-1B-Base",         "tiiuae/Falcon3-1B-Instruct"),
]
GATE_DROP = 0.15      # chat resolved mass may not fall this far below raw
NOISE_BAND = 0.05     # where family-level effects sit


def realise(tok, prompt, mode):
    if mode == "raw":
        return prompt
    content = prompt if mode == "chat" else f"Continue this text: {prompt}"
    return tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True)


def word_dist(model, tok, prompt, dev, bmask, cjk, mode):
    text = realise(tok, prompt, mode)
    w, res, _ = tc.expand(model, tok, text, dev, bmask, cjk=cjk)
    return w, res


def js(a, b):
    """Jensen-Shannon over the union of word surfaces, each renormalised."""
    ka, kb = {}, {}
    for (s, _), v in a.items():
        ka[s] = ka.get(s, 0.0) + v
    for (s, _), v in b.items():
        kb[s] = kb.get(s, 0.0) + v
    keys = sorted(set(ka) | set(kb))
    if not keys:
        return float("nan")
    p = np.array([ka.get(k, 0.0) for k in keys])
    q = np.array([kb.get(k, 0.0) for k in keys])
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    def kl(x, y):
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / y[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load(mid, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    bmask = tc.boundary_mask(tok, model.config.vocab_size)
    cjk = None
    trie = tc.load_prefix_trie()
    if trie is not None:
        cids, cstrs, lids, pids = tc.cjk_vocab(tok, model.config.vocab_size)
        if len(cids):
            cjk = (trie, cids, cstrs, lids, pids)
    return tok, model, bmask, cjk


def main(a):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    prompts = [r["prompt"] for r in __import__("csv").DictReader(
        open(os.path.join(PATH_DATA, "prompt_inventory.csv")))
        if r["source"] == "DEFAULT"][:a.n_prompts]
    print(f"pilot: {len(ARMS)} arms x {len(prompts)} prompts x 3 modes on {dev}\n")

    rows, gate_failed = [], []
    for fam, base_id, aligned_id in ARMS:
        tok, model, bmask, cjk = load(aligned_id, dev)
        if getattr(tok, "chat_template", None) is None:
            print(f"{fam}: aligned arm has NO chat_template -- DECLARED SKIP")
            del model
            tc.free()
            continue

        # ---- CONDITION (d): THE GATE. Runs on the first prompts and aborts. ----
        g_raw, g_chat = [], []
        for p in prompts[:a.gate_n]:
            wr, rr = word_dist(model, tok, p, dev, bmask, cjk, "raw")
            wc, rc = word_dist(model, tok, p, dev, bmask, cjk, "chat")
            g_raw.append(sum(wr.values()))
            g_chat.append(sum(wc.values()))
        mr, mc = float(np.median(g_raw)), float(np.median(g_chat))
        print(f"{fam:<9} GATE  resolved raw {mr:.3f}  chat {mc:.3f}  "
              f"drop {mr-mc:+.3f}", flush=True)
        if mr - mc > GATE_DROP:
            print(f"           *** GATE FAILED: chat resolved mass {mr-mc:.3f} below "
                  f"raw (limit {GATE_DROP}). The assistant-turn boundary is "
                  f"draining to `drop`. NOT MEASURING THIS ARM.")
            gate_failed.append(fam)
            del model
            tc.free()
            continue

        for p in prompts:
            d = {}
            for mode in ("raw", "chat", "continue"):
                w, res = word_dist(model, tok, p, dev, bmask, cjk, mode)
                d[mode] = (w, sum(w.values()))
            rows.append(dict(family=fam, model=aligned_id, prompt=p,
                             js_raw_chat=js(d["raw"][0], d["chat"][0]),
                             js_raw_cont=js(d["raw"][0], d["continue"][0]),
                             js_chat_cont=js(d["chat"][0], d["continue"][0]),
                             res_raw=d["raw"][1], res_chat=d["chat"][1],
                             res_cont=d["continue"][1]))
        print(f"{fam:<9} done {len(prompts)} prompts", flush=True)
        del model
        tc.free()

    if not rows:
        print("\nNO ARM PASSED THE GATE. Nothing measured, which is the "
              "correct outcome -- the boundary rule needs reworking first.")
        return
    import csv as _csv
    out = os.path.join(PATH_DATA, "mode_pilot.csv")
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'family':<10}{'n':>4}{'JS raw-chat':>13}{'JS raw-cont':>13}"
          f"{'res raw':>9}{'res chat':>10}")
    fams = sorted({r["family"] for r in rows})
    for f in fams:
        s = [r for r in rows if r["family"] == f]
        print(f"{f:<10}{len(s):>4}{np.median([r['js_raw_chat'] for r in s]):>13.4f}"
              f"{np.median([r['js_raw_cont'] for r in s]):>13.4f}"
              f"{np.median([r['res_raw'] for r in s]):>9.3f}"
              f"{np.median([r['res_chat'] for r in s]):>10.3f}")
    allrc = np.median([r["js_raw_chat"] for r in rows])
    print(f"\nOVERALL median JS(raw, chat) = {allrc:.4f}   noise band ~{NOISE_BAND}")
    print("DECISION: " + ("FORMAT EFFECT IS REAL -- the 41,418 is worth spending"
                          if allrc >= NOISE_BAND else
                          "AT/BELOW THE NOISE BAND -- this null IS the result; "
                          "do not spend the 41,418"))
    print("\nCONTINUE reported separately and NEVER pooled: it injects "
          "'Continue this text:', so raw-vs-continue confounds format with content.")
    if gate_failed:
        print(f"\nGATE-FAILED ARMS (not measured): {gate_failed}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--gate-n", type=int, default=5)
    main(ap.parse_args())
