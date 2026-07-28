"""Nonce battery: does a signifier hold value with no referent? (rung C)

Registered at docs/f20x_nonce_registration.md, adeef97, BEFORE this file existed.
Amendment 1 at 3494324 fixed five defects lacan's audit found -- read both before
changing any constant here; several are load-bearing and none is obvious from the
code.

Reuses f20x_generate's roster, sampler and SKIP set rather than restating them, so
the roster cannot drift between batteries. What differs is the PROMPT TABLE: this
battery varies the referent rather than the question, so a prompt is a full string
and not a question slotted into a rung.

FOUR THINGS THE REGISTRATION DECLARES THAT THIS FILE IMPLEMENTS:

1. N-BARE IS THE PRIMARY, NOT N-DEF. A stipulated definition sits in the context
   window, so holding it is in-context retrieval. N-bare has nothing to retrieve.
   If the effect lives only in N-def the finding is retrieval and the structuralist
   reading is withdrawn.

2. R-RARE STIPULATES SOMETHING TRUE. The first draft's rare words were sarsen,
   withy, scrim and fipple -- a sandstone block, a willow branch, a fabric and a
   recorder mouthpiece. "A kind of tool" was FALSE of four of five, and a model
   that knew better would contradict the frame, coding as drift for a reason with
   nothing to do with signification. Every word below genuinely denotes a tool.

3. TOKEN MATCHING AT SELECTION. Nonce words fragment at 2 or 3 tokens identically
   across the Llama, Qwen and OLMo tokenisers. Rare tools at 3 fragments barely
   exist (1 of 29 candidates), so the PRIMARY is the 2-fragment stratum, 9 v 9
   exactly matched, and the 3-fragment nonce words are a declared SECONDARY with
   no matched control. `stratum` is recorded on every row.

4. THE GATE IS AN OUTCOME. Aligned arms decline MORE than base (0.296 v 0.201 in
   the third-person battery) and a bare nonce question is the strongest invitation
   to decline in any battery run here. Whether a value was posed is outcome one;
   drift is conditional on it. That is an analysis-side rule, but it is why n per
   cell matters and why nothing is prefiltered at generation.

    uv run .venv/bin/python scripts/f20x_nonce_generate.py [--smoke] [--family KEY]
"""
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malign_logits import PATH_DATA  # noqa: E402
import f20x_generate as G  # noqa: E402

OUT = os.path.join(PATH_DATA, "f20x_nonce.parquet")

# Zero-frequency in en/de/fr/es/nl/sv/ru/pt/it (wordfreq). Chinese unverified:
# jieba is absent, and two roster families are Chinese-trained. Stated in the
# registration rather than papered over.
NONCE_2 = ["glorp", "quiln", "plost", "tarnu", "velbin",
           "gorpin", "flant", "zendle", "fenmit"]      # 2 fragments -- PRIMARY
# SECONDARY, 3 fragments, no token-matched control. CUT AT RELAUNCH (Amendment 2):
# the run needed to be shortened to fit `P-repeat` in the same pass, and lacan
# [167] is right that time must not be bought with the primary stimulus set. These
# six and A-abst were the only expendable material -- both are outside every
# registered prediction.
NONCE_3 = []
# Real, rare, and genuinely tools, so the stipulation is true. zipf 1.11-2.77.
RARE_2 = ["froe", "quern", "adze", "burin", "reamer",
          "bodkin", "gimlet", "mandrel", "auger"]      # 2 fragments, matched

N_SAMPLES = 5
TEMPERATURES = (0.7, 1.0)      # matches the person battery, so P-repeat replicates
# Declared seed offset. Pass 1 used G.SEED0 directly; this pass is a separate
# process and cannot reproduce that numbering once the prompt table changed, so
# the offset is stated rather than pretended away. Torch process state applies to
# all conditions equally and so is not a between-condition confound -- which is
# what Amendment 1 was actually protecting (lacan [167] amended its own wording).
SEED_OFFSET = G.SEED0 + 100000


def prompt_table():
    rows = []
    for w in NONCE_2 + NONCE_3:
        st = "primary" if w in NONCE_2 else "secondary"
        rows.append(dict(pid=f"ndef_{w}", condition="N-def", word=w, stratum=st,
                         text=f"A {w} is a kind of tool. Q: What is a {w}?\nA:"))
        rows.append(dict(pid=f"nbare_{w}", condition="N-bare", word=w, stratum=st,
                         text=f"Q: What is a {w}?\nA:"))
    for w in RARE_2:
        rows.append(dict(pid=f"rare_{w}", condition="R-rare", word=w, stratum="primary",
                         text=f"A {w} is a kind of tool. Q: What is a {w}?\nA:"))
    # P-REPEAT: the person condition, INSIDE this run. Amendment 1 to the object
    # registration requires the persons and objects terms of the contrast to share
    # a run, a coder and a roster; taking persons from the earlier battery would
    # confound the contrast with everything differing between two runs.
    # WORDING IS VERBATIM from f20x_generate.PROMPTS -- lacan [167]. A paraphrase
    # would break the replication check against the published -0.061.
    for k, q in G.PROMPTS.items():
        rows.append(dict(pid=f"prep_{k}", condition="P-repeat", word=k,
                         stratum="primary", text=G.RUNG.format(q=q)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--family")
    a = ap.parse_args()

    fams = G.roster()
    if a.family:
        fams = [f for f in fams if f["key"] == a.family]
    else:
        fams = [f for f in fams if f["key"] not in G.SKIP]
    prompts = prompt_table()
    if a.smoke:
        fams = fams[:2]
        prompts = [p for p in prompts if p["word"] in ("glorp", "froe")]
    temps = (1.0,) if a.smoke else TEMPERATURES
    n = 3 if a.smoke else N_SAMPLES

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_parquet(OUT)
        done = set(prev.model_id)
        print(f"resuming: {len(done)} model arms already written")

    print(f"{len(fams)} families, {len({f['base'] for f in fams})} bases, "
          f"{len(prompts)} prompts, {len(temps)} temps, n={n}", flush=True)
    print(f"  => {len(prompts)*len(temps)*n} completions per arm", flush=True)

    sink, cell = [], 0
    if done:
        sink = pd.read_parquet(OUT).to_dict("records")
    for fi, fam in enumerate(fams, 1):
        for arm, mid in (("base", fam["base"]), (fam["slot"], fam["aligned"])):
            if mid in done:
                continue
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    mid, dtype=torch.float16, device_map="mps", trust_remote_code=True)
                model.eval()
            except Exception as e:
                print(f"  LOAD FAIL {mid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            for p in prompts:
                for temp in temps:
                    cell += 1
                    try:
                        outs = G.sample(model, tok, p["text"], n, temp, SEED_OFFSET + cell)
                    except Exception as e:
                        print(f"  cell fail {mid} {p['pid']}: {str(e)[:90]}", flush=True)
                        continue
                    # sample() returns the TWO WINDOWS the parent battery declares:
                    # the 10-token prefix (comparable to the beam measures) and the
                    # full 60 (interpretable). Both are carried, both classified.
                    for i, r in enumerate(outs):
                        sink.append(dict(family=fam["key"], arm=arm, model_id=mid,
                                         base_model_id=fam["base"], condition=p["condition"],
                                         word=p["word"], stratum=p["stratum"], pid=p["pid"],
                                         prompt=p["text"], temperature=temp, idx_in_cell=i,
                                         text=r["text"], prefix=r["prefix"],
                                         n_tokens=r["n_tokens"], **G.classify(r["text"])))
            del model
            import gc
            gc.collect()
            torch.mps.empty_cache()
            pd.DataFrame(sink).to_parquet(OUT, compression="zstd", index=False)
            print(f"[{fi}/{len(fams)}] {fam['key']}/{arm} done, {len(sink):,} rows", flush=True)


if __name__ == "__main__":
    main()
