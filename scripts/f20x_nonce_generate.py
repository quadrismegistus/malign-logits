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
# Zero-frequency in en/de/fr/es/nl/sv/ru/pt/it (wordfreq). Chinese unverified:
# jieba is absent, and two roster families are Chinese-trained.
NONCE = ["glorp", "quiln", "plost", "tarnu", "velbin",
         "gorpin", "flant", "zendle", "fenmit"]      # 2 fragments, token-matched
# Three genuinely-tool words for the O-named cell. R-rare is GONE (Amendment 3):
# O-named IS a named tool with a true stipulation, so keeping both ran one cell
# twice under two names and inflated the object arm's n against the person arm's.
TOOLS = ["froe", "quern", "adze"]

N_SAMPLES = 10
TEMPERATURES = (1.0,)          # Amendment 3: one temperature, n doubled. The cost
# is call count, not text -- 42 prompts x 2 temps x n=5 is 11.8h for the same 420
# completions per arm that 1 temp x n=10 delivers in 5.9h. T=1.0 is also where base
# drift is LARGEST (parent -0.086 vs -0.037), so the object arm is tested at the
# temperature most favourable to finding an effect: an object null here is stronger
# evidence than an object null pooled (lacan [171]).
TEMP_CHECK = 0.7               # O-deictic only, for the dose-response check the
# parent battery established and this design would otherwise be unable to see.
SEED_OFFSET = G.SEED0 + 200000


def prompt_table():
    """The 2x2 (referent kind x referent availability) plus the rung-C nonce arm.

                    referent GIVEN            referent ABSENT
        PERSON      1P  (you)                 3P  (she)
        OBJECT      O-named (froe, stipulated) O-deictic (that)

    Person wordings are VERBATIM from f20x_generate.PROMPTS and
    f20x_generate_3p.PROMPTS -- a paraphrase would break the replication check
    against the parent's -0.061. Object cells use the same four question forms.
    The deictic set keeps the demonstrative throughout (lacan [169]).
    """
    rows = []
    for k, q in G.PROMPTS.items():                      # 1P: person, referent given
        rows.append(dict(pid=f"1p_{k}", condition="1P", word="you",
                         stratum="2x2", text=G.RUNG.format(q=q)))
    import f20x_generate_3p as G3                       # 3P: person, referent absent
    for k, (q, kind) in G3.PROMPTS.items():
        if kind != "matched":                           # the pronoun variants are
            continue                                    # a different question
        rows.append(dict(pid=f"3p_{k}", condition="3P", word="she",
                         stratum="2x2", text=G.RUNG.format(q=q)))
    OBJ_Q = ["What is {a}{w}?", "Who made the {w}?",
             "Where is the {w} from?", "What is the {w} for?"]
    for w in TOOLS:                                     # O-named: object, given
        for i, qt in enumerate(OBJ_Q):
            q = qt.format(a="a " if i == 0 else "", w=w)
            rows.append(dict(pid=f"onamed_{w}_{i}", condition="O-named", word=w,
                             stratum="2x2",
                             text=f"A {w} is a kind of tool. " + G.RUNG.format(q=q)))
    for i, q in enumerate(["What is that?", "Who made that?",
                           "Where is that from?", "What is that for?"]):
        rows.append(dict(pid=f"odeictic_{i}", condition="O-deictic", word="that",
                         stratum="2x2", text=G.RUNG.format(q=q)))
    for w in NONCE:                                     # rung C
        rows.append(dict(pid=f"ndef_{w}", condition="N-def", word=w, stratum="nonce",
                         text=f"A {w} is a kind of tool. Q: What is a {w}?\nA:"))
        rows.append(dict(pid=f"nbare_{w}", condition="N-bare", word=w, stratum="nonce",
                         text=f"Q: What is a {w}?\nA:"))
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
                # O-deictic also runs at TEMP_CHECK: the parent found the drift
                # effect roughly doubling from 0.7 to 1.0, and a single-temperature
                # battery cannot see whether that dose-response holds for objects.
                # One cell, where an object effect is most likely (lacan [171]).
                ptemps = temps if p["condition"] != "O-deictic" else tuple(temps) + (TEMP_CHECK,)
                for temp in ptemps:
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
