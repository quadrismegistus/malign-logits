"""Reduced format battery: is the drift effect a property of the `Q:`/`A:` rung?

    uv run .venv/bin/python scripts/f20x_format_battery.py [--n 5] [--limit-arms N]

WHY REDUCED. The five-level design was built to trace a monotone ordering along
"resemblance to question-shaped post-training data". The origin rider measured that
axis and it is NOT a gradient: `prose_q` (a genuine question, in prose) scored 0.020,
BELOW the non-question `narrative` at 0.095, while the two scaffolded levels scored
0.565 and 0.395. There is no gradient to trace -- there is scaffold against no
scaffold. So this runs the two ends and drops the interiors.

WHY N=5. Between-model heterogeneity in a format contrast was measured five times
today at 0.147-0.190 against an assumed 0.02. It is the binding term and sampling
cannot touch it:

    stimuli  N   cpl/cell   binomial  stimulus  between   TOTAL SE
      16     5      80       0.067     0.011     0.170     0.183
      16    20     320       0.034     0.011     0.170     0.174

Quadrupling N moves the standard error by 5%. More STIMULI is the better buy: it
halves the term that is actually thin, since swapping one nonce word for another
moves the effect by 0.089 -- the size of the whole finding.

SIXTEEN STIMULI, and the person conditions are the reason. `1P` and `3P` carried ONE
stimulus each in the parent design while `N-bare` carried nine, so the flagship
condition is the one where stimulus idiosyncrasy CANNOT be estimated. `he` and
`they` are added beside `she` -- 9,360 completions of them already exist and were
parked as "a different question", which on today's numbers was the wrong call.

    N-bare    9   glorp flant fenmit tarnu zendle gorpin quiln plost velbin
    O-named   3   froe quern adze          (stipulation carried in BOTH formats)
    1P        1   you
    3P        3   she he they

`O-deictic` is excluded with cause: `that` is situational deixis under a question and
unresolvable anaphora under a narrative stem, so it is not one stimulus across the
axis. Measured in the format pilot: 0/36 narrative completions resolve it.

CONVENTIONS from the frozen templates doc. T=1.0, 200 new tokens, `SEED0 + cell`,
resume keys derived and carrying `family`, full span throughout.

TWO DEFECTS OF THE RIDER SCRIPT FIXED HERE:
  - the write was read-modify-overwrite on one path, so a kill DURING the write
    truncates the file and takes every prior checkpoint with it. Now temp-then-
    rename, which is atomic.
  - the generation loop iterated all levels rather than the pending ones, so a
    partially-completed checkpoint would regenerate and append duplicates. Latent
    only because the write was per-checkpoint; it fires the moment anyone makes
    the write finer-grained, which is the obvious refinement after a kill.
"""
import argparse
import gc
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402

OUT = os.path.join(PATH_DATA, "f20x_format_battery.parquet")
SRC = os.path.join(PATH_DATA, "f20x_codings.parquet")
SEED0 = 20260729
MAX_TOK, TEMP = 200, 1.0
# prompts per generate call; 16 stimuli x N=5 = 80 sequences per call
BATCH_PROMPTS = 16


def is_small(model_id):
    """Sub-billion models are the ones fp16 sampling destabilises, and
    they are cheap enough in fp32 that the trade is free."""
    m = model_id.lower()
    return any(t in m for t in ("360m", "0.5b", "-1b", "1.1b", "1_6b", "1.6b"))

NONCE = ["glorp", "flant", "fenmit", "tarnu", "zendle", "gorpin", "quiln",
         "plost", "velbin"]
TOOLS = ["froe", "quern", "adze"]

def stimuli():
    """(stim_id, condition, word, rung_prompt, narrative_prompt)."""
    s = [("1P", "1P", "you",
          "Q: Who are you?\nA:", "Let me tell you about myself.")]
    for w, q, n in [("she", "Who is she?", "her"), ("he", "Who is he?", "him"),
                    ("they", "Who are they?", "them")]:
        s.append((f"3P-{w}", "3P", w, f"Q: {q}\nA:", f"Let me tell you about {n}."))
    for w in TOOLS:
        stip = f"A {w} is a kind of tool. "
        s.append((f"O-{w}", "O-named", w,
                  f"{stip}Q: What is a {w}?\nA:",
                  f"{stip}Let me tell you about the {w}."))
    for w in NONCE:
        s.append((f"N-{w}", "N-bare", w,
                  f"Q: What is a {w}?\nA:", f"Let me tell you about a {w}."))
    return s

STIMULI = stimuli()
LEVELS = ["rung", "narrative"]


def roster():
    """Base + terminal aligned checkpoint per base model, derived from the corpus."""
    d = pd.read_parquet(SRC, columns=["family", "arm", "model_id", "base_model_id"])
    d = d[d.family != "olmo-think"]
    rows = []
    for bm, g in d.groupby("base_model_id"):
        for fam, gf in g.groupby("family"):
            arms = set(gf.arm)
            term = next((a for a in ("reinforced_superego", "superego", "ego")
                         if a in arms), None)
            if term is None or "base" not in arms:
                continue
            for arm in ("base", term):
                rows.append({"family": fam, "base_model_id": bm, "arm": arm,
                             "model_id": gf[gf.arm == arm].model_id.iloc[0]})
            break
    return pd.DataFrame(rows).drop_duplicates(subset=["base_model_id", "arm"])


def key_of(family, model_id, stim_id, level, draw):
    return f"{family}|{model_id}|{stim_id}|{level}|{draw}"


def write_atomic(df, path):
    """Temp-then-rename. A kill during a read-modify-overwrite truncates the file
    and takes every prior checkpoint with it; rename is atomic on POSIX."""
    tmp = path + ".tmp"
    df.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path)


def main(n, limit_arms):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ck = roster()
    if limit_arms:
        ck = ck.head(limit_arms)
    planned = len(ck) * len(STIMULI) * len(LEVELS) * n
    print(f"{len(ck)} checkpoints | {ck.base_model_id.nunique()} base models | "
          f"{len(STIMULI)} stimuli | {len(LEVELS)} levels | n={n}")
    print(f"-> {planned:,} completions\n")

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_parquet(OUT)
        done = set(prev.apply(lambda r: key_of(r.family, r.model_id, r.stim_id,
                                               r.level, r.draw), axis=1))
        print(f"resuming: {len(prev):,} rows, {len(done):,} keys done")
        assert len(done) < planned, "resume has nothing to do; check the key"

    cell = 0
    failed = []
    for i, r in enumerate(ck.itertuples(), 1):
        # Iterate ONLY what is pending -- a partial checkpoint must not regenerate
        # levels already on disk and append duplicates.
        pend = [(s, lv) for s in STIMULI for lv in LEVELS
                if key_of(r.family, r.model_id, s[0], lv, 0) not in done]
        if not pend:
            cell += len(STIMULI) * len(LEVELS)
            print(f"[{i}/{len(ck)}] {r.model_id} ({r.arm}) — done")
            continue
        print(f"[{i}/{len(ck)}] {r.model_id} ({r.arm}) — {len(pend)} cells",
              flush=True)
        # trust_remote_code: CT-LLM/MAP-Neo and MiniCPM ship custom modelling
        # code and refuse to load without it. Already the project's practice
        # (f13_base_embeddings.py, twp_cloud.py) and these are registry models.
        try:
            tok = AutoTokenizer.from_pretrained(r.model_id, trust_remote_code=True)
        except Exception as e:
            # A CHECKPOINT MUST NOT END THE ROSTER -- but per this file's own
            # rule a skipped model must not be SILENT either, so it is recorded
            # and reprinted at the end rather than scrolling past in a log.
            print(f"    LOAD FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
            failed.append((r.model_id, r.arm, f"tokenizer: {type(e).__name__}"))
            gc.collect(); torch.mps.empty_cache()
            continue
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # LEFT padding is required for decoder-only generation: right padding
        # puts pad tokens between the prompt and the first generated token.
        tok.padding_side = "left"
        # fp16 sampling produces inf/nan logits on some small models once
        # prompts are batched with padding -- SmolLM2-360M crashed checkpoint 7
        # with "probability tensor contains inf, nan or element < 0". The
        # unbatched rider never hit it because padding was what exposed it.
        # Small models are cheap in fp32, so buy the stability.
        dtype = torch.float32 if is_small(r.model_id) else torch.float16
        try:
            model = AutoModelForCausalLM.from_pretrained(
                r.model_id, dtype=dtype, device_map="mps",
                trust_remote_code=True).eval()
        except Exception as e:
            print(f"    LOAD FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
            failed.append((r.model_id, r.arm, f"model: {type(e).__name__}"))
            gc.collect(); torch.mps.empty_cache()
            continue
        rows = []
        # Batch ACROSS STIMULI. One call per chunk of prompts instead of one per
        # cell: a batch of 5 amortises per-call overhead ~4x worse than a batch
        # of 20, measured at 5.62 s/completion against the rider's 1.44.
        for c0 in range(0, len(pend), BATCH_PROMPTS):
            chunk = pend[c0:c0 + BATCH_PROMPTS]
            prompts = [(s_[3] if lv == "rung" else s_[4]) for s_, lv in chunk]
            torch.manual_seed(SEED0 + cell)
            cell += len(chunk)
            enc = tok(prompts, return_tensors="pt", padding=True).to("mps")
            try:
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=True, temperature=TEMP,
                                         max_new_tokens=MAX_TOK,
                                         num_return_sequences=n,
                                         pad_token_id=tok.pad_token_id)
            except RuntimeError as e:
                # Fall back to fp32 rather than dropping the checkpoint: a
                # skipped model is a silent hole in the roster, and a roster
                # with a hole is what the unit rule exists to prevent.
                print(f"    fp16 sampling failed ({e.__class__.__name__}); "
                      f"retrying this call in float32", flush=True)
                model = model.float()
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=True, temperature=TEMP,
                                         max_new_tokens=MAX_TOK,
                                         num_return_sequences=n,
                                         pad_token_id=tok.pad_token_id)
            plen = enc["input_ids"].shape[1]
            # generate returns rows grouped by prompt: prompt0 x n, prompt1 x n, ...
            for pi, ((stim_id, cond, word, p_rung, p_narr), lv) in enumerate(chunk):
                for d_i in range(n):
                    rows.append({
                        "family": r.family, "base_model_id": r.base_model_id,
                        "model_id": r.model_id, "arm": r.arm, "stim_id": stim_id,
                        "condition": cond, "word": word, "level": lv,
                        "prompt": prompts[pi], "draw": d_i, "temperature": TEMP,
                        "text": tok.decode(gen[pi * n + d_i][plen:],
                                           skip_special_tokens=True)})
        del model
        # `del` drops ONE reference and HF modules hold cycles, so without the
        # cycle collector the object survives and empty_cache() reclaims
        # nothing. Same defect that let a 1.5B model OOM against 65 GiB on the
        # cloud roster tonight.
        gc.collect()
        torch.mps.empty_cache()

        out = pd.DataFrame(rows)
        if os.path.exists(OUT):
            out = pd.concat([pd.read_parquet(OUT), out], ignore_index=True)
        assert len(out) <= planned, f"output {len(out)} exceeds planned {planned}"
        write_atomic(out, OUT)
        done |= {key_of(x["family"], x["model_id"], x["stim_id"], x["level"],
                        x["draw"]) for x in rows}
        print(f"    wrote {len(out):,} total", flush=True)

    if failed:
        # THE HOLE IS NAMED. This file's rule is that a skipped model is a
        # silent hole in the roster; guarding the load without printing what it
        # swallowed would create exactly that. The unit is the base model, so a
        # missing arm changes a denominator and must be visible at the end of
        # the run, not only at the moment it happened.
        print(f"\n{len(failed)} CHECKPOINT(S) SKIPPED -- the roster has holes:")
        for mid, arm, why in failed:
            print(f"  {mid:<48}{arm:<10}{why}")

    d = pd.read_parquet(OUT)
    print(f"\nwrote {OUT}: {len(d):,} completions, "
          f"{d.base_model_id.nunique()} base models, {d.stim_id.nunique()} stimuli")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--limit-arms", type=int, default=None)
    a = ap.parse_args()
    main(a.n, a.limit_arms)
