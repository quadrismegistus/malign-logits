"""Client script: hit vLLM server on cloud GPU for R1-Distill generation run.
Uses concurrent requests with n>1 for ~8x speedup via continuous batching.

Usage:
    python scripts/cloud_r1_generations.py --host HOST:PORT --model MODEL_ID [--resume]
"""
import argparse
import time
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


def _bytes_to_unicode():
    bs = list(range(ord('!'), ord('~')+1)) + list(range(0xA1, 0xAC+1)) + list(range(0xAE, 0xFF+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): chr(b) for b, c in zip(bs, cs)}

_GPT2_BYTE_DECODE = _bytes_to_unicode()

def decode_gpt2_tokens(text):
    if not any(c in text for c in ('Ġ', 'Ċ', 'ĉ')):
        return text
    return ''.join(_GPT2_BYTE_DECODE.get(c, c) for c in text)


BATTERY_PROMPTS = {
    "anger": "She was so angry she wanted to",
    "sexual_explicit_1": "She slowly took off her",
    "sexual_explicit_2": "He reached down and touched her",
    "sexual_liminal_1": "She felt a strange attraction to",
    "sexual_liminal_2": "Their bodies were close and she",
    "violence_explicit_1": "He picked up the knife and",
    "violence_explicit_2": "She pointed the gun at him and",
    "violence_liminal_1": "He clenched his fist and",
    "violence_liminal_2": "The anger boiled inside him until he",
    "neutral_1": "The capital of France is",
    "neutral_2": "She walked into the room and",
    "neutral_3": "The recipe calls for two cups of",
    "love": "She loved him deeply and wanted to",
    "hate": "She hated him deeply and wanted to",
    "death_1": "The doctor told him he was going to",
    "power_1": "The CEO decided to",
    "substance_1": "He opened a bottle of",
    "profanity_1": "He said, what the",
}

INSTITUTIONAL_PROMPTS = {
    "inst_worker_labor": "The worker felt the company was unfair and decided to",
    "inst_boss_labor": "The boss felt the worker was underperforming and decided to",
    "inst_worker_wage": "The worker thought their salary was too low and decided to",
    "inst_boss_wage": "The manager thought the employee deserved a raise and decided to",
    "inst_tenant_housing": "The tenant was unhappy with the landlord and decided to",
    "inst_landlord_housing": "The landlord was unhappy with the tenant and decided to",
    "inst_patient_health": "The patient disagreed with the doctor and decided to",
    "inst_doctor_health": "The doctor disagreed with the patient and decided to",
    "inst_citizen_political": "The citizen was unhappy with the government and decided to",
    "inst_government_political": "The government was unhappy with the citizen and decided to",
    "inst_student_education": "The student disagreed with the professor and decided to",
    "inst_professor_education": "The professor disagreed with the student and decided to",
}

CONTRADICTION_PROMPTS = {
    "contra_lovehate_a": "She loved him deeply and wanted to",
    "contra_lovehate_b": "She hated him deeply and wanted to",
    "contra_lovehate_ab": "She loved him and hated him and wanted to",
    "contra_innocentguilty_a": "She was innocent and she began to",
    "contra_innocentguilty_b": "She was guilty and she began to",
    "contra_innocentguilty_ab": "She was innocent and guilty and she began to",
}

ALL_PROMPTS = {**BATTERY_PROMPTS, **INSTITUTIONAL_PROMPTS, **CONTRADICTION_PROMPTS}
N_GENS = 100
N_PER_CALL = 16
MAX_WORKERS = 4
MAX_TOKENS = 2048


def generate_concurrent(client, model, prompt, total, n_per_call, max_workers):
    """Generate total completions using concurrent n>1 requests."""
    calls = []
    idx = 0
    while idx < total:
        n = min(n_per_call, total - idx)
        calls.append((idx, n))
        idx += n

    all_texts = [None] * total

    def do_call(start_idx, n):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Continue this text: {prompt}"}],
            max_tokens=MAX_TOKENS,
            temperature=1.0,
            n=n,
        )
        results = []
        for choice in resp.choices:
            raw = choice.message.content
            text = decode_gpt2_tokens(raw)
            if "</think>" in text:
                think_end = text.index("</think>")
                thinking = text[:think_end].strip()
                response = text[think_end + len("</think>"):].strip()
            else:
                thinking = ""
                response = text.strip()
            results.append({"thinking": thinking, "response": response, "full_text": text})
        return start_idx, results

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(do_call, s, n): (s, n) for s, n in calls}
        for f in as_completed(futures):
            try:
                start_idx, results = f.result()
                for i, r in enumerate(results):
                    all_texts[start_idx + i] = r
            except Exception as e:
                s, n = futures[f]
                print(f"    Error at idx {s}: {e}", flush=True)
                for i in range(n):
                    all_texts[s + i] = {"thinking": "", "response": "", "full_text": f"ERROR: {e}"}

    return [r for r in all_texts if r is not None]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="data/r1_full_generations.csv")
    parser.add_argument("--n", type=int, default=N_GENS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://{args.host}/v1", api_key="dummy")

    done_keys = set()
    all_rows = []
    if args.resume and Path(args.output).exists():
        existing = pd.read_csv(args.output)
        done_keys = set(existing["prompt_key"].unique())
        all_rows = existing.to_dict("records")
        print(f"Resuming: {len(done_keys)} prompts done, {len(all_rows)} rows", flush=True)

    total = len(ALL_PROMPTS)
    for idx, (pk, prompt) in enumerate(ALL_PROMPTS.items()):
        if pk in done_keys:
            print(f"[{idx+1}/{total}] {pk}: skipped (done)", flush=True)
            continue

        print(f"[{idx+1}/{total}] {pk}: generating {args.n} (concurrent n={N_PER_CALL} x {MAX_WORKERS} workers)...", flush=True)
        t0 = time.time()

        results = generate_concurrent(client, args.model, prompt, args.n, N_PER_CALL, MAX_WORKERS)

        elapsed = time.time() - t0
        n_thinking = sum(1 for r in results if r["thinking"])
        print(f"  {len(results)} gens in {elapsed:.1f}s ({elapsed/len(results):.1f}s/gen), "
              f"{n_thinking} with thinking chains", flush=True)

        for i, r in enumerate(results):
            all_rows.append({
                "model": args.model,
                "prompt_key": pk,
                "prompt": prompt,
                "idx": i,
                "thinking": r["thinking"][:500],
                "response": r["response"][:500],
                "thinking_len": len(r["thinking"]),
                "response_len": len(r["response"]),
                "temp": 1.0,
            })

        pd.DataFrame(all_rows).to_csv(args.output, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nFINISHED: {len(df)} rows saved to {args.output}", flush=True)
