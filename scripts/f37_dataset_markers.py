"""F37 Track A: lexical marker rates across post-training corpora.

Computes 7 marker sets as length-normalized rates per corpus, per turn role,
per preference side. Lexicons frozen per spec.

Usage:
    uv run python scripts/f37_dataset_markers.py
"""
import re
import json
import os
import sys
import time
from collections import defaultdict, Counter

import pandas as pd
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# ── Frozen lexicons (from spec, no changes without TM approval) ────

MARKERS = {
    # 1. Death-naming vs severity-matched violence
    "death_naming": ["kill", "kills", "killed", "killing", "die", "dies", "died", "dying", "death", "dead"],
    "violence_nondeath": ["stab", "stabs", "stabbed", "stabbing", "shoot", "shoots", "shot", "shooting",
                          "strangle", "strangles", "strangled", "strangling",
                          "hit", "hits", "hitting", "punch", "punches", "punched", "punching"],

    # 2. Containment/interiority
    "containment": ["held it in", "deep breath", "bit her tongue", "bit his tongue",
                     "composed herself", "composed himself", "took a moment",
                     "calmed down", "regained composure", "collected herself", "collected himself",
                     "paused", "breathed", "steadied"],
    "mental_state": ["felt", "thought", "believed", "realized", "understood", "recognized",
                     "considered", "reflected", "wondered", "imagined", "feared", "hoped",
                     "wished", "desired", "needed", "wanted"],

    # 3. De-escalation vs somatic escalation
    "de_escalation": ["however", "but", "instead", "rather", "although", "nonetheless",
                      "on the other hand", "it's important to", "it is important to",
                      "keep in mind", "consider", "alternatively"],
    "somatic_escalation": ["harder", "faster", "tighter", "deeper", "couldn't stop",
                           "couldn't help", "kept going", "more and more", "growing",
                           "surging", "overwhelming", "trembling", "shaking"],

    # 4. Enunciation cells
    "first_person_present_desire": ["I want to", "I need to", "I have to", "I'm going to",
                                     "I am going to", "I wish to", "I desire to"],
    "first_person_past_desire": ["I wanted to", "I needed to", "I had to", "I wished to",
                                  "I was going to"],
    "third_person_desire": ["she wanted to", "he wanted to", "she wants to", "he wants to",
                             "they wanted to", "they want to"],
    "quotation_markers": ['she said', 'he said', 'they said', '"I', "'I",
                           'she told', 'he told', 'she asked', 'he asked'],

    # 5. Moralizing markers
    "moralizing": ["it is important to", "it's important to", "remember that",
                    "keep in mind", "please note", "be aware", "you should",
                    "it's crucial", "it is crucial", "it's essential", "it is essential",
                    "be careful", "be cautious", "make sure", "responsible",
                    "appropriate", "inappropriate", "ethical", "unethical",
                    "harmful", "dangerous", "safe", "safety"],

    # 6. Deference/proceduralization
    "deference": ["consult a professional", "seek help", "contact the authorities",
                   "proper channels", "seek professional", "talk to someone",
                   "reach out to", "qualified professional", "mental health",
                   "law enforcement", "medical professional", "healthcare provider",
                   "trained professional", "licensed"],

    # 7. F11 pole vocabulary
    "pole_prosocial": ["love", "trust", "beautiful", "obey", "pleasure",
                        "desire", "sacred", "free", "create", "innocent",
                        "rational", "faithful"],
    "pole_antisocial": ["hate", "fear", "disgusting", "rebel", "pain",
                         "disgust", "profane", "captive", "destroy", "guilty",
                         "irrational", "unfaithful"],
}


def count_markers(text, marker_set):
    """Count occurrences of markers in text, case-insensitive."""
    text_lower = text.lower()
    total = 0
    for marker in marker_set:
        if " " in marker:
            total += text_lower.count(marker.lower())
        else:
            total += len(re.findall(r'\b' + re.escape(marker.lower()) + r'\b', text_lower))
    return total


def tokenize_count(text):
    """Rough word count."""
    return len(text.split())


def process_texts(texts, corpus_name, role, side=""):
    """Process a list of texts and return marker rates."""
    results = []
    total_tokens = 0
    marker_counts = {k: 0 for k in MARKERS}
    n_valid = 0

    for i, text in enumerate(texts):
        if not text or not isinstance(text, str):
            continue
        n_tokens = tokenize_count(text)
        total_tokens += n_tokens
        n_valid += 1
        for marker_name, marker_list in MARKERS.items():
            marker_counts[marker_name] += count_markers(text, marker_list)
        if (i + 1) % 100000 == 0:
            print(f"    {corpus_name}/{role}/{side}: {i+1}/{len(texts)} texts, {total_tokens:,} tokens", flush=True)

    if total_tokens == 0:
        return []

    for marker_name, count in marker_counts.items():
        results.append({
            "corpus": corpus_name,
            "role": role,
            "side": side,
            "marker": marker_name,
            "count": count,
            "tokens": total_tokens,
            "rate_per_10k": count / total_tokens * 10000,
            "n_texts": n_valid,
        })

    return results


# ── Corpus loaders ──────────────────────────────────────────────────

def load_alpaca():
    """Alpaca 52k: instruction-output pairs, no turns."""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    assistant = [row["output"] for row in ds if row["output"]]
    user = [row["instruction"] + (" " + row["input"] if row["input"] else "") for row in ds]
    return {"assistant": assistant, "user": user}


def load_ultrachat():
    """UltraChat: multi-turn dialogues. Stream to avoid memory blowup."""
    ds = load_dataset("stingning/ultrachat", split="train", streaming=True)
    assistant, user = [], []
    for i, row in enumerate(ds):
        turns = row["data"]
        for j, turn in enumerate(turns):
            if j % 2 == 0:
                user.append(turn)
            else:
                assistant.append(turn)
        if (i + 1) % 100000 == 0:
            print(f"    UltraChat: {i+1} dialogues parsed ({len(assistant)} asst, {len(user)} user)")
    return {"assistant": assistant, "user": user}


def load_ultrafeedback():
    """UltraFeedback: instruction + multiple completions with scores."""
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    user, chosen, rejected = [], [], []
    for row in ds:
        user.append(row["instruction"])
        completions = row["completions"]
        if completions:
            scored = [(c.get("overall_score", 0) or 0, c.get("response", "")) for c in completions]
            scored.sort(key=lambda x: -x[0])
            if len(scored) >= 1:
                chosen.append(scored[0][1])
            if len(scored) >= 2:
                rejected.append(scored[-1][1])
    return {"assistant_chosen": chosen, "assistant_rejected": rejected, "user": user}


def load_pku():
    """PKU-SafeRLHF: preference pairs with safety labels."""
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    user, chosen, rejected = [], [], []
    for row in ds:
        user.append(row["prompt"])
        safer = row["safer_response_id"]
        r0, r1 = row["response_0"], row["response_1"]
        if safer == 0:
            chosen.append(r0); rejected.append(r1)
        else:
            chosen.append(r1); rejected.append(r0)
    return {"assistant_chosen": chosen, "assistant_rejected": rejected, "user": user}


def load_hh_rlhf():
    """Anthropic HH-RLHF: chosen/rejected conversation pairs."""
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    chosen_asst, rejected_asst, user_turns = [], [], []
    for row in ds:
        # Parse the conversation format: \n\nHuman: ... \n\nAssistant: ...
        for side, key in [("chosen", "chosen"), ("rejected", "rejected")]:
            text = row[key]
            parts = text.split("\n\nAssistant: ")
            for i, part in enumerate(parts[1:], 1):
                # Strip any subsequent Human: turn
                asst_text = part.split("\n\nHuman: ")[0].strip()
                if side == "chosen":
                    chosen_asst.append(asst_text)
                else:
                    rejected_asst.append(asst_text)

        # User turns from chosen side
        parts = text.split("\n\nHuman: ")
        for part in parts[1:]:
            user_text = part.split("\n\nAssistant: ")[0].strip()
            user_turns.append(user_text)

    return {"assistant_chosen": chosen_asst, "assistant_rejected": rejected_asst, "user": user_turns}


def load_tulu_sft():
    """Tulu 3 SFT mix — try the main dataset."""
    try:
        ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
        assistant, user = [], []
        for i, row in enumerate(ds):
            if i >= 200000:
                break
            msgs = row.get("messages", [])
            for msg in msgs:
                if msg.get("role") == "assistant":
                    assistant.append(msg["content"])
                elif msg.get("role") == "user":
                    user.append(msg["content"])
        return {"assistant": assistant, "user": user}
    except Exception as e:
        print(f"  Tulu SFT mix not available: {e}")
        return None


def load_sharegpt():
    """ShareGPT conversations."""
    try:
        ds = load_dataset("RyokoAI/ShareGPT52K", split="train")
        assistant, user = [], []
        for row in ds:
            convos = row.get("conversations", [])
            for msg in convos:
                if msg.get("from") == "gpt":
                    assistant.append(msg.get("value", ""))
                elif msg.get("from") == "human":
                    user.append(msg.get("value", ""))
        return {"assistant": assistant, "user": user}
    except Exception as e:
        print(f"  ShareGPT not available ({e}), trying alt...")
        try:
            ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")
            assistant, user = [], []
            for row in ds:
                user.append(row.get("prompt", ""))
                assistant.append(row.get("response_a", ""))
            return {"assistant": assistant, "user": user}
        except:
            return None


def load_oasst():
    """OpenAssistant Conversations — human-written assistant responses."""
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        assistant, user = [], []
        for row in ds:
            if row.get("role") == "assistant":
                assistant.append(row["text"])
            elif row.get("role") == "prompter":
                user.append(row["text"])
        return {"assistant": assistant, "user": user}
    except Exception as e:
        print(f"  OASST error: {e}")
        return None


def load_stackexchange():
    """StackExchange answers — human expert responses."""
    try:
        ds = load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train", streaming=True)
        answers = []
        questions = []
        for i, row in enumerate(ds):
            if i >= 100000:
                break
            questions.append(row.get("question", ""))
            ans_list = row.get("answers", [])
            if isinstance(ans_list, list) and len(ans_list) > 0:
                best = max(ans_list, key=lambda a: a.get("pm_score", 0) if isinstance(a, dict) else 0)
                text = best.get("text", "") if isinstance(best, dict) else str(best)
                answers.append(text)
            if (i + 1) % 50000 == 0:
                print(f"    StackExchange: {i+1} rows, {len(answers)} answers", flush=True)
        print(f"    StackExchange: {len(questions)} questions, {len(answers)} answers", flush=True)
        return {"assistant": answers, "user": questions}
    except Exception as e:
        print(f"  StackExchange error: {e}")
        return None


def load_reddit():
    """Reddit answers — human conversational responses."""
    try:
        ds = load_dataset("reddit", split="train", streaming=True)
        comments = []
        for i, row in enumerate(ds):
            if i >= 100000:
                break
            body = row.get("body", row.get("content", ""))
            if body and isinstance(body, str) and len(body) > 20:
                comments.append(body)
        return {"assistant": comments}
    except Exception as e:
        print(f"  Reddit error ({e}), trying webis/tldr...")
        try:
            ds = load_dataset("webis/tldr-17", split="train", streaming=True)
            comments = []
            for i, row in enumerate(ds):
                if i >= 100000:
                    break
                body = row.get("content", row.get("body", ""))
                if body and isinstance(body, str) and len(body) > 20:
                    comments.append(body)
            return {"assistant": comments}
        except Exception as e2:
            print(f"  Reddit fallback error: {e2}")
            return None


CORPUS_LOADERS = {
    "alpaca": load_alpaca,
    "ultrachat": load_ultrachat,
    "ultrafeedback": load_ultrafeedback,
    "pku_saferlhf": load_pku,
    "hh_rlhf": load_hh_rlhf,
    "sharegpt": load_sharegpt,
    "tulu_sft": load_tulu_sft,
    "oasst": load_oasst,
    "stackexchange": load_stackexchange,
    "reddit": load_reddit,
}


def main():
    out = os.path.join(DATA_DIR, "f37_marker_rates.csv")
    tc_out = os.path.join(DATA_DIR, "f37_corpus_tokens.csv")

    # Load existing results to skip completed corpora
    done = set()
    if os.path.exists(out):
        existing = pd.read_csv(out)
        done = set(existing.corpus.unique())
        print(f"Resuming: {len(done)} corpora already done: {done}")

    for corpus_name, loader in CORPUS_LOADERS.items():
        if corpus_name in done:
            print(f"\n  {corpus_name}: CACHED, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  {corpus_name}")
        print(f"{'='*60}")
        t0 = time.time()

        data = loader()
        if data is None:
            print(f"  SKIPPED")
            continue

        corpus_results = []
        corpus_tokens = []
        for role_key, texts in data.items():
            if "_chosen" in role_key:
                role = "assistant"
                side = "chosen"
            elif "_rejected" in role_key:
                role = "assistant"
                side = "rejected"
            else:
                role = role_key
                side = ""

            print(f"  {role_key}: {len(texts)} texts...")
            results = process_texts(texts, corpus_name, role, side)
            corpus_results.extend(results)

            total_tokens = sum(r["tokens"] for r in results[:1])
            corpus_tokens.append({
                "corpus": corpus_name,
                "role": role,
                "side": side,
                "n_texts": len(texts),
                "total_tokens": total_tokens,
            })

        # Append to CSVs immediately
        df_new = pd.DataFrame(corpus_results)
        header = not os.path.exists(out)
        df_new.to_csv(out, mode='a', header=header, index=False)

        tc_new = pd.DataFrame(corpus_tokens)
        tc_header = not os.path.exists(tc_out)
        tc_new.to_csv(tc_out, mode='a', header=tc_header, index=False)

        print(f"  Done ({time.time()-t0:.0f}s) — {len(df_new)} rows appended to {out}")

    # Print summary
    if os.path.exists(out):
        df = pd.read_csv(out)
        print(f"\nTotal: {len(df)} rows across {df.corpus.nunique()} corpora")
        print("\n" + "="*80)
        print("MARKER RATES (per 10k tokens, assistant role)")
        print("="*80)
        asst = df[df.role == "assistant"]
        if not asst.empty:
            pivot = asst.pivot_table(index="corpus", columns="marker", values="rate_per_10k")
            print(pivot.to_string(float_format="%.2f"))


if __name__ == "__main__":
    main()
