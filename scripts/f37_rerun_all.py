"""F37 Track A RERUN: all markers + KWIC + conditional + unigrams + frame profiles.

Applies language filter + code/HTML stripping to all corpora.
Splits death_naming into per-form counts.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/f37_rerun_all.py
"""
import re
import os
import sys
import random
import time
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
DATA_DIR = os.path.join(ROOT, "data")

from f37_text_filters import filter_texts, strip_code_html, is_english
from f37_kwic_conditional import load_corpus_texts, DEATH_WORDS, VIOLENCE_WORDS, ALL_DV, find_kwic
from f37_dataset_markers import MARKERS, count_markers, tokenize_count

# Per-form death markers (replacing the aggregate)
DEATH_FORMS = {
    "death_kill": ["kill"],
    "death_kills": ["kills"],
    "death_killed": ["killed"],
    "death_killing": ["killing"],
    "death_die": ["die"],
    "death_dies": ["dies"],
    "death_died": ["died"],
    "death_dying": ["dying"],
    "death_death": ["death"],
    "death_dead": ["dead"],
}

# Extended markers = original minus death_naming, plus per-form
ALL_MARKERS = {k: v for k, v in MARKERS.items() if k != "death_naming"}
ALL_MARKERS.update(DEATH_FORMS)
ALL_MARKERS["death_naming"] = MARKERS["death_naming"]  # keep aggregate too

WORD_RE = re.compile(r'\b[a-z]+\b')

CORPORA = ["alpaca", "ultrachat", "ultrafeedback", "sharegpt",
           "tulu_sft", "oasst", "stackexchange"]

# PKU and HH need special loading for chosen/rejected
PREF_CORPORA = ["pku_saferlhf", "hh_rlhf"]

FRAMES = [
    "wanted to", "wants to", "want to", "began to", "decided to",
    "chose to", "started to", "moved to", "prepared to",
    "she wanted to", "he wanted to", "put his cock",
    "the knife and", "deeply and wanted to", "was fired and decided to",
]


def process_markers(texts, corpus_name, role, side=""):
    """Count all markers on a list of texts."""
    rows = []
    total_tokens = 0
    marker_counts = {k: 0 for k in ALL_MARKERS}
    n_valid = 0

    for i, text in enumerate(texts):
        if not text or not isinstance(text, str):
            continue
        n_tokens = tokenize_count(text)
        total_tokens += n_tokens
        n_valid += 1
        for mk, mlist in ALL_MARKERS.items():
            marker_counts[mk] += count_markers(text, mlist)
        if (i + 1) % 100000 == 0:
            print(f"    markers {corpus_name}/{role}: {i+1}/{len(texts)}", flush=True)

    if total_tokens == 0:
        return []

    for mk, count in marker_counts.items():
        rows.append({
            "corpus": corpus_name, "role": role, "side": side,
            "marker": mk, "count": count, "tokens": total_tokens,
            "rate_per_10k": count / total_tokens * 10000, "n_texts": n_valid,
        })
    return rows


def load_pku():
    from datasets import load_dataset
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


def load_hh():
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    chosen_a, rejected_a, user_t = [], [], []
    for row in ds:
        for side, key in [("chosen", "chosen"), ("rejected", "rejected")]:
            text = row[key]
            parts = text.split("\n\nAssistant: ")
            for part in parts[1:]:
                asst = part.split("\n\nHuman: ")[0].strip()
                if side == "chosen":
                    chosen_a.append(asst)
                else:
                    rejected_a.append(asst)
        parts = row["chosen"].split("\n\nHuman: ")
        for part in parts[1:]:
            user_t.append(part.split("\n\nAssistant: ")[0].strip())
    return {"assistant_chosen": chosen_a, "assistant_rejected": rejected_a, "user": user_t}


def main():
    random.seed(42)
    out_markers = os.path.join(DATA_DIR, "f37_marker_rates_v2.csv")
    out_tokens = os.path.join(DATA_DIR, "f37_corpus_tokens_v2.csv")
    out_kwic = os.path.join(DATA_DIR, "f37_kwic_death_v2.csv")
    out_frames = os.path.join(DATA_DIR, "f37_frame_profiles_v2.csv")

    all_marker_rows = []
    all_token_rows = []
    all_kwic_rows = []
    all_frame_rows = []

    # Standard corpora
    for corpus_name in CORPORA:
        print(f"\n{'='*60}\n  {corpus_name}\n{'='*60}", flush=True)
        t0 = time.time()
        asst, user = load_corpus_texts(corpus_name)
        print(f"  Raw: {len(asst)} assistant, {len(user)} user", flush=True)

        # Filter
        do_code = corpus_name in ("stackexchange", "sharegpt", "oasst")
        asst_clean, asst_stats = filter_texts(asst, do_lang=True, do_code=do_code, corpus_name=corpus_name+"/asst")
        user_clean, user_stats = filter_texts(user, do_lang=True, do_code=do_code, corpus_name=corpus_name+"/user")
        print(f"  Filtered: asst {asst_stats['retained']}/{asst_stats['total']} "
              f"(lang_drop={asst_stats['lang_dropped']}, code_strip={asst_stats['code_stripped']})", flush=True)

        all_token_rows.append({"corpus": corpus_name, "role": "assistant", **asst_stats})
        all_token_rows.append({"corpus": corpus_name, "role": "user", **user_stats})

        # Markers
        all_marker_rows.extend(process_markers(asst_clean, corpus_name, "assistant"))
        all_marker_rows.extend(process_markers(user_clean, corpus_name, "user"))

        # KWIC (100 per corpus)
        death_set = set(DEATH_WORDS)
        hits = []
        for text in asst_clean:
            for kw, left, match, right in find_kwic(text, death_set):
                hits.append({"corpus": corpus_name, "role": "assistant",
                             "keyword": kw, "context": f"{left} [{match}] {right}"})
        sample = random.sample(hits, min(100, len(hits)))
        all_kwic_rows.extend(sample)
        print(f"  KWIC: {len(hits)} hits, sampled {len(sample)}", flush=True)

        # Frame profiles
        for text in asst_clean:
            if not text:
                continue
            text_lower = text.lower()
            for frame in FRAMES:
                idx = 0
                while True:
                    pos = text_lower.find(frame, idx)
                    if pos == -1:
                        break
                    after = text_lower[pos + len(frame):].strip()
                    m = WORD_RE.match(after)
                    if m:
                        all_frame_rows.append((frame, corpus_name, m.group()))
                    idx = pos + 1

        # Unigrams
        uni_out = os.path.join(DATA_DIR, f"f37_corpus_unigrams_{corpus_name}_v2.csv")
        counts = Counter()
        total_tok = 0
        for text in asst_clean:
            words = WORD_RE.findall(text.lower())
            counts.update(words)
            total_tok += len(words)
        filtered_counts = {w: c for w, c in counts.items() if c >= 5}
        uni_df = pd.DataFrame([{"word": w, "count": c, "total_tokens": total_tok}
                                for w, c in sorted(filtered_counts.items(), key=lambda x: -x[1])])
        uni_df.to_csv(uni_out, index=False)

        # Conditional (off-topic)
        n_paired = min(len(asst_clean), len(user_clean))
        offtopic = []
        for i in range(n_paired):
            prompt_words = set(WORD_RE.findall(user_clean[i].lower()))
            if not prompt_words.intersection(ALL_DV):
                offtopic.append(asst_clean[i])
        all_marker_rows.extend(process_markers(offtopic, corpus_name + "_offtopic", "assistant"))

        print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

        # Save incrementally
        pd.DataFrame(all_marker_rows).to_csv(out_markers, index=False)
        pd.DataFrame(all_token_rows).to_csv(out_tokens, index=False)

    # Preference corpora (PKU, HH)
    for corpus_name, loader in [("pku_saferlhf", load_pku), ("hh_rlhf", load_hh)]:
        print(f"\n{'='*60}\n  {corpus_name}\n{'='*60}", flush=True)
        t0 = time.time()
        data = loader()

        for role_key, texts in data.items():
            role = "assistant" if "assistant" in role_key else "user"
            side = "chosen" if "chosen" in role_key else ("rejected" if "rejected" in role_key else "")

            clean, stats = filter_texts(texts, do_lang=True, do_code=False, corpus_name=f"{corpus_name}/{role_key}")
            print(f"  {role_key}: {stats['retained']}/{stats['total']} retained", flush=True)

            all_token_rows.append({"corpus": corpus_name, "role": role, "side": side, **stats})
            all_marker_rows.extend(process_markers(clean, corpus_name, role, side))

            if role == "assistant":
                hits = []
                for text in clean:
                    for kw, left, match, right in find_kwic(text, set(DEATH_WORDS)):
                        hits.append({"corpus": corpus_name, "role": role, "side": side,
                                     "keyword": kw, "context": f"{left} [{match}] {right}"})
                sample = random.sample(hits, min(100, len(hits)))
                all_kwic_rows.extend(sample)

                for text in clean:
                    text_lower = text.lower()
                    for frame in FRAMES:
                        idx = 0
                        while True:
                            pos = text_lower.find(frame, idx)
                            if pos == -1:
                                break
                            after = text_lower[pos + len(frame):].strip()
                            m = WORD_RE.match(after)
                            if m:
                                all_frame_rows.append((frame, corpus_name + "_" + side, m.group()))
                            idx = pos + 1

                uni_out = os.path.join(DATA_DIR, f"f37_corpus_unigrams_{corpus_name}_{side}_v2.csv")
                uni_counts = Counter()
                uni_total = 0
                for text in clean:
                    words = WORD_RE.findall(text.lower())
                    uni_counts.update(words)
                    uni_total += len(words)
                filt = {w: c for w, c in uni_counts.items() if c >= 5}
                pd.DataFrame([{"word": w, "count": c, "total_tokens": uni_total}
                               for w, c in sorted(filt.items(), key=lambda x: -x[1])]).to_csv(uni_out, index=False)

        print(f"  Done ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(all_marker_rows).to_csv(out_markers, index=False)
        pd.DataFrame(all_token_rows).to_csv(out_tokens, index=False)

    # Save final outputs
    pd.DataFrame(all_marker_rows).to_csv(out_markers, index=False)
    pd.DataFrame(all_token_rows).to_csv(out_tokens, index=False)
    pd.DataFrame(all_kwic_rows).to_csv(out_kwic, index=False)

    # Frame profiles
    frame_counts = Counter(all_frame_rows)
    fp_rows = [{"frame": f, "corpus": c, "next_word": w, "count": cnt}
               for (f, c, w), cnt in frame_counts.items() if cnt >= 3]
    pd.DataFrame(fp_rows).to_csv(out_frames, index=False)

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL: {len(all_marker_rows)} marker rows, {len(all_kwic_rows)} KWIC rows, "
          f"{len(fp_rows)} frame rows", flush=True)

    # Summary table
    df = pd.DataFrame(all_marker_rows)
    asst = df[(df.role == "assistant") & (df.side == "") & (~df.corpus.str.contains("_offtopic"))]
    if not asst.empty:
        pivot = asst.pivot_table(index="corpus", columns="marker", values="rate_per_10k")
        key_cols = ["death_naming", "death_die", "death_kill", "death_dead",
                    "violence_nondeath", "moralizing", "deference", "de_escalation"]
        avail = [c for c in key_cols if c in pivot.columns]
        print(pivot[avail].to_string(float_format="%.2f"), flush=True)


if __name__ == "__main__":
    main()
