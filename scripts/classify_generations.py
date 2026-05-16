"""Classify BOS/The generations by genre: code, exam/MCQ, chat template, non-English, prose.

Outputs a parquet with columns: family, layer, prompt_type, idx, text, genre, language, code_lang.
"""

import re
import pandas as pd
from malign_logits import MODEL_FAMILIES
from malign_logits.cache import get_cache


# ── Genre classifier ──────────────────────────────────────────────

CODE_PATTERNS = [
    (r'```', None),
    (r'def \w+\(', 'python'),
    (r'import \w+', 'python'),
    (r'class \w+[:\(]', 'python'),
    (r'print\(', 'python'),
    (r'function\s*\w*\(', 'javascript'),
    (r'const \w+\s*=', 'javascript'),
    (r'var \w+\s*=', 'javascript'),
    (r'console\.log', 'javascript'),
    (r'public\s+(static\s+)?void', 'java'),
    (r'System\.out\.print', 'java'),
    (r'<\?php', 'php'),
    (r'#include\s*<', 'c/c++'),
    (r'int main\(', 'c/c++'),
    (r'<!DOCTYPE|<html|<div|<span', 'html'),
    (r'SELECT\s+\w+\s+FROM', 'sql'),
    (r'package\s+\w+', 'java'),
    (r'return\s+\w+;', None),
    (r'if\s*\(.+\)\s*\{', None),
    (r'\bself\.\w+', 'python'),
    (r'for\s+\w+\s+in\s+', 'python'),
]

EXAM_PATTERNS = [
    r'\bA\.\s+\S',
    r'\(a\)\s+\S',
    r'Answer:\s*[A-E]',
    r'答案[:：]\s*[A-E]',
    r'Choose the correct',
    r'Which of the following',
    r'下列.*(?:正确|错误|不正确)',
    r'选项',
]

TEMPLATE_PATTERNS = [
    r'You are a helpful',
    r'function-calling AI assistant',
    r'<\|extra_id',
    r'<\|system\|>',
    r'<\|assistant\|>',
    r'Your answer must contain exactly',
    r'your response should',
]


def detect_language(text):
    """Simple language detection without langdetect dependency."""
    cjk = len(re.findall(r'[一-鿿]', text))
    hangul = len(re.findall(r'[가-힯]', text))
    total = len(text)
    if total == 0:
        return "empty"
    if cjk / total > 0.1:
        return "zh"
    if hangul / total > 0.1:
        return "ko"
    # Check for non-ASCII European (Finnish, Dutch, etc.)
    non_ascii_latin = len(re.findall(r'[À-ÿĀ-žŽ-ž]', text))
    if non_ascii_latin / max(total, 1) > 0.03:
        return "other_euro"
    return "en"


def classify_genre(text):
    """Returns (genre, code_lang)."""
    if not text or len(text.strip()) < 10:
        return "empty", None

    # Chat template
    for pat in TEMPLATE_PATTERNS:
        if re.search(pat, text[:300]):
            return "template", None

    # Code detection: count pattern matches
    code_hits = 0
    code_langs = {}
    for pat, lang in CODE_PATTERNS:
        matches = len(re.findall(pat, text))
        if matches:
            code_hits += matches
            if lang:
                code_langs[lang] = code_langs.get(lang, 0) + matches

    # Exam/MCQ detection
    exam_hits = sum(1 for pat in EXAM_PATTERNS if re.search(pat, text))

    # Decide
    if code_hits >= 3 and code_hits > exam_hits:
        best_lang = max(code_langs, key=code_langs.get) if code_langs else "unknown"
        return "code", best_lang

    if exam_hits >= 2:
        return "exam", None

    # Check for math/LaTeX
    latex_hits = len(re.findall(r'\$\$?[^$]+\$\$?|\\frac|\\sum|\\int', text))
    if latex_hits >= 2:
        return "math", None

    return "prose", None


# ── Main ──────────────────────────────────────────────────────────

def main():
    cache = get_cache()
    rows = []

    for fam_key, fam in MODEL_FAMILIES.items():
        layers = [
            ("base", fam.base),
            ("ego", fam.ego),
            ("superego", fam.superego),
            ("instruct", fam.reinforced_superego),
        ]
        for layer_name, model_id in layers:
            if model_id is None:
                continue
            for prompt_type, prompts in [("bos", ["<|endoftext|>", "<|begin_of_text|>", "<s>"]), ("the", ["The"])]:
                for prompt in prompts:
                    n = cache.count_generations(model_id, prompt, temp=1.0)
                    if n == 0:
                        continue
                    for idx in range(n):
                        text = cache.get_generation(model_id, prompt, temp=1.0, idx=idx)
                        if text is None:
                            continue
                        genre, code_lang = classify_genre(text)
                        lang = detect_language(text)
                        rows.append({
                            "family": fam_key,
                            "layer": layer_name,
                            "model_id": model_id,
                            "prompt_type": prompt_type,
                            "idx": idx,
                            "genre": genre,
                            "language": lang,
                            "code_lang": code_lang,
                            "text": text,
                        })
                    break  # found the right BOS token

    df = pd.DataFrame(rows)
    out = "data/bos_generations.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} generations to {out}")
    print()

    # Summary
    print("=== Genre distribution (BOS) ===")
    bos = df[df["prompt_type"] == "bos"]
    print(bos.groupby(["layer", "genre"]).size().unstack(fill_value=0).to_string())
    print()

    print("=== Genre distribution (The) ===")
    the = df[df["prompt_type"] == "the"]
    print(the.groupby(["layer", "genre"]).size().unstack(fill_value=0).to_string())
    print()

    print("=== Language distribution (BOS) ===")
    print(bos.groupby(["layer", "language"]).size().unstack(fill_value=0).to_string())
    print()

    print("=== Code languages (BOS, code genre only) ===")
    code = bos[bos["genre"] == "code"]
    if len(code):
        print(code.groupby(["layer", "code_lang"]).size().unstack(fill_value=0).to_string())
    print()

    print("=== Per-family genre (BOS, base layer) ===")
    base_bos = bos[bos["layer"] == "base"]
    print(base_bos.groupby(["family", "genre"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
