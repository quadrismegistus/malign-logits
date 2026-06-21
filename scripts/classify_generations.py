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
    (r'\bA\.\s+\S', 1),
    (r'\(a\)\s+\S', 1),
    (r'Answer:\s*[A-E]', 2),
    (r'答案[:：]\s*[A-E]', 2),
    (r'Choose the correct', 2),
    (r'Which of the following', 2),
    (r'下列.*(?:正确|错误|不正确)', 2),
    (r'选项', 1),
    (r'Does it follow that', 2),
    (r'Choose your answer', 2),
    (r'Options?:\s*\n', 2),
    (r'pick from the following', 2),
    (r'Therefore,? the answer is', 2),
    (r'the answer is \w', 1),
    (r"Let's give stream of consciousness", 2),
    (r'____', 1),
    (r'fill in the (?:blank|gap)', 2),
    (r'\(\s*\)\s*(?:refers to|is the)', 1),
    (r'正确率', 1),
    (r'易错项', 1),
    (r'简述', 1),
    (r'（[^）]{1,10}）', 1),
    (r'Are these two sentences paraphrases?', 2),
    (r'(?:rephrase|paraphrase).*(?:sentence|question)', 1),
    (r'Choose from:\s*\(', 2),
    (r'(?:翻译|怎么翻译|用英文怎么说|阅读下面|回答各题)', 1),
    (r'(?:punctuated version|add in punctuation|correct punctuation)', 2),
    (r'(?:Can we conclude|is it true that|entailment|contradiction)\b', 1),
    (r'Step \d+:\s', 1),
]

REFUSAL_PATTERNS = [
    r"I (?:cannot|can't|am unable to|must decline)",
    r"I'm (?:not able|unable) to (?:continue|write|generate|create|produce)",
    r"I can't (?:continue|write|generate|create|produce)",
    r"I apologize,? but I",
    r"I appreciate you.{0,30}but I(?:'m not| cannot| can't)",
    r"not appropriate (?:to|for|in)",
    r"is not allowed",
    r"content warning",
    r"I'm sorry,? (?:but )?I (?:can't|cannot|am unable)",
    r"I'm happy to (?:help|assist) with",
    r"this (?:type of )?content is not (?:allowed|appropriate)",
    r"violates? (?:our|the) (?:policy|guidelines|terms)",
    r"(?:explicit|graphic) (?:sexual|violent|violence) content",
]

INSTRUCT_LEAK_PATTERNS = [
    r'In your response,? the (?:letter|word)',
    r'There should be \d+ paragraphs?',
    r'the (?:letter|word) \w+ should appear',
    r'Your response (?:must|should)',
    r'Do not include any',
    r'Write \d+ paragraphs?',
    r'Your answer must contain',
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

WEB_PATTERNS = [
    (r'https?://', 2),
    (r'<(?:div|span|p|a\s|br|img|table|tr|td|li|ul|ol|h[1-6]|script|style|meta|link|!--|DOCTYPE)', 2),
    (r'\[.{1,80}\]\(https?://', 2),
    (r'[\w.+-]+@[\w-]+\.[\w.-]+', 1),
    (r'\b\d{3}[-.)]\s*\d{3}[-.)]\s*\d{4}\b', 1),
    (r'(?:\{"|"\s*:\s*[\[\{"])', 1),
    (r'<!\[CDATA\[', 2),
    (r'<?xml\s', 2),
    (r'xmlns[:=]', 2),
    (r'(?:call now|click here|subscribe|free shipping|limited time|buy now|order now|discount code|escort|incall|outcall)', 1),
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

    # Exam/MCQ detection (weighted: 2 = strong signal, 1 = weak)
    exam_score = sum(w for pat, w in EXAM_PATTERNS if re.search(pat, text))

    # Decide
    if code_hits >= 3 and code_hits > exam_score:
        best_lang = max(code_langs, key=code_langs.get) if code_langs else "unknown"
        return "code", best_lang

    if exam_score >= 2:
        return "exam", None

    # Check for math/LaTeX
    latex_hits = len(re.findall(r'\$\$?[^$]+\$\$?|\\frac|\\sum|\\int', text))
    if latex_hits >= 2:
        return "math", None

    # Refusal (aligned model safety response)
    refusal_hits = sum(1 for pat in REFUSAL_PATTERNS if re.search(pat, text, re.I))
    if refusal_hits >= 2:
        return "refusal", None

    # Instruction leak (benchmark/prompt constraint bleeds into output)
    instruct_hits = sum(1 for pat in INSTRUCT_LEAK_PATTERNS if re.search(pat, text, re.I))
    if instruct_hits >= 1:
        return "instruct_leak", None

    # Web residue: URLs, HTML fragments, emails, ads, XML
    web_score = sum(w for pat, w in WEB_PATTERNS if re.search(pat, text, re.I))
    if web_score >= 2:
        return "web", None

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
