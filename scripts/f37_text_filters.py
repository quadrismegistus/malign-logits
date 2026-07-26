"""F37 text filters: language detection + code/HTML stripping.

Shared by all Track A scripts.
"""
import re
from langdetect import detect, LangDetectException

# HTML tags
HTML_RE = re.compile(r'<[^>]+>')
# Code blocks (markdown fenced + inline)
CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')
CODE_INLINE_RE = re.compile(r'`[^`]+`')
# Common code patterns
CODE_LINE_RE = re.compile(r'^[\s]*(import |from |def |class |#include|public |private |var |let |const |function |if\s*\(|for\s*\(|while\s*\(|\{|\}|;$)', re.MULTILINE)


def strip_code_html(text):
    """Remove HTML tags and code blocks from text."""
    text = CODE_BLOCK_RE.sub(' ', text)
    text = CODE_INLINE_RE.sub(' ', text)
    text = HTML_RE.sub(' ', text)
    # Remove lines that look like code
    lines = text.split('\n')
    clean_lines = [l for l in lines if not CODE_LINE_RE.match(l)]
    return ' '.join(clean_lines)


def is_english(text, min_len=20):
    """Check if text is English. Returns True/False."""
    if not text or len(text) < min_len:
        return True  # too short to detect, keep by default
    try:
        return detect(text[:500]) == 'en'
    except LangDetectException:
        return True  # detection failed, keep


def filter_texts(texts, do_lang=True, do_code=True, corpus_name=""):
    """Filter and clean a list of texts. Returns (cleaned_texts, stats)."""
    total = len(texts)
    lang_dropped = 0
    code_stripped = 0
    cleaned = []

    for i, text in enumerate(texts):
        if not text or not isinstance(text, str):
            continue

        # Language filter
        if do_lang and not is_english(text):
            lang_dropped += 1
            continue

        # Code/HTML stripping
        if do_code:
            original_len = len(text)
            text = strip_code_html(text)
            if len(text) < original_len * 0.5:
                code_stripped += 1

        if text.strip():
            cleaned.append(text)

        if (i + 1) % 50000 == 0:
            print(f"    {corpus_name}: filtered {i+1}/{total} "
                  f"(lang_dropped={lang_dropped}, code_stripped={code_stripped})", flush=True)

    stats = {
        'total': total,
        'retained': len(cleaned),
        'lang_dropped': lang_dropped,
        'code_stripped': code_stripped,
        'retain_frac': len(cleaned) / total if total > 0 else 0,
    }
    return cleaned, stats
