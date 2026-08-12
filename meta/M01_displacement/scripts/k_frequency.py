"""Frequency norms for Plan K, English and Chinese, with the register argument.

    from k_frequency import fpm, MEASURES
    fpm("scream", "en", "coca_fic")     # per-million
    fpm("裤子", "zh")                    # SUBTLEX-CH

WHICH NORM, AND WHY IT IS NOT A DETAIL. Measured 2026-08-12 over the 12,673
words of the K English population covered by all fourteen measures, Spearman:

    fpm_COCA ~ fpm_coca_mag      0.97      the top of the range
    fpm_coca_acad ~ fpm_bnc_acad 0.92      academic corpora agree with academic
    fpm_coca_fic ~ fpm_bnc_fic   0.92      fiction agrees with fiction
    fpm_coca_fic ~ fpm_bnc_acad  0.48      the bottom of the range
    SUBTLEX_US ~ fpm_SOAP        0.90      subtitles agree with soap opera
    SUBTLEX_US ~ fpm_COCA        0.80
    SUBTLEX_US ~ fpm_bnc_acad    0.54

**The measures are not interchangeable and the structure is REGISTER.** A word's
rank moves enormously between an academic corpus and a fiction one, so "control
for frequency" is underspecified until the corpus is named.

THE DEFAULT IS REGISTER-MATCHED, NOT NAME-MATCHED. K's prompts are narrative
fiction -- "She slowly took off her", "He struck the prisoner hard across the"
-- so the English default is `coca_fic` and NOT the better-known `fpm_COCA`,
which correlates with it at only 0.81. The Chinese default is SUBTLEX-CH, which
is film subtitles and therefore the closest Chinese analogue in register even
though it is a different corpus family from COCA.

    register-matched   coca_fic (en)  +  SUBTLEX-CH (zh)      <- default
    lineage-matched    SUBTLEX-US (en) + SUBTLEX-CH (zh)      <- robustness

SUBTLEX-US ~ coca_fic is 0.83, so the two English choices are a real check on
each other rather than a formality. Report the partial correlation under both;
if it survives only one, that is a finding about the norm and not about
alignment.

COVERAGE of the K rating units, which bounds what any of this can control:

    English  SUBTLEX-US 67.5%,  BYU/COCA 65.0%,  both 61.5%
    Chinese  SUBTLEX-CH 52.8%  -- 97% of single characters, 62% of two-character
             words, 14-24% of the longer compounds. The misses are numerals and
             measure phrases the tokenizer assembles (100万, 2015年, 150米),
             which are not words a corpus would list.

SOURCES
    BYU/COCA      ~/Dropbox/Prof/Code/osp/worddb.byu.txt  (13 fpm columns)
    SUBTLEX-US    norms_sources/subtlex_us/  Brysbaert & New 2009, 60,384 words
    SUBTLEX-CH    norms_sources/subtlex_ch/  Cai & Brysbaert 2010, CC-BY,
                  98,733 words, 33.5M-word subtitle corpus
"""
import csv
import functools
import os

NORMS = "/Users/rj416/Dropbox/Prof/Articles/TheoryMachines/norms_sources"
BYU = os.path.expanduser("~/Dropbox/Prof/Code/osp/worddb.byu.txt")

#: every English measure available, so a robustness sweep does not have to
#: rediscover the column names
MEASURES = ("coca_fic", "COCA", "BNC", "SOAP", "coca_spok", "coca_news",
            "coca_acad", "coca_mag", "bnc_fic", "bnc_spok", "bnc_acad",
            "COHA_1800s", "COHA_1950-89", "SUBTLEX_US")
DEFAULT = {"en": "coca_fic", "zh": "SUBTLEX_CH"}


@functools.lru_cache(maxsize=1)
def _byu():
    out = {}
    with open(BYU, encoding="utf-8", errors="ignore") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            w = (r.get("word") or "").strip().lower()
            if not w or w in out:
                continue
            d = {}
            for m in MEASURES:
                if m == "SUBTLEX_US":
                    continue
                try:
                    v = float(r.get("fpm_" + m) or 0)
                    if v > 0:
                        d[m] = v
                except (TypeError, ValueError):
                    pass
            if d:
                out[w] = d
    return out


@functools.lru_cache(maxsize=1)
def _subtlex_us():
    p = os.path.join(NORMS, "subtlex_us", "subtlex_us.tsv")
    out = {}
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                out[r["word"].strip().lower()] = float(r["fpm"])
            except (KeyError, TypeError, ValueError):
                pass
    return out


@functools.lru_cache(maxsize=1)
def _subtlex_ch():
    p = os.path.join(NORMS, "subtlex_ch", "SUBTLEX_CH_131210_CE.utf8")
    out = {}
    with open(p, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            w = (r.get("Word") or "").strip()
            try:
                out[w] = float(r["W.million"])
            except (KeyError, TypeError, ValueError):
                pass
    return out


def fpm(word, lang="en", measure=None):
    """Frequency per million, or None. NEVER 0 for an absent word.

    An absent word and a zero-frequency word are different facts, and returning
    0.0 for both puts every uncovered word at the bottom of the frequency rank
    -- which is exactly the direction that would manufacture a rarity effect in
    a study whose main confound is rarity.
    """
    m = measure or DEFAULT[lang]
    if lang == "zh" and m == "SUBTLEX_CH":
        return _subtlex_ch().get(word.strip())
    w = word.strip().lower()
    if m == "SUBTLEX_US":
        return _subtlex_us().get(w)
    return _byu().get(w, {}).get(m)


if __name__ == "__main__":
    for w in ("scream", "kill", "the", "defenestrate"):
        print("  %-14s coca_fic %-9s COCA %-9s SUBTLEX_US %s"
              % (w, fpm(w, "en"), fpm(w, "en", "COCA"), fpm(w, "en", "SUBTLEX_US")))
    for w in ("裤子", "鞋", "衣服", "高跟鞋"):
        print("  %-14s SUBTLEX-CH %s" % (w, fpm(w, "zh")))
