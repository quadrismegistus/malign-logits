"""Plan K bulk: rate every word in the N=50 population, one call per word.

    uv run python meta/M01_displacement/scripts/k_bulk.py en
    uv run python meta/M01_displacement/scripts/k_bulk.py zh

Writes `results/k/ratings_<lang>.json` and appends nothing else. Run the two
languages as separate processes; they share no state and their caches are keyed
by task name, so parallel is safe.

THE UNIT IS THE NORMALISED WORD, THE JOIN KEY IS THE TOKEN. `results/k/
normalisation_<lang>.json` holds `token_to_unit`. The rating is a property of
the word; the movement is a property of the token; one rating serves every token
that normalises onto it. English collapses 30,223 tokens to 27,242 units (9.7%),
Chinese 21,230 to 20,654 (2.6% -- no capitalisation, so only punctuation does
any work).

SCRIPT ROUTING, CHINESE ONLY. 13.6% of the "Chinese" population is English
tokens the tokenizer emits at Chinese prompts -- `offered`, `neck`, `nutrition`
-- and they genuinely move, so excluding them would select on a property. But
rating them under an instruction that says "the words are CHINESE" would be
wrong. A unit containing any CJK character goes to the ZH instrument, everything
else to the EN one. Nothing is dropped for being the wrong language.

WHAT IS NOT RATED, and it is recorded rather than dropped: units that normalise
to bare punctuation (49 English, 30 Chinese). They are not words and a rating
for them would be an invented number.

The instrument is FROZEN at `results/k/INSTRUMENT.txt`. Adding three scales once
moved `penis` vulgarity 2->4 and `defenestrate` charge 2->4 at temperature 0, so
a rating is a property of the instrument VERSION and outputs from different
versions must never be pooled. Any change means a new sha and a new file.
"""
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
SCALES = ("vulgarity", "register_level", "transgressiveness", "charge",
          "valence", "bodily_harm", "concreteness")
CJK = re.compile(r'[一-鿿㐀-䶿]')
WORKERS = 16          #: the default of 4 runs at ~2/s; 16 gives ~12.5/s


def main(lang):
    from malign_logits.tasks.rate_charge_v1 import ChargeTask7EN, ChargeTask7ZH
    norm = json.load(open(os.path.join(K, "normalisation_%s.json" % lang)))
    units = sorted(set(norm["token_to_unit"].values()))
    zh = [u for u in units if CJK.search(u)]
    en = [u for u in units if not CJK.search(u)]
    print("[%s] %d tokens -> %d rating units | %d to ZH instrument, %d to EN"
          % (lang, norm["n_tokens"], len(units), len(zh), len(en)), flush=True)
    print("[%s] instrument frozen; workers=%d" % (lang, WORKERS), flush=True)

    out, t0 = {}, time.time()
    for tag, words, task in (("zh", zh, ChargeTask7ZH), ("en", en, ChargeTask7EN)):
        if not words:
            continue
        t = task()
        print("[%s] %s instrument: %d words ..." % (lang, tag, len(words)), flush=True)
        res = t.map(words, num_workers=WORKERS)
        ok = 0
        for w, r in zip(words, res):
            if r is None:
                continue
            ok += 1
            out[w] = {s: getattr(r, s) for s in SCALES}
            out[w]["reading"] = r.reading
            out[w]["_instrument"] = tag
        print("[%s] %s instrument: %d/%d returned  (%.0fs elapsed)"
              % (lang, tag, ok, len(words), time.time() - t0), flush=True)

    path = os.path.join(K, "ratings_%s.json" % lang)
    json.dump({"_lang": lang, "_n_units": len(units), "_n_rated": len(out),
               "_n_failed": len(units) - len(out),
               "_instrument_sha256": open(os.path.join(K, "INSTRUMENT.txt")).read(),
               "_workers": WORKERS, "ratings": out},
              open(path, "w"), ensure_ascii=False)
    print("[%s] DONE %d/%d rated in %.0fs -> %s"
          % (lang, len(out), len(units), time.time() - t0, os.path.relpath(path, ROOT)),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "en"))
