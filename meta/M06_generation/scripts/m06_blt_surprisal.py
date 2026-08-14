"""BLT byte-level surprisal: the campaign's first CROSS-LINGUAL surprisal axis.

    uv run python meta/M06_generation/scripts/m06_blt_surprisal.py [--n 100] [--corpus f11_l2]
    -> results/blt_surprisal_pilot.json  + cache entries under ref_surprisal

WHY BLT AND NOT gpt2. Measured on this box: gpt2 spends 1.95 tokens per Chinese
CHARACTER against 0.23 for English, and costs 3.15x more bits per BYTE on
Chinese (1.26 -> 3.98). It is not modelling Chinese, it is spelling out UTF-8
byte fragments. BLT costs 1.19x (0.846 -> 1.007). The cross-lingual drift work
has NO surprisal axis for exactly this reason, and adding a gpt2 one would have
produced a language difference that was a tokenizer difference -- the same shape
as `total_drift`'s sentence-count reversal.

BLT COMPETENCE, PROBED BEFORE BUILDING (one sentence per condition, a
demonstration not a measurement):

    ZH natural 1.007  odd-semantics 1.442 (1.43x)  scrambled 3.519  random 5.054
    EN natural 0.846  odd-semantics 1.218 (1.44x)  scrambled 2.056  random 4.948

The semantic swap (friend -> refrigerator, children -> staircases, grammatical
throughout, common characters) costs the SAME RELATIVE amount in both languages.
A character-frequency model without comprehension would not do that.

**BITS PER BYTE, NEVER BITS PER CHARACTER.** A CJK character is ~3 UTF-8 bytes,
so bits/char inflates ~3x for Chinese mechanically and is NOT comparable across
scripts. `build_jakobson.py`'s `_bits_per_char` carries that defect. Byte is the
script-neutral unit and is what this stores.

STORAGE. `{'surprisal': float32, 'token_ids': uint16}` under the existing
`ref_surprisal` stash, keyed {ref, prompt, text} -- text-addressed, which is
correct for a reference scorer since the score is a property of the string, not
of which model emitted it. Measured: the dict is SMALLER than a bare array
(17.04 vs 19.36 KB at byte scale; the low-entropy id array compresses to less
than nothing) as well as self-verifying, since the text can be reconstructed
from the ids and checked. uint16 not uint8: ids are byte+4, range 4-259.

NOT bits-per-byte in f16: it saves 18% for 1.9e-3 nats of error, and the
smallest effect this campaign has measured (the ordering contrast) is 2-4e-3.

Context is 4096 positions, so passages (median 918-1040 bytes) fit whole. The
"1024" warning transformers emits is the TOKENIZER's default model_max_length,
not the model's capacity.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

OUTD = os.path.join(ROOT, "meta/M06_generation/results")
BLT = "itazap/blt-1b-hf"
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")
CTX = 4096
SEED = 20260814


def ch_rows(q):
    o = subprocess.run([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                       capture_output=True, text=True).stdout.strip()
    return [json.loads(l) for l in o.split("\n") if l]


def main():
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from malign_logits.cache import get_cache

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="passages PER LANGUAGE")
    ap.add_argument("--corpus", default="f11_l2")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    cache = get_cache()
    tk = AutoTokenizer.from_pretrained(BLT, trust_remote_code=True)
    tk.model_max_length = CTX
    model = AutoModelForCausalLM.from_pretrained(
        BLT, trust_remote_code=True, dtype=torch.float32).eval()

    #: ROLE IS EMPTY FOR f11_l2 -- all 228,520 rows carry role='' -- so it is
    #: derived from the pair registry the way m06_crosslingual_arms.py does,
    #: not read off the row. The passage corpus populates role; this one does
    #: not, and filtering on it silently returned zero rows.
    pairs = [q for q in json.load(open(os.path.join(ROOT, "data/base_aligned_pairs.json")))
             if not q.get("ambiguous")]
    role_of = {}
    for q in pairs:
        role_of[q["base"]] = "base"
        role_of[q["aligned"]] = "aligned"

    rows = {}
    for lang, pred in (("zh", "length(text) != lengthUTF8(text)"),
                       ("en", "length(text) = lengthUTF8(text)")):
        cand = ch_rows(
            "SELECT model, prompt, sample_idx, text FROM "
            "malign_logits.gen_sequences WHERE corpus='%s' AND forced_word='' "
            "AND %s AND length(text) BETWEEN 200 AND %d "
            "ORDER BY cityHash64(concat(model, prompt, toString(sample_idx))) "
            "LIMIT %d" % (a.corpus, pred, CTX - 64, a.n * 40))
        #: balance the arms explicitly rather than hoping the hash order does it
        picked, want = [], a.n // 2
        for want_role in ("base", "aligned"):
            got = 0
            for r in cand:
                if role_of.get(r["model"]) == want_role:
                    r["role"] = want_role
                    picked.append(r)
                    got += 1
                    if got >= want:
                        break
        rows[lang] = picked
        print("%s: %d passages (%d base / %d aligned) from %d candidates"
              % (lang, len(picked), sum(r["role"] == "base" for r in picked),
                 sum(r["role"] == "aligned" for r in picked), len(cand)))
        if not picked:
            raise SystemExit("REFUSING: no %s passages mapped to a base/aligned "
                             "role. %d candidates, %d models in the registry."
                             % (lang, len(cand), len(role_of)))

    out = {"ref": BLT, "corpus": a.corpus, "n_per_lang": a.n, "unit": "bits_per_byte",
           "ctx": CTX, "languages": {}}
    t0 = time.time()
    for lang in ("zh", "en"):
        recs, n_cached = [], 0
        for i, r in enumerate(rows[lang]):
            text, prompt = r["text"], r["prompt"]
            got = cache.get_ref_surprisal(BLT, prompt, text)
            if got is not None:
                sur = np.asarray(got["surprisal"], dtype=np.float32)
                n_cached += 1
            else:
                ids = tk(text, add_special_tokens=False)["input_ids"]
                with torch.no_grad():
                    lg = model(torch.tensor([ids])).logits[0]
                lp = torch.log_softmax(lg.float(), -1)
                idx = torch.tensor(ids[1:])
                sur = (-lp[:-1].gather(1, idx[:, None]).squeeze(1)).numpy().astype(np.float32)
                cache.set_ref_surprisal(BLT, prompt, text, {
                    "surprisal": sur,
                    "token_ids": np.asarray(ids, dtype=np.uint16)})
            nb = len(text.encode())
            recs.append({"model": r["model"], "role": r["role"], "prompt": prompt,
                         "sample_idx": r["sample_idx"], "n_bytes": nb,
                         "n_chars": len(text),
                         "bits_per_byte": float(sur.sum() / np.log(2) / nb),
                         "bits_per_char": float(sur.sum() / np.log(2) / len(text))})
            if (i + 1) % 25 == 0:
                print("  %s %d/%d  (%.1f min, %d cached)"
                      % (lang, i + 1, len(rows[lang]), (time.time() - t0) / 60, n_cached))
        import statistics as st
        by = {}
        for role in ("base", "aligned"):
            v = [x["bits_per_byte"] for x in recs if x["role"] == role]
            by[role] = {"n": len(v), "median": st.median(v) if v else None,
                        "mean": st.fmean(v) if v else None}
        out["languages"][lang] = {
            "n": len(recs), "n_cached": n_cached,
            "bits_per_byte_median": st.median([x["bits_per_byte"] for x in recs]),
            "bits_per_char_median": st.median([x["bits_per_char"] for x in recs]),
            "bytes_per_char": st.fmean([x["n_bytes"] / x["n_chars"] for x in recs]),
            "by_role": by,
            "arm_delta_bits_per_byte": (by["aligned"]["median"] - by["base"]["median"])
            if by["base"]["median"] is not None and by["aligned"]["median"] is not None else None}
        r_ = out["languages"][lang]
        print("%s: bits/byte median %.4f (bits/char %.4f, %.2f bytes/char) | "
              "base %s aligned %s -> delta %s"
              % (lang, r_["bits_per_byte_median"], r_["bits_per_char_median"],
                 r_["bytes_per_char"],
                 "%.4f" % by["base"]["median"] if by["base"]["median"] else "n/a",
                 "%.4f" % by["aligned"]["median"] if by["aligned"]["median"] else "n/a",
                 "%+.4f" % r_["arm_delta_bits_per_byte"] if r_["arm_delta_bits_per_byte"] else "n/a"))

    out["elapsed_min"] = (time.time() - t0) / 60
    os.makedirs(OUTD, exist_ok=True)
    p = os.path.join(OUTD, "blt_surprisal_pilot.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nelapsed %.1f min -> %s" % (out["elapsed_min"], p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
