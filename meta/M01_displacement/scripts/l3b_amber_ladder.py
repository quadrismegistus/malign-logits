"""L3-b PILOT: kill -> scream through the layers, on the Amber ladder, licensed by twp.

    uv run python l3b_amber_ladder.py

WHAT THIS FIXES ABOUT THE EARLIER PILOT. `l3_pilot_displacement.py` mapped the
model's final norm over EVERY hidden state, and HuggingFace's last hidden state
is already normed, so its final-layer row -- the row every headline was read off
-- was double-normed. `malign_logits.models.logit_lens` had the same defect and
now refuses unless its final layer reproduces the model's own logits. This pilot
uses that function, so the check runs on every model here.

THE PROBLEM THIS PILOT IS ABOUT. A logit lens reads TOKENS; displacement is
about WORDS. On Amber " scream" is ['sc', 'ream'], so a lens can only watch
'sc' -- a prefix shared with scare, scratch, scold. `kill` is one token and
needs no licensing; `scream`, the riser the whole finding turns on, does.

    p_L('sc')                     what the lens can actually read
    p_L('ream' | prompt + 'sc')   THE LICENSING FACTOR: if you read out at
                                  layer L, does 'sc' become 'scream'?
    product                       the early-exit word estimate

**THE LICENSING FACTOR IS ITSELF PER-LAYER AND THAT IS THE POINT.** Reporting
"at the output 'sc' is 97.6% scream" and then reading the whole trajectory as
scream assumes a constant nobody measured. Two forward passes measure it: one
pass yields every layer at once, so this costs what twp costs, not n_layers
times it.

WHAT THE PRODUCT IS AND IS NOT. Appending 'sc' and running forward inserts a
decode-and-re-encode the real network never performs at layer L: the model at
layer 7 does not emit a token, it hands a residual to layer 8. So the product is
**what a model exiting at layer L would say**, the early-exit reading, not what
layer L represents. Named here so no table below has to carry the caveat.

VALIDATION. At the final layer both quantities are known independently:
`true_word_probs` holds p(kill) and p(scream) for these exact cells. If the
lens's final layer does not reproduce them, nothing per-layer is readable.

t1 COMES FROM twp, NOT FROM `tokenizer.encode`. The stash records the first
token per word under the campaign's own boundary rule; re-deriving it is a
second policy, and `logit_lens_words`'s `ids[0]` is what that costs.

NOT A FINDING. One prompt, one family, three checkpoints, no control vocabulary.
The null -- how far an ARBITRARY word moves through a stack -- is still missing
and is the thing to build before any of this is citable.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

PROMPT = "She was so angry she wanted to"
LADDER = [("base", "LLM360/Amber"), ("SFT", "LLM360/AmberChat"), ("DPO", "LLM360/AmberSafe")]
TRACK = ("kill", "scream")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from malign_logits.models import logit_lens
    from malign_logits.cache import CacheManager

    cm = CacheManager()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device %s   prompt %r\n" % (dev, PROMPT))

    out = {}
    for arm, mid in LADDER:
        twp = cm.get_true_word_probs(mid, PROMPT)
        if twp is None:
            raise SystemExit("no twp cell for %s" % mid)
        rows = {r["word"]: r for r in twp["rows"]}
        tok = AutoTokenizer.from_pretrained(mid)

        #: t1 FROM THE STASH. For a multi-token word this is its first piece.
        t1 = {w: rows[w]["t1"] for w in TRACK if w in rows}
        pieces = {w: tok.encode(" " + w, add_special_tokens=False) for w in TRACK}
        multi = [w for w in TRACK if len(pieces.get(w, [])) > 1]

        mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(dev).eval()
        lens = [torch.softmax(v.float(), -1) for v in logit_lens(mdl, tok, PROMPT)]

        #: SECOND PASS for the licensing factor, on prompt + the first piece.
        #: ROUND-TRIP GUARD: appending the decoded piece must give exactly the
        #: prompt's ids plus t1, or the second pass measures a different string.
        lens2 = {}
        #: BUILD THE IDS, THEN DECODE THE WHOLE SEQUENCE. Decoding the piece
        #: alone and concatenating strings does not work: token 885 is ' sc'
        #: with a leading space that `decode([885])` strips, so PROMPT + 'sc'
        #: is "...wanted tosc" and tokenises to 1557 instead. The guard caught
        #: exactly that, which is what it is for.
        for w in multi:
            want = tok.encode(PROMPT) + [t1[w]]
            ext = tok.decode(want, skip_special_tokens=True)
            got = tok.encode(ext)
            if got != want:
                raise SystemExit("round-trip failed for %r: %s != %s" % (ext, got, want))
            lens2[w] = [torch.softmax(v.float(), -1) for v in logit_lens(mdl, tok, ext)]
        del mdl
        if dev == "mps":
            torch.mps.empty_cache()

        out[arm] = {"lens": lens, "lens2": lens2, "t1": t1, "pieces": pieces,
                    "twp": {w: rows[w]["p"] for w in TRACK if w in rows}, "tok": tok}
        print("%-4s %-22s layers=%d   twp: %s"
              % (arm, mid, len(lens),
                 "  ".join("%s %.6f%s" % (w, rows[w]["p"],
                                          "" if len(pieces[w]) == 1 else " [%d tok]" % len(pieces[w]))
                           for w in TRACK if w in rows)))

    print("\n" + "=" * 92)
    print("VALIDATION AT THE FINAL LAYER, against true_word_probs")
    print("=" * 92)
    print("   %-5s %-9s %12s %12s %8s   %s" % ("arm", "word", "lens", "twp", "ratio", "note"))
    for arm, _ in LADDER:
        d = out[arm]
        for w in TRACK:
            if w not in d["twp"]:
                continue
            n = len(d["pieces"][w])
            p = float(d["lens"][-1][d["t1"][w]])
            if n > 1:
                p_full = p * float(d["lens2"][w][-1][d["pieces"][w][1]])
                print("   %-5s %-9s %12.6f %12.6f %8.4f   product, t1=%r x next"
                      % (arm, w, p_full, d["twp"][w], p_full / d["twp"][w],
                         d["tok"].decode([d["t1"][w]])))
                print("   %-5s %-9s %12.6f %12.6f %8.4f   first token only"
                      % ("", "", p, d["twp"][w], p / d["twp"][w]))
            else:
                print("   %-5s %-9s %12.6f %12.6f %8.4f   single token, exact"
                      % (arm, w, p, d["twp"][w], p / d["twp"][w]))

    print("\n" + "=" * 92)
    print("THE TRAJECTORY. `sc->ream` is the per-layer licensing factor.")
    print("=" * 92)
    for arm, _ in LADDER:
        d = out[arm]
        nL = len(d["lens"])
        print("\n   %s" % arm)
        print("   %5s %11s %11s %11s %11s" % ("layer", "kill", "p(sc)", "sc->ream", "= scream"))
        for L in range(nL):
            k = float(d["lens"][L][d["t1"]["kill"]]) if "kill" in d["t1"] else float("nan")
            s = float(d["lens"][L][d["t1"]["scream"]]) if "scream" in d["t1"] else float("nan")
            if "scream" in d["lens2"]:
                lic = float(d["lens2"]["scream"][L][d["pieces"]["scream"][1]])
            else:
                lic = float("nan")
            if L >= nL - 6 or L % 8 == 0:
                print("   %5d %11.6f %11.6f %11.3f %11.6f%s"
                      % (L, k, s, lic, s * lic, "   <- output" if L == nL - 1 else ""))

    print("\n" + "=" * 92)
    print("WHERE THE PROXY IS TRUSTWORTHY")
    print("=" * 92)
    for arm, _ in LADDER:
        d = out[arm]
        if "scream" not in d["lens2"]:
            continue
        lic = [float(d["lens2"]["scream"][L][d["pieces"]["scream"][1]])
               for L in range(len(d["lens"]))]
        good = [L for L, x in enumerate(lic) if x >= 0.5]
        print("   %-5s licensing >= 0.50 at layers %s   (of 0..%d)"
              % (arm, good if good else "NONE", len(lic) - 1))
        print("   %-5s at the output %.3f;  max over interior layers %.3f at layer %d"
              % ("", lic[-1], max(lic[:-1]), lic[:-1].index(max(lic[:-1]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
